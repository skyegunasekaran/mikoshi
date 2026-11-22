"""Addressed inbox messages and one shared change feed per peer.

Inbox cursors advance after successful delivery. Page changes use the last
successfully delivered snapshot's tick, so retries and compaction cannot lose updates.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace

from mikoshi.problem import ScheduledNotificationSpec
from mikoshi.runtime.budgets import TIME_REMINDER_PERCENTAGES

EXCERPT_CHARS = 2000
DIGEST_CHARS = 6000
CHANGED_PAGE_LIMIT = 20
SCHEMA = """
CREATE TABLE inbox_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    recipient TEXT NOT NULL REFERENCES agents(id),
    sender TEXT NOT NULL REFERENCES agents(id),
    kind TEXT NOT NULL,
    source_ref TEXT NOT NULL,
    page_ref TEXT,
    message TEXT NOT NULL,
    wake_requested INTEGER NOT NULL CHECK(wake_requested IN (0,1)),
    event_id INTEGER NOT NULL REFERENCES events(id),
    UNIQUE(recipient,event_id,source_ref)
);
CREATE INDEX inbox_recipient_order ON inbox_messages(recipient,id);
"""


class Notifications:
    def __init__(self, store):
        self.store = store
        self._inbox = store.inbox

    def cursor(self, agent_id):
        return self._inbox.cursor(agent_id)

    def deliver(
        self, *, sender, kind, source_ref, message, recipients, wake, event_id, page_ref=None
    ):
        """Append atomically with the source event; caller owns the transaction."""
        from mikoshi.runtime.organization import wake_inbox

        for recipient in sorted(set(recipients)):
            self._inbox.append(
                recipient=recipient,
                sender=sender,
                kind=kind,
                source_ref=source_ref,
                page_ref=page_ref,
                message=message,
                wake=wake,
                event_id=event_id,
            )
            if wake:
                wake_inbox(self.store, recipient)

    def _item(self, row):
        """An addressed comment preview or a complete scheduled operator message."""
        item = {"kind": row["kind"], "sender": row["sender"]}
        if row["comment_id"] is not None:
            item["comment_id"] = row["comment_id"]
            item["path"] = self.store.page_path(row["page_id"])
            if row["reply_to"]:
                item["reply_to"] = row["reply_to"]
        else:
            item["reference"] = row["source_ref"]
        # Scheduled messages are delivered in full; comments longer than the preview are cut.
        limit = None if row["kind"] == "scheduled_message" else EXCERPT_CHARS
        item["message"] = row["message"][:limit]
        if limit is not None and len(row["message"]) > limit:
            item["truncated"] = True
        return item

    def digest(self, agent_id):
        """New inbox entries, plus each entry's audit source for decision evidence."""
        self.store.get_agent(agent_id)
        through = self.cursor(agent_id)
        packet, sources = {"updates": [], "has_more": False}, []
        for row in self._inbox.pending(agent_id, through):
            item = self._item(row)
            candidate = {"updates": [*packet["updates"], item], "has_more": False}
            # The first entry is always delivered so the cursor advances; later ones may wait.
            if packet["updates"] and len(json.dumps(candidate)) > DIGEST_CHARS:
                packet["has_more"] = True
                break
            packet = candidate
            sources.append(
                {
                    "id": row["id"],
                    "source_ref": row["source_ref"],
                    "truncated": item.get("truncated", False),
                }
            )
            through = row["id"]
        return packet, through, sources

    def acknowledge(self, agent_id, through):
        self.store.get_agent(agent_id)
        self._inbox.acknowledge(agent_id, through)

    def changes(self, agent_id, since_tick, through_tick):
        """References to peer edits and unaddressed public comments since delivery.

        Pages come oldest change first. When more pages changed than fit, the window
        ends at the last delivered page, so the rest arrive in the next observation.
        A deferred page counts only the comments after that point.
        """
        rows, remaining = self._inbox.changed_pages(
            agent_id, since_tick, through_tick, CHANGED_PAGE_LIMIT
        )
        if remaining:
            through_tick = rows[-1]["created_tick"]
        pages = []
        for row in reversed(rows):
            item = {
                "path": self.store.page_path(row["page_id"]),
                "title": row["title"][:200],
                "tick": row["created_tick"],
            }
            if row["revision"] is not None:
                item.update(revision=row["revision"], author_id=row["author_id"])
            if row["comments"]:
                item.update(comments=row["comments"], latest_comment_id=row["latest_comment_id"])
            pages.append(item)
        return {
            "since_tick": since_tick,
            "through_tick": through_tick,
            "pages": pages,
            "has_more": bool(remaining),
        }


TIME_REMINDER_PREFIX = "time_remaining_"
TIME_REMINDER_IDS = frozenset(
    f"{TIME_REMINDER_PREFIX}{percent}_percent" for percent in TIME_REMINDER_PERCENTAGES
)


def with_time_reminders(problem, seconds):
    """Resolve deadline-relative reminders before persisting a run's problem.

    Applies to every finite wall-time budget. Explicit scheduled messages remain
    separate. Stable IDs make repeated resolution idempotent and prevent ambiguous
    reminder overrides.
    """
    if seconds is None:
        return problem
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("time reminders require a finite positive wall-time budget")
    notifications = list(problem.scheduled_notifications)
    by_id = {item.id: item for item in notifications}
    for percent in TIME_REMINDER_PERCENTAGES:
        item = ScheduledNotificationSpec(
            id=f"{TIME_REMINDER_PREFIX}{percent}_percent",
            at_minutes=seconds * ((100 - percent) / 100) / 60,
            recipient="all",
            message=f"{percent}% of the time limit remains ({seconds * percent / 6000:g} minutes).",
        )
        if item.id in by_id:
            if by_id[item.id] != item:
                raise ValueError(
                    f"scheduled notification id {item.id!r} is reserved for time reminders"
                )
        else:
            notifications.append(item)
    return replace(problem, scheduled_notifications=tuple(notifications))


class ScheduledNotifications:
    def __init__(self, store, organization, budget):
        self.store = store
        self.db = store._connection
        self.budget = budget
        self.schedule = organization.problem.get("scheduled_notifications", [])
        if self.schedule and budget is None:
            raise ValueError("scheduled notifications require an execution budget clock")

    def _load(self, key, default):
        row = self.db.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def _save(self, key, value):
        self.db.execute(
            "INSERT INTO metadata(key,value) VALUES (?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, json.dumps(value)),
        )

    @property
    def pending(self):
        released = self._load("scheduled_notification_releases", {})
        return any(item["id"] not in released for item in self.schedule)

    def release_due(self):
        if not self.schedule:
            return
        released = self._load("scheduled_notification_releases", {})
        elapsed = self.budget.elapsed
        for item in sorted(self.schedule, key=lambda item: item["at_minutes"]):
            if item["id"] in released or elapsed / 60 < item["at_minutes"]:
                continue
            recipients = [
                row[0]
                for row in self.db.execute("SELECT id FROM agents WHERE role='peer' ORDER BY id")
                if item["recipient"] in {"all", row[0]}
            ]
            with self.db:
                self.db.execute(
                    "INSERT OR IGNORE INTO agents VALUES "
                    "('system_events','External events','Runtime-published scheduled messages.','environment')"
                )
                trigger = self.db.execute(
                    "INSERT INTO triggers(agent_id,reason,priority,created_tick,status,kind) "
                    "VALUES ('system_events','scheduled notification',0,?,'completed','environment')",
                    (self.store.current_tick + 1,),
                ).lastrowid
                activation = self.db.execute(
                    "INSERT INTO activations(agent_id,trigger_id,started_tick,causal_wave_id,status) "
                    "VALUES ('system_events',?,?,0,'environment')",
                    (trigger, self.store.current_tick + 1),
                ).lastrowid
                event, tick = self.store._record_event(
                    activation,
                    "system_events",
                    "scheduled_notification",
                    "succeeded",
                    None,
                    {
                        **item,
                        "recipients": recipients,
                        "released_at_seconds": elapsed,
                    },
                )
                self.db.execute(
                    "UPDATE activations SET ended_tick=?,action_count=1 WHERE id=?",
                    (tick, activation),
                )
                released[item["id"]] = {
                    "released_at_seconds": elapsed,
                    "recipients": recipients,
                    "event_id": event,
                }
                self._save("scheduled_notification_releases", released)
                Notifications(self.store).deliver(
                    sender="system_events",
                    kind="scheduled_message",
                    source_ref=f"event:{event}",
                    message=item["message"],
                    recipients=recipients,
                    # Time reminders are informational: idle peers see them if something
                    # else wakes them. Explicit scheduled messages still wake.
                    wake=item["id"] not in TIME_REMINDER_IDS,
                    event_id=event,
                )
            self.store._publish_committed_event(event)
