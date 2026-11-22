"""Inbox queries and cursor writes, independent of notification presentation."""

from __future__ import annotations


class InboxRepository:
    def __init__(self, connection):
        self._connection = connection

    def cursor(self, agent_id):
        row = self._connection.execute(
            "SELECT value FROM metadata WHERE key=?", (f"inbox_cursor:{agent_id}",)
        ).fetchone()
        return int(row[0]) if row else 0

    def append(
        self, *, recipient, sender, kind, source_ref, message, wake, event_id, page_ref=None
    ):
        """Append inside the caller's source-event transaction; never commit it here."""
        self._connection.execute(
            "INSERT OR IGNORE INTO inbox_messages (recipient,sender,kind,source_ref,"
            "page_ref,message,wake_requested,event_id) VALUES (?,?,?,?,?,?,?,?)",
            (recipient, sender, kind, source_ref, page_ref, message, int(wake), event_id),
        )

    def pending(self, agent_id, after):
        """Stream unread entries so the presentation layer can stop at its byte limit."""
        rows = self._connection.execute(
            "SELECT m.*, c.id AS comment_id, c.page_id, c.reply_to "
            "FROM inbox_messages m "
            "LEFT JOIN page_comments c ON m.source_ref LIKE 'comment:%' "
            "AND c.id=CAST(substr(m.source_ref,9) AS INTEGER) "
            "WHERE m.recipient=? AND m.id>? ORDER BY m.id",
            (agent_id, after),
        )
        return (dict(row) for row in rows)

    def acknowledge(self, agent_id, through):
        """Advance monotonically only after the peer successfully receives a response."""
        if type(through) is not int or through < 0:
            raise ValueError("inbox cursor must be a nonnegative integer")
        with self._connection:
            self._connection.execute(
                "INSERT INTO metadata(key,value) VALUES (?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=MAX(metadata.value,excluded.value)",
                (f"inbox_cursor:{agent_id}", through),
            )

    def changed_pages(self, agent_id, since_tick, through_tick, limit):
        """Peer revisions and unaddressed comments share one bounded delivery window."""
        rows = [
            dict(row)
            for row in self._connection.execute(
                """
                WITH edits AS (
                    SELECT r.page_id,r.revision,r.created_tick,r.author_id,
                           ROW_NUMBER() OVER (PARTITION BY r.page_id
                                              ORDER BY r.revision DESC) AS position
                    FROM page_revisions r
                    WHERE r.created_tick>? AND r.created_tick<=? AND r.author_id<>?
                ), comments AS (
                    SELECT c.page_id,COUNT(*) AS comments,MAX(c.id) AS latest_comment_id,
                           MAX(c.created_tick) AS created_tick
                    FROM page_comments c
                    WHERE c.created_tick>? AND c.created_tick<=? AND c.author_id<>?
                    AND NOT EXISTS (SELECT 1 FROM inbox_messages m
                                    WHERE m.recipient=? AND m.source_ref='comment:'||c.id)
                    GROUP BY c.page_id
                ), changed AS (
                    SELECT page_id FROM edits WHERE position=1
                    UNION SELECT page_id FROM comments
                )
                SELECT p.id AS page_id,p.title,e.revision,e.author_id,
                       COALESCE(c.comments,0) AS comments,c.latest_comment_id,
                       MAX(COALESCE(e.created_tick,0),COALESCE(c.created_tick,0)) AS created_tick,
                       COUNT(*) OVER () AS total_pages
                FROM changed JOIN shared_pages p ON p.id=changed.page_id
                LEFT JOIN edits e ON e.page_id=p.id AND e.position=1
                LEFT JOIN comments c ON c.page_id=p.id
                WHERE p.project_id IS NULL OR p.project_id IN
                      (SELECT project_id FROM project_members WHERE agent_id=?)
                ORDER BY created_tick ASC,p.id LIMIT ?
                """,
                (
                    since_tick,
                    through_tick,
                    agent_id,
                    since_tick,
                    through_tick,
                    agent_id,
                    agent_id,
                    agent_id,
                    limit,
                ),
            )
        ]
        total = rows[0]["total_pages"] if rows else 0
        return rows, total - len(rows)
