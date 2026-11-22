"""Shared Markdown knowledge and live page discussions.

SQLite commits Markdown revisions, comments and their events together.
The pages/ directory is a rebuildable Markdown export; CAS retains exact tool evidence.
Pages use Markdown bodies, attributed comments and exact-match edits.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid

from mikoshi.contracts import TEXT_FIELD_CHARS, action_schema
from mikoshi.tools.page_search import search_pages
from mikoshi.tools.text_tools import apply_edits
from mikoshi.tools.workspace import WorkspaceAccessError

PAGE_NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_. -]{0,99}")

PAGE_ACTIONS = frozenset(
    (
        "list_pages",
        "search_pages",
        "view_page",
        "add_page",
        "edit_page",
        "comment_page",
        "read_comment",
    )
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS shared_pages (
    id TEXT PRIMARY KEY COLLATE NOCASE, project_id TEXT REFERENCES projects(id),
    title TEXT NOT NULL,
    revision INTEGER NOT NULL, created_tick INTEGER NOT NULL,
    parent_id TEXT REFERENCES shared_pages(id), name TEXT NOT NULL COLLATE NOCASE
);
CREATE UNIQUE INDEX IF NOT EXISTS page_sibling_names
    ON shared_pages(COALESCE(parent_id, ''), name);
CREATE TABLE IF NOT EXISTS page_snapshots (
    token TEXT NOT NULL, page_id TEXT NOT NULL, revision INTEGER NOT NULL,
    sections TEXT NOT NULL, agent_id TEXT NOT NULL REFERENCES agents(id),
    PRIMARY KEY(agent_id, page_id)
);
CREATE TABLE IF NOT EXISTS page_revisions (
    page_id TEXT NOT NULL REFERENCES shared_pages(id), revision INTEGER NOT NULL,
    markdown TEXT NOT NULL, author_id TEXT REFERENCES agents(id),
    created_tick INTEGER NOT NULL, PRIMARY KEY(page_id, revision)
);
CREATE INDEX IF NOT EXISTS page_revisions_changed ON page_revisions(created_tick);
CREATE TABLE IF NOT EXISTS page_comments (
    id INTEGER PRIMARY KEY AUTOINCREMENT, page_id TEXT NOT NULL REFERENCES shared_pages(id),
    page_revision INTEGER NOT NULL, reply_to INTEGER REFERENCES page_comments(id),
    author_id TEXT NOT NULL REFERENCES agents(id), markdown TEXT NOT NULL,
    created_tick INTEGER NOT NULL
);
"""


def records(store):
    """Evaluator/export view. Never pass this unrestricted view to a worker."""
    return {
        name: [
            dict(row) for row in store._connection.execute(f"SELECT * FROM {name} ORDER BY rowid")
        ]
        for name in (
            "shared_pages",
            "page_revisions",
            "page_comments",
            "page_snapshots",
        )
    }


def page_path(db, page_id):
    """The path peers use for a stored Markdown page."""
    names, current = [], page_id
    while current:
        row = db.execute(
            "SELECT parent_id, name FROM shared_pages WHERE id=?", (current,)
        ).fetchone()
        if row is None:
            raise ValueError(f"page {current} does not exist")
        names.append(row["name"])
        current = row["parent_id"]
    return "/" + "/".join(reversed(names))


class SharedPages:
    def __init__(self, store, workspace):
        self.store, self.workspace = store, workspace
        self.db = store._connection
        self._exported = {}
        self._dirty_pages = set()
        store.observe_committed_events(self._project_event)

    def seed(self):
        if self.db.execute("SELECT 1 FROM shared_pages LIMIT 1").fetchone():
            self.export()
            return
        from mikoshi.runtime.organization import config

        organization = config(self.store)
        if organization is None:
            raise ValueError("shared workspace seeding requires a problem-first organization")
        objective = json.loads(organization["problem_json"])["objective"]
        with self.db:
            self._seed_page(
                "objective",
                None,
                objective["title"],
                f"# {objective['title']}\n\n{objective['description']}\n\n## Constraints\n\n"
                + "\n".join(f"- {item}" for item in objective["constraints"]),
            )
            self._seed_page(
                "coordination",
                None,
                "Coordination",
                "# Coordination\n\nA shared space for plans, findings, and discussion. "
                "Everyone can edit this page and organize it together.\n",
            )
        self.export()

    def _seed_page(self, page_id, project_id, title, markdown, *, parent_id=None, name=None):
        self.db.execute(
            "INSERT INTO shared_pages (id,project_id,title,revision,created_tick,parent_id,name) "
            "VALUES (?, ?, ?, 1, ?, ?, ?)",
            (page_id, project_id, title, self.store.current_tick, parent_id, name or page_id),
        )
        self.db.execute(
            "INSERT INTO page_revisions VALUES (?, 1, ?, NULL, ?)",
            (page_id, markdown, self.store.current_tick),
        )

    def _member(self, agent_id, project_id):
        self.store.get_agent(agent_id)
        if project_id is not None and not self.store.is_project_member(agent_id, project_id):
            raise WorkspaceAccessError(f"{agent_id} is not a member of this page's project")

    def _page(self, agent_id, page_id):
        row = self.db.execute("SELECT * FROM shared_pages WHERE id=?", (page_id,)).fetchone()
        if row is None:
            raise ValueError(f"page {page_id} does not exist")
        self._member(agent_id, row["project_id"])
        return dict(row)

    def _actor(self, state):
        row = self.db.execute(
            "SELECT agent_id, status FROM activations WHERE id=?", (state.activation_id,)
        ).fetchone()
        if row is None or row["status"] != "running" or row["agent_id"] != state.trigger.agent_id:
            raise WorkspaceAccessError("page tools need an active activation")

    def entry_points(self, agent_id):
        return [
            self._reference(page)
            for page in self._accessible(agent_id)
            if page["id"] in {"objective", "coordination"}
        ]

    def _accessible(self, agent_id):
        self.store.get_agent(agent_id)
        pages = [
            dict(row)
            for row in self.db.execute(
                """
            SELECT p.* FROM shared_pages p WHERE p.project_id IS NULL OR EXISTS (
                SELECT 1 FROM project_members m WHERE m.project_id=p.project_id AND m.agent_id=?)
             ORDER BY p.id""",
                (agent_id,),
            )
        ]
        return pages

    def context(self, state):
        return {"entry_points": self.entry_points(state.trigger.agent_id)}

    def schemas(self, content_limit):
        text = {"type": "string", "minLength": 1}
        path = {
            **text,
            "maxLength": 1000,
            "description": "Page path, like /objective or /notes/plan.",
        }
        body = {**text, "maxLength": min(content_limit, self.workspace.limits.max_file_bytes // 4)}

        def offset(description):
            return {"type": "integer", "minimum": 0, "description": description}

        listing = offset("Pass next_offset from the previous result to continue.")
        characters = offset(
            "Character to start from. Pass next_offset from the previous result to continue."
        )
        peers = {
            "type": "array",
            "maxItems": 256,
            "items": {"type": "string", "minLength": 1, "maxLength": 100},
        }
        actions = [
            action_schema(
                "list_pages",
                {
                    "path": {**path, "description": "Parent page path; / by default."},
                    "offset": listing,
                },
                required=[],
            ),
            action_schema(
                "search_pages",
                {
                    "query": {**text, "maxLength": 200, "description": "Words to search for."},
                    "path": {**path, "description": "Only search under this page; / by default."},
                    "offset": listing,
                },
                required=["query"],
            ),
            action_schema("view_page", {"path": path, "offset": characters}, required=["path"]),
            action_schema(
                "add_page",
                {
                    "path": {
                        **path,
                        "description": "Path of the new page, like /notes/plan. "
                        "The parent page must exist.",
                    },
                    "title": {**text, "maxLength": 200},
                    "markdown": {**body, "description": "Page body in Markdown."},
                },
                required=["path", "title", "markdown"],
            ),
            action_schema(
                "edit_page",
                {
                    "path": path,
                    "edits": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 50,
                        "description": "Replacements, each matched against the page as it was "
                        "before this edit.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "old_text": {
                                    **body,
                                    "description": "Exact text to replace; must occur once.",
                                },
                                "new_text": {
                                    "type": "string",
                                    "maxLength": body["maxLength"],
                                    "description": "Replacement text.",
                                },
                            },
                            "required": ["old_text", "new_text"],
                            "additionalProperties": False,
                        },
                    },
                },
                required=["path", "edits"],
            ),
            action_schema(
                "comment_page",
                {
                    "path": path,
                    "markdown": {
                        **text,
                        "maxLength": TEXT_FIELD_CHARS,
                        "description": "Public comment text in Markdown.",
                    },
                    "reply_to": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Comment ID to reply to. Omit for a new thread.",
                    },
                    "notify": {
                        **peers,
                        "description": "Exact roster peer IDs to notify and wake. "
                        "Omit to post without sending notifications.",
                    },
                },
                required=["path", "markdown"],
            ),
            action_schema(
                "read_comment",
                {"comment_id": {"type": "integer", "minimum": 1}, "offset": characters},
                required=["comment_id"],
            ),
        ]

        descriptions = {
            "list_pages": "List the pages directly under a page, 20 at a time.",
            "search_pages": "Search page text and comments under a page, 20 results at a time.",
            "view_page": "Read a page and its public comments. Long pages come in parts; "
            "offset continues the view you opened "
            "without an offset, even if the page has changed since.",
            "add_page": "Create a Markdown page for plans, notes, shared boards or anything else.",
            "edit_page": "Replace exact text in a page's markdown. Comments can't be edited.",
            "comment_page": "Post a public comment or threaded reply on a page. Only peers "
            "listed in notify receive an inbox message and are activated when idle.",
            "read_comment": "Read one comment in full.",
        }
        for schema in actions:
            schema["description"] = descriptions[schema["properties"]["action"]["enum"][0]]
        return actions

    def _resolve(self, state, path):
        actor = state.trigger.agent_id
        if not isinstance(path, str) or not path or len(path) > 1000:
            raise ValueError("path must contain 1 to 1000 characters")
        current = None
        for part in path.split("/"):
            if part in ("", "."):
                continue
            if part == "..":
                current = self._page(actor, current)["parent_id"] if current else None
                continue
            row = self.db.execute(
                "SELECT * FROM shared_pages WHERE parent_id IS ? AND name=?",
                (current, part),
            ).fetchone()
            if row is None:
                raise ValueError(f"no page at {path}; list_pages shows the pages that exist")
            self._member(actor, row["project_id"])
            current = row["id"]
        return self._page(actor, current) if current else None

    def _ancestors(self, page):
        """Walk the stored parent chain, including the selected page."""
        while page:
            yield page
            row = self.db.execute(
                "SELECT * FROM shared_pages WHERE id=?", (page["parent_id"],)
            ).fetchone()
            page = dict(row) if row else None

    def _path(self, page):
        return page_path(self.db, page["id"])

    def _reference(self, page):
        return {
            "path": self._path(page),
            "title": page["title"][:200],
            "revision": page["revision"],
        }

    def _destination(self, state, path):
        if not isinstance(path, str) or not path.strip("/"):
            raise ValueError("path must name the new page, like /notes/plan")
        parent_path, _, name = path.rstrip("/").rpartition("/")
        if name in (".", "..") or not PAGE_NAME.fullmatch(name):
            raise ValueError(
                "page names must be 1 to 100 letters, digits, spaces, dots, hyphens or "
                "underscores, and start with a letter, digit or underscore"
            )
        parent = self._resolve(state, parent_path or ("/" if path.startswith("/") else "."))
        if self.db.execute(
            "SELECT 1 FROM shared_pages WHERE parent_id IS ? AND name=?",
            (parent["id"] if parent else None, name),
        ).fetchone():
            raise ValueError(f"a page already exists at {path}; use edit_page to change it")
        return parent, name

    def _descends_from(self, page, parent_id):
        return any(ancestor["id"] == parent_id for ancestor in self._ancestors(page))

    def _creation_scope(self, state):
        """Creating a page never grants new project membership or global visibility."""
        from mikoshi.runtime.organization import config

        scope = state.trigger.project_id
        if scope is None:
            organization = config(self.store)
            scope = organization["scope_id"] if organization else None
        if scope is None:
            raise WorkspaceAccessError("page creation requires an authorized project scope")
        self._member(state.trigger.agent_id, scope)
        return scope

    def _sections(self, page):
        row = self.db.execute(
            "SELECT * FROM page_revisions WHERE page_id=? AND revision=?",
            (page["id"], page["revision"]),
        ).fetchone()
        if row is None:
            raise ValueError(f"revision {page['revision']} of {self._path(page)} is missing")
        comments = list(
            self.db.execute(
                "SELECT * FROM page_comments WHERE page_id=? ORDER BY id", (page["id"],)
            )
        )
        text = "\n\n## Comments\n" if comments else ""
        for comment in comments:
            details = [f"by @{comment['author_id']}", f"on revision {comment['page_revision']}"]
            if comment["reply_to"]:
                details.append(f"reply to comment {comment['reply_to']}")
            text += (
                f"\n### Comment {comment['id']}, {', '.join(details)}\n\n{comment['markdown']}\n"
            )
        return {"markdown": row["markdown"], "comments": text}

    def render(self, page):
        return "".join(self._sections(page).values())

    def _page_path(self, page):
        return f"pages/{page['id']}.md"

    def _capture(self, page):
        return self.workspace._capture_bytes(self._page_path(page), self.render(page).encode())

    def _append_comment(self, page, author, markdown, *, reply_to=None):
        """Append a validated public comment; the caller owns the transaction."""
        cursor = self.db.execute(
            """INSERT INTO page_comments
            (page_id, page_revision, reply_to, author_id, markdown, created_tick)
            VALUES (?, ?, ?, ?, ?, ?)""",
            (
                page["id"],
                page["revision"],
                reply_to,
                author,
                markdown,
                self.store.current_tick + 1,
            ),
        )
        return int(cursor.lastrowid)

    def _reply_parent(self, page, decision):
        reply_to = decision.get("reply_to")
        if reply_to is None:
            return None
        parent = self.db.execute(
            "SELECT * FROM page_comments WHERE id=? AND page_id=?",
            (reply_to, page["id"]),
        ).fetchone()
        if parent is None:
            raise ValueError(f"comment {reply_to} is not a comment on this page")
        return parent

    def _view(self, page, decision, actor):
        """Offset 0 opens a fresh view; a later offset continues that same view."""
        offset = decision.get("offset", 0)
        saved = None
        if offset:
            saved = self.db.execute(
                "SELECT * FROM page_snapshots WHERE page_id=? AND agent_id=?",
                (page["id"], actor),
            ).fetchone()
        if saved is not None:
            revision, token = saved["revision"], saved["token"]
            sections = json.loads(saved["sections"])
        else:
            revision = page["revision"]
            sections = self._sections(page)
            serialized = json.dumps(sections, ensure_ascii=False, sort_keys=True)
            token = hashlib.sha256(
                f"{actor}:{page['id']}:{revision}:{serialized}".encode()
            ).hexdigest()
            self.db.execute(
                "INSERT OR REPLACE INTO page_snapshots VALUES (?, ?, ?, ?, ?)",
                (token, page["id"], revision, serialized, actor),
            )
        total = sum(map(len, sections.values()))
        if offset > total:
            raise ValueError(f"offset {offset} is past the end of the page ({total} characters)")
        limit = self.workspace.limits.max_read_chars
        result = {
            **self._reference(page),
            "revision": revision,
            "offset": offset,
            "next_offset": offset + limit if offset + limit < total else None,
        }
        position = 0
        for key in ("markdown", "comments"):
            value = sections[key]
            start, end = max(0, offset - position), max(0, offset + limit - position)
            if key == "markdown" or value[start:end]:
                result[key] = value[start:end]
            position += len(value)
        observed = "".join(result.get(key, "") for key in ("markdown", "comments"))
        version = self.workspace._capture_bytes(self._page_path(page), observed.encode())
        return result, version, len(observed), token

    def apply(self, state, decision):
        self._actor(state)
        action, actor = decision["action"], state.trigger.agent_id
        if action not in PAGE_ACTIONS:
            raise ValueError(f"unknown page tool {action!r}")
        if action == "read_comment":
            return self._read_comment(state, decision)
        payload = {
            "project_id": state.trigger.project_id,
        }
        page_id = None
        recipients = []
        with self.db:
            if action in {"list_pages", "search_pages"}:
                page = self._resolve(state, decision.get("path", "/"))
                parent_id = page["id"] if page else None
                if page:
                    payload["project_id"] = page["project_id"]
                offset = decision.get("offset", 0)
                if action == "list_pages":
                    children = [
                        self._reference(p)
                        for p in self._accessible(actor)
                        if p["parent_id"] == parent_id
                    ]
                    children.sort(key=lambda p: p["path"].casefold())
                    result = {
                        "path": self._path(page) if page else "/",
                        "pages": children[offset : offset + 20],
                        "total": len(children),
                        "next_offset": offset + 20 if offset + 20 < len(children) else None,
                    }
                else:
                    available = [
                        p
                        for p in self._accessible(actor)
                        if parent_id is None or self._descends_from(p, parent_id)
                    ]
                    found = search_pages(available, self.render, decision["query"], offset)
                    references = {p["id"]: self._reference(p) for p in available}
                    result = {
                        "path": self._path(page) if page else "/",
                        "matches": [
                            {**references[m["page_id"]], "excerpt": m["excerpt"]}
                            for m in found["matches"]
                        ],
                        "total": found["total"],
                        "match_mode": found["match_mode"],
                        "next_offset": found["next_offset"],
                    }
                payload.update(result, offset=offset)
            elif action == "add_page":
                parent, name = self._destination(state, decision["path"])
                scope = (
                    parent["project_id"]
                    if parent and parent["project_id"]
                    else self._creation_scope(state)
                )
                self._member(actor, scope)
                self._check_body(decision["markdown"])
                if not decision["title"].strip():
                    raise ValueError("title must not be blank")
                page_id = f"page-{uuid.uuid4().hex}"
                self._seed_page(
                    page_id,
                    scope,
                    decision["title"],
                    decision["markdown"],
                    parent_id=parent["id"] if parent else None,
                    name=name,
                )
                payload["project_id"] = scope
                self.db.execute(
                    "UPDATE page_revisions SET author_id=?, created_tick=? WHERE page_id=?",
                    (actor, self.store.current_tick + 1, page_id),
                )
                self.db.execute(
                    "UPDATE shared_pages SET created_tick=? WHERE id=?",
                    (self.store.current_tick + 1, page_id),
                )
                page = self._page(actor, page_id)
                result = self._reference(page)
                payload.update(result)
            else:
                page = self._resolve(state, decision["path"])
                if page is None:
                    raise ValueError("/ lists the top-level pages; name a page to use it")
                page_id = page["id"]
                path = self._path(page)
                payload["project_id"] = page["project_id"]
                if action == "view_page":
                    result, version, count, snapshot = self._view(page, decision, actor)
                    payload.update(
                        resource_version=version,
                        offset=result["offset"],
                        included_characters=count,
                        revision=result["revision"],
                        snapshot=snapshot,
                    )
                elif action == "edit_page":
                    if page_id == "objective":
                        raise WorkspaceAccessError(
                            f"{path} is read-only; comment_page can still comment on it"
                        )
                    payload["previous_resource_version"] = self._capture(page)
                    markdown = self._sections(page)["markdown"]
                    markdown = apply_edits(markdown, decision["edits"], path)
                    self._check_body(markdown)
                    self.db.execute(
                        "INSERT INTO page_revisions VALUES (?, ?, ?, ?, ?)",
                        (
                            page_id,
                            page["revision"] + 1,
                            markdown,
                            actor,
                            self.store.current_tick + 1,
                        ),
                    )
                    self.db.execute(
                        "UPDATE shared_pages SET revision=revision+1 WHERE id=?", (page_id,)
                    )
                    page = self._page(actor, page_id)
                    result = self._reference(page)
                    payload.update(result)
                elif action == "comment_page":
                    body = decision["markdown"]
                    if not body.strip():
                        raise ValueError("markdown must not be blank")
                    from mikoshi.runtime.organization import resolve_peers

                    parent = self._reply_parent(page, decision)
                    recipients = resolve_peers(
                        self.store,
                        actor,
                        page["project_id"],
                        decision.get("notify", []),
                        field="notify",
                    )
                    payload["previous_resource_version"] = self._capture(page)
                    comment_id = self._append_comment(
                        page, actor, body, reply_to=parent["id"] if parent else None
                    )
                    result = {"path": path, "comment_id": comment_id}
                    busy = {
                        row[0]
                        for row in self.db.execute(
                            "SELECT agent_id FROM triggers WHERE status IN ('queued','running')"
                        )
                    }
                    result["notified"] = recipients
                    result["woken"] = [peer for peer in recipients if peer not in busy]
                    payload.update(result)
                    payload["resource_version"] = self._capture(page)
                else:
                    raise ValueError(f"unknown page tool {action!r}")
            payload["page_id"] = page_id
            if action in {"list_pages", "search_pages"}:
                payload["resource_version"] = self.workspace._capture_bytes(
                    "pages/directory.json",
                    json.dumps(result, sort_keys=True, ensure_ascii=False).encode(),
                )
            event_id, _ = self.store._record_event(
                state.activation_id, actor, action, "succeeded", page_id, payload
            )
            if action in {"add_page", "edit_page"}:
                payload["resource_version"] = self._capture(page)
                self.db.execute(
                    "UPDATE events SET payload=? WHERE id=?",
                    (json.dumps(payload, sort_keys=True), event_id),
                )
            if recipients:
                from mikoshi.runtime.notifications import Notifications

                Notifications(self.store).deliver(
                    sender=actor,
                    kind="comment",
                    source_ref=f"comment:{comment_id}",
                    page_ref=path,
                    message=body,
                    recipients=recipients,
                    wake=True,
                    event_id=event_id,
                )
        self.store._publish_committed_event(event_id)
        return False, result

    def _read_comment(self, state, decision):
        action, actor = decision["action"], state.trigger.agent_id
        with self.db:
            row = self.db.execute(
                "SELECT * FROM page_comments WHERE id=?",
                (decision["comment_id"],),
            ).fetchone()
            if row is None:
                raise ValueError(f"comment {decision['comment_id']} is not a comment you can read")
            page = self._page(actor, row["page_id"])
            offset = decision.get("offset", 0)
            body = row["markdown"]
            if offset > len(body):
                raise ValueError(
                    f"offset {offset} is past the end of the comment ({len(body)} characters)"
                )
            end = offset + self.workspace.limits.max_read_chars
            result = {
                "path": self._path(page),
                "comment_id": row["id"],
                "author_id": row["author_id"],
            }
            if row["reply_to"]:
                result["reply_to"] = row["reply_to"]
            result.update(
                markdown=body[offset:end],
                offset=offset,
                next_offset=end if end < len(body) else None,
                total_characters=len(body),
            )
            version = self.workspace._capture_bytes(
                f"pages/discussion/{action}.json",
                json.dumps(result, sort_keys=True, ensure_ascii=False).encode(),
            )
            event, _ = self.store._record_event(
                state.activation_id,
                actor,
                action,
                "succeeded",
                page["id"],
                {
                    "page_id": page["id"],
                    "project_id": page["project_id"],
                    **result,
                    "resource_version": version,
                },
            )
        self.store._publish_committed_event(event)
        return False, result

    def _check_body(self, markdown):
        if not markdown.strip() or len(markdown.encode()) > self.workspace.limits.max_file_bytes:
            raise ValueError(
                f"markdown must be nonblank and at most {self.workspace.limits.max_file_bytes} bytes"
            )

    def _export_file(self, path, text):
        self.workspace._reject_symlink_components(path)
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        try:
            info = path.stat()
        except FileNotFoundError:
            identity = None
        else:
            identity = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
        if self._exported.get(path) == (digest, identity):
            return
        temporary = path.with_suffix(".tmp")
        self.workspace._reject_symlink_components(temporary)
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
        info = path.stat()
        self._exported[path] = (
            digest,
            (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns),
        )

    def _project_event(self, event):
        action, payload = event["action_type"], event["payload"]
        if event["outcome"] == "succeeded":
            if action in {"add_page", "edit_page", "comment_page"}:
                self._dirty_pages.add(payload["page_id"])
        for page_id in sorted(self._dirty_pages):
            page = dict(
                self.db.execute("SELECT * FROM shared_pages WHERE id=?", (page_id,)).fetchone()
            )
            self._export_page(page)
            self._dirty_pages.remove(page_id)

    def _export_page(self, page):
        path = self.workspace.root / self._page_path(page)
        self.workspace._reject_symlink_components(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._export_file(path, self.render(page))

    def export(self):
        """Reconcile all projections and write a complete history snapshot."""
        root = self.workspace.root / "pages"
        self.workspace._reject_symlink_components(root)
        root.mkdir(parents=True, exist_ok=True)
        for row in self.db.execute("SELECT * FROM shared_pages ORDER BY id"):
            self._export_page(dict(row))
        self._export_file(root / "history.json", json.dumps(records(self.store), indent=2) + "\n")
        self._dirty_pages.clear()
