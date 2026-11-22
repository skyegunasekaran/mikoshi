"""Durable conversation exchanges and per-peer repository guidance checkpoints."""

from __future__ import annotations

import json


class ConversationRepository:
    def __init__(self, connection):
        self._connection = connection

    def require_supported_format(self):
        if not self._connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='agent_memory_exchanges'"
        ).fetchone():
            raise ValueError("run database is missing conversation history; refusing to reset it")
        columns = {
            row[1] for row in self._connection.execute("PRAGMA table_info(agent_memory_exchanges)")
        }
        if not {
            "full_state_json",
            "conversation_id",
            "decision_id",
            "request_id",
            "request_ids",
            "tool_calls",
        }.issubset(columns):
            raise ValueError(
                "run database has an unsupported agent observation format; start a fresh trial"
            )

    def append(
        self,
        *,
        agent_id,
        conversation_id,
        exchange_key,
        decision_id,
        request_id,
        request_ids,
        tool_calls,
        messages,
        full_state_json,
    ):
        """Commit once per exchange key; return its sequence only for a new row.

        Callers publish exchange evidence after this method returns, so observers
        cannot see an exchange which subsequently rolls back.
        """
        with self._connection:
            inserted = self._connection.execute(
                "INSERT OR IGNORE INTO agent_memory_exchanges(agent_id,conversation_id,exchange_key,"
                "decision_id,request_id,request_ids,tool_calls,messages,full_state_json) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    agent_id,
                    conversation_id,
                    exchange_key,
                    decision_id,
                    request_id,
                    json.dumps(request_ids),
                    json.dumps(tool_calls),
                    json.dumps(messages, ensure_ascii=False),
                    full_state_json,
                ),
            )
        return inserted.lastrowid if inserted.rowcount else None

    def latest_state_json(self, agent_id):
        row = self._connection.execute(
            "SELECT full_state_json FROM agent_memory_exchanges "
            "WHERE agent_id=? AND conversation_id IS NOT NULL ORDER BY sequence DESC LIMIT 1",
            (agent_id,),
        ).fetchone()
        return row["full_state_json"] if row else None

    def latest_exchange(self, agent_id):
        row = self._connection.execute(
            "SELECT sequence,messages FROM agent_memory_exchanges "
            "WHERE agent_id=? AND conversation_id IS NOT NULL ORDER BY sequence DESC LIMIT 1",
            (agent_id,),
        ).fetchone()
        return dict(row) if row else None

    def messages(self, agent_id, sequence):
        row = self._connection.execute(
            "SELECT messages FROM agent_memory_exchanges WHERE sequence=? AND agent_id=?",
            (sequence, agent_id),
        ).fetchone()
        return json.loads(row[0]) if row else []


class GuidanceRepository:
    def __init__(self, connection):
        self._connection = connection

    def load(self, agent_id):
        row = self._connection.execute(
            "SELECT value FROM metadata WHERE key=?", (f"repository-guidance:{agent_id}",)
        ).fetchone()
        return json.loads(row[0]) if row else {"seen": {}, "checkpoint": None}

    def save(self, agent_id, value):
        with self._connection:
            self._connection.execute(
                "INSERT OR REPLACE INTO metadata(key,value) VALUES (?,?)",
                (f"repository-guidance:{agent_id}", json.dumps(value)),
            )
