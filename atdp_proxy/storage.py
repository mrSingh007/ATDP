from __future__ import annotations

import hashlib
import json
from pathlib import Path
from threading import RLock
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from atdp_proxy.config import Settings
from atdp_proxy.schemas import ATDPEvent, ATDPEventCreate, RewardSignal, RewardUpdateRecord, SessionSummary, new_id, utc_now


def _json(value: Any) -> str:
    if isinstance(value, str):
        return value
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _loads(value: str | None, default: Any) -> Any:
    if not value:
        return default
    return json.loads(value)


def _hash_event(event: ATDPEvent) -> str:
    data = event.model_dump(mode="json")
    data["event_hash"] = None
    data["latest_reward"] = None
    data["reward_updates"] = []
    canonical = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _hash_reward_update(update: dict[str, Any]) -> str:
    data = dict(update)
    data["update_hash"] = None
    canonical = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ATDPStorage:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.data_dir = settings.data_path
        self.sqlite_path = settings.resolved_sqlite_path
        self.jsonl_path = settings.resolved_jsonl_path
        self._lock = RLock()
        self.engine: Engine | None = None

    def initialize(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{self.sqlite_path}", future=True)
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id TEXT PRIMARY KEY,
                        created_at TEXT NOT NULL,
                        ended_at TEXT,
                        metadata_json TEXT NOT NULL DEFAULT '{}'
                    )
                    """
                )
            )
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS events (
                        id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        step INTEGER NOT NULL,
                        type TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        parent_event_id TEXT,
                        trace_id TEXT,
                        span_id TEXT,
                        replay_mode TEXT,
                        reward_json TEXT,
                        event_json TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        UNIQUE(session_id, step)
                    )
                    """
                )
            )
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_events_session_step ON events(session_id, step)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_events_type ON events(type)"))
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS reward_updates (
                        id TEXT PRIMARY KEY,
                        event_id TEXT NOT NULL,
                        session_id TEXT NOT NULL,
                        value REAL,
                        label TEXT,
                        critique TEXT,
                        source TEXT NOT NULL,
                        metadata_json TEXT NOT NULL,
                        update_json TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    )
                    """
                )
            )
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_reward_updates_event ON reward_updates(event_id)"))

    @property
    def _engine(self) -> Engine:
        if self.engine is None:
            raise RuntimeError("ATDPStorage.initialize() has not been called")
        return self.engine

    def append_event(self, event_create: ATDPEventCreate) -> ATDPEvent:
        with self._lock:
            if not event_create.session_id:
                raise ValueError("session_id is required before persisting an ATDP event")
            event_id = event_create.id or new_id()
            step = event_create.step
            with self._engine.begin() as conn:
                conn.execute(
                    text(
                        """
                        INSERT OR IGNORE INTO sessions(session_id, created_at, metadata_json)
                        VALUES(:session_id, :created_at, :metadata_json)
                        """
                    ),
                    {
                        "session_id": event_create.session_id,
                        "created_at": utc_now().isoformat(),
                        "metadata_json": "{}",
                    },
                )
                if step is None:
                    row = conn.execute(
                        text("SELECT COALESCE(MAX(step), 0) + 1 FROM events WHERE session_id = :session_id"),
                        {"session_id": event_create.session_id},
                    ).one()
                    step = int(row[0])

                previous_row = conn.execute(
                    text(
                        """
                        SELECT event_json
                        FROM events
                        WHERE session_id = :session_id AND step < :step
                        ORDER BY step DESC
                        LIMIT 1
                        """
                    ),
                    {"session_id": event_create.session_id, "step": step},
                ).first()
                previous_hash = None
                if previous_row is not None:
                    previous_hash = _loads(previous_row[0], {}).get("event_hash")

                event_data = event_create.model_dump(exclude={"id", "step"})
                if event_data.get("trajectory_id") is None:
                    event_data["trajectory_id"] = event_create.session_id
                event_data["previous_event_hash"] = previous_hash
                event_data["event_hash"] = None
                event = ATDPEvent(
                    **event_data,
                    id=event_id,
                    step=step,
                )
                event.event_hash = _hash_event(event)
                event_json = event.model_dump(mode="json")
                conn.execute(
                    text(
                        """
                        INSERT INTO events(
                            id, session_id, step, type, timestamp, parent_event_id, trace_id, span_id,
                            replay_mode, reward_json, event_json, created_at
                        )
                        VALUES(
                            :id, :session_id, :step, :type, :timestamp, :parent_event_id, :trace_id, :span_id,
                            :replay_mode, :reward_json, :event_json, :created_at
                        )
                        """
                    ),
                    {
                        "id": event.id,
                        "session_id": event.session_id,
                        "step": event.step,
                        "type": event.type,
                        "timestamp": event.timestamp.isoformat(),
                        "parent_event_id": event.parent_event_id,
                        "trace_id": event.trace_id,
                        "span_id": event.span_id,
                        "replay_mode": event.replay.mode.value,
                        "reward_json": _json(event.reward) if event.reward else None,
                        "event_json": _json(event_json),
                        "created_at": utc_now().isoformat(),
                    },
                )
            self._append_jsonl({"record_type": "event", "event": event_json})
            return event

    def get_event(self, event_id: str) -> ATDPEvent | None:
        with self._engine.begin() as conn:
            row = conn.execute(text("SELECT event_json, reward_json FROM events WHERE id = :id"), {"id": event_id}).first()
            updates = self._reward_updates_for_event(conn, event_id)
        if row is None:
            return None
        event_data = _loads(row[0], {})
        reward_data = _loads(row[1], None)
        if reward_data is not None:
            event_data["latest_reward"] = reward_data
        event_data["reward_updates"] = [update.model_dump(mode="json") for update in updates]
        return ATDPEvent.model_validate(event_data)

    def events_for_session(self, session_id: str) -> list[ATDPEvent]:
        with self._engine.begin() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT id, event_json, reward_json
                    FROM events
                    WHERE session_id = :session_id
                    ORDER BY step ASC
                    """
                ),
                {"session_id": session_id},
            ).all()
            updates_by_event = self._reward_updates_for_session(conn, session_id)
        events: list[ATDPEvent] = []
        for row in rows:
            event_data = _loads(row[1], {})
            reward_data = _loads(row[2], None)
            if reward_data is not None:
                event_data["latest_reward"] = reward_data
            event_data["reward_updates"] = [
                update.model_dump(mode="json") for update in updates_by_event.get(row[0], [])
            ]
            events.append(ATDPEvent.model_validate(event_data))
        return events

    def list_sessions(self) -> list[SessionSummary]:
        with self._engine.begin() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT s.session_id, s.created_at, s.ended_at, s.metadata_json, COUNT(e.id) AS event_count
                    FROM sessions s
                    LEFT JOIN events e ON s.session_id = e.session_id
                    GROUP BY s.session_id, s.created_at, s.ended_at, s.metadata_json
                    ORDER BY s.created_at DESC
                    """
                )
            ).all()
        return [
            SessionSummary(
                session_id=row[0],
                created_at=row[1],
                ended_at=row[2],
                metadata=_loads(row[3], {}),
                event_count=int(row[4] or 0),
            )
            for row in rows
        ]

    def mark_session_ended(self, session_id: str, metadata: dict[str, Any] | None = None) -> None:
        with self._lock:
            now = utc_now().isoformat()
            with self._engine.begin() as conn:
                conn.execute(
                    text(
                        """
                        INSERT INTO sessions(session_id, created_at, ended_at, metadata_json)
                        VALUES(:session_id, :created_at, :ended_at, :metadata_json)
                        ON CONFLICT(session_id) DO UPDATE SET
                            ended_at = excluded.ended_at,
                            metadata_json = excluded.metadata_json
                        """
                    ),
                    {
                        "session_id": session_id,
                        "created_at": now,
                        "ended_at": now,
                        "metadata_json": _json(metadata or {}),
                    },
                )

    def append_reward_update(self, event_id: str, reward: RewardSignal) -> ATDPEvent | None:
        with self._lock:
            event = self.get_event(event_id)
            if event is None:
                return None
            reward_json = reward.model_dump(mode="json")
            update = {
                "id": new_id(),
                "event_id": event_id,
                "session_id": event.session_id,
                "reward": reward_json,
                "created_at": utc_now().isoformat(),
                "update_hash": None,
            }
            update["update_hash"] = _hash_reward_update(update)
            with self._engine.begin() as conn:
                conn.execute(
                    text(
                        """
                        INSERT INTO reward_updates(
                            id, event_id, session_id, value, label, critique, source,
                            metadata_json, update_json, created_at
                        )
                        VALUES(
                            :id, :event_id, :session_id, :value, :label, :critique, :source,
                            :metadata_json, :update_json, :created_at
                        )
                        """
                    ),
                    {
                        "id": update["id"],
                        "event_id": event_id,
                        "session_id": event.session_id,
                        "value": reward.value,
                        "label": reward.label,
                        "critique": reward.critique,
                        "source": reward.source,
                        "metadata_json": _json(reward.metadata),
                        "update_json": _json(update),
                        "created_at": update["created_at"],
                    },
                )
                conn.execute(
                    text("UPDATE events SET reward_json = :reward_json WHERE id = :event_id"),
                    {"event_id": event_id, "reward_json": _json(reward_json)},
                )
            self._append_jsonl({"record_type": "reward.update", **update})
            return self.get_event(event_id)

    def export_jsonl(self, session_id: str | None = None) -> str:
        if session_id is None:
            if not self.jsonl_path.exists():
                return ""
            return self.jsonl_path.read_text(encoding="utf-8")
        with self._engine.begin() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT created_at, 'event' AS record_type, event_json AS record_json
                    FROM events
                    WHERE session_id = :session_id
                    UNION ALL
                    SELECT created_at, 'reward.update' AS record_type, update_json AS record_json
                    FROM reward_updates
                    WHERE session_id = :session_id
                    ORDER BY created_at ASC
                    """
                ),
                {"session_id": session_id},
            ).all()
        lines = []
        for row in rows:
            if row[1] == "event":
                lines.append(_json({"record_type": "event", "event": _loads(row[2], {})}))
            else:
                lines.append(_json({"record_type": "reward.update", **_loads(row[2], {})}))
        return "\n".join(lines) + ("\n" if lines else "")

    def _append_jsonl(self, record: dict[str, Any]) -> None:
        line = _json(record)
        with self.jsonl_path.open("a", encoding="utf-8") as fh:
            fh.write(line)
            fh.write("\n")

    def _reward_updates_for_event(self, conn, event_id: str) -> list[RewardUpdateRecord]:
        rows = conn.execute(
            text(
                """
                SELECT update_json
                FROM reward_updates
                WHERE event_id = :event_id
                ORDER BY created_at ASC
                """
            ),
            {"event_id": event_id},
        ).all()
        return [RewardUpdateRecord.model_validate(_loads(row[0], {})) for row in rows]

    def _reward_updates_for_session(self, conn, session_id: str) -> dict[str, list[RewardUpdateRecord]]:
        rows = conn.execute(
            text(
                """
                SELECT event_id, update_json
                FROM reward_updates
                WHERE session_id = :session_id
                ORDER BY created_at ASC
                """
            ),
            {"session_id": session_id},
        ).all()
        updates: dict[str, list[RewardUpdateRecord]] = {}
        for row in rows:
            updates.setdefault(row[0], []).append(RewardUpdateRecord.model_validate(_loads(row[1], {})))
        return updates
