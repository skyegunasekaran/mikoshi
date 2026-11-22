"""Mechanical reference checks for the public record stream."""

from mikoshi.persistence.content import content_bytes, read_content


def pointer_value(value, pointer):
    if pointer == "":
        return value
    if not pointer.startswith("/"):
        raise ValueError("submitted content needs a JSON Pointer")
    try:
        for part in pointer[1:].split("/"):
            part = part.replace("~1", "/").replace("~0", "~")
            if isinstance(value, list):
                index = int(part)
                if index < 0 or str(index) != part:
                    raise ValueError("invalid array index")
                value = value[index]
            else:
                value = value[part]
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise ValueError(f"invalid submitted-content pointer: {pointer}") from error
    return value


class EvidenceLinks:
    def __init__(self, root):
        self.root = root
        self.decisions = {}
        self.requests = {}
        self.request_order = {}
        self.exchanges = []
        self.actions = {}
        self.notifications = []

    def observe(self, record):
        data, kind = record["data"], record["type"]
        if kind == "decision":
            if data["decision_id"] in self.decisions:
                raise ValueError("duplicate decision identity")
            self.decisions[data["decision_id"]] = data
        elif kind in {"request", "response", "usage"}:
            events = self.requests.setdefault(data["request_id"], {})
            if data["kind"] in events:
                raise ValueError("duplicate provider request event")
            events[data["kind"]] = data
            if data["kind"] == "provider_request_prepared":
                self.request_order[data["request_id"]] = record["sequence"]
        elif kind == "exchange":
            self.exchanges.append(data)
        elif kind == "action":
            self.actions[data["id"]] = data
        elif kind == "notification":
            self.notifications.append(data)
        if kind in {"decision", "request", "response", "usage"}:
            if record["agent_id"] != data.get("agent_id") or record["activation_id"] != data.get(
                "activation_id"
            ):
                raise ValueError("record owner differs from its evidence payload")

    def decision(self, identity, agent_id, activation_id):
        decision = self.decisions.get(identity)
        if (
            decision is None
            or decision["agent_id"] != agent_id
            or decision["activation_id"] != activation_id
        ):
            raise ValueError(f"unknown or mismatched decision: {identity}")
        return decision

    def request(self, identity, decision_id):
        prepared = self.requests.get(identity, {}).get("provider_request_prepared")
        if prepared is None or prepared["decision_id"] != decision_id:
            raise ValueError(f"unknown or mismatched request: {identity}")
        return prepared

    def finish(self, run):
        for identity, events in self.requests.items():
            prepared = events.get("provider_request_prepared")
            if prepared is None:
                raise ValueError(f"request lacks its prepared record: {identity}")
            decision = self.decision(
                prepared["decision_id"], prepared.get("agent_id"), prepared.get("activation_id")
            )
            previous = prepared["previous_request_id"]
            if previous is not None:
                self.request(previous, prepared["decision_id"])
                if self.request_order[previous] >= self.request_order[identity]:
                    raise ValueError("request predecessor must be an earlier prepared request")
            for event in events.values():
                if any(
                    event.get(key) != prepared.get(key)
                    for key in (
                        "decision_id",
                        "agent_id",
                        "activation_id",
                        "activation_step",
                        "previous_request_id",
                    )
                ):
                    raise ValueError(f"request identity changed between events: {identity}")
            submitted = events.get("provider_request_submitted")
            if submitted:
                self.submitted_content(submitted, decision)
            for response_kind in ("provider_response_received", "provider_response_decoded"):
                if response_kind in events and submitted is None:
                    raise ValueError("response has no submission record")
            usage = events.get("provider_usage_recorded")
            if usage and (
                usage["entry"].get("request_id") != identity
                or usage["entry"].get("decision_id") != prepared["decision_id"]
            ):
                raise ValueError("usage entry belongs to another request")
            completed = events.get("provider_request_completed")
            if completed:
                timing = read_content(self.root, completed["timing_ref"])
                index = completed["timing_index"]
                try:
                    matches = (
                        timing["request_ids"][index] == identity
                        and timing["decision_ids"][index] == prepared["decision_id"]
                    )
                except (IndexError, KeyError, TypeError):
                    matches = False
                if not matches:
                    raise ValueError("request timing belongs to another request")
        exchanges_by_decision = {}
        for exchange in self.exchanges:
            decision_id = exchange.get("decision_id")
            if decision_id is None:
                continue
            exchanges_by_decision[decision_id] = exchange
            self.decision(
                decision_id, exchange["agent_id"], int(exchange["exchange_key"].split(":")[0])
            )
            for identity in exchange["request_ids"]:
                self.request(identity, decision_id)
            if exchange["request_id"] is not None:
                self.request(exchange["request_id"], decision_id)
        for action in self.actions.values():
            payload = action["payload"]
            if payload.get("decision_id") is None:
                continue
            self.decision(payload["decision_id"], action["agent_id"], action["activation_id"])
            if payload.get("request_id") is not None:
                self.request(payload["request_id"], payload["decision_id"])
            exchange = exchanges_by_decision.get(payload["decision_id"])
            if exchange is not None and payload.get("tool_call_id") is not None:
                calls = {call["index"]: call["id"] for call in exchange["tool_calls"]}
                if calls.get(payload.get("tool_call_index")) != payload["tool_call_id"]:
                    raise ValueError("action refers to another authored native tool call")
                if payload.get("request_id") != exchange["request_id"]:
                    raise ValueError(
                        "action and native tool call have different originating requests"
                    )
        for notification in self.notifications:
            if notification["event_id"] not in self.actions:
                raise ValueError("queued notification has no source action")
        for entry in (run.get("budget") or {}).get("entries", []):
            if entry.get("request_id"):
                self.request(entry["request_id"], entry.get("decision_id"))
        self.provider_errors(run.get("provider_metrics") or {})

    def submitted_content(self, submitted, decision):
        payload = read_content(self.root, submitted["payload_ref"])
        for included in submitted.get("included_content", []):
            value = pointer_value(payload, included["path"])
            if content_bytes(value) != content_bytes(
                read_content(self.root, included["content_ref"])
            ):
                raise ValueError("submitted content differs from its referenced value")
            for field in included.get("source_fields", []):
                if included["content_ref"] != decision[field + "_ref"]:
                    raise ValueError("included observation refers to another prepared value")
            for source in included.get("notification_sources", []):
                if source not in decision["metadata"].get("observation_notification_sources", []):
                    raise ValueError(
                        "submitted notification source differs from the prepared observation"
                    )

    def provider_errors(self, metrics):
        if metrics.get("evidence_errors"):
            raise ValueError("provider reports incomplete evidence export")
        for child in metrics.get("by_model", {}).values():
            self.provider_errors(child)
