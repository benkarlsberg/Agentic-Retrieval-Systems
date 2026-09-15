"""Action parsing for single-agent loops."""

from __future__ import annotations

import re
from enum import Enum

from pydantic import BaseModel


class ActionType(str, Enum):
    RETRIEVE = "retrieve"
    REFORMULATE = "reformulate"
    STOP = "stop"
    ABSTAIN = "abstain"
    SYNTHESIZE = "synthesize"


class Action(BaseModel):
    type: ActionType
    query: str | None = None
    reason: str | None = None


def parse_action(text: str) -> Action:
    t = text.strip()
    m = re.search(r"ACTION:\s*(\w+)", t, re.I)
    action_raw = (m.group(1) if m else "").lower()
    q = None
    mq = re.search(r"QUERY:\s*(.+?)(?:\n|$)", t, re.I)
    if mq:
        q = mq.group(1).strip()
    reason = None
    mr = re.search(r"REASON:\s*(.+?)(?:\n|$)", t, re.I)
    if mr:
        reason = mr.group(1).strip()

    mapping = {
        "retrieve": ActionType.RETRIEVE,
        "reformulate": ActionType.REFORMULATE,
        "stop": ActionType.STOP,
        "abstain": ActionType.ABSTAIN,
        "synthesize": ActionType.SYNTHESIZE,
    }
    at = mapping.get(action_raw, ActionType.STOP)
    return Action(type=at, query=q, reason=reason)
