"""Unit tests for TokenTracker adoption in the proofreader."""
from unittest.mock import MagicMock

import pytest

from token_tracker import TokenTracker


def _fake_aimessage(input_tokens, output_tokens):
    """Build a stand-in for a langchain-core AIMessage with usage_metadata."""
    msg = MagicMock()
    msg.usage_metadata = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }
    msg.response_metadata = {"model_name": "google/gemini-2.5-pro"}
    return msg


def test_record_from_response_captures_token_counts():
    tracker = TokenTracker()
    msg = _fake_aimessage(100, 50)

    tracker.record_from_response(
        source="proofread", model="google/gemini-2.5-pro", raw_response=msg
    )

    events = tracker.events()
    assert len(events) == 1
    assert events[0].input_tokens == 100
    assert events[0].output_tokens == 50
    assert events[0].source == "proofread"


def test_per_source_rollup():
    tracker = TokenTracker()
    for _ in range(3):
        tracker.record_from_response(
            source="proofread",
            model="google/gemini-2.5-pro",
            raw_response=_fake_aimessage(100, 50),
        )
    for _ in range(2):
        tracker.record_from_response(
            source="verify",
            model="google/gemini-2.5-pro",
            raw_response=_fake_aimessage(200, 75),
        )

    totals = tracker.totals()
    assert totals["proofread"]["calls"] == 3
    assert totals["proofread"]["input_tokens"] == 300
    assert totals["proofread"]["output_tokens"] == 150
    assert totals["verify"]["calls"] == 2
    assert totals["verify"]["input_tokens"] == 400
    assert totals["verify"]["output_tokens"] == 150


def test_cost_estimate_math():
    tracker = TokenTracker()
    tracker.record_from_response(
        source="proofread",
        model="google/gemini-2.5-pro",
        raw_response=_fake_aimessage(1_000_000, 500_000),
    )

    pricing = {"google/gemini-2.5-pro": {"input": 1.25, "output": 5.0}}
    cost = tracker.cost_estimate(pricing)

    # 1M @ $1.25/M = $1.25 input; 500k @ $5/M = $2.50 output; total $3.75
    assert cost["input_usd"] == pytest.approx(1.25)
    assert cost["output_usd"] == pytest.approx(2.50)
    assert cost["total_usd"] == pytest.approx(3.75)


def test_missing_usage_metadata_noop():
    tracker = TokenTracker()
    msg = MagicMock()
    msg.usage_metadata = None

    tracker.record_from_response(
        source="proofread", model="google/gemini-2.5-pro", raw_response=msg
    )

    assert len(tracker.events()) == 0


def test_unknown_model_in_pricing_dict():
    """Unknown models contribute 0 to cost_estimate, no error raised."""
    tracker = TokenTracker()
    tracker.record_from_response(
        source="proofread",
        model="some/unknown-model",
        raw_response=_fake_aimessage(100, 50),
    )

    cost = tracker.cost_estimate({"google/gemini-2.5-pro": {"input": 1.25, "output": 5.0}})

    # Per token-tracker contract: unknown models silently skipped
    assert cost["total_usd"] == 0.0
