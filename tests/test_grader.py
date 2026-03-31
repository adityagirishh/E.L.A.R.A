"""
test_grader.py — Tests for the 5-dimension grader
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "app"))

import pytest
from environment import ElaraEnv
from models import Action, EpisodeState
from grader import grade


def run_episode(task_id: str, actions: list) -> dict:
    """Helper: run a full episode and return grader result."""
    env = ElaraEnv()
    env.reset(task_id=task_id)
    for action in actions:
        env.step(action)
    return grade(EpisodeState(**env.state()))


class TestGraderEasy:
    def test_good_agent_passes(self):
        result = run_episode("easy", [
            Action(
                action_type="send_email",
                target_lead_id="L-001",
                subject="Intro — E.L.A.R.A. for NovaTech",
                body=(
                    "Hi Arun, reaching out about E.L.A.R.A. — cuts lead response "
                    "time by 60% and unifies email, calls and messages. "
                    "Happy to set up a demo!"
                ),
                goal="intro",
                priority="high",
            ),
        ])
        assert result["pass"] is True
        assert result["score"] >= 0.60
        assert result["task_completed"] is True

    def test_bad_agent_fails(self):
        result = run_episode("easy", [
            Action(action_type="make_call", target_lead_id="L-001", body=""),
            Action(action_type="make_call", target_lead_id="L-001", body="hi"),
            Action(action_type="wait", target_lead_id="L-001"),
        ])
        # Bad agent should score significantly lower
        assert result["score"] < 0.90

    def test_grader_returns_all_dimensions(self):
        result = run_episode("easy", [
            Action(action_type="send_email", target_lead_id="L-001", body="hi"),
        ])
        assert "dimensions" in result
        dims = result["dimensions"]
        assert "task_completion" in dims
        assert "channel_correctness" in dims
        assert "crm_accuracy" in dims
        assert "compliance" in dims
        assert "efficiency" in dims
        # Each dimension has score, weight, reason
        for dim_data in dims.values():
            assert "score" in dim_data
            assert "weight" in dim_data
            assert "reason" in dim_data


class TestGraderMedium:
    def test_good_agent_passes(self):
        result = run_episode("medium", [
            Action(
                action_type="send_email",
                target_lead_id="L-004",
                subject="RE: Pricing — SwiftOps",
                body="Hi Priya, ROI calculator shows 10x return in 3 months at ₹2,999/seat.",
                goal="handle_objection",
                priority="high",
            ),
            Action(
                action_type="request_documents",
                target_lead_id="L-004",
                subject="Documents for SwiftOps",
                body="Hi Priya, please share the NDA and tooling overview.",
                goal="get_documents",
            ),
            Action(
                action_type="update_crm",
                target_lead_id="L-004",
                goal="log_interaction",
                metadata={"note": "Handled pricing objection. Doc request sent."},
            ),
        ])
        assert result["pass"] is True
        assert result["score"] >= 0.60

    def test_missing_crm_update_hurts(self):
        result_no_crm = run_episode("medium", [
            Action(action_type="send_email", target_lead_id="L-004", body="hi Priya"),
            Action(action_type="request_documents", target_lead_id="L-004", body="docs pls"),
            Action(action_type="wait", target_lead_id="L-004"),
            Action(action_type="wait", target_lead_id="L-004"),
        ])
        result_with_crm = run_episode("medium", [
            Action(action_type="send_email", target_lead_id="L-004", body="hi Priya"),
            Action(action_type="request_documents", target_lead_id="L-004", body="docs pls"),
            Action(action_type="update_crm", target_lead_id="L-004", metadata={"note": "done"}),
            Action(action_type="wait", target_lead_id="L-004"),
        ])
        assert result_with_crm["score"] > result_no_crm["score"]


class TestGraderHard:
    def test_good_agent_passes(self):
        result = run_episode("hard", [
            Action(
                action_type="make_call", target_lead_id="L-007",
                body="Hi Rajan, following up on the proposal for Apex Ventures.",
                goal="proposal_followup", priority="high",
            ),
            Action(
                action_type="request_documents", target_lead_id="L-007",
                subject="Documents — Apex Ventures",
                body="Hi Rajan, need the signed NDA and procurement form.",
                goal="get_documents",
            ),
            Action(
                action_type="send_message", target_lead_id="L-007",
                body="Hi Rajan — sent the docs request via email. Let me know.",
                goal="keep_warm",
            ),
            Action(
                action_type="update_crm", target_lead_id="L-007",
                goal="log_interaction",
                metadata={"note": "Follow-up done. Docs requested. Targeting close."},
            ),
            Action(
                action_type="make_call", target_lead_id="L-007",
                body="Hi Rajan, documents received — ready to move forward?",
                goal="close_deal",
                metadata={"docs_received": True},
            ),
        ])
        assert result["pass"] is True
        assert result["score"] >= 0.60


class TestGraderScoreRange:
    def test_score_between_0_and_1(self):
        for task_id in ["easy", "medium", "hard"]:
            result = run_episode(task_id, [
                Action(action_type="wait", target_lead_id={
                    "easy": "L-001", "medium": "L-004", "hard": "L-007"
                }[task_id]),
            ])
            assert 0.0 <= result["score"] <= 1.0

    def test_grader_deterministic(self):
        """Running the same episode twice should produce the same score."""
        actions = [
            Action(action_type="send_email", target_lead_id="L-001",
                   body="Hi Arun, E.L.A.R.A. demo", goal="intro"),
        ]
        r1 = run_episode("easy", actions)
        r2 = run_episode("easy", actions)
        assert r1["score"] == r2["score"]
