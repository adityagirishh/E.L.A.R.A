"""
test_reset_step.py — Tests for ElaraEnv reset() and step()
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "app"))

import pytest
from environment import ElaraEnv
from models import Action


class TestReset:
    def test_reset_easy(self):
        env = ElaraEnv()
        obs = env.reset(task_id="easy")
        assert obs.task_id == "easy"
        assert obs.lead_id == "L-001"
        assert obs.lead_name == "Arun Sharma"
        assert obs.lead_stage == "new"
        assert obs.consent is True

    def test_reset_medium(self):
        env = ElaraEnv()
        obs = env.reset(task_id="medium")
        assert obs.task_id == "medium"
        assert obs.lead_id == "L-004"
        assert obs.lead_stage == "contacted"

    def test_reset_hard(self):
        env = ElaraEnv()
        obs = env.reset(task_id="hard")
        assert obs.task_id == "hard"
        assert obs.lead_id == "L-007"
        assert obs.lead_stage == "proposal_sent"

    def test_reset_invalid_task(self):
        env = ElaraEnv()
        with pytest.raises(ValueError, match="not found"):
            env.reset(task_id="nonexistent")

    def test_reset_clears_previous_state(self):
        env = ElaraEnv()
        env.reset(task_id="easy")
        env.step(Action(action_type="send_email", target_lead_id="L-001", body="hello"))
        # Reset should clear everything
        obs = env.reset(task_id="easy")
        assert obs.step_count == 0
        assert obs.lead_stage == "new"

    def test_observation_has_product_context(self):
        env = ElaraEnv()
        obs = env.reset(task_id="easy")
        assert "name" in obs.product_context
        assert obs.product_context["name"] == "E.L.A.R.A."
        assert "features" in obs.product_context
        assert "value_props" in obs.product_context

    def test_observation_has_available_actions(self):
        env = ElaraEnv()
        obs = env.reset(task_id="easy")
        assert "send_email" in obs.available_actions
        assert "make_call" in obs.available_actions
        assert "update_crm" in obs.available_actions
        assert len(obs.available_actions) == 9

    def test_observation_has_task_hint(self):
        env = ElaraEnv()
        obs = env.reset(task_id="easy")
        assert obs.task_hint != ""
        assert "L-001" in obs.task_hint


class TestStep:
    def test_step_before_reset_raises(self):
        env = ElaraEnv()
        with pytest.raises(RuntimeError, match="reset"):
            env.step(Action(action_type="send_email", target_lead_id="L-001"))

    def test_step_returns_tuple(self):
        env = ElaraEnv()
        env.reset(task_id="easy")
        result = env.step(Action(action_type="send_email", target_lead_id="L-001", body="hi"))
        assert len(result) == 4
        obs, reward, done, info = result
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_step_advances_stage(self):
        env = ElaraEnv()
        env.reset(task_id="easy")
        obs, _, _, info = env.step(Action(
            action_type="send_email", target_lead_id="L-001", body="hello Arun"
        ))
        assert info["stage_after"] in ("contacted", "awaiting_docs")

    def test_step_wrong_lead_id(self):
        env = ElaraEnv()
        env.reset(task_id="easy")
        obs, reward, done, info = env.step(Action(
            action_type="send_email", target_lead_id="INVALID"
        ))
        assert reward < 0
        assert done is True

    def test_episode_ends_at_max_steps(self):
        env = ElaraEnv()
        env.reset(task_id="easy")  # max_steps = 3
        for i in range(3):
            obs, _, done, _ = env.step(Action(
                action_type="wait", target_lead_id="L-001"
            ))
        assert done is True

    def test_step_after_done_returns_zero_reward(self):
        env = ElaraEnv()
        env.reset(task_id="easy")
        for _ in range(3):
            env.step(Action(action_type="wait", target_lead_id="L-001"))
        # Extra step after done
        obs, reward, done, info = env.step(Action(
            action_type="wait", target_lead_id="L-001"
        ))
        assert reward == 0.0
        assert done is True

    def test_step_updates_total_reward(self):
        env = ElaraEnv()
        env.reset(task_id="easy")
        env.step(Action(
            action_type="send_email", target_lead_id="L-001",
            body="Hi Arun from NovaTech, E.L.A.R.A. cuts lead response time by 60%.",
            goal="intro", priority="high",
        ))
        state = env.state()
        assert state["total_reward"] != 0.0

    def test_state_returns_episode_log(self):
        env = ElaraEnv()
        env.reset(task_id="easy")
        env.step(Action(action_type="send_email", target_lead_id="L-001", body="hi"))
        state = env.state()
        assert len(state["episode_log"]) == 1
        assert state["episode_log"][0]["action_type"] == "send_email"


class TestState:
    def test_state_before_reset_raises(self):
        env = ElaraEnv()
        with pytest.raises(RuntimeError, match="reset"):
            env.state()

    def test_state_returns_dict(self):
        env = ElaraEnv()
        env.reset(task_id="easy")
        s = env.state()
        assert isinstance(s, dict)
        assert "product" in s
        assert "leads" in s
        assert "task_id" in s
        assert "step_count" in s
        assert "done" in s
