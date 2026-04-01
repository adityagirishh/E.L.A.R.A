"""
test_rewards.py — Tests for the reward engine
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "app"))

import pytest
from models import Action, LeadProfile, ProductProfile
from reward_engine import (
    calculate_reward,
    reward_channel_choice,
    reward_timing,
    reward_message_quality,
    reward_crm_update,
    reward_lead_progression,
    reward_policy_compliance,
    reward_duplicate_outreach,
    reward_looping,
)


def make_lead(**overrides) -> LeadProfile:
    defaults = dict(
        lead_id="L-001", lead_name="Arun Sharma",
        company="NovaTech Solutions", role="Head of Sales",
        lead_stage="new", last_contact_channel="none",
        days_since_last_contact=0, next_followup_due=3,
        consent=True, sentiment="neutral",
        documents_pending=True, preferred_channel="email",
    )
    defaults.update(overrides)
    return LeadProfile(**defaults)


def make_product() -> ProductProfile:
    return ProductProfile(
        product_id="P-001", product_name="E.L.A.R.A.",
        description="Sales ops platform",
        features=["multi-channel outreach", "lead tracking"],
        value_props=["Cut lead response time by 60%"],
        objections=["too expensive"],
        compliance_notes=["Respect consent"],
    )


class TestChannelChoice:
    def test_correct_channel(self):
        lead = make_lead(preferred_channel="email")
        action = Action(action_type="send_email", target_lead_id="L-001")
        score, _ = reward_channel_choice(action, lead)
        assert score == 0.2

    def test_wrong_channel(self):
        lead = make_lead(preferred_channel="email")
        action = Action(action_type="make_call", target_lead_id="L-001")
        score, _ = reward_channel_choice(action, lead)
        assert score == -0.2

    def test_any_preference(self):
        lead = make_lead(preferred_channel="any")
        action = Action(action_type="send_email", target_lead_id="L-001")
        score, _ = reward_channel_choice(action, lead)
        assert score == 0.2

    def test_non_contact_action(self):
        lead = make_lead(preferred_channel="email")
        action = Action(action_type="update_crm", target_lead_id="L-001")
        score, _ = reward_channel_choice(action, lead)
        assert score == 0.0


class TestTiming:
    def test_first_contact_valid(self):
        lead = make_lead(last_contact_channel="none")
        action = Action(action_type="send_email", target_lead_id="L-001")
        score, _ = reward_timing(action, lead)
        assert score == 0.1

    def test_too_soon(self):
        lead = make_lead(last_contact_channel="email", days_since_last_contact=0)
        action = Action(action_type="send_email", target_lead_id="L-001")
        score, _ = reward_timing(action, lead)
        assert score == -0.1

    def test_on_time(self):
        lead = make_lead(last_contact_channel="email", days_since_last_contact=4, next_followup_due=3)
        action = Action(action_type="send_email", target_lead_id="L-001")
        score, _ = reward_timing(action, lead)
        assert score == 0.1


class TestMessageQuality:
    def test_empty_body(self):
        lead = make_lead()
        product = make_product()
        action = Action(action_type="send_email", target_lead_id="L-001", body="")
        score, _ = reward_message_quality(action, lead, product)
        assert score == 0.0

    def test_body_present(self):
        lead = make_lead()
        product = make_product()
        action = Action(action_type="send_email", target_lead_id="L-001", body="Hello there")
        score, _ = reward_message_quality(action, lead, product)
        assert score >= 0.05

    def test_personalised_body(self):
        lead = make_lead()
        product = make_product()
        action = Action(
            action_type="send_email", target_lead_id="L-001",
            body="Hi Arun, reaching out about multi-channel outreach for NovaTech"
        )
        score, _ = reward_message_quality(action, lead, product)
        assert abs(score - 0.15) < 1e-9  # body + personalised + product-relevant (no objection to address)


class TestCrmUpdate:
    def test_crm_update_action(self):
        action = Action(action_type="update_crm", target_lead_id="L-001")
        score, _ = reward_crm_update(action)
        assert score == 0.1

    def test_non_crm_action(self):
        action = Action(action_type="send_email", target_lead_id="L-001")
        score, _ = reward_crm_update(action)
        assert score == 0.0


class TestLeadProgression:
    def test_progression(self):
        lead = make_lead(lead_stage="new")
        action = Action(action_type="send_email", target_lead_id="L-001")
        score, _ = reward_lead_progression(action, lead, "contacted")
        assert score == 0.3

    def test_no_change(self):
        lead = make_lead(lead_stage="new")
        action = Action(action_type="wait", target_lead_id="L-001")
        score, _ = reward_lead_progression(action, lead, "new")
        assert score == 0.0

    def test_regression(self):
        lead = make_lead(lead_stage="contacted")
        action = Action(action_type="wait", target_lead_id="L-001")
        score, _ = reward_lead_progression(action, lead, "new")
        assert score == -0.1


class TestPolicyCompliance:
    def test_compliant(self):
        lead = make_lead(consent=True)
        action = Action(action_type="send_email", target_lead_id="L-001")
        score, _ = reward_policy_compliance(action, lead)
        assert score == 0.1

    def test_consent_violation(self):
        lead = make_lead(consent=False)
        action = Action(action_type="send_email", target_lead_id="L-001")
        score, _ = reward_policy_compliance(action, lead)
        assert score == -0.5


class TestDuplicateOutreach:
    def test_no_duplicate(self):
        lead = make_lead()
        action = Action(action_type="send_email", target_lead_id="L-001")
        score, _ = reward_duplicate_outreach(action, lead)
        assert score == 0.0

    def test_duplicate(self):
        # New API: duplicate detection uses prev_action_type + prev_goal, not lead history
        lead = make_lead()
        action = Action(action_type="send_email", target_lead_id="L-001", goal="intro")
        score, _ = reward_duplicate_outreach(
            action, lead,
            prev_action_type="send_email",
            prev_goal="intro",
        )
        assert score == -0.2

    def test_different_goal_not_duplicate(self):
        lead = make_lead()
        action = Action(action_type="send_email", target_lead_id="L-001", goal="get_documents")
        score, _ = reward_duplicate_outreach(
            action, lead,
            prev_action_type="send_email",
            prev_goal="intro",
        )
        assert score == 0.0


class TestLooping:
    def test_no_looping(self):
        score, _ = reward_looping(step_count=1, max_steps=5)
        assert score == 0.0

    def test_looping_detected(self):
        score, _ = reward_looping(step_count=4, max_steps=5)
        assert score == -0.1


class TestCalculateReward:
    def test_full_calculation(self):
        lead = make_lead()
        product = make_product()
        action = Action(
            action_type="send_email", target_lead_id="L-001",
            body="Hi Arun from NovaTech, multi-channel outreach platform demo"
        )
        total, breakdown = calculate_reward(
            action=action, lead_before=lead, lead_after_stage="contacted",
            product=product, step_count=1, max_steps=3, done=False,
        )
        assert isinstance(total, float)
        assert "channel" in breakdown
        assert "timing" in breakdown
        assert "message" in breakdown
        assert "compliance" in breakdown
        assert total > 0  # good action should have positive reward
