"""
reward_engine.py
Implements the exact reward table from the ELARA spec.

Positive:
  correct channel      +0.2
  correct timing       +0.1
  good message         +0.2
  CRM update           +0.1
  lead progression     +0.3
  policy compliance    +0.1

Penalties:
  duplicate outreach   -0.2
  wrong channel        -0.2
  consent violation    -0.5
  looping/useless      -0.1
"""

from typing import Any, Dict, List, Tuple
from models import Action, LeadProfile, ProductProfile


# ─────────────────────────────────────────────
# Channel logic
# ─────────────────────────────────────────────

# Actions that count as "contacting" a lead
CONTACT_ACTIONS = {"send_email", "make_call", "send_message", "request_documents", "run_campaign"}

# Map action_type → channel for duplicate/preferred-channel checks
ACTION_CHANNEL_MAP = {
    "send_email": "email",
    "make_call": "call",
    "send_message": "message",
    "request_documents": "email",   # document requests go via email by default
    "run_campaign": "email",
}

# Non-contact actions — always safe, small neutral reward
NON_CONTACT_ACTIONS = {"update_crm", "schedule_followup", "escalate", "wait"}


def _recent_channels(lead: LeadProfile, window: int = 1) -> List[str]:
    """Return channels used in the last `window` interactions."""
    history = lead.conversation_history[-window:] if lead.conversation_history else []
    return [h.get("channel", "") for h in history]


# ─────────────────────────────────────────────
# Individual reward components
# ─────────────────────────────────────────────

def reward_channel_choice(action: Action, lead: LeadProfile) -> Tuple[float, str]:
    """
    +0.2 correct channel (matches preferred or best fit for context)
    -0.2 wrong channel
    """
    action_channel = ACTION_CHANNEL_MAP.get(action.action_type)
    if action_channel is None:
        return 0.0, "non-contact action, no channel reward"

    preferred = lead.preferred_channel
    # "any" means no preference — reward any channel used thoughtfully
    if preferred == "any":
        return 0.2, f"channel {action_channel} acceptable (no preference set)"

    if action_channel == preferred:
        return 0.2, f"correct channel: {action_channel} matches lead preference"
    else:
        return -0.2, f"wrong channel: used {action_channel}, lead prefers {preferred}"


def reward_timing(action: Action, lead: LeadProfile) -> Tuple[float, str]:
    """
    +0.1 contacting within or at the followup window
    -0.1 contacting too soon (< 1 day since last contact, not a new lead)
    """
    if action.action_type not in CONTACT_ACTIONS:
        return 0.0, "non-contact action, no timing reward"

    days = lead.days_since_last_contact
    due = lead.next_followup_due

    # Brand new lead — first contact is always on time
    if lead.last_contact_channel == "none":
        return 0.1, "first contact — timing valid"

    # Too soon
    if days < 1:
        return -0.1, f"contacted too soon ({days}d since last contact)"

    # On time or overdue
    if days >= due:
        return 0.1, f"follow-up on time ({days}d elapsed, due at {due}d)"

    # Slightly early but not spammy
    return 0.05, f"slightly early follow-up ({days}d elapsed, due at {due}d)"


def reward_message_quality(action: Action, lead: LeadProfile, product: ProductProfile) -> Tuple[float, str]:
    """
    +0.2 message body is non-empty, mentions lead name, relevant to context
    Partial: +0.1 if body present but generic
    0.0 if body empty
    """
    if action.action_type in NON_CONTACT_ACTIONS and action.action_type != "run_campaign":
        return 0.0, "non-message action"

    body = (action.body or "").strip()
    if not body:
        return 0.0, "empty message body"

    score = 0.0
    reasons = []

    # Body exists
    score += 0.1
    reasons.append("body present")

    # Personalised — mentions lead name or company
    name_lower = lead.lead_name.split()[0].lower()
    company_lower = lead.company.lower()
    body_lower = body.lower()

    if name_lower in body_lower or company_lower in body_lower:
        score += 0.05
        reasons.append("personalised")

    # Relevant — mentions a product feature, value prop, or addresses objection
    product_terms = (
        [f.lower() for f in product.features] +
        [v.lower() for v in product.value_props] +
        [o.lower() for o in lead.objections]
    )
    if any(term[:8] in body_lower for term in product_terms if len(term) >= 5):
        score += 0.05
        reasons.append("product-relevant")

    return min(score, 0.2), ", ".join(reasons)


def reward_crm_update(action: Action) -> Tuple[float, str]:
    """
    +0.1 when agent explicitly calls update_crm
    """
    if action.action_type == "update_crm":
        return 0.1, "CRM updated"
    return 0.0, "no CRM update"


def reward_lead_progression(action: Action, lead: LeadProfile, new_stage: str) -> Tuple[float, str]:
    """
    +0.3 if the action caused the lead to advance a stage
    0.0  if stage unchanged
    -0.1 if stage regressed (shouldn't happen but guard it)
    """
    STAGE_ORDER = [
        "new", "contacted", "qualified", "awaiting_docs",
        "proposal_sent", "negotiating", "closed_won"
    ]
    old_idx = STAGE_ORDER.index(lead.lead_stage) if lead.lead_stage in STAGE_ORDER else 0
    new_idx = STAGE_ORDER.index(new_stage) if new_stage in STAGE_ORDER else 0

    if new_idx > old_idx:
        return 0.3, f"lead progressed: {lead.lead_stage} → {new_stage}"
    if new_idx < old_idx:
        return -0.1, f"lead regressed: {lead.lead_stage} → {new_stage}"
    return 0.0, "stage unchanged"


def reward_policy_compliance(action: Action, lead: LeadProfile) -> Tuple[float, str]:
    """
    +0.1 compliant action
    -0.5 consent violated
    -0.2 contacted via explicitly wrong channel when preference is set
    """
    # Consent check — hardest penalty
    if not lead.consent and action.action_type in CONTACT_ACTIONS:
        return -0.5, "CONSENT VIOLATION: lead has consent=False"

    return 0.1, "policy compliant"


def reward_duplicate_outreach(action: Action, lead: LeadProfile) -> Tuple[float, str]:
    """
    -0.2 if the same channel was used in the immediately preceding step
    """
    if action.action_type not in CONTACT_ACTIONS:
        return 0.0, "non-contact action"

    recent = _recent_channels(lead, window=1)
    action_channel = ACTION_CHANNEL_MAP.get(action.action_type, "")

    if action_channel and recent and recent[-1] == action_channel:
        return -0.2, f"duplicate outreach: {action_channel} used in last step too"

    return 0.0, "no duplicate"


def reward_looping(step_count: int, max_steps: int, done: bool) -> Tuple[float, str]:
    """
    -0.1 if agent is burning steps without progressing
    (applied when > 80% of budget used and episode not done)
    """
    if not done and step_count >= int(max_steps * 0.8):
        return -0.1, f"inefficient: {step_count}/{max_steps} steps used, not done"
    return 0.0, "step budget ok"


# ─────────────────────────────────────────────
# Master reward calculator
# ─────────────────────────────────────────────

def calculate_reward(
    action: Action,
    lead_before: LeadProfile,
    lead_after_stage: str,
    product: ProductProfile,
    step_count: int,
    max_steps: int,
    done: bool,
) -> Tuple[float, Dict[str, Any]]:
    """
    Returns (total_reward, breakdown_dict).
    breakdown_dict has each component for transparency.
    """
    components: Dict[str, Tuple[float, str]] = {}

    components["channel"]     = reward_channel_choice(action, lead_before)
    components["timing"]      = reward_timing(action, lead_before)
    components["message"]     = reward_message_quality(action, lead_before, product)
    components["crm_update"]  = reward_crm_update(action)
    components["progression"] = reward_lead_progression(action, lead_before, lead_after_stage)
    components["compliance"]  = reward_policy_compliance(action, lead_before)
    components["duplicate"]   = reward_duplicate_outreach(action, lead_before)
    components["looping"]     = reward_looping(step_count, max_steps, done)

    total = sum(v for v, _ in components.values())
    total = round(max(-1.0, min(1.5, total)), 4)  # soft clamp

    breakdown = {
        k: {"reward": round(v, 4), "reason": r}
        for k, (v, r) in components.items()
    }

    return total, breakdown