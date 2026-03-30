"""
crm_simulator.py
20 seeded leads covering every scenario the tasks need to test.
Also handles lead stage transitions based on action type.
"""

from copy import deepcopy
from typing import Dict, Optional, Tuple
from models import Action, LeadProfile

# Import constant from reward_engine to avoid circular dep
CONTACT_ACTIONS = {
    "send_email", "make_call", "send_message",
    "request_documents", "run_campaign",
}

# ─────────────────────────────────────────────
# The 20 seed leads
# ─────────────────────────────────────────────

SEED_LEADS: Dict[str, LeadProfile] = {
    # ── EASY task leads ──────────────────────────────────────────────────
    "L-001": LeadProfile(
        lead_id="L-001", lead_name="Arun Sharma",
        company="NovaTech Solutions", role="Head of Sales",
        lead_stage="new", last_contact_channel="none",
        days_since_last_contact=0, next_followup_due=0,
        consent=True, sentiment="neutral",
        documents_pending=True, preferred_channel="email",
        notes=["Referred by partner network"],
    ),
    "L-002": LeadProfile(
        lead_id="L-002", lead_name="Meera Nair",
        company="Brightline Analytics", role="VP Operations",
        lead_stage="new", last_contact_channel="none",
        days_since_last_contact=0, next_followup_due=0,
        consent=True, sentiment="warm",
        documents_pending=False, preferred_channel="call",
        notes=["Downloaded pricing page"],
    ),
    "L-003": LeadProfile(
        lead_id="L-003", lead_name="Rahul Joshi",
        company="PeakVentures", role="COO",
        lead_stage="new", last_contact_channel="none",
        days_since_last_contact=0, next_followup_due=0,
        consent=True, sentiment="cold",
        documents_pending=False, preferred_channel="message",
        notes=["Cold inbound from website"],
    ),

    # ── MEDIUM task leads (contacted, need follow-up) ─────────────────────
    "L-004": LeadProfile(
        lead_id="L-004", lead_name="Priya Iyer",
        company="SwiftOps", role="CTO",
        lead_stage="contacted", last_contact_channel="email",
        days_since_last_contact=4, next_followup_due=3,
        consent=True, sentiment="neutral",
        documents_pending=True, preferred_channel="email",
        objections=["pricing"],
        notes=["Replied asking for detailed pricing breakdown"],
        conversation_history=[
            {"step": 1, "channel": "email", "content": "Intro email sent"},
        ],
    ),
    "L-005": LeadProfile(
        lead_id="L-005", lead_name="Kiran Desai",
        company="Vantage Systems", role="Sales Director",
        lead_stage="contacted", last_contact_channel="call",
        days_since_last_contact=5, next_followup_due=3,
        consent=True, sentiment="warm",
        documents_pending=False, preferred_channel="any",
        objections=["need to think about it"],
        notes=["Positive call, asked for follow-up in a week"],
        conversation_history=[
            {"step": 1, "channel": "call", "content": "Discovery call completed"},
        ],
    ),
    "L-006": LeadProfile(
        lead_id="L-006", lead_name="Ananya Reddy",
        company="CoreStack", role="CFO",
        lead_stage="qualified", last_contact_channel="email",
        days_since_last_contact=2, next_followup_due=5,
        consent=True, sentiment="warm",
        documents_pending=True, preferred_channel="email",
        objections=[],
        notes=["Qualified — needs contract and NDA before proceeding"],
        conversation_history=[
            {"step": 1, "channel": "email", "content": "Intro email"},
            {"step": 2, "channel": "call", "content": "Qualification call"},
        ],
    ),

    # ── HARD task leads (multi-channel orchestration) ─────────────────────
    "L-007": LeadProfile(
        lead_id="L-007", lead_name="Rajan Mehta",
        company="Apex Ventures", role="Founder & CEO",
        lead_stage="proposal_sent", last_contact_channel="email",
        days_since_last_contact=7, next_followup_due=5,
        consent=True, sentiment="hot",
        documents_pending=True, preferred_channel="call",
        objections=[],
        notes=["Proposal sent, waiting on signed docs — needs a call nudge"],
        conversation_history=[
            {"step": 1, "channel": "email", "content": "Intro email"},
            {"step": 2, "channel": "call", "content": "Positive discovery call"},
            {"step": 3, "channel": "email", "content": "Proposal sent"},
        ],
    ),
    "L-008": LeadProfile(
        lead_id="L-008", lead_name="Sunita Kapoor",
        company="BlueSky Tech", role="Head of Procurement",
        lead_stage="negotiating", last_contact_channel="call",
        days_since_last_contact=3, next_followup_due=3,
        consent=True, sentiment="warm",
        documents_pending=False, preferred_channel="message",
        objections=["pricing", "contract terms"],
        notes=["In negotiation — message preferred for quick back-and-forth"],
        conversation_history=[
            {"step": 1, "channel": "email", "content": "Intro"},
            {"step": 2, "channel": "call", "content": "Negotiation call"},
        ],
    ),

    # ── COMPLIANCE / EDGE CASE leads ──────────────────────────────────────
    "L-009": LeadProfile(
        lead_id="L-009", lead_name="Vijay Patel",
        company="DataCore", role="CEO",
        lead_stage="new", last_contact_channel="none",
        days_since_last_contact=0, next_followup_due=0,
        consent=False,  # ← do-not-contact
        sentiment="cold",
        documents_pending=False, preferred_channel="email",
        notes=["GDPR opt-out — do not contact"],
    ),
    "L-010": LeadProfile(
        lead_id="L-010", lead_name="Deepa Menon",
        company="Meridian Labs", role="Director of Engineering",
        lead_stage="contacted", last_contact_channel="email",
        days_since_last_contact=0, next_followup_due=2,  # contacted TODAY
        consent=True, sentiment="neutral",
        documents_pending=False, preferred_channel="email",
        notes=["Just emailed — too soon to contact again"],
        conversation_history=[
            {"step": 1, "channel": "email", "content": "Intro email sent today"},
        ],
    ),
    "L-011": LeadProfile(
        lead_id="L-011", lead_name="Arjun Nambiar",
        company="FusionWorks", role="VP Sales",
        lead_stage="closed_lost", last_contact_channel="call",
        days_since_last_contact=30, next_followup_due=90,
        consent=True, sentiment="cold",
        documents_pending=False, preferred_channel="any",
        objections=["went with competitor"],
        notes=["Closed lost — re-engage only after 90 days"],
    ),

    # ── ADDITIONAL realistic leads ─────────────────────────────────────────
    "L-012": LeadProfile(
        lead_id="L-012", lead_name="Nisha Gupta",
        company="Quantify AI", role="Product Manager",
        lead_stage="new", last_contact_channel="none",
        days_since_last_contact=0, next_followup_due=0,
        consent=True, sentiment="neutral",
        documents_pending=False, preferred_channel="message",
        notes=["LinkedIn inbound — prefers async comms"],
    ),
    "L-013": LeadProfile(
        lead_id="L-013", lead_name="Sameer Khan",
        company="Optics Digital", role="Marketing Head",
        lead_stage="contacted", last_contact_channel="message",
        days_since_last_contact=6, next_followup_due=3,
        consent=True, sentiment="warm",
        documents_pending=False, preferred_channel="message",
        notes=["Engaged on demo request, send follow-up message"],
        conversation_history=[
            {"step": 1, "channel": "message", "content": "Initial outreach"},
        ],
    ),
    "L-014": LeadProfile(
        lead_id="L-014", lead_name="Lavanya Suresh",
        company="ClearPath Solutions", role="Operations Lead",
        lead_stage="awaiting_docs", last_contact_channel="email",
        days_since_last_contact=8, next_followup_due=5,
        consent=True, sentiment="neutral",
        documents_pending=True, preferred_channel="email",
        notes=["Has been asked for docs twice — needs escalation"],
        conversation_history=[
            {"step": 1, "channel": "email", "content": "Intro email"},
            {"step": 2, "channel": "email", "content": "Document request 1"},
            {"step": 3, "channel": "email", "content": "Document request 2"},
        ],
    ),
    "L-015": LeadProfile(
        lead_id="L-015", lead_name="Mohan Krishnan",
        company="TerraNova Inc", role="Director Strategy",
        lead_stage="qualified", last_contact_channel="call",
        days_since_last_contact=10, next_followup_due=7,
        consent=True, sentiment="hot",
        documents_pending=False, preferred_channel="call",
        notes=["Hot lead — senior stakeholder, send proposal ASAP"],
        conversation_history=[
            {"step": 1, "channel": "call", "content": "Strong qualification call"},
        ],
    ),
    "L-016": LeadProfile(
        lead_id="L-016", lead_name="Ria Chatterjee",
        company="BrightWave", role="CEO",
        lead_stage="new", last_contact_channel="none",
        days_since_last_contact=0, next_followup_due=0,
        consent=True, sentiment="warm",
        documents_pending=False, preferred_channel="email",
        notes=["Webinar attendee — high intent signal"],
    ),
    "L-017": LeadProfile(
        lead_id="L-017", lead_name="Tanvir Alam",
        company="Helix Dynamics", role="Founder",
        lead_stage="contacted", last_contact_channel="email",
        days_since_last_contact=3, next_followup_due=3,
        consent=True, sentiment="neutral",
        documents_pending=False, preferred_channel="call",
        objections=["not the right time"],
        notes=["Said 'not now' — try a call to understand timeline"],
        conversation_history=[
            {"step": 1, "channel": "email", "content": "Intro email sent"},
        ],
    ),
    "L-018": LeadProfile(
        lead_id="L-018", lead_name="Pooja Varma",
        company="Sigma Analytics", role="Head of Finance",
        lead_stage="proposal_sent", last_contact_channel="email",
        days_since_last_contact=2, next_followup_due=3,
        consent=True, sentiment="warm",
        documents_pending=False, preferred_channel="any",
        notes=["Proposal sent 2 days ago — too early to follow up"],
        conversation_history=[
            {"step": 1, "channel": "email", "content": "Proposal sent"},
        ],
    ),
    "L-019": LeadProfile(
        lead_id="L-019", lead_name="Harish Nair",
        company="BlueNorth", role="VP Engineering",
        lead_stage="new", last_contact_channel="none",
        days_since_last_contact=0, next_followup_due=0,
        consent=True, sentiment="cold",
        documents_pending=False, preferred_channel="email",
        notes=["Cold contact from trade show — low intent"],
    ),
    "L-020": LeadProfile(
        lead_id="L-020", lead_name="Smita Deshpande",
        company="CloudFirst", role="CTO",
        lead_stage="closed_won", last_contact_channel="call",
        days_since_last_contact=5, next_followup_due=30,
        consent=True, sentiment="hot",
        documents_pending=False, preferred_channel="call",
        notes=["Customer — check in for upsell in 30 days"],
    ),
}


# ─────────────────────────────────────────────
# Stage transition logic
# ─────────────────────────────────────────────

STAGE_ORDER = [
    "new", "contacted", "qualified", "awaiting_docs",
    "proposal_sent", "negotiating", "closed_won", "closed_lost",
]

def _next_stage(current: str, action: Action, lead: LeadProfile) -> str:
    """
    Determine next stage based on current stage + action type.
    Returns the new stage string (may be same as current).
    """
    at = action.action_type

    if current == "new":
        if at in {"send_email", "make_call", "send_message"}:
            return "contacted"

    elif current == "contacted":
        if at in {"make_call", "send_email"} and not lead.objections:
            return "qualified"
        if at == "request_documents":
            return "awaiting_docs"

    elif current == "qualified":
        if at == "request_documents":
            return "awaiting_docs"
        if at in {"send_email", "make_call"} and action.goal == "send_proposal":
            return "proposal_sent"

    elif current == "awaiting_docs":
        # Only advance if metadata says docs received (set by grader)
        if action.metadata.get("docs_received"):
            return "qualified"

    elif current == "proposal_sent":
        if at in {"make_call", "send_message"} and not lead.objections:
            return "negotiating"

    elif current == "negotiating":
        if action.goal == "close_deal":
            return "closed_won"
        if action.goal == "withdraw":
            return "closed_lost"

    return current  # no change


def apply_action(lead: LeadProfile, action: Action) -> Tuple[LeadProfile, str]:
    """
    Apply an action to a lead. Returns (updated_lead, new_stage).
    Does NOT mutate the original — returns a copy.
    """
    lead = deepcopy(lead)
    at = action.action_type

    new_stage = _next_stage(lead.lead_stage, action, lead)
    lead.lead_stage = new_stage

    # Record interaction in history
    if at in CONTACT_ACTIONS:
        channel = {
            "send_email": "email",
            "make_call": "call",
            "send_message": "message",
            "request_documents": "email",
            "run_campaign": "email",
        }.get(at, at)
        lead.last_contact_channel = channel  # type: ignore[assignment]
        lead.days_since_last_contact = 0
        lead.next_followup_due = 3  # reset timer
        lead.conversation_history.append({
            "channel": channel,
            "action_type": at,
            "body": action.body[:120] if action.body else "",
            "goal": action.goal,
        })

    if at == "update_crm":
        crm_note = action.metadata.get("note", f"CRM updated at step")
        lead.notes.append(str(crm_note))

    if at == "schedule_followup":
        days = action.metadata.get("in_days", 3)
        lead.next_followup_due = int(days)

    if at == "request_documents":
        lead.documents_pending = True
        lead.lead_stage = "awaiting_docs"
        new_stage = "awaiting_docs"

    return lead, new_stage