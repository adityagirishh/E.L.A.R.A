from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────
# Product
# ─────────────────────────────────────────────

class ProductProfile(BaseModel):
    product_id: str
    product_name: str
    website_url: str = "local://hardcoded"
    description: str
    features: List[str] = Field(default_factory=list)
    target_users: List[str] = Field(default_factory=list)
    pricing_summary: str = ""
    objections: List[str] = Field(default_factory=list)
    compliance_notes: List[str] = Field(default_factory=list)
    value_props: List[str] = Field(default_factory=list)
    faqs: List[Dict[str, str]] = Field(default_factory=list)


# ─────────────────────────────────────────────
# Lead
# ─────────────────────────────────────────────

LeadStage = Literal[
    "new", "contacted", "qualified", "awaiting_docs",
    "proposal_sent", "negotiating", "closed_won", "closed_lost"
]

Channel = Literal["none", "email", "call", "message"]

class LeadProfile(BaseModel):
    lead_id: str
    lead_name: str
    company: str = ""
    role: str = ""
    lead_stage: LeadStage = "new"
    last_contact_channel: Channel = "none"
    days_since_last_contact: int = 0
    next_followup_due: int = 3
    consent: bool = True
    sentiment: Literal["cold", "neutral", "warm", "hot"] = "neutral"
    objections: List[str] = Field(default_factory=list)
    documents_pending: bool = False
    preferred_channel: Literal["email", "call", "message", "any"] = "any"
    conversation_history: List[Dict[str, Any]] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


# ─────────────────────────────────────────────
# Action  (all 9 types from spec)
# ─────────────────────────────────────────────

ActionType = Literal[
    "send_email",
    "make_call",
    "send_message",
    "request_documents",
    "update_crm",
    "schedule_followup",
    "run_campaign",
    "escalate",
    "wait",
]

class Action(BaseModel):
    action_type: ActionType
    target_lead_id: str
    subject: Optional[str] = None
    body: str = ""
    goal: str = ""
    priority: Literal["low", "medium", "high"] = "medium"
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ─────────────────────────────────────────────
# Lead summary (for multi-lead observations)
# ─────────────────────────────────────────────

class LeadSummary(BaseModel):
    lead_id: str
    lead_name: str
    company: str
    lead_stage: str
    sentiment: str
    preferred_channel: str
    documents_pending: bool
    consent: bool
    days_since_last_contact: int
    objections: List[str] = Field(default_factory=list)


# ─────────────────────────────────────────────
# Observation
# ─────────────────────────────────────────────

class PolicyConstraints(BaseModel):
    must_use_channel: Optional[str] = None
    min_days_since_contact: int = 0
    consent_required: bool = True
    allow_campaign: bool = False
    max_steps_remaining: int = 5

class Observation(BaseModel):
    task_id: str
    step_count: int = 0
    # Primary lead (the one last acted on or the default)
    lead_id: str
    lead_name: str
    company: str
    role: str
    lead_stage: str
    last_contact_channel: str
    days_since_last_contact: int
    next_followup_due: int
    consent: bool
    sentiment: str
    documents_pending: bool
    preferred_channel: str
    objections: List[str] = Field(default_factory=list)
    # Multi-lead: summaries of ALL active leads in this task
    active_leads: List[LeadSummary] = Field(default_factory=list)
    # Dynamic responses from leads after contact actions
    lead_responses: List[Dict[str, Any]] = Field(default_factory=list)
    # Product, history, policy, actions, hint
    product_context: Dict[str, Any] = Field(default_factory=dict)
    recent_history: List[Dict[str, Any]] = Field(default_factory=list)
    policy_constraints: PolicyConstraints = Field(default_factory=PolicyConstraints)
    available_actions: List[str] = Field(default_factory=list)
    task_hint: str = ""


# ─────────────────────────────────────────────
# Step result
# ─────────────────────────────────────────────

class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


# ─────────────────────────────────────────────
# Full episode state
# ─────────────────────────────────────────────

class EpisodeState(BaseModel):
    product: ProductProfile
    leads: Dict[str, LeadProfile]
    current_lead_id: str
    active_lead_ids: List[str] = Field(default_factory=list)
    task_id: str
    step_count: int = 0
    max_steps: int = 5
    done: bool = False
    total_reward: float = 0.0
    episode_log: List[Dict[str, Any]] = Field(default_factory=list)
    lead_responses: List[Dict[str, Any]] = Field(default_factory=list)
    seed: Optional[int] = None
