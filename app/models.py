from typing import Any, Dict, List, Literal, Optional
<<<<<<< HEAD
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────
# Product
# ─────────────────────────────────────────────
=======

from pydantic import BaseModel, Field


class LeadProfile(BaseModel):
    lead_id: str
    lead_name: str
    lead_stage: str = "new"
    last_contact_channel: str = "none"
    days_since_last_contact: int = 0
    consent: bool = True
    sentiment: str = "neutral"
    objections: List[str] = Field(default_factory=list)
    documents_pending: bool = False
    conversation_history: List[Dict[str, Any]] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)

>>>>>>> 8cffedc4c8dc68dd752a6a32295fc5d3f86bccf5

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
<<<<<<< HEAD
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
    product_context: Dict[str, Any] = Field(default_factory=dict)
    recent_history: List[Dict[str, Any]] = Field(default_factory=list)
    policy_constraints: PolicyConstraints = Field(default_factory=PolicyConstraints)
    available_actions: List[str] = Field(default_factory=list)
    task_hint: str = ""


# ─────────────────────────────────────────────
# Step result
# ─────────────────────────────────────────────
=======


class Observation(BaseModel):
    task_id: str
    product_name: str
    lead_id: str
    lead_name: str
    lead_stage: str
    last_contact_channel: str
    days_since_last_contact: int
    consent: bool
    sentiment: str
    documents_pending: bool
    product_context: Dict[str, Any] = Field(default_factory=dict)
    recent_history: List[Dict[str, Any]] = Field(default_factory=list)
    available_actions: List[str] = Field(default_factory=list)


class Action(BaseModel):
    action_type: Literal["email", "call", "message"]
    target_lead_id: str
    content: str = ""
    subject: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

>>>>>>> 8cffedc4c8dc68dd752a6a32295fc5d3f86bccf5

class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


<<<<<<< HEAD
# ─────────────────────────────────────────────
# Full episode state
# ─────────────────────────────────────────────

class EpisodeState(BaseModel):
=======
class State(BaseModel):
>>>>>>> 8cffedc4c8dc68dd752a6a32295fc5d3f86bccf5
    product: ProductProfile
    leads: Dict[str, LeadProfile]
    current_lead_id: str
    task_id: str
    step_count: int = 0
<<<<<<< HEAD
    max_steps: int = 5
    done: bool = False
    total_reward: float = 0.0
    episode_log: List[Dict[str, Any]] = Field(default_factory=list)
    seed: Optional[int] = None
=======
    done: bool = False
    episode_log: List[Dict[str, Any]] = Field(default_factory=list)
>>>>>>> 8cffedc4c8dc68dd752a6a32295fc5d3f86bccf5
