from typing import Any, Dict, List, Literal, Optional

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


class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


class State(BaseModel):
    product: ProductProfile
    leads: Dict[str, LeadProfile]
    current_lead_id: str
    task_id: str
    step_count: int = 0
    done: bool = False
    episode_log: List[Dict[str, Any]] = Field(default_factory=list)