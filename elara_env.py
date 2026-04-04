# elara_env.py — OpenEnv client for E.L.A.R.A.
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from openenv.core import EnvClient
from openenv.core.client_types import StepResult


@dataclass
class ElaraAction:
    action_type: str
    target_lead_id: str
    subject: Optional[str] = None
    body: str = ""
    goal: str = ""
    priority: str = "medium"
    metadata: Dict[str, Any] = field(default_factory=dict)


class ElaraEnv(EnvClient):

    def _step_payload(self, action: ElaraAction) -> Dict[str, Any]:
        action_dict = {k: v for k, v in asdict(action).items() if v is not None}
        return {"action": action_dict}

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult:
        return StepResult(
            observation=payload.get("observation", {}),
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return payload
