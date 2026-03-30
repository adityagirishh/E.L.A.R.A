<<<<<<< HEAD
"""
app.py — E.L.A.R.A. FastAPI server
Endpoints: /health /reset /step /state /tasks /grader
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "app"))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.environment import ElaraEnv
from app.models import Action, EpisodeState
from app.grader import grade

app = FastAPI(
    title="E.L.A.R.A.",
    description="Extensive LLM Applied Reasoning Agent — sandboxed sales ops environment",
    version="0.1.0",
)

env = ElaraEnv()


# ─────────────────────────────────────────────
# Request/response wrappers
# ─────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: str = "easy"
    seed: int | None = None

=======
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from environment import ElaraEnv
from models import Action

app = FastAPI(title="E.L.A.R.A.", version="0.1.0")
env = ElaraEnv()


>>>>>>> 8cffedc4c8dc68dd752a6a32295fc5d3f86bccf5
class StepRequest(BaseModel):
    action: Action


<<<<<<< HEAD
# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "project": "E.L.A.R.A.", "version": "0.1.0"}


@app.post("/reset")
def reset(req: ResetRequest = ResetRequest()):
    try:
        obs = env.reset(task_id=req.task_id, seed=req.seed)
        return {"observation": obs.model_dump()}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step")
def step(req: StepRequest):
    try:
        obs, reward, done, info = env.step(req.action)
        return {
            "observation": obs.model_dump(),
=======
@app.get("/health")
def health():
    return {"status": "ok", "project": "E.L.A.R.A."}


@app.post("/reset")
def reset():
    obs = env.reset()
    return {"observation": obs.model_dump()}


@app.post("/step")
def step(payload: StepRequest):
    try:
        observation, reward, done, info = env.step(payload.action)
        return {
            "observation": observation.model_dump(),
>>>>>>> 8cffedc4c8dc68dd752a6a32295fc5d3f86bccf5
            "reward": reward,
            "done": done,
            "info": info,
        }
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state")
def state():
    try:
        return env.state()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/tasks")
def tasks():
<<<<<<< HEAD
    import json
    from pathlib import Path
    task_dir = Path(__file__).parent / "tasks"
    result = []
    for tid in ["easy", "medium", "hard"]:
        p = task_dir / f"{tid}.json"
        if p.exists():
            result.append(json.loads(p.read_text()))
    return {"tasks": result}


@app.post("/grader")
def grader():
    try:
        raw = env.state()
        s = EpisodeState(**raw)
        result = grade(s)
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
=======
    return {
        "tasks": [
            {
                "task_id": "easy",
                "name": "First-touch outreach",
                "difficulty": "easy",
                "goal": "Choose the right channel and send a relevant first contact.",
            },
            {
                "task_id": "medium",
                "name": "Follow-up handling",
                "difficulty": "medium",
                "goal": "Respond correctly to lead context and request missing docs.",
            },
            {
                "task_id": "hard",
                "name": "Multi-channel orchestration",
                "difficulty": "hard",
                "goal": "Sequence call/email/message actions with timing and compliance.",
            },
        ],
        "action_schema": {
            "action_type": ["email", "call", "message"],
            "target_lead_id": "string",
            "content": "string",
            "subject": "optional string",
            "metadata": "optional object",
        },
    }


@app.get("/grader")
def grader():
    if env.state_obj is None:
        raise HTTPException(status_code=400, detail="Environment not reset.")
    return {
        "status": "ready",
        "step_count": env.state_obj.step_count,
        "done": env.state_obj.done,
        "note": "Day 1 grader stub. Full task scoring comes next.",
    }
>>>>>>> 8cffedc4c8dc68dd752a6a32295fc5d3f86bccf5
