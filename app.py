from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from environment import ElaraEnv
from models import Action

app = FastAPI(title="E.L.A.R.A.", version="0.1.0")
env = ElaraEnv()


class StepRequest(BaseModel):
    action: Action


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