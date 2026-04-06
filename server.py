"""
server.py — E.L.A.R.A. FastAPI server
Endpoints: /health /reset /step /state /tasks /grader /ws
"""

import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "app"))

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from environment import ElaraEnv
from models import Action, EpisodeState
from grader import grade

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


class StepRequest(BaseModel):
    action: Action


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
    import json
    from pathlib import Path
    task_dir = Path(__file__).parent / "tasks"
    result = []
    for tid in ["easy", "medium", "hard", "consent"]:
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


# ─────────────────────────────────────────────
# WebSocket endpoint (OpenEnv SDK)
# ─────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            msg_type = msg.get("type", "")
            data = msg.get("data", {})

            if msg_type == "reset":
                task_id = data.get("task_id", "easy")
                seed = data.get("seed")
                obs = env.reset(task_id=task_id, seed=seed)
                await ws.send_text(json.dumps({
                    "type": "reset",
                    "data": {
                        "observation": obs.model_dump(),
                        "reward": 0.0,
                        "done": False,
                    },
                }))

            elif msg_type == "step":
                action_data = data.get("action", data)
                action = Action(**action_data)
                obs, reward, done, info = env.step(action)
                await ws.send_text(json.dumps({
                    "type": "step",
                    "data": {
                        "observation": obs.model_dump(),
                        "reward": reward,
                        "done": done,
                        "info": info,
                    },
                }))

            elif msg_type == "state":
                state_data = env.state()
                await ws.send_text(json.dumps({
                    "type": "state",
                    "data": state_data,
                }))

            elif msg_type == "close":
                await ws.close()
                break

            else:
                await ws.send_text(json.dumps({
                    "type": "error",
                    "data": {"message": f"Unknown message type: {msg_type}"},
                }))

    except WebSocketDisconnect:
        pass
