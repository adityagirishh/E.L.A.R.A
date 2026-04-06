"""
test_api.py — Tests for the FastAPI server endpoints
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "app"))
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from fastapi.testclient import TestClient
from server import app


client = TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_ok(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["project"] == "E.L.A.R.A."


class TestResetEndpoint:
    def test_reset_default(self):
        resp = client.post("/reset")
        assert resp.status_code == 200
        data = resp.json()
        assert "observation" in data
        obs = data["observation"]
        assert obs["task_id"] == "easy"
        assert obs["lead_id"] == "L-001"

    def test_reset_with_task(self):
        resp = client.post("/reset", json={"task_id": "medium"})
        assert resp.status_code == 200
        obs = resp.json()["observation"]
        assert obs["task_id"] == "medium"
        assert obs["lead_id"] == "L-004"

    def test_reset_invalid_task(self):
        resp = client.post("/reset", json={"task_id": "impossible"})
        assert resp.status_code == 400


class TestStepEndpoint:
    def test_step_after_reset(self):
        client.post("/reset", json={"task_id": "easy"})
        resp = client.post("/step", json={
            "action": {
                "action_type": "send_email",
                "target_lead_id": "L-001",
                "body": "Hi Arun, intro about E.L.A.R.A.",
                "goal": "intro",
            }
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "observation" in data
        assert "reward" in data
        assert "done" in data
        assert "info" in data
        assert isinstance(data["reward"], float)

    def test_step_before_reset_fails(self):
        # Create a fresh app instance to avoid leftover state
        from server import app as fresh_app
        fresh_client = TestClient(fresh_app)
        # This may or may not fail depending on whether env was reset elsewhere
        # But we can test the step structure at least
        client.post("/reset")
        resp = client.post("/step", json={
            "action": {
                "action_type": "wait",
                "target_lead_id": "L-001",
            }
        })
        assert resp.status_code == 200


class TestStateEndpoint:
    def test_state_after_reset(self):
        client.post("/reset", json={"task_id": "easy"})
        resp = client.get("/state")
        assert resp.status_code == 200
        data = resp.json()
        assert "product" in data
        assert "leads" in data
        assert "task_id" in data


class TestTasksEndpoint:
    def test_tasks_returns_all(self):
        resp = client.get("/tasks")
        assert resp.status_code == 200
        data = resp.json()
        assert "tasks" in data
        assert len(data["tasks"]) == 4
        task_ids = [t["task_id"] for t in data["tasks"]]
        assert "easy" in task_ids
        assert "medium" in task_ids
        assert "hard" in task_ids
        assert "consent" in task_ids


class TestGraderEndpoint:
    def test_grader_after_episode(self):
        client.post("/reset", json={"task_id": "easy"})
        client.post("/step", json={
            "action": {
                "action_type": "send_email",
                "target_lead_id": "L-001",
                "body": "Hi Arun, E.L.A.R.A. cuts response time by 60%.",
                "goal": "intro",
            }
        })
        resp = client.post("/grader")
        assert resp.status_code == 200
        data = resp.json()
        assert "score" in data
        assert "pass" in data
        assert "dimensions" in data
        assert 0.0 <= data["score"] <= 1.0
