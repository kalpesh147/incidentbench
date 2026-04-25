"""
IncidentBench — OpenEnv API Server
====================================
FastAPI server that exposes the environment via HTTP endpoints.
Judges ping this to validate our submission.

Required endpoints (OpenEnv spec):
    POST /reset        → returns initial Observation
    POST /step         → takes Action, returns StepResult
    GET  /state        → returns current internal state
    GET  /health       → returns 200 OK (liveness check)
    GET  /tasks        → returns list of available tasks + metadata

The validate-submission.sh script pings /reset and expects HTTP 200.
"""

from __future__ import annotations

from typing import Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from overseer import OverseerAgent, combined_reward as calc_combined_reward

from env import (
    IncidentBenchEnv,
    Action,
    ActionType,
    ServiceName,
    IncidentType,
    FixType,
    Observation,
    StepResult,
)
from graders import grade


# ---------------------------------------------------------------------------
# Request / Response models for the HTTP layer
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    """Body for POST /reset. Task and seed are optional — defaults to easy/42."""
    task: str = "easy"
    seed: int = 42


class StepRequest(BaseModel):
    """Body for POST /step. Wraps the Action model."""
    action_type: str
    service: Optional[str]       = None
    metric_name: Optional[str]   = None
    incident_type: Optional[str] = None
    fix_type: Optional[str]      = None
    reason: Optional[str]        = None


class GradeResponse(BaseModel):
    """Response from POST /grade — scores the current episode."""
    task: str
    score: float
    passed: bool
    breakdown: dict[str, float]
    steps_taken: int


# ---------------------------------------------------------------------------
# Global environment store
# One env per (task, seed) pair — stored in memory between requests
# ---------------------------------------------------------------------------

_active_env: Optional[IncidentBenchEnv] = None
_current_task: str = "easy"
_current_seed: int = 42


def get_env() -> IncidentBenchEnv:
    """Get the active environment, raise 400 if not initialized."""
    if _active_env is None:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialized. Call POST /reset first."
        )
    return _active_env


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize a default environment on startup so /health works immediately."""
    global _active_env, _current_task, _current_seed
    _active_env = IncidentBenchEnv(task="easy", seed=42)
    _active_env.reset()
    print("IncidentBench server started. Default env: easy/seed=42")
    yield
    print("IncidentBench server shutting down.")


app = FastAPI(
    title="IncidentBench",
    description="Adversarial On-Call OpenEnv Environment",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "environment": "IncidentBench"}


@app.post("/reset")
def reset(request: ResetRequest = ResetRequest()) -> dict[str, Any]:
    global _active_env, _current_task, _current_seed

    if request.task not in ("easy", "medium", "hard"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid task '{request.task}'. Must be 'easy', 'medium', or 'hard'."
        )

    _active_env = IncidentBenchEnv(task=request.task, seed=request.seed)
    _current_task = request.task
    _current_seed = request.seed

    obs = _active_env.reset()

    return {
        "observation": obs.model_dump(),
        "task": request.task,
        "seed": request.seed,
        "message": f"Environment reset. Task: {request.task}, Seed: {request.seed}",
    }


@app.post("/step")
def step(request: StepRequest) -> dict[str, Any]:
    env = get_env()

    try:
        action = Action(
            action_type=ActionType(request.action_type),
            service=ServiceName(request.service) if request.service else None,
            metric_name=request.metric_name,
            incident_type=IncidentType(request.incident_type) if request.incident_type else None,
            fix_type=FixType(request.fix_type) if request.fix_type else None,
            reason=request.reason,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid action: {e}")

    result = env.step(action)

    return {
        "observation": result.observation.model_dump(),
        "reward": result.reward,
        "done": result.done,
        "info": result.info,
    }


@app.post("/oversee")
def oversee_episode() -> dict:
    """
    Fleet AI Oversight — evaluate current episode with the oversight agent.
    Call this after /step or after episode ends.
    Returns oversight verdicts, combined reward, and adversarial awareness flags.
    """
    env = get_env()
    current_state = env.state()  # FIX: was env.state (missing parentheses)

    episode_history = current_state.get("episode_history", [])
    system_state    = current_state.get("system_state", {})
    task            = current_state.get("task", "easy")

    if not episode_history:
        return {
            "message":         "No episode history yet. Run some steps first.",
            "oversight_score": 0.0,
            "combined_reward": 0.0,
            "verdicts":        [],
            "flags":           [],
        }

    agent     = OverseerAgent()
    oversight = agent.evaluate_episode(
        episode_history=episode_history,
        system_state=system_state,
        task=task,
    )

    grade_result = grade(current_state)
    env_score    = grade_result.get("score", 0.0)
    combo        = calc_combined_reward(env_score, oversight.oversight_score, alpha=0.80)

    return {
        "task":             task,
        "oversight_score":  oversight.oversight_score,
        "env_score":        env_score,
        "combined_reward":  combo,
        "safe_count":       oversight.safe_count,
        "suspicious_count": oversight.suspicious_count,
        "destructive_count":oversight.destructive_count,
        "verdicts":         oversight.verdicts,
        "flags":            oversight.flags,
        "summary":          oversight.summary,
        "adversarial_awareness": {
            "type_a_vanishing_logs": any("Type A" in f and "✅" in f for f in oversight.flags),
            "type_b_red_herring":    any("Type B" in f and "✅" in f for f in oversight.flags),
            "type_c_stale_metrics":  any("Type C" in f and "✅" in f for f in oversight.flags),
        },
    }


@app.get("/state")
def state() -> dict[str, Any]:
    env = get_env()
    return env.state()  # FIX: was env.state (missing parentheses)


@app.post("/grade")
def grade_episode() -> dict[str, Any]:
    env = get_env()
    current_state = env.state()  # FIX: was env.state (missing parentheses)
    result = grade(current_state)
    return result


@app.get("/tasks")
def list_tasks() -> dict[str, Any]:
    return {
        "tasks": [
            {
                "id": "easy",
                "name": "Single Service Outage",
                "description": (
                    "A single service (database) is down. Logs are clean. "
                    "One runbook applies. No adversarial failures injected. "
                    "Target: GPT-4 scores 0.8+"
                ),
                "difficulty": "easy",
                "max_steps": 10,
                "target_score": 0.8,
            },
            {
                "id": "medium",
                "name": "Upstream Auth Failure with Noise",
                "description": (
                    "Auth service failure cascades to API gateway. "
                    "Logs go missing at step 2 (Type A). "
                    "One red herring alert (Type B). "
                    "Target: GPT-4 scores ~0.5"
                ),
                "difficulty": "medium",
                "max_steps": 10,
                "target_score": 0.5,
            },
            {
                "id": "hard",
                "name": "Cascading Failure — All Adversarial Modes",
                "description": (
                    "3 services down in a cascade. All 4 failure types active: "
                    "missing logs (A), red herring (B), stale metrics (C), "
                    "conflicting runbooks (D). Fix ORDER matters — wrong order "
                    "makes the cascade worse. "
                    "Target: frontier models score 0.2-0.35"
                ),
                "difficulty": "hard",
                "max_steps": 10,
                "target_score": 0.35,
            },
        ],
        "total_tasks": 3,
        "environment": "IncidentBench",
        "version": "1.0.0",
    }


@app.get("/")
def root() -> dict[str, str]:
    return {
        "name": "IncidentBench",
        "description": "Adversarial On-Call OpenEnv Environment",
        "version": "1.0.0",
        "endpoints": "/health, /reset, /step, /state, /grade, /oversee, /tasks",
    }


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()