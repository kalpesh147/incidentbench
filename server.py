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

# We keep a single active environment instance
# Judges run one episode at a time so this is sufficient
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

# Allow all origins — needed for HuggingFace Spaces
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
    """
    Liveness check. validate-submission.sh pings this.
    Must return 200 always.
    """
    return {"status": "ok", "environment": "IncidentBench"}


@app.post("/reset")
def reset(request: ResetRequest = ResetRequest()) -> dict[str, Any]:
    """
    Reset the environment to a fresh episode.
    Returns the initial Observation as a dict.

    This is the FIRST endpoint validate-submission.sh tests.
    Must return HTTP 200 with a valid observation.
    """
    global _active_env, _current_task, _current_seed

    # Validate task
    if request.task not in ("easy", "medium", "hard"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid task '{request.task}'. Must be 'easy', 'medium', or 'hard'."
        )

    # Create fresh environment
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
    """
    Execute one action in the environment.
    Returns StepResult: observation, reward, done, info.
    """
    env = get_env()

    # Convert request into Action — validate enum values
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


@app.get("/state")
def state() -> dict[str, Any]:
    """
    Return full internal environment state.
    Used by graders to score the episode after it ends.
    """
    env = get_env()
    return env.state()


@app.post("/grade")
def grade_episode() -> dict[str, Any]:
    """
    Grade the current episode. Call after episode is done.
    Returns score breakdown for the current task.
    """
    env = get_env()
    current_state = env.state()
    result = grade(current_state)
    return result


@app.get("/tasks")
def list_tasks() -> dict[str, Any]:
    """
    List all available tasks with metadata.
    Required by OpenEnv spec.
    """
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
    """Root endpoint — basic info."""
    return {
        "name": "IncidentBench",
        "description": "Adversarial On-Call OpenEnv Environment",
        "version": "1.0.0",
        "endpoints": "/health, /reset, /step, /state, /grade, /tasks",
    }