---
title: IncidentBench
colorFrom: red
colorTo: pink
sdk: docker
pinned: false
---

# IncidentBench

**Adversarial On-Call Environment for AI Agent Evaluation**

An OpenEnv-compatible environment where an AI agent plays the role of an on-call engineer responding to production incidents. The environment injects typed, seeded failures designed to challenge and differentiate frontier models.

---

## Motivation

Every SRE and on-call engineer faces cascading production incidents. Current AI agent benchmarks test either pure reasoning or pure tool use — IncidentBench tests both together under adversarial noise. The environment is directly useful for evaluating and training agents for real-world DevOps automation.

---

## Environment Description

The agent receives PagerDuty-style alerts and must diagnose and fix a production incident using 5 tools. The environment injects up to 4 adversarial failure types to prevent simple pattern-matching:

| Failure Type | Description |
|---|---|
| Type A | Logs vanish mid-episode (after step 1 or 2) |
| Type B | Red herring alert fires for an unrelated service |
| Type C | All metrics are 10 minutes stale — current data unavailable |
| Type D | Conflicting runbooks give contradictory remediation advice |

---

## Action Space

The agent may take exactly one action per step, chosen from:

| Action | Required Params | Description |
|---|---|---|
| `query_logs` | `service` | Fetch recent log lines for a service |
| `query_metrics` | `service`, `metric_name` | Fetch a metric value for a service |
| `read_runbook` | `incident_type` | Read the remediation runbook for an incident type |
| `apply_fix` | `service`, `fix_type` | Apply a remediation fix |
| `escalate` | `reason` (optional) | Escalate to a human — ends the episode |

**Valid service names:** `api_gateway`, `auth_service`, `database`, `cache`

**Valid incident types:** `high_latency`, `auth_failure`, `db_connection`, `cache_miss_spike`

**Valid fix types:** `restart_service`, `rollback_deploy`, `flush_cache`, `scale_up`, `rotate_credentials`

---

## Observation Space

Each step returns a structured observation containing:

| Field | Type | Description |
|---|---|---|
| `active_alerts` | list | Firing alerts with severity, service, message, timestamp |
| `tool_response` | dict | Result of the last action (logs, metrics, runbook, or error) |
| `system_state` | dict | Current health of each service: `healthy` / `degraded` / `down` |
| `step_count` | int | Current step number |
| `max_steps` | int | Maximum steps allowed (10) |
| `last_action_error` | str or null | Error message if the last action was invalid |

---

## Tasks

### Easy — Single Service Outage
- **Scenario:** Database is down. Logs are clean. One runbook applies.
- **Adversarial failures:** None
- **Target score:** 0.8 for frontier models
- **Scoring:** Root cause (0.4) + correct fix (0.4) + efficiency (0.2) − destructive penalty

### Medium — Upstream Auth Failure with Noise
- **Scenario:** Auth service JWT key expiry cascades to API gateway.
- **Adversarial failures:** Type A (logs vanish after step 2) + Type B (red herring cache alert)
- **Target score:** 0.5 for frontier models
- **Scoring:** Root cause (0.3) + red herring ignored (0.2) + correct fix (0.4) + efficiency (0.1)

### Hard — Cascading Failure, All Adversarial Modes
- **Scenario:** Three services failing in cascade. Fix ORDER matters — applying cache fix before auth fix worsens the outage.
- **Adversarial failures:** All 4 types active (A + B + C + D)
- **Note on stale metrics (Type C):** The hard task deliberately returns metrics showing `hsm_connected=1` and `error_rate=0.0` for auth_service while logs clearly show the HSM is down. This is intentional — the metrics pipeline is 10 minutes stale. An agent that cross-references metrics then logs (in that order) will detect the staleness. This is not a bug in the environment.
- **Target score:** 0.2–0.35 for frontier models
- **Scoring:** Root cause (0.25) + red herring ignored (0.15) + stale metrics detected (0.15) + correct fixes in order (0.45)

---

## Reward Function

Dense reward — signal provided at every step, not just episode end.

| Event | Reward |
|---|---|
| Each step taken | −0.01 (step cost) |
| Relevant tool use (first time) | +0.10 |
| Root cause identified (with prior evidence) | +0.20 |
| Correct fix applied | +0.40 |
| Efficiency bonus (≤3 steps on easy) | +0.20 |
| Destructive action | −0.15 to −0.20 |
| Early escalation (step 1) | −0.20 |
| Late escalation on hard (step 5+) | +0.10 |

All episode scores are clamped to [0.0, 1.0].

---

## Baseline Scores

Measured using `inference.py` with `meta-llama/Llama-3.3-70B-Instruct` via HuggingFace router. Actual scores may vary by ±0.1 depending on model temperature and API response variance.

| Task | Score | Passed |
|---|---|---|
| Easy | 0.90 | ✅ |
| Medium | 0.50 | ✅ |
| Hard | 0.25 | ✅ |
| **Average** | **0.55** | |

---

## Setup & Usage

### Requirements

- Python 3.10+
- Docker (for containerized run)
- HuggingFace account + token

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the server locally

```bash
uvicorn server:app --host 0.0.0.0 --port 7860
```

### Run with Docker

```bash
docker build -t incidentbench .
docker run -p 7860:7860 incidentbench
```

### Run the baseline inference script

```bash
export HF_TOKEN=your_token_here
export MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct
export API_BASE_URL=https://router.huggingface.co/v1
export ENV_BASE_URL=https://your-space.hf.space

python inference.py
```

### API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Liveness check — returns 200 |
| POST | `/reset` | Reset environment `{"task": "easy", "seed": 42}` |
| POST | `/step` | Execute action `{"action_type": "query_logs", "service": "database"}` |
| GET | `/state` | Full internal state for grading |
| POST | `/grade` | Grade the current episode |
| GET | `/tasks` | List all tasks with metadata |

### Run the pre-submission validator

```bash
./validate-submission.sh https://your-space.hf.space
```

---

## Project Structure

```
incidentbench/
├── env.py           # Core environment logic, Pydantic models, adversarial injections
├── graders.py       # Deterministic task graders — scores 0.0 to 1.0
├── server.py        # FastAPI server exposing OpenEnv HTTP endpoints
├── inference.py     # Baseline inference script — runs all 3 tasks
├── openenv.yaml     # OpenEnv spec metadata
├── requirements.txt # Pinned Python dependencies
├── Dockerfile       # Container definition for HuggingFace Spaces
└── README.md        # This file
```