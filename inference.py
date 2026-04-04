"""
IncidentBench — Baseline Inference Script
==========================================
MANDATORY requirements met:
    - Uses OpenAI client for all LLM calls
    - Reads API_BASE_URL, MODEL_NAME, HF_TOKEN from environment variables
    - Named inference.py, placed in root directory
    - Produces reproducible scores on all 3 tasks
    - Runs in under 20 minutes on 2vCPU / 8GB RAM
    - Emits exact [START] / [STEP] / [END] stdout format required by spec

IMPORTANT — NO LOCAL IMPORTS:
    This script talks to the deployed HF Space over HTTP.
    It does NOT import env.py or graders.py.
    This ensures it works in the automated evaluation environment
    where only inference.py is executed against the live server.
"""

import os
import json
import re
import textwrap
import requests
from typing import Optional, List

from openai import OpenAI

# ---------------------------------------------------------------------------
# Config — all from environment variables as required by spec
# ---------------------------------------------------------------------------

# The URL of your deployed HuggingFace Space.
# Set ENV_BASE_URL to your HF Space URL before running.
# Defaults to localhost for local development.
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")

# LLM config
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
MODEL_NAME   = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")

BENCHMARK   = "incidentbench"
MAX_STEPS   = 10    # Hard cap — keeps runtime under 20min

# TEMPERATURE = 0.0 for maximum reproducibility.
# The grader is deterministic — score variance comes entirely from LLM
# non-determinism. Setting temp=0 ensures the same model produces the
# same action sequence on every run, giving reproducible scores as
# required by the Phase 2 variance check.
TEMPERATURE = 0.0

MAX_TOKENS  = 300   # Action responses are short

# Fixed seeds for reproducibility — same seed = same episode every time
TASK_SEEDS = {
    "easy":   42,
    "medium": 42,
    "hard":   42,
}

# HTTP timeout — generous to handle HF Space cold starts
HTTP_TIMEOUT = 30


# ---------------------------------------------------------------------------
# HTTP client — wraps /reset, /step, /state calls to the deployed server
# ---------------------------------------------------------------------------

def http_reset(task: str, seed: int) -> dict:
    """POST /reset — returns the initial observation dict."""
    url = f"{ENV_BASE_URL.rstrip('/')}/reset"
    resp = requests.post(
        url,
        json={"task": task, "seed": seed},
        timeout=HTTP_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def http_step(action_dict: dict) -> dict:
    """POST /step — sends action, returns {observation, reward, done, info}."""
    url = f"{ENV_BASE_URL.rstrip('/')}/step"
    resp = requests.post(url, json=action_dict, timeout=HTTP_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def http_state() -> dict:
    """GET /state — returns full internal state for grading."""
    url = f"{ENV_BASE_URL.rstrip('/')}/state"
    resp = requests.get(url, timeout=HTTP_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def http_grade() -> dict:
    """POST /grade — grades the current episode, returns score breakdown."""
    url = f"{ENV_BASE_URL.rstrip('/')}/grade"
    resp = requests.post(url, timeout=HTTP_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# MANDATORY structured stdout logging — spec requires EXACT format
# Any deviation causes incorrect evaluation scoring
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    """Emit the [START] line. Must be called once at episode begin."""
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    """
    Emit one [STEP] line immediately after env.step() returns.
    - reward formatted to 2 decimal places
    - done is lowercase boolean: true or false
    - error is the raw last_action_error string, or null if none
    """
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float], score: float = 0.0) -> None:
    """
    Emit the [END] line after episode ends. Always emitted, even on exception.
    - rewards is comma-separated, each formatted to 2 decimal places
    - guard against empty rewards list to avoid trailing comma
    """
    # Guard: if no steps were taken (crash before first step), emit 0.00
    if rewards:
        rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    else:
        rewards_str = "0.00"
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert on-call engineer responding to a production incident.
    You have access to 5 tools to diagnose and fix the problem.

    AVAILABLE ACTIONS (respond with exactly one JSON object):

    1. Query logs for a service:
       {"action_type": "query_logs", "service": "<service_name>"}

    2. Query a metric for a service:
       {"action_type": "query_metrics", "service": "<service_name>", "metric_name": "<metric>"}

    3. Read a runbook for an incident type:
       {"action_type": "read_runbook", "incident_type": "<incident_type>"}

    4. Apply a fix:
       {"action_type": "apply_fix", "service": "<service_name>", "fix_type": "<fix_type>"}

    5. Escalate if you cannot resolve:
       {"action_type": "escalate", "reason": "<your reason>"}

    VALID SERVICE NAMES:
        api_gateway, auth_service, database, cache

    VALID INCIDENT TYPES:
        high_latency, auth_failure, db_connection, cache_miss_spike

    VALID FIX TYPES:
        restart_service, rollback_deploy, flush_cache, scale_up, rotate_credentials

    STRATEGY:
    - Start by querying logs for the most suspicious service based on alerts
    - Read the relevant runbook once you suspect the incident type
    - Apply the fix the runbook recommends
    - If logs are missing or metrics look stale, cross-reference with other tools
    - Do NOT act on alerts that seem unrelated to the main incident
    - Fix root causes in order — upstream services before downstream

    CRITICAL: Respond with ONLY a valid JSON object. No explanation. No markdown. No extra text.
    Example: {"action_type": "query_logs", "service": "database"}
""").strip()


# ---------------------------------------------------------------------------
# Prompt builder — works directly with the observation dict from HTTP
# ---------------------------------------------------------------------------

def build_prompt(obs: dict, step: int, history: List[str]) -> str:
    alerts = obs.get("active_alerts", [])
    alert_lines = []
    for alert in alerts:
        alert_lines.append(
            f"  [{alert.get('severity', 'unknown').upper()}] {alert.get('service')} — "
            f"{alert.get('message')} (id: {alert.get('alert_id')}, time: {alert.get('timestamp')})"
        )
    alerts_text = "\n".join(alert_lines) if alert_lines else "  No active alerts."

    system_state = obs.get("system_state", {})
    state_lines = [f"  {service}: {status}" for service, status in system_state.items()]
    state_text = "\n".join(state_lines) if state_lines else "  No state available."

    tool_response = obs.get("tool_response")
    if tool_response:
        tool_text = json.dumps(tool_response, indent=2)
        if len(tool_text) > 1500:
            tool_text = tool_text[:1500] + "\n... (truncated)"
    else:
        tool_text = "  No tool response yet — this is the first step."

    max_steps = obs.get("max_steps", MAX_STEPS)

    if history:
        history_text = "\n".join(f"  {h}" for h in history[-5:])
    else:
        history_text = "  No actions taken yet."

    prompt = textwrap.dedent(f"""
        STEP {step} of {max_steps}

        === ACTIVE ALERTS ===
{alerts_text}

        === SYSTEM STATE ===
{state_text}

        === LAST TOOL RESPONSE ===
{tool_text}

        === ACTION HISTORY ===
{history_text}

        Based on the above, what is your next action?
        Respond with exactly one JSON object.
    """).strip()

    return prompt


# ---------------------------------------------------------------------------
# Action parser — parses LLM response into a dict for HTTP /step
# ---------------------------------------------------------------------------

# Valid enum values — used for validation without importing env.py
VALID_ACTION_TYPES   = {"query_logs", "query_metrics", "read_runbook", "apply_fix", "escalate"}
VALID_SERVICES       = {"api_gateway", "auth_service", "database", "cache"}
VALID_INCIDENT_TYPES = {"high_latency", "auth_failure", "db_connection", "cache_miss_spike"}
VALID_FIX_TYPES      = {"restart_service", "rollback_deploy", "flush_cache", "scale_up", "rotate_credentials"}


def parse_action(response_text: str) -> Optional[dict]:
    """
    Parse LLM response into a flat dict ready to POST to /step.
    Returns None if parsing fails — caller should use fallback.
    """
    if not response_text or not response_text.strip():
        return None

    text = response_text.strip()
    text = re.sub(r"```(?:json)?", "", text).strip()
    text = text.strip("`").strip()

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None

    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

    action_type = data.get("action_type", "")
    if action_type not in VALID_ACTION_TYPES:
        return None

    # Validate optional fields — drop invalid values rather than crashing
    action: dict = {"action_type": action_type}

    if "service" in data:
        if data["service"] in VALID_SERVICES:
            action["service"] = data["service"]
        else:
            return None  # invalid service = bad action

    if "metric_name" in data:
        action["metric_name"] = str(data["metric_name"])

    if "incident_type" in data:
        if data["incident_type"] in VALID_INCIDENT_TYPES:
            action["incident_type"] = data["incident_type"]
        else:
            return None

    if "fix_type" in data:
        if data["fix_type"] in VALID_FIX_TYPES:
            action["fix_type"] = data["fix_type"]
        else:
            return None

    if "reason" in data:
        action["reason"] = str(data["reason"])[:200]  # truncate for safety

    return action


# Fallback actions when LLM response can't be parsed
FALLBACK_ACTIONS = [
    {"action_type": "query_logs",    "service": "api_gateway"},
    {"action_type": "query_logs",    "service": "auth_service"},
    {"action_type": "query_metrics", "service": "database", "metric_name": "error_rate"},
    {"action_type": "read_runbook",  "incident_type": "high_latency"},
    {"action_type": "escalate",      "reason": "Unable to parse model response — escalating."},
]


def fallback_action(step: int) -> dict:
    return FALLBACK_ACTIONS[step % len(FALLBACK_ACTIONS)]


def action_to_str(action: dict) -> str:
    """Convert action dict to a compact string for the [STEP] log line."""
    parts = [action.get("action_type", "unknown")]
    if "service" in action:
        parts.append(action["service"])
    if "fix_type" in action:
        parts.append(action["fix_type"])
    if "incident_type" in action:
        parts.append(action["incident_type"])
    if "metric_name" in action:
        parts.append(action["metric_name"])
    if "reason" in action:
        parts.append(action["reason"][:40].replace("\n", " "))
    return "(" + ",".join(parts) + ")"


# ---------------------------------------------------------------------------
# Single episode runner — all env calls go over HTTP
# ---------------------------------------------------------------------------

def run_episode(client: OpenAI, task: str, seed: int) -> dict:
    """
    Run one full episode for a task against the deployed HTTP server.
    Emits [START], one [STEP] per step, [END] — always, even on exception.
    Returns the grade result dict.
    """
    history: List[str]   = []
    rewards: List[float] = []
    steps_taken          = 0
    success              = False
    grade_result         = {"score": 0.0, "passed": False}

    # [START] — emitted once, before anything else
    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

    try:
        # Reset the environment via HTTP
        reset_response = http_reset(task=task, seed=seed)
        obs = reset_response.get("observation", reset_response)

        for step in range(1, MAX_STEPS + 1):
            user_prompt = build_prompt(obs, step, history)

            # LLM call
            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": user_prompt},
                    ],
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                    stream=False,
                )
                response_text = completion.choices[0].message.content or ""
            except Exception as exc:
                print(f"[DEBUG] LLM call failed at step {step}: {exc}", flush=True)
                response_text = ""

            # Parse LLM response into action dict
            action = parse_action(response_text)
            if action is None:
                action = fallback_action(step)

            # Step the environment via HTTP
            step_response = http_step(action)

            reward = step_response.get("reward", 0.0)
            done   = step_response.get("done", False)
            obs    = step_response.get("observation", {})
            error  = obs.get("last_action_error")

            rewards.append(reward)
            steps_taken = step

            # [STEP] — emitted immediately after env.step(), every step
            log_step(
                step=step,
                action=action_to_str(action),
                reward=reward,
                done=done,
                error=error,
            )

            # Track history for prompt context
            action_summary = (
                f"Step {step}: {action.get('action_type')}"
                + (f"({action['service']})" if "service" in action else "")
                + (f" fix={action['fix_type']}" if "fix_type" in action else "")
                + f" reward={reward:+.2f}"
            )
            history.append(action_summary)

            if done:
                break

        # Grade the completed episode via HTTP
        grade_result = http_grade()
        grade_result["total_reward"] = round(sum(rewards), 3)
        grade_result["seed"]         = seed
        success = grade_result.get("passed", False)

    except Exception as exc:
        print(f"[ERROR] Episode '{task}' crashed: {exc}", flush=True)
        success = False

    finally:
        # [END] — always emitted, even on exception
        episode_score = grade_result.get("score", 0.0) if grade_result else 0.0
        log_end(success=success, steps=steps_taken, rewards=rewards, score=episode_score)

    return grade_result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not API_KEY:
        # WARNING only — do not exit. Automated runners may inject the key
        # via a different mechanism. Let the OpenAI client fail naturally
        # if the key is truly missing, so [START]/[STEP]/[END] lines are
        # still emitted and the baseline is not silently disqualified.
        print("WARNING: No API key found. Set HF_TOKEN or API_KEY. Continuing...", flush=True)

    # Verify the server is reachable before starting.
    # WARNING: do not SystemExit here — if we exit before run_episode(),
    # no [START]/[END] lines are emitted and the automated evaluator
    # marks the baseline as failed. Instead, log the warning and let
    # each episode fail gracefully with its own [START]/[END] pair.
    try:
        health_url = f"{ENV_BASE_URL.rstrip('/')}/health"
        resp = requests.get(health_url, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        print(f"[INFO] Server reachable at {ENV_BASE_URL}", flush=True)
    except Exception as exc:
        print(f"WARNING: Cannot reach environment server at {ENV_BASE_URL}: {exc}", flush=True)
        print(f"         Set ENV_BASE_URL to your deployed HF Space URL. Continuing...", flush=True)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    all_results = {}

    for task in ["easy", "medium", "hard"]:
        seed = TASK_SEEDS[task]
        try:
            result = run_episode(client, task=task, seed=seed)
            all_results[task] = result
        except Exception as exc:
            print(f"[ERROR] Task '{task}' failed: {exc}", flush=True)
            all_results[task] = {"score": 0.0, "passed": False, "error": str(exc)}

    # Human-readable summary (does NOT interfere with [START]/[STEP]/[END] parsing)
    print("\n========================================", flush=True)
    print("  FINAL SCORES — IncidentBench Baseline", flush=True)
    print("========================================", flush=True)
    total = 0.0
    for task, result in all_results.items():
        score  = result.get("score", 0.0)
        passed = result.get("passed", False)
        status = "PASS" if passed else "FAIL"
        print(f"  {task.upper():<8} score={score:.3f}  [{status}]", flush=True)
        total += score

    avg = total / len(all_results)
    print(f"  {'AVERAGE':<8} score={avg:.3f}", flush=True)
    print("========================================", flush=True)

    # Machine-readable scores
    print("\nJSON_SCORES:", json.dumps({
        task: result.get("score", 0.0)
        for task, result in all_results.items()
    }), flush=True)


if __name__ == "__main__":
    main()