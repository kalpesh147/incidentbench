"""
IncidentBench — Task Graders v1.1
===================================
Deterministic scoring for all 3 tasks.

Changes from v1.0:
    - FIX 1: Root cause credit requires prior evidence (queried_services check)
    - FIX 2: Reward farming already blocked in env — grader reflects clean scores
    - FIX 3: Auto-success already handled in env — grader checks step efficiency
    - FIX 4: Stale metric detection now checks SEQUENCE (metrics → then logs)
    - FIX 5: Early escalation penalty reflected in grader scoring
    - FIX 6: Step penalty in env — grader rewards fewer steps more aggressively
"""

from __future__ import annotations
from typing import Any


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


# ---------------------------------------------------------------------------
# Easy grader
# ---------------------------------------------------------------------------

def grade_easy(state: dict[str, Any]) -> dict[str, Any]:
    """
    Easy task scoring (total = 1.0):
        0.4  correct root cause identified WITH prior evidence (FIX 1)
        0.4  correct fix applied
        0.2  efficiency — fewer steps = higher bonus (FIX 6)

    Penalties:
        -0.3  destructive action taken
    """
    score = 0.0
    breakdown = {}

    # Root cause (0.4) — FIX 1: must have gathered evidence first
    if state["root_cause_identified"]:
        score += 0.4
        breakdown["root_cause"] = 0.4
    else:
        breakdown["root_cause"] = 0.0

    # Correct fix (0.4)
    correct_fixes = set(state["scenario_correct_fixes"])
    applied_fixes = set(state["correct_fixes_applied"])
    if correct_fixes & applied_fixes:
        score += 0.4
        breakdown["correct_fix"] = 0.4
    else:
        breakdown["correct_fix"] = 0.0

    # Efficiency (0.2) — FIX 6: tighter step thresholds
    steps = state["step_count"]
    if steps <= 3:
        score += 0.2
        breakdown["efficiency"] = 0.2
    elif steps <= 5:
        score += 0.1
        breakdown["efficiency"] = 0.1
    else:
        breakdown["efficiency"] = 0.0

    # Destructive action penalty
    if state["destructive_actions"] > 0:
        penalty = min(0.3, state["destructive_actions"] * 0.15)
        score -= penalty
        breakdown["destructive_penalty"] = -penalty
    else:
        breakdown["destructive_penalty"] = 0.0

    final = clamp(score)
    return {
        "task":       "easy",
        "score":      final,
        "breakdown":  breakdown,
        "passed":     final >= 0.6,
        "steps_taken": steps,
    }


# ---------------------------------------------------------------------------
# Medium grader
# ---------------------------------------------------------------------------

def grade_medium(state: dict[str, Any]) -> dict[str, Any]:
    """
    Medium task scoring (total = 1.0):
        0.3  correct root cause WITH evidence (FIX 1)
        0.2  red herring NOT acted upon
        0.4  correct fix applied
        0.1  efficiency

    Penalties:
        -0.2  acted on red herring
        -0.2  destructive action
    """
    score = 0.0
    breakdown = {}

    # Root cause (0.3) — FIX 1 enforced in env, grader just checks the flag
    if state["root_cause_identified"]:
        score += 0.3
        breakdown["root_cause"] = 0.3
    else:
        breakdown["root_cause"] = 0.0

    # Red herring ignored (0.2)
    acted_on_red_herring = any(
        step["action"].get("action_type") == "apply_fix" and
        step["action"].get("service") == "cache"
        for step in state["episode_history"]
    )
    if not acted_on_red_herring:
        score += 0.2
        breakdown["red_herring_ignored"] = 0.2
    else:
        score -= 0.2
        breakdown["red_herring_ignored"] = -0.2

    # Correct fix (0.4)
    correct_fixes = set(state["scenario_correct_fixes"])
    applied_fixes = set(state["correct_fixes_applied"])
    if correct_fixes & applied_fixes:
        score += 0.4
        breakdown["correct_fix"] = 0.4
    else:
        breakdown["correct_fix"] = 0.0

    # Efficiency (0.1) — FIX 6
    steps = state["step_count"]
    if steps <= 5:
        score += 0.1
        breakdown["efficiency"] = 0.1
    else:
        breakdown["efficiency"] = 0.0

    # Destructive penalty
    if state["destructive_actions"] > 0:
        penalty = min(0.2, state["destructive_actions"] * 0.1)
        score -= penalty
        breakdown["destructive_penalty"] = -penalty
    else:
        breakdown["destructive_penalty"] = 0.0

    final = clamp(score)
    return {
        "task":       "medium",
        "score":      final,
        "breakdown":  breakdown,
        "passed":     final >= 0.5,
        "steps_taken": steps,
    }


# ---------------------------------------------------------------------------
# Hard grader
# ---------------------------------------------------------------------------

def grade_hard(state: dict[str, Any]) -> dict[str, Any]:
    """
    Hard task scoring (total = 1.0):
        0.25  correct root cause WITH evidence (FIX 1)
        0.15  red herring (database) NOT acted upon
        0.15  stale metrics detected via SEQUENCE check (FIX 4)
              — requires metrics queried THEN logs, not just both
        0.45  correct fixes in correct ORDER

    Penalties:
        -0.3  fixes applied in wrong order
        -0.2  each destructive action
    """
    score = 0.0
    breakdown = {}

    # Root cause (0.25) — FIX 1: evidence required, enforced in env
    if state["root_cause_identified"]:
        score += 0.25
        breakdown["root_cause"] = 0.25
    else:
        breakdown["root_cause"] = 0.0

    # Red herring ignored (0.15)
    acted_on_db_red_herring = any(
        step["action"].get("action_type") == "apply_fix" and
        step["action"].get("service") == "database"
        for step in state["episode_history"]
    )
    if not acted_on_db_red_herring:
        score += 0.15
        breakdown["red_herring_ignored"] = 0.15
    else:
        score -= 0.15
        breakdown["red_herring_ignored"] = -0.15

    # FIX 4 — stale metric detection: SEQUENCE matters
    # Agent must have queried metrics FIRST, then queried logs after
    # Both flags must be True — this proves intentional cross-referencing
    metrics_first = state.get("metrics_queried_first", False)
    logs_after    = state.get("logs_queried_after_metrics", False)

    if metrics_first and logs_after:
        # Agent saw stale metrics warning and then went to cross-check logs
        score += 0.15
        breakdown["stale_metrics_detected"] = 0.15
    else:
        breakdown["stale_metrics_detected"] = 0.0

    # Correct fixes in correct order (0.45)
    applied   = state["correct_fixes_applied"]
    auth_fix  = "rotate_credentials:auth_service"
    cache_fix = "flush_cache:cache"

    auth_applied  = auth_fix  in applied
    cache_applied = cache_fix in applied

    if auth_applied and cache_applied:
        auth_idx  = applied.index(auth_fix)
        cache_idx = applied.index(cache_fix)
        if auth_idx < cache_idx:
            score += 0.45
            breakdown["correct_fixes_ordered"] = 0.45
        else:
            score += 0.15
            score -= 0.3
            breakdown["correct_fixes_ordered"] = 0.15
            breakdown["wrong_order_penalty"]   = -0.3
    elif auth_applied:
        score += 0.2
        breakdown["correct_fixes_ordered"] = 0.2
    elif cache_applied:
        score -= 0.1
        breakdown["correct_fixes_ordered"] = -0.1
    else:
        breakdown["correct_fixes_ordered"] = 0.0

    # Destructive penalty
    if state["destructive_actions"] > 0:
        penalty = min(0.4, state["destructive_actions"] * 0.2)
        score -= penalty
        breakdown["destructive_penalty"] = -penalty
    else:
        breakdown["destructive_penalty"] = 0.0

    final = clamp(score)
    return {
        "task":       "hard",
        "score":      final,
        "breakdown":  breakdown,
        "passed":     final >= 0.20,
        "steps_taken": state["step_count"],
        "note":       "Target: frontier models score 0.20-0.35 on this task.",
    }


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def grade(state: dict[str, Any]) -> dict[str, Any]:
    task = state.get("task", "easy")
    if task == "easy":
        return grade_easy(state)
    elif task == "medium":
        return grade_medium(state)
    elif task == "hard":
        return grade_hard(state)
    else:
        raise ValueError(f"Unknown task: '{task}'")


# ---------------------------------------------------------------------------
# Self-test — run directly to verify all 6 fixes work correctly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from env import (
        IncidentBenchEnv, Action, ActionType,
        ServiceName, IncidentType, FixType
    )

    print("=" * 60)
    print("IncidentBench Grader Self-Test v1.1 — all 6 fixes")
    print("=" * 60)

    # --- FIX 1: Blind runbook guessing gets NO credit ---
    print("\n[FIX 1] Blind runbook guess — no evidence gathered:")
    env = IncidentBenchEnv(task="easy", seed=42)
    env.reset()
    # Agent reads correct runbook WITHOUT querying logs/metrics first
    env.step(Action(action_type=ActionType.READ_RUNBOOK,
                    incident_type=IncidentType.DB_CONNECTION))
    result = grade(env.state())
    assert result["breakdown"]["root_cause"] == 0.0, "FIX 1 FAILED"
    print(f"  root_cause credit = {result['breakdown']['root_cause']} (expected 0.0) ✓")

    # --- FIX 1: Evidence first then runbook = full credit ---
    print("\n[FIX 1] Evidence first, then runbook — gets credit:")
    env = IncidentBenchEnv(task="easy", seed=42)
    env.reset()
    env.step(Action(action_type=ActionType.QUERY_LOGS,
                    service=ServiceName.DATABASE))
    env.step(Action(action_type=ActionType.READ_RUNBOOK,
                    incident_type=IncidentType.DB_CONNECTION))
    result = grade(env.state())
    assert result["breakdown"]["root_cause"] == 0.4, "FIX 1 FAILED"
    print(f"  root_cause credit = {result['breakdown']['root_cause']} (expected 0.4) ✓")

    # --- FIX 2: Reward farming blocked ---
    print("\n[FIX 2] Spam same action — reward only once:")
    env = IncidentBenchEnv(task="easy", seed=42)
    env.reset()
    r1 = env.step(Action(action_type=ActionType.QUERY_LOGS,
                         service=ServiceName.DATABASE))
    r2 = env.step(Action(action_type=ActionType.QUERY_LOGS,
                         service=ServiceName.DATABASE))
    r3 = env.step(Action(action_type=ActionType.QUERY_LOGS,
                         service=ServiceName.DATABASE))
    # First call: -0.01 step + 0.1 reward = 0.09
    # Second+: -0.01 step + 0.0 = -0.01
    assert r1.reward > r2.reward, "FIX 2 FAILED"
    print(f"  step1={r1.reward:.3f}, step2={r2.reward:.3f}, step3={r3.reward:.3f} ✓")

    # --- FIX 3: Auto success ends episode ---
    print("\n[FIX 3] Auto success — episode ends when system fixed:")
    env = IncidentBenchEnv(task="easy", seed=42)
    env.reset()
    env.step(Action(action_type=ActionType.QUERY_LOGS,
                    service=ServiceName.DATABASE))
    env.step(Action(action_type=ActionType.READ_RUNBOOK,
                    incident_type=IncidentType.DB_CONNECTION))
    result = env.step(Action(action_type=ActionType.APPLY_FIX,
                             service=ServiceName.DATABASE,
                             fix_type=FixType.RESTART_SERVICE))
    assert result.done, "FIX 3 FAILED — episode should be done"
    assert result.info.get("termination_reason") == "success_all_services_healthy"
    print(f"  done={result.done}, reason={result.info['termination_reason']} ✓")

    # --- FIX 4: Stale metric sequence check ---
    print("\n[FIX 4] Sequence check — metrics THEN logs = credit:")
    env = IncidentBenchEnv(task="hard", seed=42)
    env.reset()
    env.step(Action(action_type=ActionType.QUERY_METRICS,
                    service=ServiceName.AUTH_SERVICE,
                    metric_name="error_rate"))
    env.step(Action(action_type=ActionType.QUERY_LOGS,
                    service=ServiceName.AUTH_SERVICE))
    s = env.state()
    assert s["metrics_queried_first"] and s["logs_queried_after_metrics"], "FIX 4 FAILED"
    print(f"  metrics_first={s['metrics_queried_first']}, logs_after={s['logs_queried_after_metrics']} ✓")

    print("\n[FIX 4] Logs ONLY — no sequence credit:")
    env = IncidentBenchEnv(task="hard", seed=42)
    env.reset()
    env.step(Action(action_type=ActionType.QUERY_LOGS,
                    service=ServiceName.AUTH_SERVICE))
    s = env.state()
    assert not s["logs_queried_after_metrics"], "FIX 4 FAILED"
    print(f"  logs_after={s['logs_queried_after_metrics']} (expected False) ✓")

    # --- FIX 5: Early escalation penalty ---
    print("\n[FIX 5] Early escalation (step 1) — penalty applied:")
    env = IncidentBenchEnv(task="easy", seed=42)
    env.reset()
    result = env.step(Action(action_type=ActionType.ESCALATE,
                             reason="giving up"))
    # step penalty -0.01 + escalation penalty -0.2 = -0.21
    assert result.reward < 0, "FIX 5 FAILED"
    print(f"  reward={result.reward:.3f} (expected negative) ✓")

    print("\n[FIX 5] Late escalation (step 5+) on hard — small reward:")
    env = IncidentBenchEnv(task="hard", seed=42)
    env.reset()
    for _ in range(4):
        env.step(Action(action_type=ActionType.QUERY_LOGS,
                        service=ServiceName.AUTH_SERVICE))
    result = env.step(Action(action_type=ActionType.ESCALATE,
                             reason="cannot resolve cascading failure"))
    # -0.01 step + 0.1 escalation reward = 0.09
    assert result.reward > 0, "FIX 5 FAILED"
    print(f"  reward={result.reward:.3f} (expected positive) ✓")

    # --- FIX 6: Step penalty ---
    print("\n[FIX 6] Step penalty — every step costs -0.01:")
    env = IncidentBenchEnv(task="easy", seed=42)
    env.reset()
    result = env.step(Action(action_type=ActionType.QUERY_LOGS,
                             service=ServiceName.CACHE))
    # irrelevant service: -0.01 step + 0.0 reward = -0.01
    assert result.reward == -0.01, f"FIX 6 FAILED: got {result.reward}"
    print(f"  irrelevant action reward={result.reward:.3f} (expected -0.01) ✓")

    print()
    print("=" * 60)
    print("ALL 6 FIXES VERIFIED — graders.py is clean")
    print("=" * 60)