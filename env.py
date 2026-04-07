"""
IncidentBench — Adversarial On-Call Environment
================================================
OpenEnv-compliant environment where an AI agent acts as an on-call engineer.

Fixes applied (v1.4):
    FIX 1  — Root cause detection now requires EVIDENCE before credit
              Agent must query logs/metrics for the right service BEFORE
              reading the runbook. Blind runbook guessing gives 0 credit.
    FIX 2  — Reward farming blocked
              Each (action_type, service) pair rewarded only on FIRST use.
              Spamming same action gives 0 after first call.
    FIX 3  — Auto success condition
              Episode ends automatically when all relevant services are healthy.
              No more "fixed but still running" episodes.
    FIX 4  — Stale metric detection requires correct SEQUENCE
              Agent must call query_metrics THEN query_logs (in that order)
              to prove it detected staleness and cross-referenced.
    FIX 5  — Escalation penalty for early bail-out
              Escalating before step 4 = penalty. Late escalation = small reward.
    FIX 6  — Step penalty added
              -0.01 per step to discourage wandering and reward efficiency.
    FIX 7  — Red herring flag stripped from public Alert model
              is_red_herring is internal only — never sent to agent.
    FIX 8  — Wrong-order fix now gives partial credit (v1.2)
              Applying a correct fix out of order: records it, heals the service,
              but returns -0.1 instead of +0.4. Grader's wrong_order_penalty
              (-0.3) handles the score hit.
    FIX 9  — Medium task logs vanish after step 1 (v1.2, was step 2)
              Forces agents to gather evidence at step 1 before logs disappear,
              making the Type A failure injection actually challenging.
    FIX 10 — Per-service metrics tracking added (v1.3)
              metrics_queried_services tracks which services had query_metrics called.
              Used by grader to require metric verification for full fix credit on medium.
    FIX 11 — Medium alert message de-obfuscated (v1.3)
              Alert no longer says "JWT signing key rotation failed" explicitly.
              Agent must query logs to identify the specific cause.
    FIX 12 — Medium root cause gate tightened (v1.4)
              Agent must query BOTH logs AND metrics for auth_service before
              getting runbook credit on medium. Querying logs alone (step 1)
              then jumping to runbook is no longer enough. This forces the agent
              to burn a second step on metrics — guaranteeing it either:
                (a) hits the Type A log disappearance when it re-queries logs, OR
                (b) queries metrics first then logs, proving real investigation.
              Without this, the optimal 4-step path bypasses the adversarial
              mechanic entirely by never re-querying logs after step 1.
    FIX 13 — Medium logs_queried_count tracking added (v1.4)
              Tracks how many times the agent queries logs for auth_service.
              If count >= 2, agent encountered the Type A failure (logs vanished).
              Exposed in state() for grader's logs_vanished_observed bonus.
    FIX 14 — Hard task: restart_service:auth_service added to destructive_pairs (v1.4)
              The conflicting legacy runbook explicitly tells the agent to restart
              auth_service. The correct runbook says do NOT restart auth_service.
              Previously this wrong action only gave -0.05 (no_effect penalty).
              Now it correctly triggers the -0.2 destructive penalty, making the
              Type D failure injection actually penalise the intended wrong action.
"""

from __future__ import annotations

import random
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    QUERY_LOGS    = "query_logs"
    QUERY_METRICS = "query_metrics"
    READ_RUNBOOK  = "read_runbook"
    APPLY_FIX     = "apply_fix"
    ESCALATE      = "escalate"


class ServiceName(str, Enum):
    API_GATEWAY  = "api_gateway"
    AUTH_SERVICE = "auth_service"
    DATABASE     = "database"
    CACHE        = "cache"


class HealthStatus(str, Enum):
    HEALTHY  = "healthy"
    DEGRADED = "degraded"
    DOWN     = "down"


class IncidentType(str, Enum):
    HIGH_LATENCY     = "high_latency"
    AUTH_FAILURE     = "auth_failure"
    DB_CONNECTION    = "db_connection"
    CACHE_MISS_SPIKE = "cache_miss_spike"


class FixType(str, Enum):
    RESTART_SERVICE    = "restart_service"
    ROLLBACK_DEPLOY    = "rollback_deploy"
    FLUSH_CACHE        = "flush_cache"
    SCALE_UP           = "scale_up"
    ROTATE_CREDENTIALS = "rotate_credentials"


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class Action(BaseModel):
    action_type:   ActionType
    service:       Optional[ServiceName]  = None
    metric_name:   Optional[str]          = None
    incident_type: Optional[IncidentType] = None
    fix_type:      Optional[FixType]      = None
    reason:        Optional[str]          = None


class Alert(BaseModel):
    """Public alert — is_red_herring intentionally absent (FIX 7)."""
    alert_id:  str
    service:   ServiceName
    severity:  str
    message:   str
    timestamp: str


class _InternalAlert(BaseModel):
    """Internal alert with red herring flag — never sent to agent (FIX 7)."""
    alert_id:       str
    service:        ServiceName
    severity:       str
    message:        str
    timestamp:      str
    is_red_herring: bool = False

    def to_public(self) -> Alert:
        return Alert(
            alert_id=self.alert_id,
            service=self.service,
            severity=self.severity,
            message=self.message,
            timestamp=self.timestamp,
        )


class Observation(BaseModel):
    active_alerts:     list[Alert]
    tool_response:     Optional[dict[str, Any]] = None
    system_state:      dict[str, str] = Field(default_factory=dict)
    step_count:        int = 0
    max_steps:         int = 10
    last_action_error: Optional[str] = None


class StepResult(BaseModel):
    observation: Observation
    reward:      float
    done:        bool
    info:        dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# The Environment
# ---------------------------------------------------------------------------

class IncidentBenchEnv:
    """
    IncidentBench — Adversarial On-Call OpenEnv Environment v1.4
    """

    MAX_STEPS = 10

    def __init__(self, task: str = "easy", seed: int = 42):
        assert task in ("easy", "medium", "hard"), \
            f"task must be 'easy', 'medium', or 'hard', got '{task}'"

        self.task = task
        self.seed = seed
        self._rng = random.Random(seed)

        # Core state
        self._step_count: int = 0
        self._done: bool = False
        self._system_state: dict[str, str] = {}
        self._alerts: list[_InternalAlert] = []
        self._episode_history: list[dict] = []

        # Tracking for graders
        self._root_cause_identified: bool = False
        self._correct_fixes_applied: list[str] = []
        self._destructive_actions: int = 0
        self._ignored_critical_steps: int = 0
        self._last_tool_response: Optional[dict] = None

        # FIX 1 — evidence tracking before runbook credit
        self._queried_services: set[str] = set()

        # FIX 2 — reward farming prevention
        self._rewarded_actions: set[str] = set()

        # FIX 4 — stale metric sequence tracking (hard task)
        self._metrics_queried_first: bool = False
        self._logs_queried_after_metrics: bool = False

        # FIX 10 — per-service metrics query tracking
        self._metrics_queried_services: set[str] = set()

        # FIX 12 — per-service logs query tracking for medium evidence gate
        self._logs_queried_services: set[str] = set()

        # FIX 13 — auth_service log query counter for Type A detection
        self._auth_logs_query_count: int = 0

        # Failure injection flags
        self._logs_go_missing_after: Optional[int] = None
        self._red_herring_alert_id: Optional[str] = None
        self._metrics_staleness_minutes: Optional[int] = None
        self._conflicting_runbook_active: bool = False

        self._scenario = self._load_scenario()

    # ------------------------------------------------------------------
    # OpenEnv contract
    # ------------------------------------------------------------------

    def reset(self) -> Observation:
        self._rng = random.Random(self.seed)
        self._step_count = 0
        self._done = False
        self._episode_history = []
        self._root_cause_identified = False
        self._correct_fixes_applied = []
        self._destructive_actions = 0
        self._ignored_critical_steps = 0
        self._last_tool_response = None
        self._queried_services = set()
        self._rewarded_actions = set()
        self._metrics_queried_first = False
        self._logs_queried_after_metrics = False
        self._metrics_queried_services = set()
        self._logs_queried_services = set()
        self._auth_logs_query_count = 0

        self._scenario = self._load_scenario()
        self._system_state = dict(self._scenario["initial_system_state"])
        self._alerts = [_InternalAlert(**a) for a in self._scenario["initial_alerts"]]

        return self._build_observation()

    def step(self, action: Action) -> StepResult:
        if self._done:
            return StepResult(
                observation=self._build_observation(),
                reward=0.001,
                done=True,
                info={"warning": "step() called after episode is done"}
            )

        self._step_count += 1

        # FIX 6 — small step penalty every step
        reward = -0.01
        error_msg = None
        info: dict[str, Any] = {}

        validation_error = self._validate_action(action)
        if validation_error:
            error_msg = validation_error
            reward -= 0.05
        else:
            step_reward, info = self._execute_action(action)
            reward += step_reward

        # Track ignored critical alerts
        critical_alerts = [
            a for a in self._alerts
            if a.severity == "critical" and not a.is_red_herring
        ]
        if critical_alerts and action.action_type not in (
            ActionType.QUERY_LOGS, ActionType.QUERY_METRICS,
            ActionType.READ_RUNBOOK, ActionType.APPLY_FIX
        ):
            self._ignored_critical_steps += 1
            if self._ignored_critical_steps >= 3:
                reward -= 0.3

        # Record history
        self._episode_history.append({
            "step":                  self._step_count,
            "action":                action.model_dump(),
            "reward":                round(reward, 4),
            "root_cause_identified": self._root_cause_identified,
            "correct_fixes":         list(self._correct_fixes_applied),
            "system_state":          dict(self._system_state),
        })

        # FIX 3 — auto success: episode ends only when ALL required fixes applied
        required_fixes = set(self._scenario["correct_fixes"])
        applied_fixes  = set(self._correct_fixes_applied)
        if required_fixes and required_fixes.issubset(applied_fixes):
            self._done = True
            info["termination_reason"] = "success_all_services_healthy"

        if self._step_count >= self.MAX_STEPS:
            self._done = True
            info["termination_reason"] = info.get(
                "termination_reason", "max_steps_reached"
            )

        if action.action_type == ActionType.ESCALATE:
            self._done = True
            info["termination_reason"] = "escalated"

        obs = self._build_observation(error_msg=error_msg)
        clamped_reward = max(0.001, min(0.999, round(reward, 4)))
        return StepResult(
            observation=obs,
            reward=clamped_reward,
            done=self._done,
            info=info,
        )

    def state(self) -> dict[str, Any]:
        return {
            "task":                       self.task,
            "seed":                       self.seed,
            "step_count":                 self._step_count,
            "done":                       self._done,
            "system_state":               {k: v for k, v in self._system_state.items()},
            "root_cause_identified":      self._root_cause_identified,
            "correct_fixes_applied":      list(self._correct_fixes_applied),
            "destructive_actions":        self._destructive_actions,
            "ignored_critical_steps":     self._ignored_critical_steps,
            "episode_history":            list(self._episode_history),
            "scenario_root_cause":        self._scenario["root_cause"],
            "scenario_correct_fixes":     self._scenario["correct_fixes"],
            "queried_services":           list(self._queried_services),
            "logs_queried_services":      list(self._logs_queried_services),
            "metrics_queried_first":      self._metrics_queried_first,
            "logs_queried_after_metrics": self._logs_queried_after_metrics,
            "metrics_queried_services":   list(self._metrics_queried_services),
            # FIX 13 — exposed for grader's logs_vanished_observed bonus
            "auth_logs_query_count":      self._auth_logs_query_count,
        }

    # ------------------------------------------------------------------
    # Action execution
    # ------------------------------------------------------------------

    def _execute_action(self, action: Action) -> tuple[float, dict]:
        if action.action_type == ActionType.QUERY_LOGS:
            return self._handle_query_logs(action)
        elif action.action_type == ActionType.QUERY_METRICS:
            return self._handle_query_metrics(action)
        elif action.action_type == ActionType.READ_RUNBOOK:
            return self._handle_read_runbook(action)
        elif action.action_type == ActionType.APPLY_FIX:
            return self._handle_apply_fix(action)
        elif action.action_type == ActionType.ESCALATE:
            return self._handle_escalate(action)
        return 0.001, {}

    def _handle_query_logs(self, action: Action) -> tuple[float, dict]:
        service = action.service.value

        # FIX 4 — track sequence: was metrics queried before this?
        if self._metrics_queried_first and not self._logs_queried_after_metrics:
            self._logs_queried_after_metrics = True

        # FIX 1 — record evidence BEFORE Type A check
        self._queried_services.add(service)

        # FIX 12 — track per-service log queries (separate set, for medium gate)
        self._logs_queried_services.add(service)

        # FIX 13 — count auth_service log queries specifically
        if service == ServiceName.AUTH_SERVICE.value:
            self._auth_logs_query_count += 1

        # Type A failure injection — logs vanish after N steps
        if (self._logs_go_missing_after is not None and
                self._step_count > self._logs_go_missing_after):
            self._last_tool_response = {
                "service": service,
                "logs":    [],
                "note":    "Log pipeline unavailable — no data returned.",
            }
            return 0.001, {"failure_injected": "type_a_missing_logs"}

        logs = self._scenario["logs"].get(service, [])
        self._last_tool_response = {"service": service, "logs": logs}

        # FIX 2 — reward only first meaningful query per (action, service)
        reward_key = f"query_logs:{service}"
        relevant   = self._scenario.get("relevant_services", [])
        if service in relevant and reward_key not in self._rewarded_actions:
            self._rewarded_actions.add(reward_key)
            return 0.1, {}

        return 0.001, {}

    def _handle_query_metrics(self, action: Action) -> tuple[float, dict]:
        service = action.service.value
        metric  = action.metric_name or "error_rate"

        # FIX 4 — mark metrics as queried first (for hard task sequence check)
        if not self._metrics_queried_first:
            self._metrics_queried_first = True

        # FIX 10 — track per-service metrics queries
        self._metrics_queried_services.add(service)

        metrics = self._scenario["metrics"].get(service, {})
        value   = metrics.get(metric, 0.0)

        response: dict[str, Any] = {"service": service, "metric": metric, "value": value}

        if self._metrics_staleness_minutes is not None:
            response["staleness_warning"] = (
                f"Data is {self._metrics_staleness_minutes} minutes old. "
                "Real-time pipeline may be degraded."
            )
            response["stale"] = True

        self._last_tool_response = response

        # FIX 2 — reward only first meaningful query per (action, service)
        reward_key = f"query_metrics:{service}"
        relevant   = self._scenario.get("relevant_services", [])
        if service in relevant and reward_key not in self._rewarded_actions:
            self._rewarded_actions.add(reward_key)
            return 0.1, {}

        return 0.001, {}

    def _handle_read_runbook(self, action: Action) -> tuple[float, dict]:
        incident_type = action.incident_type.value
        runbooks      = self._scenario["runbooks"]

        if incident_type not in runbooks:
            self._last_tool_response = {
                "incident_type": incident_type,
                "error":         "No runbook found for this incident type.",
            }
            return 0.001, {}

        runbook = runbooks[incident_type]

        # Hard task: conflicting runbook injection
        if self._conflicting_runbook_active:
            conflicting = self._scenario.get("conflicting_runbook", {})
            if not self._metrics_queried_first:
                self._last_tool_response = {
                    "incident_type": incident_type,
                    "runbook":       conflicting,
                    "note":          "Served from runbook cache. Real-time pipeline may have newer version.",
                }
            else:
                self._last_tool_response = {
                    "incident_type":     incident_type,
                    "runbook":           runbook,
                    "alternate_runbook": conflicting,
                    "warning":           "Multiple runbooks found. Verify which applies.",
                }
        else:
            self._last_tool_response = {
                "incident_type": incident_type,
                "runbook":       runbook,
            }

        correct_incident_type = self._scenario.get("correct_incident_type")
        relevant_services     = self._scenario.get("relevant_services", [])

        # FIX 1 — root cause credit ONLY if evidence was gathered first
        has_logs_evidence = any(
            svc in self._logs_queried_services for svc in relevant_services
        )

        # FIX 12 — MEDIUM TASK: require BOTH logs AND metrics for auth_service,
        # AND require correct sequence (metrics FIRST, then logs).
        # This forces the agent to query metrics before re-querying logs.
        # Since logs vanish after step 1, re-querying logs after metrics
        # guarantees the agent hits the Type A disappearance — closing the
        # bypass where logs→runbook_attempt→metrics→runbook never re-queries logs.
        if self.task == "medium":
            auth = ServiceName.AUTH_SERVICE.value
            logs_done    = auth in self._logs_queried_services
            metrics_done = auth in self._metrics_queried_services
            correct_order = self._metrics_queried_first and self._logs_queried_after_metrics
            has_evidence  = logs_done and metrics_done and correct_order
        elif self._conflicting_runbook_active:
            # HARD TASK: require metrics→logs sequence proof
            sequence_done = self._metrics_queried_first and self._logs_queried_after_metrics
            has_evidence  = has_logs_evidence and sequence_done
        else:
            has_evidence = has_logs_evidence

        if incident_type == correct_incident_type:
            if has_evidence:
                self._root_cause_identified = True
                return 0.2, {"root_cause_identified": True}
            else:
                if self.task == "medium":
                    auth = ServiceName.AUTH_SERVICE.value
                    missing = []
                    if auth not in self._logs_queried_services:
                        missing.append("auth_service logs")
                    if auth not in self._metrics_queried_services:
                        missing.append("auth_service metrics")
                    if not (self._metrics_queried_first and self._logs_queried_after_metrics):
                        missing.append("correct sequence (query metrics first, then re-query logs)")
                    note = (
                        f"Correct runbook but insufficient evidence. "
                        f"Still need: {', '.join(missing)}. "
                        "Query metrics first, then re-query logs to confirm current state."
                    )
                elif self._conflicting_runbook_active:
                    note = (
                        "Correct runbook identified but stale metrics not cross-referenced. "
                        "Query metrics first, then logs to confirm real-time state. No credit."
                    )
                else:
                    note = "Correct runbook but no prior evidence gathered. No credit."
                return 0.001, {
                    "root_cause_identified": False,
                    "note": note,
                }

        return 0.05, {}

    def _handle_apply_fix(self, action: Action) -> tuple[float, dict]:
        service           = action.service.value
        fix               = action.fix_type.value
        correct_fixes     = self._scenario["correct_fixes"]
        also_valid        = self._scenario.get("also_valid_fixes", [])
        destructive_pairs = self._scenario.get("destructive_pairs", [])
        fix_key           = f"{fix}:{service}"

        # HARD TASK: enforce fix ORDER (FIX 8)
        if self._conflicting_runbook_active and fix_key in correct_fixes:
            fix_index    = correct_fixes.index(fix_key)
            already_done = set(self._correct_fixes_applied)
            prior_fixes  = correct_fixes[:fix_index]
            if prior_fixes and not all(f in already_done for f in prior_fixes):
                self._correct_fixes_applied.append(fix_key)
                if service in self._system_state:
                    self._system_state[service] = HealthStatus.HEALTHY.value
                return -0.1, {
                    "fix_applied": fix_key,
                    "result":      "wrong_order — applied but worsened cascade. Penalty applied.",
                }

        if fix_key in correct_fixes or fix_key in also_valid:
            self._correct_fixes_applied.append(fix_key)
            if service in self._system_state:
                self._system_state[service] = HealthStatus.HEALTHY.value
            return 0.4, {"fix_applied": fix_key, "result": "success"}

        if fix_key in destructive_pairs:
            self._destructive_actions += 1
            return -0.2, {"fix_applied": fix_key, "result": "destructive_action"}

        return -0.05, {"fix_applied": fix_key, "result": "no_effect"}

    def _handle_escalate(self, action: Action) -> tuple[float, dict]:
        # FIX 5 — penalise early escalation, allow late escalation on hard
        if self._step_count < 4:
            return -0.2, {
                "escalation_reason": action.reason,
                "note": "Early escalation penalty — insufficient investigation.",
            }
        reward = 0.1 if self.task == "hard" else 0.001
        return reward, {"escalation_reason": action.reason}

    # ------------------------------------------------------------------
    # Scenario loading
    # ------------------------------------------------------------------

    def _load_scenario(self) -> dict[str, Any]:
        if self.task == "easy":
            return self._scenario_easy()
        elif self.task == "medium":
            return self._scenario_medium()
        else:
            return self._scenario_hard()

    def _scenario_easy(self) -> dict[str, Any]:
        self._logs_go_missing_after      = None
        self._red_herring_alert_id       = None
        self._metrics_staleness_minutes  = None
        self._conflicting_runbook_active = False

        return {
            "root_cause":            "database_connection_exhausted",
            "correct_incident_type": IncidentType.DB_CONNECTION.value,
            "correct_fixes": [
                f"{FixType.RESTART_SERVICE.value}:{ServiceName.DATABASE.value}",
            ],
            "also_valid_fixes": [
                f"{FixType.SCALE_UP.value}:{ServiceName.DATABASE.value}",
            ],
            "destructive_pairs": [
                f"{FixType.RESTART_SERVICE.value}:{ServiceName.API_GATEWAY.value}",
            ],
            "relevant_services": [
                ServiceName.DATABASE.value,
                ServiceName.API_GATEWAY.value,
            ],
            "initial_system_state": {
                ServiceName.API_GATEWAY.value:  HealthStatus.DEGRADED.value,
                ServiceName.AUTH_SERVICE.value: HealthStatus.HEALTHY.value,
                ServiceName.DATABASE.value:     HealthStatus.DOWN.value,
                ServiceName.CACHE.value:        HealthStatus.HEALTHY.value,
            },
            "initial_alerts": [
                {
                    "alert_id":       "alert_001",
                    "service":        ServiceName.API_GATEWAY.value,
                    "severity":       "warning",
                    "message":        "API gateway response time > 2000ms",
                    "timestamp":      "2024-01-15T14:32:00Z",
                    "is_red_herring": False,
                },
                {
                    "alert_id":       "alert_002",
                    "service":        ServiceName.DATABASE.value,
                    "severity":       "critical",
                    "message":        "Database connection pool exhausted — 0/100 connections available",
                    "timestamp":      "2024-01-15T14:31:45Z",
                    "is_red_herring": False,
                },
            ],
            "logs": {
                ServiceName.DATABASE.value: [
                    "[ERROR] 14:31:40 Connection pool exhausted. Max connections: 100",
                    "[ERROR] 14:31:41 New connection request rejected — pool full",
                    "[ERROR] 14:31:43 Timeout waiting for connection after 30s",
                    "[WARN]  14:31:44 Active queries: 100, Queued: 47",
                ],
                ServiceName.API_GATEWAY.value: [
                    "[WARN]  14:31:50 Upstream database timeout on /api/users",
                    "[WARN]  14:31:51 Upstream database timeout on /api/orders",
                    "[ERROR] 14:31:55 Circuit breaker OPEN for database dependency",
                ],
            },
            "metrics": {
                ServiceName.DATABASE.value: {
                    "error_rate":           0.98,
                    "connection_pool_used": 100,
                    "connection_pool_max":  100,
                    "query_latency_p99_ms": 30000,
                },
                ServiceName.API_GATEWAY.value: {
                    "error_rate":           0.43,
                    "response_time_p99_ms": 2800,
                },
            },
            "runbooks": {
                IncidentType.DB_CONNECTION.value: {
                    "title": "Database connection pool exhaustion",
                    "steps": [
                        "1. Verify connection pool metrics via query_metrics(database, connection_pool_used)",
                        "2. Check for long-running queries in database logs",
                        "3. Apply fix: restart_service on database to reset pool",
                        "4. Alternatively: scale_up database to increase pool size",
                        "5. Monitor error_rate post-fix — should drop below 0.05 within 2 minutes",
                    ]
                },
                IncidentType.HIGH_LATENCY.value: {
                    "title": "High latency — generic",
                    "steps": [
                        "1. Identify slowest service via metrics",
                        "2. Check for resource saturation (CPU, memory, connections)",
                        "3. Consider scale_up or rollback_deploy if recent deployment",
                    ]
                },
            },
        }

    def _scenario_medium(self) -> dict[str, Any]:
        self._logs_go_missing_after      = 1
        self._metrics_staleness_minutes  = None
        self._conflicting_runbook_active = False

        scenario = {
            "root_cause":            "auth_service_token_expiry",
            "correct_incident_type": IncidentType.AUTH_FAILURE.value,
            "correct_fixes": [
                f"{FixType.ROTATE_CREDENTIALS.value}:{ServiceName.AUTH_SERVICE.value}",
            ],
            "destructive_pairs": [
                f"{FixType.RESTART_SERVICE.value}:{ServiceName.DATABASE.value}",
                f"{FixType.FLUSH_CACHE.value}:{ServiceName.CACHE.value}",
            ],
            "relevant_services": [
                ServiceName.AUTH_SERVICE.value,
                ServiceName.API_GATEWAY.value,
            ],
            "initial_system_state": {
                ServiceName.API_GATEWAY.value:  HealthStatus.DEGRADED.value,
                ServiceName.AUTH_SERVICE.value: HealthStatus.DEGRADED.value,
                ServiceName.DATABASE.value:     HealthStatus.HEALTHY.value,
                ServiceName.CACHE.value:        HealthStatus.HEALTHY.value,
            },
            "initial_alerts": [
                {
                    "alert_id":       "alert_001",
                    "service":        ServiceName.API_GATEWAY.value,
                    "severity":       "critical",
                    "message":        "API gateway 401 error rate > 80% on all authenticated endpoints",
                    "timestamp":      "2024-01-15T16:10:00Z",
                    "is_red_herring": False,
                },
                {
                    "alert_id":       "alert_002",
                    "service":        ServiceName.AUTH_SERVICE.value,
                    "severity":       "critical",
                    "message":        "Auth service: elevated authentication failure rate detected",
                    "timestamp":      "2024-01-15T16:09:50Z",
                    "is_red_herring": False,
                },
                {
                    "alert_id":       "alert_003",
                    "service":        ServiceName.CACHE.value,
                    "severity":       "warning",
                    "message":        "Cache miss rate elevated: 34% (baseline 12%)",
                    "timestamp":      "2024-01-15T16:10:05Z",
                    "is_red_herring": True,
                },
            ],
            "logs": {
                ServiceName.AUTH_SERVICE.value: [
                    "[ERROR] 16:09:45 JWT signing key expired — issued tokens are invalid",
                    "[ERROR] 16:09:46 Key rotation attempt failed: permission denied on secrets vault",
                    "[ERROR] 16:09:48 All token validation requests returning 401",
                ],
                ServiceName.API_GATEWAY.value: [
                    "[ERROR] 16:09:55 Auth validation failed for request to /api/dashboard",
                    "[ERROR] 16:09:56 Auth validation failed for request to /api/profile",
                    "[WARN]  16:09:57 Circuit breaker threshold approaching for auth dependency",
                ],
                ServiceName.CACHE.value: [
                    "[WARN]  16:09:50 Cache miss rate increasing — possible cold cache",
                    "[INFO]  16:09:52 Eviction policy: LRU, max memory: 2GB, used: 1.8GB",
                ],
            },
            "metrics": {
                ServiceName.AUTH_SERVICE.value: {
                    "error_rate":                        0.999,
                    "token_validation_failures_per_min": 847,
                    "key_rotation_failures":             3,
                },
                ServiceName.API_GATEWAY.value: {
                    "error_rate":           0.82,
                    "http_401_rate":        0.81,
                    "response_time_p99_ms": 450,
                },
                ServiceName.CACHE.value: {
                    "miss_rate":      0.34,
                    "hit_rate":       0.66,
                    "memory_used_gb": 1.8,
                },
            },
            "runbooks": {
                IncidentType.AUTH_FAILURE.value: {
                    "title": "Auth service failure — token / credential issues",
                    "steps": [
                        "1. Check auth service logs for token validation errors",
                        "2. Verify JWT signing key status in metrics",
                        "3. If key expired: apply rotate_credentials on auth_service",
                        "4. Monitor http_401_rate — should drop to <0.01 within 60s",
                        "5. Do NOT restart auth_service — clears in-flight token cache",
                    ]
                },
                IncidentType.CACHE_MISS_SPIKE.value: {
                    "title": "Cache miss spike",
                    "steps": [
                        "1. Check if cache miss spike correlates with a deployment",
                        "2. If cache was flushed: wait for warm-up (5-10 min)",
                        "3. If persistent: flush_cache to force re-population",
                    ]
                },
            },
        }

        self._red_herring_alert_id = "alert_003"
        return scenario

    def _scenario_hard(self) -> dict[str, Any]:
        self._logs_go_missing_after      = 1
        self._red_herring_alert_id       = "alert_004"
        self._metrics_staleness_minutes  = 10
        self._conflicting_runbook_active = True

        return {
            "root_cause":            "cascading_auth_then_cache_then_api",
            "correct_incident_type": IncidentType.AUTH_FAILURE.value,
            "correct_fixes": [
                f"{FixType.ROTATE_CREDENTIALS.value}:{ServiceName.AUTH_SERVICE.value}",
                f"{FixType.FLUSH_CACHE.value}:{ServiceName.CACHE.value}",
            ],
            # FIX 14 — restart_service:auth_service added to destructive_pairs.
            # The conflicting legacy runbook says "restart auth_service immediately".
            # The correct runbook says do NOT restart auth_service.
            # Previously this wrong action only returned -0.05 (no_effect).
            # Now it correctly triggers -0.2 (destructive), making the Type D
            # failure injection actually penalise the intended wrong action.
            "destructive_pairs": [
                f"{FixType.RESTART_SERVICE.value}:{ServiceName.AUTH_SERVICE.value}",
                f"{FixType.RESTART_SERVICE.value}:{ServiceName.API_GATEWAY.value}",
                f"{FixType.ROLLBACK_DEPLOY.value}:{ServiceName.DATABASE.value}",
            ],
            "relevant_services": [
                ServiceName.AUTH_SERVICE.value,
                ServiceName.CACHE.value,
                ServiceName.API_GATEWAY.value,
            ],
            "initial_system_state": {
                ServiceName.API_GATEWAY.value:  HealthStatus.DOWN.value,
                ServiceName.AUTH_SERVICE.value: HealthStatus.DOWN.value,
                ServiceName.DATABASE.value:     HealthStatus.DEGRADED.value,
                ServiceName.CACHE.value:        HealthStatus.DOWN.value,
            },
            "initial_alerts": [
                {
                    "alert_id":       "alert_001",
                    "service":        ServiceName.API_GATEWAY.value,
                    "severity":       "critical",
                    "message":        "API gateway DOWN — all requests returning 503",
                    "timestamp":      "2024-01-15T22:15:00Z",
                    "is_red_herring": False,
                },
                {
                    "alert_id":       "alert_002",
                    "service":        ServiceName.AUTH_SERVICE.value,
                    "severity":       "critical",
                    "message":        "Auth service DOWN — signing infrastructure failure",
                    "timestamp":      "2024-01-15T22:14:30Z",
                    "is_red_herring": False,
                },
                {
                    "alert_id":       "alert_003",
                    "service":        ServiceName.CACHE.value,
                    "severity":       "critical",
                    "message":        "Cache cluster DOWN — all nodes unreachable",
                    "timestamp":      "2024-01-15T22:14:45Z",
                    "is_red_herring": False,
                },
                {
                    "alert_id":       "alert_004",
                    "service":        ServiceName.DATABASE.value,
                    "severity":       "warning",
                    "message":        "Database query latency elevated: p99=850ms (threshold: 500ms)",
                    "timestamp":      "2024-01-15T22:15:05Z",
                    "is_red_herring": True,
                },
            ],
            "logs": {
                ServiceName.AUTH_SERVICE.value: [
                    "[CRITICAL] 22:14:28 HSM (Hardware Security Module) connection lost",
                    "[CRITICAL] 22:14:29 Cannot access signing keys — HSM unreachable",
                    "[ERROR]    22:14:30 All JWT operations suspended",
                ],
                ServiceName.API_GATEWAY.value: [
                    "[ERROR] 22:15:01 Auth dependency unreachable — all auth validation failing",
                    "[ERROR] 22:15:02 Returning 503 to all downstream clients",
                    "[WARN]  22:15:03 Circuit breaker OPEN for auth_service dependency",
                ],
                ServiceName.CACHE.value: [
                    "[ERROR] 22:14:50 All cache nodes unreachable — connection refused",
                    "[WARN]  22:14:51 Falling back to direct DB queries — latency will spike",
                    "[INFO]  22:14:52 Cache cluster last healthy at 22:04:45 — 10min ago",
                ],
            },
            "metrics": {
                ServiceName.AUTH_SERVICE.value: {
                    "error_rate":    0.001,
                    "hsm_connected": 1,
                },
                ServiceName.CACHE.value: {
                    "hit_rate": 0.89,
                    "nodes_up": 3,
                },
                ServiceName.API_GATEWAY.value: {
                    "error_rate":           0.12,
                    "response_time_p99_ms": 340,
                },
            },
            "runbooks": {
                IncidentType.AUTH_FAILURE.value: {
                    "title": "Auth service failure — production runbook v2.1",
                    "steps": [
                        "1. Check HSM connectivity in auth service logs",
                        "2. If HSM unreachable: rotate_credentials on auth_service",
                        "3. After auth restored: flush_cache to clear stale tokens",
                        "4. Monitor api_gateway error_rate — recovers within 90s",
                    ]
                },
            },
            "conflicting_runbook": {
                "title": "Auth service failure — LEGACY runbook v1.0 (DO NOT USE)",
                "steps": [
                    "1. Immediately restart_service on auth_service to force key reload",
                    "2. If cache is down: flush_cache first to prevent stale propagation",
                    "3. Scale up api_gateway to handle retry storm",
                ]
            },
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_observation(self, error_msg: Optional[str] = None) -> Observation:
        return Observation(
            active_alerts=[a.to_public() for a in self._alerts],
            tool_response=self._last_tool_response,
            system_state={k: v for k, v in self._system_state.items()},
            step_count=self._step_count,
            max_steps=self.MAX_STEPS,
            last_action_error=error_msg,
        )

    def _validate_action(self, action: Action) -> Optional[str]:
        if action.action_type in (ActionType.QUERY_LOGS, ActionType.QUERY_METRICS):
            if action.service is None:
                return f"{action.action_type} requires 'service' field"
        if action.action_type == ActionType.READ_RUNBOOK:
            if action.incident_type is None:
                return "read_runbook requires 'incident_type' field"
        if action.action_type == ActionType.APPLY_FIX:
            if action.service is None or action.fix_type is None:
                return "apply_fix requires both 'service' and 'fix_type' fields"
        return None