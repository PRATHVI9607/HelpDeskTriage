# SupportOps Arena — Product Requirements Document (PRD)
## OpenEnv Hackathon 2026 · Meta × PyTorch × Hugging Face
### Build Target: Claude Sonnet 4.5 via GitHub Copilot CLI

---

## TABLE OF CONTENTS

1. [Project Overview](#1-project-overview)
2. [Tech Stack & Tooling Rules](#2-tech-stack--tooling-rules)
3. [Repository Structure](#3-repository-structure)
4. [Environment Core (OpenEnv Spec)](#4-environment-core-openenv-spec)
5. [Pydantic Models & Type System](#5-pydantic-models--type-system)
6. [State Machine & Transitions](#6-state-machine--transitions)
7. [Reward Function](#7-reward-function)
8. [Task Definitions](#8-task-definitions)
9. [Adaptive Adversary Module](#9-adaptive-adversary-module)
10. [Grading System](#10-grading-system)
11. [Baseline Agent](#11-baseline-agent)
12. [FastAPI Server (OpenEnv HTTP Spec)](#12-fastapi-server-openenv-http-spec)
13. [openenv.yaml](#13-openenvyaml)
14. [Dockerfile & Containerization](#14-dockerfile--containerization)
15. [Frontend Dashboard UI](#15-frontend-dashboard-ui)
16. [inference.py (Baseline Inference Script)](#16-inferencepy-baseline-inference-script)
17. [Testing Suite](#17-testing-suite)
18. [README.md Requirements](#18-readmemd-requirements)
19. [Hugging Face Spaces Deployment](#19-hugging-face-spaces-deployment)
20. [Pre-Submission Checklist Compliance](#20-pre-submission-checklist-compliance)
21. [Design System & UI Rules](#21-design-system--ui-rules)
22. [Copilot CLI Prompt Strategy](#22-copilot-cli-prompt-strategy)
23. [Error Handling & Edge Cases](#23-error-handling--edge-cases)
24. [Absolute Rules & Prohibitions](#24-absolute-rules--prohibitions)

---

## 1. PROJECT OVERVIEW

### 1.1 What Is SupportOps Arena

SupportOps Arena is a **production-grade, OpenEnv-compliant reinforcement learning benchmark environment** that simulates enterprise IT support operations. An AI agent acts as a Level-1/Level-2 IT support operator. It receives incomplete, noisy incident tickets and must:

- Gather evidence through sequential tool use (diagnostic actions)
- Reason about hidden root causes under partial observability
- Apply risk-weighted remediation actions
- Escalate when autonomous resolution is unsafe
- Generalize policy across an adaptive adversary that targets agent weaknesses

### 1.2 Why It Exists

This is NOT a toy. IT support is a $50B+ industry. Agents deployed here face genuine operational consequences — wrong actions disrupt real users, wrong escalations waste L3 engineer time, wrong closures leave root causes unresolved. The environment tests:

- Long-horizon planning (10–24 step episodes)
- Partial observability (hidden root causes, noisy logs)
- Risk-aware action selection
- Cross-episode pattern recognition (hard task)
- Policy robustness under adversarial pressure

### 1.3 Hackathon Evaluation Weights

| Criterion | Weight |
|-----------|--------|
| Real-world utility | 30% |
| Task & grader quality | 25% |
| Environment design | 20% |
| Code quality & spec compliance | 15% |
| Creativity & novelty | 10% |

**Every design decision must maximize these weights.**

---

## 2. TECH STACK & TOOLING RULES

### 2.1 Required Stack

| Layer | Technology | Version | Rule |
|-------|-----------|---------|------|
| Language | Python | 3.11+ | REQUIRED. No 3.10 or below. |
| Type models | Pydantic | v2 | REQUIRED. All models must use `model_config`, `model_validator`, `field_validator`. No v1 syntax. |
| API server | FastAPI | 0.110+ | REQUIRED. All endpoints async. |
| ASGI server | Uvicorn | latest | Run with `--host 0.0.0.0 --port 7860` |
| Containerization | Docker | latest | Single-stage or multi-stage. Must work with `docker build . && docker run -p 7860:7860` |
| Testing | pytest + pytest-asyncio | latest | All env tests must be async. |
| LLM calls | OpenAI Python SDK | v1.x | Use `AsyncOpenAI`. Read `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN` from env vars. |
| Frontend | Vanilla HTML/CSS/JS | — | Single `index.html` served by FastAPI. No React needed for UI. |
| Logging | Python `logging` | stdlib | Structured JSON logs to stdout. |
| Random seed | Python `random` + `numpy` | — | All stochastic elements must accept a `seed` param for reproducibility. |

### 2.2 Environment Variables (MANDATORY)

The following must be read from environment at runtime. Never hardcode:

```python
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
```

### 2.3 Prohibited Patterns

- ❌ No `from pydantic import validator` (v1 style)
- ❌ No synchronous `requests` calls anywhere in async paths
- ❌ No hardcoded API keys in any file
- ❌ No `print()` for logging — use `logging.getLogger(__name__)`
- ❌ No global mutable state that breaks concurrent requests
- ❌ No `time.sleep()` in async code — use `asyncio.sleep()`
- ❌ No circular imports

---

## 3. REPOSITORY STRUCTURE

Create **exactly** this directory structure. No deviations:

```
supportops-arena/
├── env/
│   ├── __init__.py
│   ├── environment.py          # Core OpenEnv class (SupportOpsArena)
│   ├── state.py                # All Pydantic models
│   ├── actions.py              # Action enum + metadata
│   ├── rewards.py              # RewardCalculator class
│   ├── transitions.py          # StateMachine class
│   └── scenarios.py            # TicketGenerator class
├── adversary/
│   ├── __init__.py
│   ├── adversary.py            # AdaptiveAdversary class
│   └── policy_tracker.py       # PolicyTracker class
├── graders/
│   ├── __init__.py
│   ├── programmatic.py         # ProgrammaticGrader class
│   ├── llm_grader.py           # LLMGrader class
│   └── adversarial_grader.py   # AdversarialGrader class
├── baseline/
│   ├── __init__.py
│   └── baseline_agent.py       # BaselineAgent class
├── tests/
│   ├── __init__.py
│   ├── conftest.py             # pytest fixtures
│   ├── test_env.py             # Environment unit tests
│   ├── test_graders.py         # Grader accuracy tests
│   ├── test_transitions.py     # State machine tests
│   └── test_adversary.py       # Adversary behavior tests
├── app/
│   ├── __init__.py
│   ├── server.py               # FastAPI app
│   └── static/
│       └── index.html          # Frontend dashboard
├── openenv.yaml                # OpenEnv metadata spec
├── inference.py                # Baseline inference script (ROOT LEVEL)
├── Dockerfile
├── requirements.txt
├── .env.example
├── PRD.md                      # This file
└── README.md
```

---

## 4. ENVIRONMENT CORE (OPENENV SPEC)

### 4.1 `env/environment.py` — Full Specification

The main class `SupportOpsArena` must implement the full OpenEnv interface. Here is the complete contract:

```python
class SupportOpsArena:
    """
    OpenEnv-compliant environment for enterprise IT incident triage.
    
    Implements:
    - async reset(task_level: str) -> EnvObservation
    - async step(action: Action) -> StepResult
    - async state() -> EnvState
    - Pydantic v2 typed models throughout
    - Dense reward shaping
    - Adaptive adversary integration
    - 3 task levels: easy, medium, hard
    """
```

#### 4.1.1 `reset()` Method

```python
async def reset(self, task_level: str = "easy", seed: int | None = None) -> EnvObservation:
```

Rules:
- `task_level` must be one of `["easy", "medium", "hard"]`. Raise `ValueError` otherwise.
- Sample a new ticket scenario using `TicketGenerator` with the adversary's current sampling weights.
- Set `step_count = 0`.
- Set `episode_id` as a UUID4 string.
- Set `escalation_allowed = False` initially (unlocked after minimum diagnostic steps per task).
- Store `hidden_state` (root cause, correct remediation) in `EnvState` but NOT in `EnvObservation`.
- Log episode start with `episode_id`, `task_level`, `seed`.
- Return an `EnvObservation` with all fields populated. `system_logs` starts with 3–5 entries (may include one misleading entry for medium/hard).
- `action_history` starts as empty list.

#### 4.1.2 `step()` Method

```python
async def step(self, action: Action) -> StepResult:
```

`StepResult` is a named tuple / dataclass: `(observation: EnvObservation, reward: float, done: bool, info: dict)`

Rules:
- Validate action is a valid `ActionType` enum member. Return penalty reward if invalid.
- Check if episode is already done. Raise `RuntimeError("Episode already terminated. Call reset().")` if so.
- Pass action through `StateMachine.transition()` to get next state.
- Calculate reward via `RewardCalculator.calculate()`.
- Increment `step_count`.
- Append action to `action_history`.
- Check terminal conditions:
  - `action.name == ActionType.RESOLVE_TICKET` → done, apply terminal reward.
  - `action.name == ActionType.ESCALATE_TICKET` → done, apply escalation reward.
  - `action.name == ActionType.CLOSE_WITHOUT_FIX` → done, apply heavy penalty.
  - `step_count >= max_steps[task_level]` → done, apply budget-exceeded penalty.
- Update adversary policy tracker with this action.
- `info` dict must contain: `{"episode_id": ..., "step": ..., "root_cause": None, "reward_breakdown": {...}}`. Do NOT reveal `root_cause` until `done=True`.
- When `done=True`, `info["root_cause"]` = actual root cause string, `info["correct"]` = bool.

#### 4.1.3 `state()` Method

```python
async def state(self) -> EnvState:
```

Returns full `EnvState` including `hidden_state`. Used by graders only. Never expose to agent during active episode.

#### 4.1.4 Step Budgets

```python
MAX_STEPS = {
    "easy": 10,
    "medium": 16,
    "hard": 24
}
```

#### 4.1.5 Escalation Unlock Logic

```python
ESCALATION_UNLOCK_AFTER = {
    "easy": 3,    # Must take 3+ diagnostic actions first
    "medium": 4,  # Must take 4+ diagnostic actions first
    "hard": 5     # Must take 5+ diagnostic actions first
}
```

`escalation_allowed` flips to `True` in the observation after the unlock threshold is met. If agent tries to escalate before unlock, return penalty `-0.20` and do NOT terminate episode.

---

## 5. PYDANTIC MODELS & TYPE SYSTEM

### 5.1 `env/state.py` — Complete Model Definitions

Every model uses Pydantic v2. No exceptions.

```python
from pydantic import BaseModel, Field, field_validator, model_validator
from enum import Enum
from datetime import datetime
import uuid

# ─── Enums ───────────────────────────────────────────────────

class NetworkStatus(str, Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    DOWN = "DOWN"
    UNKNOWN = "UNKNOWN"

class VPNStatus(str, Enum):
    CONNECTED = "CONNECTED"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"
    NA = "N/A"

class AuthStatus(str, Enum):
    OK = "OK"
    LOCKED = "LOCKED"
    EXPIRED = "EXPIRED"
    MFA_FAIL = "MFA_FAIL"

class ServiceHealth(str, Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    DOWN = "DOWN"
    UNKNOWN = "UNKNOWN"

class TaskLevel(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"

# ─── Sub-models ────────────────────────────────────────────── 

class UserContext(BaseModel):
    department: str
    role: str
    device_type: str
    os_version: str
    location: str  # "onsite" | "remote"

class LogEntry(BaseModel):
    timestamp: str
    level: str        # "INFO" | "WARN" | "ERROR" | "DEBUG"
    service: str
    message: str
    is_misleading: bool = False  # Hidden from agent, used by grader

class ActionRecord(BaseModel):
    """Record of a past action in action_history."""
    step: int
    action_name: str
    rationale: str | None
    reward_received: float
    timestamp: str

# ─── Core Models ───────────────────────────────────────────── 

class EnvObservation(BaseModel):
    """What the AGENT sees. No hidden info."""
    ticket_id: str
    ticket_summary: str
    user_context: UserContext
    network_status: NetworkStatus
    vpn_status: VPNStatus
    auth_status: AuthStatus
    service_health: dict[str, ServiceHealth]
    system_logs: list[LogEntry]
    action_history: list[ActionRecord]
    step_count: int
    escalation_allowed: bool
    confidence_score: float | None = None  # Optional agent field
    task_level: TaskLevel
    steps_remaining: int

class HiddenState(BaseModel):
    """Internal ground truth. NEVER sent to agent."""
    root_cause: str
    root_cause_category: str  # "network" | "auth" | "dns" | "vpn" | "sso" | "hardware"
    correct_remediation: str  # Which ActionType resolves this
    correct_remediation_alts: list[str]  # Acceptable alternates
    misleading_log_index: int | None  # Which log entry is the red herring
    severity: str  # "low" | "medium" | "high" | "critical"
    affected_users_count: int  # Side-effect blast radius
    ticket_variant_id: str  # For adversary tracking

class EnvState(BaseModel):
    """Full internal state. Graders only."""
    episode_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_level: TaskLevel
    observation: EnvObservation
    hidden: HiddenState
    cumulative_reward: float = 0.0
    done: bool = False
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    action_log: list[ActionRecord] = Field(default_factory=list)
    diagnostic_steps_taken: int = 0  # Tracks unlock threshold
    
class StepResult(BaseModel):
    observation: EnvObservation
    reward: float
    done: bool
    info: dict
```

---

## 6. STATE MACHINE & TRANSITIONS

### 6.1 `env/actions.py`

Define all 15 actions with metadata:

```python
from enum import Enum
from dataclasses import dataclass

class ActionType(str, Enum):
    INSPECT_NETWORK     = "inspect_network"
    INSPECT_LOGS        = "inspect_logs"
    CHECK_AUTHENTICATION = "check_authentication"
    CHECK_PERMISSIONS   = "check_permissions"
    QUERY_DEVICE_STATUS = "query_device_status"
    SEARCH_INTERNAL_KB  = "search_internal_kb"
    CONTACT_USER        = "contact_user_for_info"
    RUN_DIAGNOSTIC      = "run_diagnostic_script"
    FLUSH_DNS           = "flush_dns"
    RECONFIGURE_CLIENT  = "reconfigure_client"
    RESTART_SERVICE     = "restart_service"
    RESET_CREDENTIALS   = "reset_credentials"
    ESCALATE_TICKET     = "escalate_ticket"
    RESOLVE_TICKET      = "resolve_ticket"
    CLOSE_WITHOUT_FIX   = "close_without_fix"

@dataclass
class ActionMetadata:
    risk_level: str       # "LOW" | "MED" | "HIGH" | "NONE"
    info_yield: str       # "HIGH" | "MED" | "LOW" | "NONE"
    step_cost: int        # 1 for all, kept for future
    requires_prior_diagnostic: bool
    can_disrupt_other_users: bool
    is_terminal: bool

ACTION_METADATA: dict[ActionType, ActionMetadata] = {
    ActionType.INSPECT_NETWORK:      ActionMetadata("LOW",  "HIGH", 1, False, False, False),
    ActionType.INSPECT_LOGS:         ActionMetadata("LOW",  "HIGH", 1, False, False, False),
    ActionType.CHECK_AUTHENTICATION: ActionMetadata("LOW",  "HIGH", 1, False, False, False),
    ActionType.CHECK_PERMISSIONS:    ActionMetadata("LOW",  "MED",  1, False, False, False),
    ActionType.QUERY_DEVICE_STATUS:  ActionMetadata("LOW",  "MED",  1, False, False, False),
    ActionType.SEARCH_INTERNAL_KB:   ActionMetadata("LOW",  "MED",  1, False, False, False),
    ActionType.CONTACT_USER:         ActionMetadata("LOW",  "MED",  1, False, False, False),
    ActionType.RUN_DIAGNOSTIC:       ActionMetadata("MED",  "HIGH", 1, True,  False, False),
    ActionType.FLUSH_DNS:            ActionMetadata("MED",  "LOW",  1, False, False, False),
    ActionType.RECONFIGURE_CLIENT:   ActionMetadata("MED",  "LOW",  1, False, False, False),
    ActionType.RESTART_SERVICE:      ActionMetadata("HIGH", "LOW",  1, True,  True,  False),
    ActionType.RESET_CREDENTIALS:    ActionMetadata("HIGH", "LOW",  1, True,  False, False),
    ActionType.ESCALATE_TICKET:      ActionMetadata("NONE", "NONE", 1, False, False, True),
    ActionType.RESOLVE_TICKET:       ActionMetadata("NONE", "NONE", 1, False, False, True),
    ActionType.CLOSE_WITHOUT_FIX:    ActionMetadata("NONE", "NONE", 1, False, False, True),
}
```

### 6.2 `env/transitions.py` — StateMachine

The state machine enforces action preconditions and generates next observations:

```python
class StateMachine:
    async def transition(self, state: EnvState, action: Action) -> tuple[EnvState, dict]:
        """
        Apply action to state. Returns (next_state, reward_events).
        
        Precondition rules (enforce strictly):
        1. If action requires_prior_diagnostic and diagnostic_steps_taken == 0:
           → Inject reward event: RISKY_WITHOUT_DIAGNOSTIC = -0.15
           → Still apply the action (don't block it), but penalize
        
        2. If action is ESCALATE_TICKET and not escalation_allowed:
           → Inject reward event: PREMATURE_ESCALATION = -0.20
           → Do NOT terminate. Keep episode running.
        
        3. If same action has been taken before (action.name in past action_history):
           → Inject reward event: REDUNDANT_ACTION = -0.05
        
        Observation updates per action (what new info is revealed):
        - inspect_network: Update network_status, vpn_status with true values (or noisy version)
        - inspect_logs: Return next batch of log entries (some may be misleading)
        - check_authentication: Update auth_status with true values
        - check_permissions: Add permission info to service_health
        - query_device_status: Update user_context.device details
        - search_internal_kb: Append KB result as a special LogEntry
        - contact_user: Append user response as LogEntry (costs 1 step)
        - run_diagnostic_script: Return targeted diagnostic result (high accuracy)
        - flush_dns: Side effect: network_status may briefly show DEGRADED
        - reconfigure_client: Partial fix (may resolve DNS-related causes)
        - restart_service: Updates service_health, but may trigger side effects
        - reset_credentials: Forces auth_status to OK but logs the action
        - escalate_ticket: Terminal if allowed
        - resolve_ticket: Terminal always
        - close_without_fix: Terminal always
        
        For diagnostic actions that match root cause category:
        → Inject reward event: CORRECT_TARGETED_DIAGNOSTIC = +0.20
        
        For any diagnostic action:
        → Increment state.diagnostic_steps_taken
        → Check if escalation_allowed should flip to True
        → Inject reward event: DIAGNOSTIC_ACTION = +0.10
        """
```

Noise model for `inspect_logs`:
- Easy task: 0% misleading entries
- Medium task: 1 misleading entry injected at random position
- Hard task: 1–2 misleading entries, plus one contradictory entry

---

## 7. REWARD FUNCTION

### 7.1 `env/rewards.py` — RewardCalculator

All reward events must be tracked individually in a `reward_breakdown` dict for the `info` return:

```python
REWARD_EVENTS = {
    # Positive
    "DIAGNOSTIC_ACTION":            +0.10,
    "CORRECT_TARGETED_DIAGNOSTIC":  +0.20,
    "EVIDENCE_BEFORE_RISKY":        +0.05,
    "CORRECT_REMEDIATION":          +0.40,
    "OPTIMAL_RESOLUTION":           +0.20,  # Only if steps_used <= min_required_steps
    "JUSTIFIED_ESCALATION":         +0.30,
    "TERMINAL_CORRECT":             +1.00,
    
    # Negative
    "REDUNDANT_ACTION":             -0.05,
    "RISKY_WITHOUT_DIAGNOSTIC":     -0.15,
    "INCORRECT_REMEDIATION":        -0.25,
    "HARMFUL_ACTION":               -0.35,
    "PREMATURE_CLOSURE":            -0.40,
    "BUDGET_EXCEEDED_PER_STEP":     -0.02,
    "PREMATURE_ESCALATION":         -0.20,  # Escalate before unlock
    "UNJUSTIFIED_ESCALATION":       -0.20,  # Escalate without enough evidence
}

class RewardCalculator:
    def calculate(self, events: list[str], state: EnvState, action: Action) -> tuple[float, dict]:
        """
        Takes list of reward event keys, computes total reward.
        Returns (total_reward, breakdown_dict).
        
        Normalization:
        max_possible_reward per task level (precomputed):
        - easy: 2.05  (DIAGNOSTIC×3 + CORRECT_TARGETED + CORRECT_REMEDIATION + TERMINAL + OPTIMAL)
        - medium: 2.35
        - hard: 2.75
        
        Episode score = clamp(cumulative_reward / max_possible[task], 0.0, 1.0)
        """
```

### 7.2 Minimum Steps for Optimal Resolution Bonus

```python
MIN_STEPS_FOR_OPTIMAL = {
    "easy": 4,    # Any solution in ≤4 steps gets the +0.20 bonus
    "medium": 7,
    "hard": 12
}
```

---

## 8. TASK DEFINITIONS

### 8.1 `env/scenarios.py` — TicketGenerator

#### Task 1 — Easy: Wi-Fi Connectivity Failure

```python
TASK_EASY_ROOT_CAUSES = [
    {
        "id": "wifi_adapter_disabled",
        "description": "Network adapter disabled by OS update",
        "category": "hardware",
        "correct_remediation": ActionType.RECONFIGURE_CLIENT,
        "diagnostic_path": [ActionType.QUERY_DEVICE_STATUS, ActionType.INSPECT_NETWORK],
        "severity": "low",
        "affected_users": 1
    },
    {
        "id": "wifi_wrong_credentials",
        "description": "Incorrect saved credentials for SSID",
        "category": "auth",
        "correct_remediation": ActionType.RESET_CREDENTIALS,
        "diagnostic_path": [ActionType.CHECK_AUTHENTICATION, ActionType.INSPECT_LOGS],
        "severity": "low",
        "affected_users": 1
    },
    {
        "id": "wifi_dns_corruption",
        "description": "Local DNS cache corruption",
        "category": "dns",
        "correct_remediation": ActionType.FLUSH_DNS,
        "diagnostic_path": [ActionType.INSPECT_NETWORK, ActionType.INSPECT_LOGS],
        "severity": "low",
        "affected_users": 1
    },
    {
        "id": "wifi_airplane_mode",
        "description": "Airplane mode enabled by accident",
        "category": "hardware",
        "correct_remediation": ActionType.RECONFIGURE_CLIENT,
        "diagnostic_path": [ActionType.QUERY_DEVICE_STATUS],
        "severity": "low",
        "affected_users": 1
    },
    {
        "id": "wifi_dhcp_exhausted",
        "description": "DHCP lease pool exhausted",
        "category": "network",
        "correct_remediation": ActionType.RESTART_SERVICE,
        "diagnostic_path": [ActionType.INSPECT_NETWORK, ActionType.RUN_DIAGNOSTIC],
        "severity": "medium",
        "affected_users": 15
    },
    {
        "id": "wifi_wrong_ssid",
        "description": "Client connected to wrong SSID (guest vs corporate)",
        "category": "network",
        "correct_remediation": ActionType.RECONFIGURE_CLIENT,
        "diagnostic_path": [ActionType.QUERY_DEVICE_STATUS, ActionType.INSPECT_NETWORK],
        "severity": "low",
        "affected_users": 1
    }
]
```

Ticket template for easy:
```
"ticket_summary": "Hi, I can't connect to Wi-Fi and need help urgently."
"user_context": randomized department/role/device/OS
"initial_logs": 3 generic log entries, 0 misleading
```

#### Task 2 — Medium: VPN Access Failure

Root causes (one OR two combined per episode):

```python
TASK_MEDIUM_ROOT_CAUSES = [
    "expired_password",           # auth_status: EXPIRED
    "vpn_version_mismatch",       # service_health["vpn_client"]: DEGRADED
    "dns_split_tunnel_conflict",  # network shows HEALTHY but vpn_status: FAILED
    "mfa_token_desync",           # auth_status: MFA_FAIL
    "firewall_subnet_block",      # network_status: DEGRADED, specific subnet
    "certificate_expiry",         # service_health["vpn_cert"]: DOWN
]
```

Key challenge implementation:
- Always inject exactly 1 misleading log entry at position `random.randint(1, len(logs)-1)`
- The misleading entry must point toward a plausible but incorrect root cause
- The agent must gather evidence from ≥3 sources before the grader gives full credit

Ticket template:
```
"ticket_summary": "I'm working from home and can't access any internal tools through VPN. 
                   It was working fine yesterday."
"initial_logs": 5 entries, 1 is misleading (flagged in HiddenState.misleading_log_index)
```

#### Task 3 — Hard: Cross-System Access Failure with Shared Infrastructure

This task delivers **two sequential tickets** in the same episode. The agent must recognize they share a root cause.

Implementation:
```python
class HardTaskState(BaseModel):
    ticket_a: EnvObservation  # Email access failure
    ticket_b: EnvObservation  # Enterprise tools access failure
    shared_root_cause: str    # "sso_token_service_degradation"
    cross_ticket_recognized: bool = False  # Set to True if agent escalates with both ticket IDs
    
# Ticket A symptoms:
# - auth_status: EXPIRED (misleading — it's not expired, it's SSO backend failing)
# - service_health["mail"]: DEGRADED
# - vpn_status: N/A

# Ticket B symptoms:  
# - auth_status: OK (misleading — cached credentials still valid)
# - service_health["sharepoint"]: DOWN
# - service_health["teams"]: DEGRADED

# True root cause: SSO token service degradation on datacenter-east
# Correct resolution: ESCALATE with summary mentioning both ticket IDs and SSO hypothesis
```

The adversary for hard task:
- Reads the agent's action history from last 5 episodes
- If agent always does inspect_logs → inject double-misleading-log variant
- If agent always escalates at step 5 → generate scenario where early escalation is penalized
- If agent scores > 0.85 on medium → unlock "contradictory evidence" hard variant

---

## 9. ADAPTIVE ADVERSARY MODULE

### 9.1 `adversary/policy_tracker.py`

```python
class PolicyTracker:
    """
    Tracks agent behavior across episodes.
    Maintains a sliding window of last N=20 episodes.
    """
    
    def record_episode(self, episode_id: str, task_level: str, actions: list[ActionRecord], score: float):
        """Record completed episode into tracker."""
    
    def get_action_frequencies(self, task_level: str) -> dict[str, float]:
        """Returns normalized frequency of each action type."""
    
    def get_first_action_distribution(self) -> dict[str, float]:
        """What does the agent always do first?"""
    
    def get_escalation_step_distribution(self) -> list[int]:
        """At which steps does agent escalate?"""
    
    def get_score_by_root_cause_category(self) -> dict[str, float]:
        """Which root cause categories does agent struggle with?"""
```

### 9.2 `adversary/adversary.py`

```python
class AdaptiveAdversary:
    """
    Analyzes policy_tracker and adjusts ticket sampling weights.
    Stateless per-episode. Reads from policy_tracker at episode start.
    """
    
    def get_sampling_weights(self, task_level: str) -> dict[str, float]:
        """
        Returns weights for each root_cause_id.
        
        Rules:
        - If agent scores > 0.85 on dns_corruption: increase weight of mfa_token_desync
        - If agent always runs inspect_logs first: enable misleading_log_boost=True
        - If agent escalates at step < unlock_threshold + 1: increase 
          UNJUSTIFIED_ESCALATION scenario weight
        - If agent uses restart_service > 40% of episodes: enable cascade_failure_mode=True
        - If agent has < 0.40 score on any category: boost that category by 2x
        
        Default: uniform weights across all root causes.
        """
    
    def get_injection_config(self, agent_profile: dict) -> dict:
        """
        Returns episode configuration overrides:
        - misleading_log_count: 0 | 1 | 2
        - contradictory_evidence: bool
        - cascade_failure_enabled: bool
        - variant_id: str (for tracking)
        """
```

---

## 10. GRADING SYSTEM

### 10.1 `graders/programmatic.py`

```python
class ProgrammaticGrader:
    """
    Rule-based ground truth checker.
    Used for all three tasks.
    """
    
    async def grade(self, state: EnvState) -> GradeResult:
        """
        Checks:
        1. root_cause_identified: bool
           → Was the correct remediation applied (or correct escalation reason given)?
        
        2. action_sequence_safe: bool
           → Were all HIGH-risk actions preceded by ≥1 diagnostic action?
        
        3. step_budget_respected: bool
           → step_count <= MAX_STEPS[task_level]
        
        4. no_harmful_actions: bool
           → No RESTART_SERVICE that triggered cascade?
        
        5. evidence_breadth_score: float [0.0-1.0]
           → For medium/hard: # unique diagnostic categories used / required_categories
        
        Returns GradeResult with score 0.0-1.0 and breakdown.
        """

class GradeResult(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    root_cause_identified: bool
    action_sequence_safe: bool
    step_budget_respected: bool
    no_harmful_actions: bool
    evidence_breadth_score: float
    breakdown: dict[str, float]
```

### 10.2 `graders/llm_grader.py`

Used for medium and hard tasks. Grades the agent's `rationale` strings.

```python
class LLMGrader:
    """
    LLM judge using OpenAI-compatible API.
    Reads API_BASE_URL, MODEL_NAME, OPENAI_API_KEY from env.
    """
    
    SYSTEM_PROMPT = """
    You are an expert IT operations grader evaluating an AI agent's performance on an 
    incident triage task. You will be given:
    1. The hidden root cause (ground truth)
    2. The agent's action history with rationale strings
    3. The final escalation summary (if escalated)
    
    Grade the following dimensions on a scale of 0.0 to 1.0:
    - diagnostic_coherence: Are the agent's rationales logically consistent with the observations?
    - evidence_sufficiency: Did the agent gather enough evidence before committing to a fix?
    - root_cause_narration: If escalated, does the escalation summary correctly identify the cause?
    
    Return ONLY a JSON object with keys: diagnostic_coherence, evidence_sufficiency, 
    root_cause_narration, overall, reasoning (string).
    Do not include any other text.
    """
    
    async def grade(self, state: EnvState) -> LLMGradeResult:
        """
        Calls LLM API with action history + rationales.
        Parses JSON response.
        Returns LLMGradeResult.
        Falls back to score=0.5 if API call fails.
        """
```

### 10.3 `graders/adversarial_grader.py`

Used for hard task only. Tests policy robustness.

```python
class AdversarialGrader:
    """
    After episode completion, generates a harder variant of the same ticket.
    Checks if the agent's learned policy (reconstructed from action history) 
    would still succeed on the variant.
    """
    
    async def grade(self, state: EnvState, adversary: AdaptiveAdversary) -> float:
        """
        1. Generate variant ticket (same root cause, different surface presentation)
        2. Replay agent's action policy (deterministic from action_history)
        3. Score: would that same action sequence work on the variant?
        Returns 0.0-1.0 robustness score.
        """
```

### 10.4 Score Aggregation

```python
def aggregate_scores(
    programmatic: float,
    llm: float | None,
    adversarial: float | None,
    task_level: str
) -> float:
    if task_level == "easy":
        return programmatic
    elif task_level == "medium":
        return 0.6 * programmatic + 0.4 * (llm or 0.5)
    else:  # hard
        return 0.5 * programmatic + 0.3 * (llm or 0.5) + 0.2 * (adversarial or 0.5)

OVERALL_BENCHMARK_SCORE_WEIGHTS = {"easy": 0.2, "medium": 0.3, "hard": 0.5}
```

---

## 11. BASELINE AGENT

### 11.1 `baseline/baseline_agent.py`

This is a **scripted heuristic agent** — no ML model required. Must produce deterministic, reproducible scores.

```python
class BaselineAgent:
    """
    Scripted heuristic baseline agent.
    No API calls. No learned model. Fully deterministic given a seed.
    
    Strategy:
    Step 1: Always run inspect_logs + check_authentication (safe, high-yield)
    Step 2: Based on observation fields, apply decision tree:
        - If auth_status in [LOCKED, EXPIRED, MFA_FAIL] → check_permissions → reset_credentials
        - If network_status in [DEGRADED, DOWN] → inspect_network → run_diagnostic_script
        - If vpn_status in [FAILED, TIMEOUT] → reconfigure_client → flush_dns
        - If service_health has DOWN services → run_diagnostic_script → restart_service
        - Default: query_device_status → search_internal_kb
    Step 3: Choose lowest-risk remediation matching most likely root cause
    Step 4: If diagnostic_steps >= 60% of budget and uncertainty high → escalate_ticket
    
    Expected scores:
    - Easy:   0.65 – 0.75
    - Medium: 0.40 – 0.55
    - Hard:   0.20 – 0.35
    """
    
    async def select_action(self, observation: EnvObservation) -> Action:
        """Pure function. Returns Action with a rationale string."""
    
    async def run_episode(self, env: SupportOpsArena, task_level: str) -> float:
        """Run one full episode. Returns episode score."""
```

---

## 12. FASTAPI SERVER (OPENENV HTTP SPEC)

### 12.1 `app/server.py` — Complete Endpoint Specification

```python
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

app = FastAPI(
    title="SupportOps Arena",
    description="OpenEnv-compliant IT incident triage RL environment",
    version="1.0.0"
)
```

#### Required Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Serve `index.html` (dashboard UI) |
| GET | `/health` | Returns `{"status": "ok", "version": "1.0.0"}` |
| POST | `/reset` | Reset environment, return initial observation |
| POST | `/step` | Take action, return observation + reward |
| GET | `/state` | Return full env state (grader use) |
| GET | `/scores` | Return scores for all 3 tasks from last baseline run |
| POST | `/grade` | Run grader on current episode |
| GET | `/openenv.yaml` | Serve openenv.yaml as text |
| GET | `/tasks` | List all 3 tasks with metadata |
| POST | `/baseline/run` | Run baseline agent on all 3 tasks |
| GET | `/docs` | FastAPI auto-docs (keep enabled) |

#### `/reset` Request/Response

```python
class ResetRequest(BaseModel):
    task_level: str = "easy"  # "easy" | "medium" | "hard"
    seed: int | None = None

# Response: EnvObservation (full typed model)
```

#### `/step` Request/Response

```python
class StepRequest(BaseModel):
    action_name: str  # ActionType enum value
    rationale: str | None = None

# Response: StepResult (observation, reward, done, info)
```

#### Session Management

Each HTTP session must maintain its own environment instance. Use a session ID (UUID) passed as a header `X-Session-ID`. If no session ID, auto-generate and return it in response headers. Store sessions in a dict with TTL of 30 minutes.

```python
sessions: dict[str, SupportOpsArena] = {}
SESSION_TTL_SECONDS = 1800
```

#### CORS Configuration

```python
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## 13. OPENENV.YAML

The file at root level must validate against `openenv validate`:

```yaml
name: supportops-arena
version: 1.0.0
description: >
  Enterprise IT incident triage RL environment. An AI agent acts as a 
  Level-1/2 IT support operator, triaging incidents under partial 
  observability with risk-weighted actions and an adaptive adversary.
author: SupportOps Arena Team
license: Apache-2.0
tags:
  - openenv
  - rl-environment
  - enterprise
  - it-support
  - partial-observability
  - multi-task
tasks:
  - id: easy
    name: Wi-Fi Connectivity Failure
    description: Triage a single Wi-Fi failure ticket. 6 possible root causes.
    max_steps: 10
    score_range: [0.0, 1.0]
    grader: programmatic
    difficulty: easy
  - id: medium
    name: VPN Access Failure
    description: Multi-cause VPN failure with one misleading log entry.
    max_steps: 16
    score_range: [0.0, 1.0]
    grader: programmatic+llm
    difficulty: medium
  - id: hard
    name: Cross-System Access Failure
    description: Two tickets share a hidden SSO infrastructure root cause.
    max_steps: 24
    score_range: [0.0, 1.0]
    grader: programmatic+llm+adversarial
    difficulty: hard
observation_type: structured_json
action_type: discrete_enum
action_count: 15
reward_type: dense_shaped
reward_range: [-0.40, 1.00]
grader: programmatic+llm+adversarial
deployment:
  type: huggingface_spaces
  space_sdk: docker
  port: 7860
docker: true
baseline_agent: included
api_server:
  type: fastapi
  base_url: /
  endpoints:
    reset: POST /reset
    step: POST /step
    state: GET /state
    health: GET /health
```

---

## 14. DOCKERFILE & CONTAINERIZATION

### 14.1 `Dockerfile`

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python deps first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Create non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Run
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
```

### 14.2 `requirements.txt`

```
fastapi==0.110.0
uvicorn[standard]==0.27.1
pydantic==2.6.4
openai==1.14.0
pytest==8.1.0
pytest-asyncio==0.23.5
httpx==0.27.0
pyyaml==6.0.1
python-dotenv==1.0.1
numpy==1.26.4
```

### 14.3 Build & Run Validation

The container must:
- Build cleanly: `docker build -t supportops-arena .`
- Run cleanly: `docker run -p 7860:7860 supportops-arena`
- Respond to: `curl http://localhost:7860/health` → `{"status": "ok"}`
- Respond to: `curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{"task_level": "easy"}'`

---

## 15. FRONTEND DASHBOARD UI

### 15.1 Overview

Build a stunning, interactive dashboard at `app/static/index.html`. This is served by FastAPI at `/`. It allows humans to:
- Watch the environment in action
- Step through episodes manually
- View real-time reward signals
- Run the baseline agent and watch it play
- See scores across all 3 tasks

### 15.2 Design System

**Color Palette (STRICT — do not deviate):**

```css
:root {
  /* Backgrounds */
  --bg-primary:     #FFFFFF;              /* Pure white */
  --bg-secondary:   #F0F8FF;              /* Alice blue (very light sky) */
  --bg-card:        #FFFFFF;              /* White cards */
  --bg-glass:       rgba(255,255,255,0.8);/* Frosted glass */
  
  /* Gradient (primary usage) */
  --gradient-main:  linear-gradient(135deg, #FFFFFF 0%, #E0F4FF 40%, #B8E4FF 100%);
  --gradient-hero:  linear-gradient(160deg, #FFFFFF 0%, #D6EEFF 50%, #87CEEB 100%);
  --gradient-card:  linear-gradient(145deg, rgba(255,255,255,0.95), rgba(224,244,255,0.6));
  
  /* Sky Blues */
  --sky-100:        #E0F4FF;
  --sky-200:        #B8E4FF;
  --sky-300:        #87CEEB;              /* Sky blue */
  --sky-400:        #5BB8E8;
  --sky-500:        #2196F3;              /* Primary accent */
  --sky-600:        #1976D2;
  --sky-700:        #1565C0;
  
  /* Text */
  --text-primary:   #0D1B2A;             /* Near-black, deep navy */
  --text-secondary: #3A5068;             /* Muted navy */
  --text-muted:     #7B9BB5;             /* Light muted */
  --text-on-accent: #FFFFFF;
  
  /* Status colors */
  --success:        #10B981;
  --warning:        #F59E0B;
  --danger:         #EF4444;
  --info:           #3B82F6;
  
  /* Borders */
  --border-light:   rgba(135, 206, 235, 0.3);
  --border-medium:  rgba(91, 184, 232, 0.5);
  
  /* Shadows */
  --shadow-sm:      0 2px 8px rgba(33, 150, 243, 0.08);
  --shadow-md:      0 4px 20px rgba(33, 150, 243, 0.12);
  --shadow-lg:      0 8px 40px rgba(33, 150, 243, 0.16);
  --shadow-glow:    0 0 30px rgba(135, 206, 235, 0.4);
  
  /* Sizing */
  --radius-sm:      8px;
  --radius-md:      12px;
  --radius-lg:      20px;
  --radius-xl:      28px;
  
  /* Transitions */
  --transition-fast: 0.15s cubic-bezier(0.4, 0, 0.2, 1);
  --transition-med:  0.3s cubic-bezier(0.4, 0, 0.2, 1);
  --transition-slow: 0.5s cubic-bezier(0.4, 0, 0.2, 1);
}
```

**Typography:**

```css
/* Import from Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&family=Sora:wght@600;700;800&display=swap');

body { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'Sora', sans-serif; }
code, .mono { font-family: 'DM Mono', monospace; }
```

**Global Rules:**
- Background is always the `--gradient-main` or pure white — NEVER dark
- Cards use `--gradient-card` with `backdrop-filter: blur(12px)`
- Borders are always light sky blue tinted — `--border-light`
- All interactive elements have smooth hover transitions
- Scrollbars are styled (thin, sky blue thumb)
- No black backgrounds anywhere

### 15.3 Layout Structure

```
┌─────────────────────────────────────────────────────────┐
│  HEADER: Logo + "SupportOps Arena" + Status Badge       │
├──────────────────┬──────────────────────────────────────┤
│  LEFT SIDEBAR    │  MAIN CONTENT AREA                  │
│  (280px fixed)   │                                      │
│                  │  ┌──────────────────────────────┐   │
│  Task Selector   │  │  TICKET VIEWER               │   │
│  [easy/med/hard] │  │  Current ticket details      │   │
│                  │  │  User context                │   │
│  Score Tracker   │  │  Network/Auth status badges  │   │
│  Easy:  [bar]    │  └──────────────────────────────┘   │
│  Med:   [bar]    │                                      │
│  Hard:  [bar]    │  ┌──────────────────────────────┐   │
│                  │  │  ACTION PANEL                │   │
│  Episode Info    │  │  15 action buttons           │   │
│  Steps: 3/10     │  │  Rationale text input        │   │
│  Reward: +0.45   │  │  [Take Action] button        │   │
│  Status: Active  │  └──────────────────────────────┘   │
│                  │                                      │
│  [▶ Run Baseline]│  ┌──────────────────────────────┐   │
│  [⟳ Reset]       │  │  SYSTEM LOGS                 │   │
│                  │  │  Scrollable log viewer       │   │
│                  │  │  Color-coded by level        │   │
└──────────────────┴──┴──────────────────────────────┘   │
│  FOOTER: Reward History Graph (Chart.js sparkline)      │
└─────────────────────────────────────────────────────────┘
```

### 15.4 Component Specifications

#### Header
```css
.header {
  background: rgba(255, 255, 255, 0.9);
  backdrop-filter: blur(20px);
  border-bottom: 1px solid var(--border-light);
  box-shadow: var(--shadow-sm);
  height: 64px;
  display: flex;
  align-items: center;
  padding: 0 24px;
  position: sticky;
  top: 0;
  z-index: 100;
}
```

Logo: SVG of a stylized checkmark shield in sky blue gradient.
"SupportOps Arena" in Sora 600 weight, deep navy text.
Status badge: pill showing "LIVE" with animated green dot.

#### Sidebar
```css
.sidebar {
  background: var(--gradient-card);
  backdrop-filter: blur(16px);
  border-right: 1px solid var(--border-light);
  box-shadow: var(--shadow-md);
  width: 280px;
  padding: 20px;
  overflow-y: auto;
}
```

Task selector: Three pill buttons (Easy / Medium / Hard). Active state has sky blue gradient fill + shadow glow.

Score bars: Custom CSS progress bars with gradient fill `linear-gradient(90deg, #87CEEB, #2196F3)`.

Episode info: Mini stat cards with white background, thin sky border, showing Steps/Reward/Status.

#### Ticket Viewer
```css
.ticket-card {
  background: white;
  border: 1px solid var(--border-light);
  border-radius: var(--radius-lg);
  box-shadow: var(--shadow-md);
  padding: 24px;
  margin-bottom: 16px;
  transition: var(--transition-med);
}

.ticket-card:hover {
  box-shadow: var(--shadow-lg);
  transform: translateY(-2px);
}
```

Status badges (Network/Auth/VPN): Pill badges with color-coded background:
- HEALTHY/OK/CONNECTED: `#D1FAE5` bg, `#065F46` text
- DEGRADED/EXPIRED: `#FEF3C7` bg, `#92400E` text  
- DOWN/FAILED/LOCKED: `#FEE2E2` bg, `#991B1B` text
- UNKNOWN/N/A: `#F1F5F9` bg, `#475569` text

#### Action Panel

15 action buttons in a 3×5 grid. Each button:
```css
.action-btn {
  background: white;
  border: 1px solid var(--border-light);
  border-radius: var(--radius-md);
  padding: 10px 14px;
  cursor: pointer;
  font-family: 'DM Mono', monospace;
  font-size: 12px;
  font-weight: 500;
  color: var(--text-secondary);
  transition: var(--transition-fast);
  text-align: left;
}

.action-btn:hover {
  background: var(--gradient-card);
  border-color: var(--sky-300);
  color: var(--sky-600);
  box-shadow: var(--shadow-sm);
  transform: translateY(-1px);
}

/* Risk level indicators */
.action-btn[data-risk="HIGH"] { border-left: 3px solid var(--danger); }
.action-btn[data-risk="MED"]  { border-left: 3px solid var(--warning); }
.action-btn[data-risk="LOW"]  { border-left: 3px solid var(--success); }
.action-btn[data-risk="NONE"] { border-left: 3px solid var(--sky-300); }
```

Rationale textarea: Clean white, sky blue focus ring, DM Mono font.

Take Action button:
```css
.btn-primary {
  background: linear-gradient(135deg, #5BB8E8, #2196F3);
  color: white;
  border: none;
  border-radius: var(--radius-md);
  padding: 12px 28px;
  font-family: 'Sora', sans-serif;
  font-weight: 600;
  font-size: 14px;
  cursor: pointer;
  box-shadow: 0 4px 15px rgba(33, 150, 243, 0.3);
  transition: var(--transition-fast);
}

.btn-primary:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 20px rgba(33, 150, 243, 0.4);
}

.btn-primary:active {
  transform: translateY(0);
}
```

#### System Logs Viewer

```css
.log-viewer {
  background: #FAFCFF;
  border: 1px solid var(--border-light);
  border-radius: var(--radius-lg);
  height: 240px;
  overflow-y: auto;
  padding: 16px;
  font-family: 'DM Mono', monospace;
  font-size: 12px;
}

/* Custom scrollbar */
.log-viewer::-webkit-scrollbar { width: 4px; }
.log-viewer::-webkit-scrollbar-track { background: transparent; }
.log-viewer::-webkit-scrollbar-thumb { background: var(--sky-300); border-radius: 2px; }

.log-entry { padding: 4px 0; border-bottom: 1px solid var(--bg-secondary); }
.log-entry.ERROR { color: var(--danger); }
.log-entry.WARN  { color: var(--warning); }
.log-entry.INFO  { color: var(--text-secondary); }
.log-entry.DEBUG { color: var(--text-muted); }
```

#### Reward History Chart

Use Chart.js (CDN). Sparkline area chart at the bottom:
- Line color: `#2196F3`
- Fill: gradient from `rgba(33, 150, 243, 0.15)` to transparent
- Grid lines: very faint sky blue
- No x-axis labels (steps only)
- White background card with shadow

#### Animations

Entry animations (use CSS keyframes):
```css
@keyframes slideIn {
  from { opacity: 0; transform: translateY(12px); }
  to   { opacity: 1; transform: translateY(0); }
}

@keyframes fadeIn {
  from { opacity: 0; }
  to   { opacity: 1; }
}

@keyframes pulse-glow {
  0%, 100% { box-shadow: 0 0 0 0 rgba(33, 150, 243, 0.3); }
  50%       { box-shadow: 0 0 0 8px rgba(33, 150, 243, 0); }
}

.ticket-card { animation: slideIn 0.3s ease forwards; }
.log-entry   { animation: fadeIn 0.2s ease forwards; }
.live-badge  { animation: pulse-glow 2s infinite; }
```

Reward feedback flash: When a positive reward comes in, flash the reward display green for 500ms. Negative → flash red.

#### Baseline Run Mode

When user clicks "▶ Run Baseline":
- Show a progress overlay (semi-transparent white with spinning sky-blue circle)
- Stream step-by-step actions as log entries in real time (SSE or polling)
- Show final scores in animated number counters that count up from 0

### 15.5 Responsive Behavior

- Desktop (>1200px): Full 3-column layout as shown
- Tablet (768–1200px): Sidebar collapses to icon mode
- Mobile (<768px): Single column, sidebar becomes bottom drawer

### 15.6 JavaScript Architecture

Use vanilla JS with a clean module pattern:

```javascript
const API_BASE = window.location.origin;

const State = {
  sessionId: null,
  taskLevel: 'easy',
  currentObs: null,
  rewardHistory: [],
  episodeDone: false,
};

const API = {
  async reset(taskLevel) { ... },
  async step(actionName, rationale) { ... },
  async getScores() { ... },
  async runBaseline() { ... },
};

const UI = {
  renderObservation(obs) { ... },
  renderLogs(logs) { ... },
  renderActionHistory(history) { ... },
  updateScores(scores) { ... },
  flashReward(value) { ... },
  updateChart(rewardHistory) { ... },
};
```

---

## 16. INFERENCE.PY (BASELINE INFERENCE SCRIPT)

This file MUST be at the root level. It is named exactly `inference.py`. The hackathon automated system will look for it here.

### 16.1 Required stdout log format

Every log line must follow this EXACT format. Any deviation causes evaluation failure:

```
[START] {"episode_id": "...", "task_level": "easy", "timestamp": "..."}
[STEP] {"step": 1, "action": "inspect_logs", "rationale": "...", "reward": 0.10, "done": false}
[STEP] {"step": 2, "action": "check_authentication", "rationale": "...", "reward": 0.20, "done": false}
...
[END] {"episode_id": "...", "task_level": "easy", "final_score": 0.72, "steps_used": 6}
```

### 16.2 `inference.py` Full Specification

```python
#!/usr/bin/env python3
"""
SupportOps Arena — Baseline Inference Script
Runs baseline agent against all 3 task levels.
Produces [START]/[STEP]/[END] logs to stdout.
Uses OpenAI-compatible API (API_BASE_URL, MODEL_NAME, OPENAI_API_KEY from env).
Must complete in < 20 minutes total.
Must run on vcpu=2, memory=8gb.
"""

import asyncio
import json
import os
import sys
import logging
from datetime import datetime

# Must use OpenAI client
from openai import AsyncOpenAI

# Read from environment
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:7860")
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# The inference script can use either:
# 1. The local env directly (no HTTP), OR
# 2. HTTP calls to the running server
# Prefer option 1 for speed (import env directly)

async def run_task(task_level: str, env, agent) -> dict:
    """Run one task level. Returns score dict."""
    obs = await env.reset(task_level=task_level)
    episode_id = obs.ticket_id  # Use as episode ID
    
    # [START] log
    print(json.dumps({"type": "[START]", "episode_id": episode_id, 
                       "task_level": task_level, 
                       "timestamp": datetime.utcnow().isoformat()}), flush=True)
    
    total_reward = 0.0
    step = 0
    done = False
    
    while not done:
        action = await agent.select_action(obs)
        obs, reward, done, info = await env.step(action)
        total_reward += reward
        step += 1
        
        # [STEP] log — REQUIRED FORMAT
        print(json.dumps({
            "type": "[STEP]",
            "step": step,
            "action": action.name,
            "rationale": action.rationale or "",
            "reward": round(reward, 4),
            "cumulative_reward": round(total_reward, 4),
            "done": done
        }), flush=True)
    
    final_score = info.get("episode_score", 0.0)
    
    # [END] log
    print(json.dumps({
        "type": "[END]",
        "episode_id": episode_id,
        "task_level": task_level,
        "final_score": round(final_score, 4),
        "steps_used": step,
        "correct": info.get("correct", False)
    }), flush=True)
    
    return {"task_level": task_level, "score": final_score}

async def main():
    from env.environment import SupportOpsArena
    from baseline.baseline_agent import BaselineAgent
    
    env = SupportOpsArena()
    agent = BaselineAgent()
    
    results = []
    for task_level in ["easy", "medium", "hard"]:
        result = await run_task(task_level, env, agent)
        results.append(result)
    
    # Final summary
    overall = sum(r["score"] * w for r, w in 
                  zip(results, [0.2, 0.3, 0.5]))
    print(json.dumps({
        "type": "[SUMMARY]",
        "easy_score": results[0]["score"],
        "medium_score": results[1]["score"],
        "hard_score": results[2]["score"],
        "overall_score": round(overall, 4)
    }), flush=True)

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 17. TESTING SUITE

### 17.1 `tests/conftest.py`

```python
import pytest
import pytest_asyncio
from env.environment import SupportOpsArena

@pytest_asyncio.fixture
async def env():
    return SupportOpsArena()

@pytest_asyncio.fixture
async def env_easy(env):
    await env.reset(task_level="easy", seed=42)
    return env

@pytest_asyncio.fixture
async def env_medium(env):
    await env.reset(task_level="medium", seed=42)
    return env

@pytest_asyncio.fixture
async def env_hard(env):
    await env.reset(task_level="hard", seed=42)
    return env
```

### 17.2 `tests/test_env.py` — Required Tests

```python
# Must all pass for submission

async def test_reset_returns_valid_observation(env):
    obs = await env.reset("easy")
    assert obs.ticket_id is not None
    assert obs.step_count == 0
    assert obs.task_level == "easy"
    assert obs.escalation_allowed == False
    assert len(obs.action_history) == 0

async def test_step_returns_valid_types(env_easy):
    from env.actions import Action, ActionType
    action = Action(name=ActionType.INSPECT_LOGS)
    result = await env_easy.step(action)
    assert 0.0 <= result.reward  # may be positive
    assert isinstance(result.done, bool)
    assert result.observation.step_count == 1

async def test_reward_in_valid_range(env_easy):
    from env.actions import Action, ActionType
    for _ in range(5):
        action = Action(name=ActionType.INSPECT_LOGS)
        result = await env_easy.step(action)
        assert -1.0 <= result.reward <= 2.0

async def test_done_on_resolve(env_easy):
    from env.actions import Action, ActionType
    result = await env_easy.step(Action(name=ActionType.RESOLVE_TICKET))
    assert result.done == True

async def test_state_contains_hidden_info(env_easy):
    state = await env_easy.state()
    assert state.hidden.root_cause is not None
    assert state.hidden.correct_remediation is not None

async def test_escalation_not_allowed_early(env_easy):
    from env.actions import Action, ActionType
    result = await env_easy.step(Action(name=ActionType.ESCALATE_TICKET))
    # Should NOT terminate (not enough diagnostic steps)
    assert result.done == False
    assert result.reward <= 0  # Penalized

async def test_all_task_levels_reset(env):
    for level in ["easy", "medium", "hard"]:
        obs = await env.reset(task_level=level)
        assert obs.task_level == level

async def test_redundant_action_penalty(env_easy):
    from env.actions import Action, ActionType
    action = Action(name=ActionType.INSPECT_LOGS)
    r1 = await env_easy.step(action)
    r2 = await env_easy.step(action)
    assert r2.reward < r1.reward  # Redundant action penalized

async def test_step_count_increments(env_easy):
    from env.actions import Action, ActionType
    for i in range(3):
        action = Action(name=ActionType.INSPECT_LOGS)
        result = await env_easy.step(action)
        assert result.observation.step_count == i + 1
```

### 17.3 `tests/test_graders.py`

```python
async def test_programmatic_grader_returns_valid_score(env_easy):
    from graders.programmatic import ProgrammaticGrader
    state = await env_easy.state()
    grader = ProgrammaticGrader()
    result = await grader.grade(state)
    assert 0.0 <= result.score <= 1.0

async def test_grade_result_has_all_fields(env_easy):
    from graders.programmatic import ProgrammaticGrader
    state = await env_easy.state()
    grader = ProgrammaticGrader()
    result = await grader.grade(state)
    assert hasattr(result, 'root_cause_identified')
    assert hasattr(result, 'action_sequence_safe')
    assert hasattr(result, 'step_budget_respected')
```

### 17.4 Run Tests Command

```bash
pytest tests/ -v --asyncio-mode=auto
```

All tests must pass before submission.

---

## 18. README.MD REQUIREMENTS

The README must contain ALL of the following sections in order:

### Required Sections

1. **Project Title & Banner** — ASCII art or SVG badge of "SupportOps Arena"
2. **Overview** — 3-4 sentences on what this environment does
3. **Motivation** — Why IT support triage? What gap does this fill?
4. **Task Descriptions**
   - Task 1 (Easy): Description, step budget, root causes, grader type
   - Task 2 (Medium): Description, step budget, challenge, grader type
   - Task 3 (Hard): Description, step budget, novel mechanics, grader type
5. **Observation Space** — Table of all observation fields with types
6. **Action Space** — Table of all 15 actions with risk/info yield
7. **Reward Function** — Full reward table (positive + negative)
8. **Setup Instructions**
   ```bash
   git clone <repo>
   cd supportops-arena
   cp .env.example .env
   # Edit .env with your API keys
   docker build -t supportops-arena .
   docker run -p 7860:7860 --env-file .env supportops-arena
   ```
9. **Running Without Docker**
   ```bash
   pip install -r requirements.txt
   uvicorn app.server:app --host 0.0.0.0 --port 7860
   ```
10. **Running Inference Script**
    ```bash
    python inference.py
    ```
11. **Baseline Scores** — Table of expected baseline agent scores per task
12. **Novel Mechanics** — Adaptive adversary, risk-weighted actions, escalation, cross-ticket memory
13. **Environment Variables** — Table of all env vars with descriptions
14. **API Reference** — Table of all HTTP endpoints
15. **Testing** — How to run tests
16. **Evaluation** — Scoring methodology

---

## 19. HUGGING FACE SPACES DEPLOYMENT

### 19.1 Space Configuration

Add `README.md` front matter at the very top:

```yaml
---
title: SupportOps Arena
emoji: 🎯
colorFrom: blue
colorTo: sky
sdk: docker
app_port: 7860
tags:
  - openenv
  - reinforcement-learning
  - it-support
  - environment
  - benchmark
pinned: false
---
```

### 19.2 Space Requirements

- Must respond to `GET /health` with 200 OK
- Must respond to `POST /reset` with valid EnvObservation JSON
- Must have the UI accessible at `/`
- Port must be 7860
- Must start within 60 seconds of container boot

### 19.3 Secrets in HF Spaces

Set these in Space Settings → Repository Secrets:
- `OPENAI_API_KEY`
- `API_BASE_URL`
- `MODEL_NAME`

---

## 20. PRE-SUBMISSION CHECKLIST COMPLIANCE

This section maps every checklist item to the implementation requirement:

| Checklist Item | Implementation Requirement | How to Verify |
|----------------|--------------------------|---------------|
| HF Space deploys | Dockerfile + port 7860 + health endpoint | `curl https://<space>/health` → 200 |
| OpenEnv spec compliance | openenv.yaml + typed models + step/reset/state | `openenv validate openenv.yaml` |
| Dockerfile builds | Single-stage Python 3.11 Dockerfile | `docker build . && docker run -p 7860:7860 .` |
| Baseline reproduces | `inference.py` at root level | `python inference.py` → stdout logs |
| 3+ tasks with graders | Easy/Medium/Hard in openenv.yaml + graders/ | `GET /tasks` → 3 tasks |
| API_BASE_URL env var | Read in server.py + inference.py | `echo $API_BASE_URL` |
| MODEL_NAME env var | Read in inference.py | `echo $MODEL_NAME` |
| HF_TOKEN env var | Read in inference.py | `echo $HF_TOKEN` |
| inference.py named correctly | File at root: `./inference.py` | `ls inference.py` |
| OpenAI client used | `from openai import AsyncOpenAI` in inference.py | Code review |
| [START]/[STEP]/[END] format | JSON with type field on every stdout line | Run inference.py and check |
| Graders return 0.0-1.0 | `score: float = Field(ge=0.0, le=1.0)` | Test suite |
| Runtime < 20 min | Baseline agent is scripted (fast) | `time python inference.py` |
| vcpu=2, memory=8gb compatible | No GPU code, async only, no heavy ML | Docker run with `--cpus=2 --memory=8g` |

---

## 21. DESIGN SYSTEM & UI RULES

### 21.1 Absolute UI Rules

1. **Background is ALWAYS light** — pure white or light sky blue gradient. No exceptions.
2. **Primary gradient** = `linear-gradient(135deg, #FFFFFF 0%, #E0F4FF 40%, #B8E4FF 100%)`
3. **All cards have glass morphism** = `backdrop-filter: blur(12px)` + semi-transparent white bg
4. **Typography hierarchy**: Sora for headings, DM Sans for body, DM Mono for code/logs
5. **Sky blue is the only accent color**. No purples, no oranges, no reds (except error states)
6. **All interactive elements have hover + active states**
7. **Reward is color-coded**: green for positive, red for negative, with CSS transitions
8. **Scrollbars are styled** — thin (4px), sky blue thumb
9. **Loading states** use sky-blue spinning rings, never gray spinners
10. **Error messages** appear in soft red pill badges, never raw text

### 21.2 Animation Principles

- Entry animations: 0.3s ease slide-up
- Hover transitions: 0.15s cubic-bezier(0.4, 0, 0.2, 1)
- Reward flash: 0.5s ease pulse (green or red glow on reward display)
- Baseline run progress: indeterminate sky-blue progress bar
- New log entries: 0.2s fade in from slightly below

---

## 22. COPILOT CLI PROMPT STRATEGY

### 22.1 How to Use This PRD with GitHub Copilot CLI

Open your terminal in the project root. Use this master prompt to kick off:

```bash
gh copilot suggest "Build the SupportOps Arena OpenEnv environment following PRD.md exactly. 
Start with: env/state.py (all Pydantic v2 models), then env/actions.py (ActionType enum + metadata), 
then env/rewards.py (RewardCalculator), then env/transitions.py (StateMachine), 
then env/scenarios.py (TicketGenerator), then env/environment.py (SupportOpsArena main class). 
Use Python 3.11+, Pydantic v2, async everywhere. Follow every rule in PRD.md."
```

### 22.2 File-by-File Prompt Sequence

Use these prompts in order. Complete each file before moving to the next:

**Phase 1: Core Models**
```
Build env/state.py for SupportOps Arena. Include: NetworkStatus, VPNStatus, AuthStatus, 
ServiceHealth, TaskLevel enums; UserContext, LogEntry, ActionRecord, EnvObservation, 
HiddenState, EnvState, StepResult Pydantic v2 models. EnvObservation must NOT contain 
HiddenState fields. All enums use str,Enum. Use field_validator and model_validator where needed.
```

**Phase 2: Actions**
```
Build env/actions.py. Create ActionType enum with exactly 15 values. Create ActionMetadata 
dataclass with risk_level, info_yield, step_cost, requires_prior_diagnostic, 
can_disrupt_other_users, is_terminal. Create ACTION_METADATA dict mapping every ActionType 
to its metadata. Create Action Pydantic v2 model with name:ActionType and rationale:str|None.
```

**Phase 3: Rewards**
```
Build env/rewards.py. Create REWARD_EVENTS dict with all 15 reward events (values from PRD). 
Create MAX_POSSIBLE_REWARD dict for easy/medium/hard. Create MIN_STEPS_FOR_OPTIMAL dict. 
Create RewardCalculator class with async calculate(events, state, action) method that returns 
(total_reward: float, breakdown: dict). Include episode score normalization with clamp to [0,1].
```

**Phase 4: Scenarios**
```
Build env/scenarios.py. Create TASK_EASY_ROOT_CAUSES list with 6 root cause dicts. 
Create TASK_MEDIUM_ROOT_CAUSES list with 6 root causes (one or two combined per episode). 
Create TicketGenerator class with async generate(task_level, adversary_weights, seed) method. 
Must return (EnvObservation, HiddenState) tuple. Easy task: 0 misleading logs. 
Medium: 1 misleading log. Hard: two tickets, 1-2 misleading logs.
```

**Phase 5: State Machine**
```
Build env/transitions.py. Create StateMachine class with async transition(state, action) method. 
Must: enforce requires_prior_diagnostic precondition, handle escalation_allowed unlock logic, 
detect redundant actions, update diagnostic_steps_taken counter, generate observation updates 
per action type (each action reveals different info), return (next_EnvState, list[reward_event_keys]).
```

**Phase 6: Main Environment**
```
Build env/environment.py. Create SupportOpsArena class. Constructor takes no required args. 
Implement: async reset(task_level, seed) -> EnvObservation, async step(action) -> StepResult, 
async state() -> EnvState. Use TicketGenerator, StateMachine, RewardCalculator, AdaptiveAdversary. 
Track episode state internally. Raise RuntimeError on step after done. Log all episodes.
```

**Phase 7: Adversary**
```
Build adversary/policy_tracker.py (PolicyTracker with sliding window of 20 episodes) and 
adversary/adversary.py (AdaptiveAdversary with get_sampling_weights and get_injection_config). 
Adversary is stateless per-episode. Reads from tracker. Adjusts ticket sampling weights based 
on agent's past action frequencies, first-action distribution, and score by root cause category.
```

**Phase 8: Graders**
```
Build graders/programmatic.py (ProgrammaticGrader), graders/llm_grader.py (LLMGrader using 
AsyncOpenAI with API_BASE_URL/MODEL_NAME from env), graders/adversarial_grader.py 
(AdversarialGrader). All return GradeResult with score: float Field(ge=0.0, le=1.0). 
LLM grader falls back to 0.5 on API failure. Add aggregate_scores function.
```

**Phase 9: Baseline Agent**
```
Build baseline/baseline_agent.py. BaselineAgent must be fully scripted (no ML). 
Decision tree: inspect_logs+check_authentication first, then branch on auth_status, 
network_status, vpn_status, service_health. Lowest-risk remediation per root cause category. 
Escalate if diagnostic_steps >= 60% of budget and done < 40% of budget remaining. 
Expected scores: easy 0.65-0.75, medium 0.40-0.55, hard 0.20-0.35.
```

**Phase 10: FastAPI Server**
```
Build app/server.py. Mount static files at /static. Serve index.html at /. Endpoints: 
GET /health, POST /reset, POST /step, GET /state, GET /scores, POST /grade, GET /openenv.yaml, 
GET /tasks, POST /baseline/run. Session management with X-Session-ID header (UUID), 
dict storage with 30min TTL. CORS middleware with allow_origins=["*"].
```

**Phase 11: Frontend**
```
Build app/static/index.html. Single file, vanilla HTML/CSS/JS. Design: pure white + light sky 
blue gradient (#FFFFFF → #E0F4FF → #B8E4FF). Fonts: Sora (headings), DM Sans (body), DM Mono (mono). 
Layout: sticky header, 280px sidebar (task selector + score bars + episode info), main area 
(ticket viewer + action panel + log viewer), bottom chart. 15 action buttons in 3x5 grid with 
risk-level color coding. Chart.js sparkline for reward history. Reward flash animation on step. 
Baseline run button with progress overlay.
```

**Phase 12: Inference Script**
```
Build inference.py at root level. Runs baseline agent on all 3 tasks sequentially. 
Uses AsyncOpenAI client (API_BASE_URL, MODEL_NAME, OPENAI_API_KEY from env). 
Prints [START]/[STEP]/[END] logs as JSON to stdout. Final [SUMMARY] with all 3 scores 
and overall weighted score (easy×0.2 + medium×0.3 + hard×0.5).
```

**Phase 13: Tests**
```
Build tests/conftest.py, tests/test_env.py, tests/test_graders.py, tests/test_transitions.py. 
All tests async with pytest-asyncio. test_env.py: test reset, step, reward range, terminal 
conditions, escalation unlock, redundant action penalty. test_graders.py: grader returns 
valid score, all fields present. pytest.ini with asyncio_mode=auto.
```

**Phase 14: Config Files**
```
Build: openenv.yaml (exact format from PRD section 13), Dockerfile (Python 3.11-slim, 
port 7860, non-root user, healthcheck), requirements.txt (exact versions from PRD), 
.env.example (all env vars with placeholder values), pytest.ini (asyncio_mode=auto).
```

**Phase 15: README**
```
Build README.md. Must include all 16 sections from PRD section 18. Include HF Spaces 
front matter at top. Include setup instructions, baseline scores table, action space table, 
observation space table, reward table, env vars table, API endpoints table.
```

### 22.3 Verification Commands After Each Phase

```bash
# After Phase 1-6 (env core):
python -c "from env.environment import SupportOpsArena; import asyncio; \
  env = SupportOpsArena(); \
  obs = asyncio.run(env.reset('easy')); \
  print('Reset OK:', obs.ticket_id)"

# After Phase 7 (adversary):
python -c "from adversary.adversary import AdaptiveAdversary; \
  a = AdaptiveAdversary(); \
  w = a.get_sampling_weights('easy'); print('Adversary OK:', w)"

# After Phase 8 (graders):
python -c "from graders.programmatic import ProgrammaticGrader; print('Graders OK')"

# After Phase 10 (server):
uvicorn app.server:app --host 0.0.0.0 --port 7860 &
sleep 2
curl http://localhost:7860/health

# After Phase 12 (inference):
python inference.py 2>&1 | head -20

# After Phase 13 (tests):
pytest tests/ -v --asyncio-mode=auto

# Final validation:
docker build -t supportops-arena .
docker run -p 7860:7860 supportops-arena &
sleep 10
curl http://localhost:7860/health
curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" \
  -d '{"task_level": "easy"}' | python -m json.tool
```

---

## 23. ERROR HANDLING & EDGE CASES

### 23.1 Environment Error Handling

```python
# In environment.py
class SupportOpsArena:
    async def step(self, action: Action) -> StepResult:
        if self._state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        if self._state.done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")
        
        # Invalid action type
        if action.name not in [a.value for a in ActionType]:
            return StepResult(
                observation=self._state.observation,
                reward=-0.05,  # Small penalty for invalid action
                done=False,
                info={"error": f"Invalid action: {action.name}"}
            )
```

### 23.2 API Error Handling

```python
# In server.py
@app.exception_handler(RuntimeError)
async def runtime_error_handler(request, exc):
    return JSONResponse(status_code=400, content={"error": str(exc)})

@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(status_code=422, content={"error": str(exc)})
```

### 23.3 LLM Grader Fallback

```python
async def grade(self, state: EnvState) -> LLMGradeResult:
    try:
        # ... API call ...
    except Exception as e:
        logger.warning(f"LLM grader API call failed: {e}. Using fallback score.")
        return LLMGradeResult(
            score=0.5,
            diagnostic_coherence=0.5,
            evidence_sufficiency=0.5,
            root_cause_narration=0.5,
            reasoning="LLM grader unavailable — fallback score applied"
        )
```

### 23.4 Session Cleanup

```python
# In server.py
import asyncio
from datetime import datetime, timedelta

async def cleanup_stale_sessions():
    """Run every 5 minutes. Remove sessions older than 30 minutes."""
    while True:
        await asyncio.sleep(300)
        cutoff = datetime.utcnow() - timedelta(seconds=SESSION_TTL_SECONDS)
        stale = [sid for sid, data in sessions.items() 
                 if data["created_at"] < cutoff]
        for sid in stale:
            del sessions[sid]
        if stale:
            logger.info(f"Cleaned up {len(stale)} stale sessions")

@app.on_event("startup")
async def startup():
    asyncio.create_task(cleanup_stale_sessions())
```

---

## 24. ABSOLUTE RULES & PROHIBITIONS

### 24.1 Rules That Cannot Be Broken

1. **`inference.py` must be at root level** — not in any subdirectory
2. **[START]/[STEP]/[END] log format is EXACT** — JSON with `type` field, no extra text
3. **All graders must return scores in [0.0, 1.0]** — validated by Pydantic Field constraint
4. **Port is always 7860** — hardcoded in Dockerfile CMD
5. **`openenv.yaml` must be at root level** — not inside any subdirectory
6. **Pydantic v2 only** — no v1 syntax anywhere
7. **No synchronous I/O in async paths** — use `asyncio`, `httpx`, `aiofiles`
8. **Session ID header is `X-Session-ID`** — exact casing
9. **Graders that always return the same score = DISQUALIFICATION** — reward must vary
10. **No API keys in source code** — always from environment variables

### 24.2 Quality Gates Before Submission

Run this full checklist manually:

```bash
# 1. Tests pass
pytest tests/ -v --asyncio-mode=auto

# 2. Docker builds and runs
docker build -t supportops-arena . && \
docker run -d -p 7860:7860 --name test-arena supportops-arena && \
sleep 10 && \
curl -f http://localhost:7860/health && \
docker stop test-arena && docker rm test-arena

# 3. Inference script runs and produces correct format
python inference.py | grep -E "^\{" | python -c "
import sys, json
for line in sys.stdin:
    data = json.loads(line)
    assert 'type' in data, f'Missing type field: {data}'
    assert data['type'] in ['[START]', '[STEP]', '[END]', '[SUMMARY]']
print('Log format: OK')
"

# 4. openenv.yaml is valid YAML
python -c "import yaml; yaml.safe_load(open('openenv.yaml')); print('YAML: OK')"

# 5. All scores in range
python -c "
import asyncio
from env.environment import SupportOpsArena
from baseline.baseline_agent import BaselineAgent
async def check():
    env = SupportOpsArena()
    agent = BaselineAgent()
    for level in ['easy', 'medium', 'hard']:
        obs = await env.reset(level)
        done = False
        while not done:
            action = await agent.select_action(obs)
            obs, reward, done, info = await env.step(action)
        score = info.get('episode_score', 0)
        assert 0.0 <= score <= 1.0, f'Score out of range for {level}: {score}'
        print(f'{level}: {score:.3f} ✓')
asyncio.run(check())
"
```

---

## APPENDIX A: Expected File Sizes

| File | Expected Lines | Note |
|------|---------------|------|
| env/state.py | 120–160 | All models |
| env/actions.py | 60–80 | Enum + metadata |
| env/rewards.py | 80–100 | Calculator class |
| env/transitions.py | 150–200 | Full state machine |
| env/scenarios.py | 200–280 | All root causes |
| env/environment.py | 180–220 | Main class |
| adversary/adversary.py | 100–140 | Adaptive adversary |
| graders/programmatic.py | 100–130 | Rule-based grader |
| graders/llm_grader.py | 80–100 | LLM judge |
| app/server.py | 200–250 | FastAPI endpoints |
| app/static/index.html | 600–900 | Full dashboard UI |
| inference.py | 80–120 | Inference script |
| tests/test_env.py | 100–140 | All env tests |

---

## APPENDIX B: Baseline Score Targets

| Task | Min Score | Max Score | Target |
|------|-----------|-----------|--------|
| Easy | 0.60 | 0.80 | 0.70 |
| Medium | 0.35 | 0.55 | 0.45 |
| Hard | 0.15 | 0.35 | 0.25 |
| **Overall** | **0.30** | **0.50** | **0.40** |

If baseline scores are outside these ranges, the reward function or baseline agent has a bug.

---

*PRD Version 1.0 · SupportOps Arena · OpenEnv Hackathon 2026*
*Build with Claude Sonnet 4.5 via GitHub Copilot CLI*
