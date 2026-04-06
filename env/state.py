"""
Pydantic v2 models for SupportOps Arena environment state.
All models use Pydantic v2 syntax exclusively.
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from enum import Enum
from datetime import datetime
import uuid
from typing import Any


# ─── Enums ───────────────────────────────────────────────────

class NetworkStatus(str, Enum):
    """Network connectivity status."""
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    DOWN = "DOWN"
    UNKNOWN = "UNKNOWN"


class VPNStatus(str, Enum):
    """VPN connection status."""
    CONNECTED = "CONNECTED"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"
    NA = "N/A"


class AuthStatus(str, Enum):
    """Authentication system status."""
    OK = "OK"
    LOCKED = "LOCKED"
    EXPIRED = "EXPIRED"
    MFA_FAIL = "MFA_FAIL"


class ServiceHealth(str, Enum):
    """Service health status."""
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    DOWN = "DOWN"
    UNKNOWN = "UNKNOWN"


class TaskLevel(str, Enum):
    """Task difficulty level."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


# ─── Sub-models ────────────────────────────────────────────── 

class UserContext(BaseModel):
    """User context information."""
    department: str
    role: str
    device_type: str
    os_version: str
    location: str  # "onsite" | "remote"

    model_config = {"frozen": False}


class LogEntry(BaseModel):
    """System log entry."""
    timestamp: str
    level: str  # "INFO" | "WARN" | "ERROR" | "DEBUG"
    service: str
    message: str
    is_misleading: bool = False  # Hidden from agent, used by grader

    model_config = {"frozen": False}


class ActionRecord(BaseModel):
    """Record of a past action in action_history."""
    step: int
    action_name: str
    rationale: str | None = None
    reward_received: float
    timestamp: str

    model_config = {"frozen": False}


# ─── Core Models ───────────────────────────────────────────── 

class EnvObservation(BaseModel):
    """
    What the AGENT sees. No hidden info.
    This is the observation returned to the agent at each step.
    """
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

    model_config = {"frozen": False}


class HiddenState(BaseModel):
    """
    Internal ground truth. NEVER sent to agent.
    Only accessible via state() method for graders.
    """
    root_cause: str
    root_cause_category: str  # "network" | "auth" | "dns" | "vpn" | "sso" | "hardware"
    correct_remediation: str  # Which ActionType resolves this
    correct_remediation_alts: list[str] = Field(default_factory=list)  # Acceptable alternates
    misleading_log_index: int | None = None  # Which log entry is the red herring
    severity: str  # "low" | "medium" | "high" | "critical"
    affected_users_count: int  # Side-effect blast radius
    ticket_variant_id: str  # For adversary tracking

    model_config = {"frozen": False}


class EnvState(BaseModel):
    """
    Full internal state. Graders only.
    Contains both observable and hidden state.
    """
    episode_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_level: TaskLevel
    observation: EnvObservation
    hidden: HiddenState
    cumulative_reward: float = 0.0
    done: bool = False
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    action_log: list[ActionRecord] = Field(default_factory=list)
    diagnostic_steps_taken: int = 0  # Tracks unlock threshold

    model_config = {"frozen": False}


class StepResult(BaseModel):
    """
    Result returned from environment step() method.
    Standard RL tuple: (observation, reward, done, info).
    """
    observation: EnvObservation
    reward: float
    done: bool
    info: dict[str, Any]

    model_config = {"frozen": False}
