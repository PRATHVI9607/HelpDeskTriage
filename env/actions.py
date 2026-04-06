"""
Action definitions and metadata for SupportOps Arena.
Defines all 15 actions available to agents with their characteristics.
"""

from enum import Enum
from dataclasses import dataclass


class ActionType(str, Enum):
    """All available actions in the environment."""
    # Diagnostic Actions (Information Gathering)
    INSPECT_NETWORK = "inspect_network"
    INSPECT_LOGS = "inspect_logs"
    CHECK_AUTHENTICATION = "check_authentication"
    CHECK_PERMISSIONS = "check_permissions"
    QUERY_DEVICE_STATUS = "query_device_status"
    SEARCH_INTERNAL_KB = "search_internal_kb"
    CONTACT_USER = "contact_user_for_info"
    RUN_DIAGNOSTIC = "run_diagnostic_script"
    
    # Remediation Actions (Risk-bearing)
    FLUSH_DNS = "flush_dns"
    RECONFIGURE_CLIENT = "reconfigure_client"
    RESTART_SERVICE = "restart_service"
    RESET_CREDENTIALS = "reset_credentials"
    
    # Terminal Actions
    ESCALATE_TICKET = "escalate_ticket"
    RESOLVE_TICKET = "resolve_ticket"
    CLOSE_WITHOUT_FIX = "close_without_fix"


@dataclass
class ActionMetadata:
    """
    Metadata for each action type.
    
    Attributes:
        risk_level: Impact risk - "LOW" | "MED" | "HIGH" | "NONE"
        info_yield: Information value - "HIGH" | "MED" | "LOW" | "NONE"
        step_cost: Step count cost (always 1 currently)
        requires_prior_diagnostic: Must have diagnostic evidence first
        can_disrupt_other_users: May affect other users
        is_terminal: Ends the episode
    """
    risk_level: str
    info_yield: str
    step_cost: int
    requires_prior_diagnostic: bool
    can_disrupt_other_users: bool
    is_terminal: bool


# Action metadata lookup table
ACTION_METADATA: dict[ActionType, ActionMetadata] = {
    # Diagnostic actions - low risk, high information
    ActionType.INSPECT_NETWORK: ActionMetadata(
        risk_level="LOW",
        info_yield="HIGH",
        step_cost=1,
        requires_prior_diagnostic=False,
        can_disrupt_other_users=False,
        is_terminal=False
    ),
    ActionType.INSPECT_LOGS: ActionMetadata(
        risk_level="LOW",
        info_yield="HIGH",
        step_cost=1,
        requires_prior_diagnostic=False,
        can_disrupt_other_users=False,
        is_terminal=False
    ),
    ActionType.CHECK_AUTHENTICATION: ActionMetadata(
        risk_level="LOW",
        info_yield="HIGH",
        step_cost=1,
        requires_prior_diagnostic=False,
        can_disrupt_other_users=False,
        is_terminal=False
    ),
    ActionType.CHECK_PERMISSIONS: ActionMetadata(
        risk_level="LOW",
        info_yield="MED",
        step_cost=1,
        requires_prior_diagnostic=False,
        can_disrupt_other_users=False,
        is_terminal=False
    ),
    ActionType.QUERY_DEVICE_STATUS: ActionMetadata(
        risk_level="LOW",
        info_yield="MED",
        step_cost=1,
        requires_prior_diagnostic=False,
        can_disrupt_other_users=False,
        is_terminal=False
    ),
    ActionType.SEARCH_INTERNAL_KB: ActionMetadata(
        risk_level="LOW",
        info_yield="MED",
        step_cost=1,
        requires_prior_diagnostic=False,
        can_disrupt_other_users=False,
        is_terminal=False
    ),
    ActionType.CONTACT_USER: ActionMetadata(
        risk_level="LOW",
        info_yield="MED",
        step_cost=1,
        requires_prior_diagnostic=False,
        can_disrupt_other_users=False,
        is_terminal=False
    ),
    ActionType.RUN_DIAGNOSTIC: ActionMetadata(
        risk_level="MED",
        info_yield="HIGH",
        step_cost=1,
        requires_prior_diagnostic=True,
        can_disrupt_other_users=False,
        is_terminal=False
    ),
    
    # Remediation actions - medium/high risk
    ActionType.FLUSH_DNS: ActionMetadata(
        risk_level="MED",
        info_yield="LOW",
        step_cost=1,
        requires_prior_diagnostic=False,
        can_disrupt_other_users=False,
        is_terminal=False
    ),
    ActionType.RECONFIGURE_CLIENT: ActionMetadata(
        risk_level="MED",
        info_yield="LOW",
        step_cost=1,
        requires_prior_diagnostic=False,
        can_disrupt_other_users=False,
        is_terminal=False
    ),
    ActionType.RESTART_SERVICE: ActionMetadata(
        risk_level="HIGH",
        info_yield="LOW",
        step_cost=1,
        requires_prior_diagnostic=True,
        can_disrupt_other_users=True,
        is_terminal=False
    ),
    ActionType.RESET_CREDENTIALS: ActionMetadata(
        risk_level="HIGH",
        info_yield="LOW",
        step_cost=1,
        requires_prior_diagnostic=True,
        can_disrupt_other_users=False,
        is_terminal=False
    ),
    
    # Terminal actions
    ActionType.ESCALATE_TICKET: ActionMetadata(
        risk_level="NONE",
        info_yield="NONE",
        step_cost=1,
        requires_prior_diagnostic=False,
        can_disrupt_other_users=False,
        is_terminal=True
    ),
    ActionType.RESOLVE_TICKET: ActionMetadata(
        risk_level="NONE",
        info_yield="NONE",
        step_cost=1,
        requires_prior_diagnostic=False,
        can_disrupt_other_users=False,
        is_terminal=True
    ),
    ActionType.CLOSE_WITHOUT_FIX: ActionMetadata(
        risk_level="NONE",
        info_yield="NONE",
        step_cost=1,
        requires_prior_diagnostic=False,
        can_disrupt_other_users=False,
        is_terminal=True
    ),
}


def get_diagnostic_actions() -> list[ActionType]:
    """Return all diagnostic (information gathering) actions."""
    return [
        ActionType.INSPECT_NETWORK,
        ActionType.INSPECT_LOGS,
        ActionType.CHECK_AUTHENTICATION,
        ActionType.CHECK_PERMISSIONS,
        ActionType.QUERY_DEVICE_STATUS,
        ActionType.SEARCH_INTERNAL_KB,
        ActionType.CONTACT_USER,
        ActionType.RUN_DIAGNOSTIC,
    ]


def get_remediation_actions() -> list[ActionType]:
    """Return all remediation (fix) actions."""
    return [
        ActionType.FLUSH_DNS,
        ActionType.RECONFIGURE_CLIENT,
        ActionType.RESTART_SERVICE,
        ActionType.RESET_CREDENTIALS,
    ]


def get_terminal_actions() -> list[ActionType]:
    """Return all terminal actions."""
    return [
        ActionType.ESCALATE_TICKET,
        ActionType.RESOLVE_TICKET,
        ActionType.CLOSE_WITHOUT_FIX,
    ]
