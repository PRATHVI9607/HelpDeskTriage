"""
State machine and transition logic for SupportOps Arena.
Handles action preconditions, observation updates, and reward event generation.
"""

import copy
from datetime import datetime
from typing import Any
from env.state import (
    EnvState, EnvObservation, LogEntry, TaskLevel,
    NetworkStatus, VPNStatus, AuthStatus, ServiceHealth
)
from env.actions import ActionType, ACTION_METADATA, get_diagnostic_actions


# ─── Escalation Unlock Thresholds ────────────────────────────

ESCALATION_UNLOCK_AFTER = {
    TaskLevel.EASY: 3,
    TaskLevel.MEDIUM: 4,
    TaskLevel.HARD: 5,
}


class StateMachine:
    """
    Manages state transitions and observation updates.
    Enforces action preconditions and generates reward events.
    """
    
    def __init__(self):
        self.diagnostic_actions = set(get_diagnostic_actions())
    
    async def transition(
        self,
        state: EnvState,
        action: ActionType
    ) -> tuple[EnvState, list[str]]:
        """
        Apply action to state and return next state with reward events.
        
        Args:
            state: Current environment state
            action: Action to apply
        
        Returns:
            Tuple of (next_state, reward_events_list)
        """
        # Create a copy of state to modify
        next_state = copy.deepcopy(state)
        reward_events = []
        
        # Get action metadata
        action_meta = ACTION_METADATA.get(action)
        if not action_meta:
            reward_events.append("HARMFUL_ACTION")
            return next_state, reward_events
        
        # Check preconditions and add penalties
        reward_events.extend(self._check_preconditions(next_state, action, action_meta))
        
        # Update observation based on action type
        self._update_observation(next_state, action)
        
        # Handle diagnostic actions
        if action in self.diagnostic_actions:
            reward_events.append("DIAGNOSTIC_ACTION")
            next_state.diagnostic_steps_taken += 1
            
            # Check if diagnostic targets root cause category
            if self._is_targeted_diagnostic(next_state, action):
                reward_events.append("CORRECT_TARGETED_DIAGNOSTIC")
            
            # Check if escalation should be unlocked
            unlock_threshold = ESCALATION_UNLOCK_AFTER[next_state.task_level]
            if next_state.diagnostic_steps_taken >= unlock_threshold:
                next_state.observation.escalation_allowed = True
        
        return next_state, reward_events
    
    def _check_preconditions(
        self,
        state: EnvState,
        action: ActionType,
        action_meta: Any
    ) -> list[str]:
        """Check action preconditions and return penalty events."""
        events = []
        
        # Check if action requires prior diagnostic but none taken
        if action_meta.requires_prior_diagnostic and state.diagnostic_steps_taken == 0:
            events.append("RISKY_WITHOUT_DIAGNOSTIC")
        
        # Check if escalating before allowed
        if action == ActionType.ESCALATE_TICKET and not state.observation.escalation_allowed:
            events.append("PREMATURE_ESCALATION")
        
        # Check for redundant actions
        past_actions = [record.action_name for record in state.observation.action_history]
        if action.value in past_actions:
            events.append("REDUNDANT_ACTION")
        
        return events
    
    def _update_observation(self, state: EnvState, action: ActionType) -> None:
        """Update observation based on action taken."""
        obs = state.observation
        hidden = state.hidden
        
        if action == ActionType.INSPECT_NETWORK:
            self._handle_inspect_network(obs, hidden, state.task_level)
        
        elif action == ActionType.INSPECT_LOGS:
            self._handle_inspect_logs(obs, hidden, state.task_level)
        
        elif action == ActionType.CHECK_AUTHENTICATION:
            self._handle_check_authentication(obs, hidden)
        
        elif action == ActionType.CHECK_PERMISSIONS:
            self._handle_check_permissions(obs, hidden)
        
        elif action == ActionType.QUERY_DEVICE_STATUS:
            self._handle_query_device_status(obs, hidden)
        
        elif action == ActionType.SEARCH_INTERNAL_KB:
            self._handle_search_kb(obs, hidden)
        
        elif action == ActionType.CONTACT_USER:
            self._handle_contact_user(obs, hidden)
        
        elif action == ActionType.RUN_DIAGNOSTIC:
            self._handle_run_diagnostic(obs, hidden)
        
        elif action == ActionType.FLUSH_DNS:
            self._handle_flush_dns(obs, hidden)
        
        elif action == ActionType.RECONFIGURE_CLIENT:
            self._handle_reconfigure_client(obs, hidden)
        
        elif action == ActionType.RESTART_SERVICE:
            self._handle_restart_service(obs, hidden)
        
        elif action == ActionType.RESET_CREDENTIALS:
            self._handle_reset_credentials(obs, hidden)
    
    def _handle_inspect_network(
        self,
        obs: EnvObservation,
        hidden: Any,
        task_level: TaskLevel
    ) -> None:
        """Update network status information."""
        category = hidden.root_cause_category
        
        if category == "network":
            obs.network_status = NetworkStatus.DEGRADED
        elif category == "dns":
            obs.network_status = NetworkStatus.HEALTHY
        else:
            obs.network_status = NetworkStatus.HEALTHY
        
        # Update VPN status if relevant
        if category == "vpn":
            obs.vpn_status = VPNStatus.FAILED
        elif "vpn" in obs.service_health:
            obs.vpn_status = VPNStatus.NA
    
    def _handle_inspect_logs(
        self,
        obs: EnvObservation,
        hidden: Any,
        task_level: TaskLevel
    ) -> None:
        """Add more log entries (may include misleading ones)."""
        # Logs were already populated during reset
        # This action reveals more details in existing logs
        now = datetime.utcnow()
        category = hidden.root_cause_category
        
        new_log = LogEntry(
            timestamp=now.isoformat(),
            level="INFO",
            service="LogAnalyzer",
            message=f"Analyzed logs: {len(obs.system_logs)} entries found related to {category} subsystem",
            is_misleading=False
        )
        obs.system_logs.append(new_log)
    
    def _handle_check_authentication(
        self,
        obs: EnvObservation,
        hidden: Any
    ) -> None:
        """Update authentication status."""
        category = hidden.root_cause_category
        
        if category == "auth":
            if "expired" in hidden.root_cause.lower():
                obs.auth_status = AuthStatus.EXPIRED
            elif "mfa" in hidden.root_cause.lower():
                obs.auth_status = AuthStatus.MFA_FAIL
            elif "locked" in hidden.root_cause.lower():
                obs.auth_status = AuthStatus.LOCKED
            else:
                obs.auth_status = AuthStatus.EXPIRED
        else:
            obs.auth_status = AuthStatus.OK
    
    def _handle_check_permissions(
        self,
        obs: EnvObservation,
        hidden: Any
    ) -> None:
        """Check permissions and update service health."""
        now = datetime.utcnow()
        log = LogEntry(
            timestamp=now.isoformat(),
            level="INFO",
            service="PermissionChecker",
            message="User permissions validated: Standard access level confirmed",
            is_misleading=False
        )
        obs.system_logs.append(log)
    
    def _handle_query_device_status(
        self,
        obs: EnvObservation,
        hidden: Any
    ) -> None:
        """Query device status and reveal hardware info."""
        category = hidden.root_cause_category
        now = datetime.utcnow()
        
        if category == "hardware":
            log = LogEntry(
                timestamp=now.isoformat(),
                level="WARN",
                service="DeviceManager",
                message=f"Device check: {hidden.root_cause}",
                is_misleading=False
            )
        else:
            log = LogEntry(
                timestamp=now.isoformat(),
                level="INFO",
                service="DeviceManager",
                message="Device status: All hardware components operational",
                is_misleading=False
            )
        obs.system_logs.append(log)
    
    def _handle_search_kb(
        self,
        obs: EnvObservation,
        hidden: Any
    ) -> None:
        """Search internal knowledge base."""
        category = hidden.root_cause_category
        now = datetime.utcnow()
        
        # KB provides relevant but not complete information
        kb_results = {
            "network": "KB Article 2034: DHCP issues can cause intermittent connectivity",
            "auth": "KB Article 1523: Password expiration policies and reset procedures",
            "dns": "KB Article 3012: DNS cache corruption troubleshooting guide",
            "vpn": "KB Article 4056: VPN client compatibility and configuration",
            "sso": "KB Article 5078: SSO token service architecture and dependencies",
            "hardware": "KB Article 1209: Hardware configuration and driver updates",
        }
        
        message = kb_results.get(category, "KB Article 9999: General troubleshooting steps")
        
        log = LogEntry(
            timestamp=now.isoformat(),
            level="INFO",
            service="KnowledgeBase",
            message=f"Search result: {message}",
            is_misleading=False
        )
        obs.system_logs.append(log)
    
    def _handle_contact_user(
        self,
        obs: EnvObservation,
        hidden: Any
    ) -> None:
        """Contact user for more information."""
        now = datetime.utcnow()
        
        # User provides additional context
        user_responses = [
            "User reports: Issue started after latest system update",
            "User reports: Problem occurs only when connecting from home",
            "User reports: Other colleagues experiencing similar issues",
            "User reports: Was working fine until this morning",
        ]
        
        import random
        message = random.choice(user_responses)
        
        log = LogEntry(
            timestamp=now.isoformat(),
            level="INFO",
            service="UserContact",
            message=message,
            is_misleading=False
        )
        obs.system_logs.append(log)
    
    def _handle_run_diagnostic(
        self,
        obs: EnvObservation,
        hidden: Any
    ) -> None:
        """Run targeted diagnostic script."""
        category = hidden.root_cause_category
        now = datetime.utcnow()
        
        # Diagnostic provides accurate targeted information
        log = LogEntry(
            timestamp=now.isoformat(),
            level="WARN",
            service="DiagnosticScript",
            message=f"Diagnostic result: Detected issue in {category} subsystem - {hidden.root_cause}",
            is_misleading=False
        )
        obs.system_logs.append(log)
    
    def _handle_flush_dns(
        self,
        obs: EnvObservation,
        hidden: Any
    ) -> None:
        """Flush DNS cache."""
        now = datetime.utcnow()
        
        # DNS flush may temporarily show degraded status
        if hidden.root_cause_category == "dns":
            obs.network_status = NetworkStatus.HEALTHY
        
        log = LogEntry(
            timestamp=now.isoformat(),
            level="INFO",
            service="DNSService",
            message="DNS cache flushed successfully",
            is_misleading=False
        )
        obs.system_logs.append(log)
    
    def _handle_reconfigure_client(
        self,
        obs: EnvObservation,
        hidden: Any
    ) -> None:
        """Reconfigure client settings."""
        now = datetime.utcnow()
        
        log = LogEntry(
            timestamp=now.isoformat(),
            level="INFO",
            service="ConfigManager",
            message="Client reconfiguration applied",
            is_misleading=False
        )
        obs.system_logs.append(log)
    
    def _handle_restart_service(
        self,
        obs: EnvObservation,
        hidden: Any
    ) -> None:
        """Restart service (high risk action)."""
        now = datetime.utcnow()
        
        # Update service health
        for service_name in obs.service_health:
            if obs.service_health[service_name] == ServiceHealth.DEGRADED:
                obs.service_health[service_name] = ServiceHealth.HEALTHY
        
        log = LogEntry(
            timestamp=now.isoformat(),
            level="WARN",
            service="ServiceManager",
            message=f"Service restart completed. {hidden.affected_users_count} users may have been affected.",
            is_misleading=False
        )
        obs.system_logs.append(log)
    
    def _handle_reset_credentials(
        self,
        obs: EnvObservation,
        hidden: Any
    ) -> None:
        """Reset user credentials."""
        obs.auth_status = AuthStatus.OK
        now = datetime.utcnow()
        
        log = LogEntry(
            timestamp=now.isoformat(),
            level="INFO",
            service="AuthService",
            message="User credentials reset successfully",
            is_misleading=False
        )
        obs.system_logs.append(log)
    
    def _is_targeted_diagnostic(self, state: EnvState, action: ActionType) -> bool:
        """Check if diagnostic action targets the root cause category."""
        category = state.hidden.root_cause_category
        
        # Map actions to categories they target
        action_category_map = {
            ActionType.INSPECT_NETWORK: ["network", "dns", "vpn"],
            ActionType.INSPECT_LOGS: ["auth", "network", "dns", "vpn", "sso", "hardware"],
            ActionType.CHECK_AUTHENTICATION: ["auth", "sso"],
            ActionType.CHECK_PERMISSIONS: ["auth"],
            ActionType.QUERY_DEVICE_STATUS: ["hardware", "network"],
            ActionType.SEARCH_INTERNAL_KB: ["auth", "network", "dns", "vpn", "sso", "hardware"],
            ActionType.CONTACT_USER: ["auth", "network", "dns", "vpn", "hardware"],
            ActionType.RUN_DIAGNOSTIC: ["network", "dns", "vpn", "sso", "hardware"],
        }
        
        targeted_categories = action_category_map.get(action, [])
        return category in targeted_categories
