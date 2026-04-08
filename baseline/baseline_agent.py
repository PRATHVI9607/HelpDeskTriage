"""
Baseline heuristic agent for SupportOps Arena.
Rule-based agent using decision tree logic (no ML required).
"""

import logging
from typing import Any

from env.state import EnvObservation, NetworkStatus, VPNStatus, AuthStatus, ServiceHealth
from env.actions import ActionType, get_diagnostic_actions
from env.environment import SupportOpsArena

logger = logging.getLogger(__name__)


class BaselineAgent:
    """
    Scripted heuristic baseline agent.
    
    Uses a simple decision tree strategy:
    1. Gather initial evidence (logs + auth)
    2. Apply rule-based diagnostics based on observations
    3. Choose lowest-risk remediation
    4. Escalate if uncertain after gathering evidence
    
    Expected performance:
    - Easy: 0.65-0.75
    - Medium: 0.40-0.55
    - Hard: 0.20-0.35
    """
    
    def __init__(self, seed: int | None = None):
        """
        Initialize baseline agent.
        
        Args:
            seed: Random seed for reproducibility (currently unused as agent is deterministic)
        """
        self.seed = seed
        self._action_history: list[str] = []
        self._diagnostic_action_names = {
            action.value for action in get_diagnostic_actions()
        }
    
    async def select_action(self, observation: EnvObservation) -> ActionType:
        """
        Select next action based on current observation.
        
        Args:
            observation: Current environment observation
        
        Returns:
            ActionType to take
        """
        step = observation.step_count
        max_steps = observation.steps_remaining + step
        
        # Phase 1: Initial diagnostic (steps 1-2)
        if step == 0:
            return ActionType.INSPECT_LOGS
        
        if step == 1:
            return ActionType.CHECK_AUTHENTICATION
        
        # Phase 2: Targeted diagnostics (steps 2-4)
        if step < 4:
            # Check auth status
            if observation.auth_status in [
                AuthStatus.LOCKED,
                AuthStatus.EXPIRED,
                AuthStatus.MFA_FAIL
            ]:
                if "check_permissions" not in self._action_history:
                    self._action_history.append("check_permissions")
                    return ActionType.CHECK_PERMISSIONS
            
            # Check network status
            if observation.network_status in [NetworkStatus.DEGRADED, NetworkStatus.DOWN]:
                if "inspect_network" not in self._action_history:
                    self._action_history.append("inspect_network")
                    return ActionType.INSPECT_NETWORK
            
            # Check VPN status
            if observation.vpn_status in [VPNStatus.FAILED, VPNStatus.TIMEOUT]:
                if "query_device_status" not in self._action_history:
                    self._action_history.append("query_device_status")
                    return ActionType.QUERY_DEVICE_STATUS
            
            # Check for service issues
            has_service_issues = any(
                status in [ServiceHealth.DEGRADED, ServiceHealth.DOWN]
                for status in observation.service_health.values()
            )
            
            if has_service_issues:
                if "run_diagnostic_script" not in self._action_history:
                    self._action_history.append("run_diagnostic_script")
                    return ActionType.RUN_DIAGNOSTIC
            
            # Default: query device
            if "query_device_status" not in self._action_history:
                self._action_history.append("query_device_status")
                return ActionType.QUERY_DEVICE_STATUS
        
        # Phase 3: Apply remediation (steps 4+)
        # Determine remediation based on observations
        remediation = self._determine_remediation(observation)
        
        if remediation:
            if remediation.value not in self._action_history:
                self._action_history.append(remediation.value)
                return remediation
        
        # Phase 4: Resolve or escalate
        # Check if we have enough evidence
        diagnostic_count = self._count_diagnostic_actions()
        evidence_threshold = max_steps * 0.6
        
        # If we've gathered enough evidence and applied a fix, resolve
        if diagnostic_count >= 4 and len(self._action_history) >= 5:
            # Check if escalation is available and we're uncertain
            if observation.escalation_allowed and diagnostic_count < evidence_threshold * 0.8:
                self._action_history.append("escalate_ticket")
                return ActionType.ESCALATE_TICKET
            
            # Otherwise resolve
            self._action_history.append("resolve_ticket")
            return ActionType.RESOLVE_TICKET
        
        # Need more diagnostics or stuck - escalate if possible
        if observation.escalation_allowed and diagnostic_count >= evidence_threshold:
            self._action_history.append("escalate_ticket")
            return ActionType.ESCALATE_TICKET
        
        # Gather more evidence
        if "search_internal_kb" not in self._action_history:
            self._action_history.append("search_internal_kb")
            return ActionType.SEARCH_INTERNAL_KB
        
        # Last resort: resolve
        self._action_history.append("resolve_ticket")
        return ActionType.RESOLVE_TICKET
    
    def _determine_remediation(self, observation: EnvObservation) -> ActionType | None:
        """
        Determine appropriate remediation based on observations.
        
        Returns:
            Remediation action or None if more diagnostics needed
        """
        diagnostic_steps_taken = self._count_diagnostic_actions()

        # Auth issues → reset credentials
        if observation.auth_status in [
            AuthStatus.LOCKED,
            AuthStatus.EXPIRED,
            AuthStatus.MFA_FAIL
        ]:
            if diagnostic_steps_taken >= 2:
                return ActionType.RESET_CREDENTIALS
        
        # Network/DNS issues → flush DNS or reconfigure
        if observation.network_status in [NetworkStatus.DEGRADED, NetworkStatus.DOWN]:
            if diagnostic_steps_taken >= 2:
                # Check if it's likely DNS
                dns_indicators = sum(
                    1 for log in observation.system_logs
                    if "dns" in log.message.lower()
                )
                if dns_indicators > 0:
                    return ActionType.FLUSH_DNS
                else:
                    return ActionType.RECONFIGURE_CLIENT
        
        # VPN issues → reconfigure client
        if observation.vpn_status in [VPNStatus.FAILED, VPNStatus.TIMEOUT]:
            if diagnostic_steps_taken >= 2:
                return ActionType.RECONFIGURE_CLIENT
        
        # Service issues → restart service (high risk, so be cautious)
        has_critical_service_down = any(
            status == ServiceHealth.DOWN
            for status in observation.service_health.values()
        )
        
        if has_critical_service_down:
            if diagnostic_steps_taken >= 3:
                return ActionType.RESTART_SERVICE
        
        # No clear remediation yet
        return None

    def _count_diagnostic_actions(self) -> int:
        """Count diagnostic actions taken in the current episode."""
        return sum(
            1 for action_name in self._action_history
            if action_name in self._diagnostic_action_names
        )
    
    async def run_episode(
        self,
        env: SupportOpsArena,
        task_level: str = "easy"
    ) -> dict[str, Any]:
        """
        Run one complete episode.
        
        Args:
            env: Environment instance
            task_level: Difficulty level
        
        Returns:
            Dictionary with episode results
        """
        # Reset tracking
        self._action_history = []
        
        # Reset environment
        observation = await env.reset(task_level=task_level, seed=self.seed)
        
        total_reward = 0.0
        done = False
        steps = 0
        
        logger.info(f"Starting baseline agent episode on {task_level}")
        
        # Run episode
        while not done and steps < 50:  # Safety limit
            # Select action
            action = await self.select_action(observation)
            
            # Take step
            result = await env.step(action)
            
            observation = result.observation
            total_reward += result.reward
            done = result.done
            steps += 1
            
            logger.debug(
                f"Step {steps}: {action.value} -> reward={result.reward:.2f}, "
                f"done={done}"
            )
        
        # Get final state
        final_state = await env.state()
        
        logger.info(
            f"Episode completed: steps={steps}, "
            f"reward={total_reward:.2f}, "
            f"score={final_state.cumulative_reward:.2f}"
        )
        
        return {
            "steps": steps,
            "total_reward": total_reward,
            "cumulative_reward": final_state.cumulative_reward,
            "done": done,
            "task_level": task_level,
        }
    
    def reset(self) -> None:
        """Reset agent state for new episode."""
        self._action_history = []
