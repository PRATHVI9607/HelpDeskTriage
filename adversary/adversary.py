"""
Adaptive adversary system for SupportOps Arena.
Analyzes agent behavior and adjusts task difficulty dynamically.
"""

from typing import Any
from collections import defaultdict

from adversary.policy_tracker import PolicyTracker
from env.state import TaskLevel, ActionRecord


class AdaptiveAdversary:
    """
    Analyzes agent behavior and adjusts ticket sampling weights.
    Makes episodes harder in areas where agent is strong,
    and tests weaknesses to drive learning.
    """
    
    def __init__(self):
        """Initialize adaptive adversary."""
        self.policy_tracker = PolicyTracker(window_size=20)
        self._default_weights = {}
    
    async def update_policy_tracker(
        self,
        ticket_variant_id: str,
        actions: list[ActionRecord],
        score: float
    ) -> None:
        """
        Update policy tracker with completed episode.
        
        Args:
            ticket_variant_id: ID of the ticket scenario
            actions: List of actions taken in episode
            score: Episode score (0-1)
        """
        # Extract metadata from ticket variant ID
        # Format: category_description (e.g., "wifi_dns_corruption", "expired_password")
        parts = ticket_variant_id.split("_")
        
        # Infer category from ID
        category = self._infer_category_from_id(ticket_variant_id)
        task_level = "easy"  # This should be passed in properly, simplified for now
        
        self.policy_tracker.record_episode(
            episode_id="",  # Would need to be passed
            task_level=task_level,
            root_cause_category=category,
            root_cause_id=ticket_variant_id,
            actions=actions,
            score=score
        )
    
    def get_sampling_weights(self, task_level: TaskLevel) -> dict[str, float]:
        """
        Get sampling weights for ticket scenarios.
        
        Adjusts weights based on agent's historical performance:
        - Boost scenarios in weak categories (score < 0.40) by 2x
        - Increase difficulty if agent is strong (score > 0.85) on certain types
        - Default to uniform weights if insufficient data
        
        Args:
            task_level: Current task difficulty level
        
        Returns:
            Dictionary mapping root_cause_id to sampling weight
        """
        # Need at least 5 episodes for meaningful adaptation
        if self.policy_tracker.get_episode_count() < 5:
            return {}  # Return empty dict for uniform sampling
        
        weights = defaultdict(lambda: 1.0)
        
        # Identify weak categories (< 0.40 average score)
        weak_categories = self.policy_tracker.get_weakness_categories(threshold=0.40)
        
        # Boost scenarios in weak categories by 2x
        for category in weak_categories:
            # This is simplified - in reality would map categories to specific IDs
            weights[f"{category}_*"] = 2.0
        
        # Check for strong performance on specific types
        scores_by_category = self.policy_tracker.get_score_by_root_cause_category()
        
        for category, avg_score in scores_by_category.items():
            if avg_score > 0.85:
                # Agent is strong here, introduce harder variants
                if task_level == TaskLevel.EASY:
                    # Boost medium-difficulty variants of same category
                    weights[f"{category}_complex"] = 1.5
                elif task_level == TaskLevel.MEDIUM:
                    # Enable contradictory evidence
                    weights[f"{category}_contradictory"] = 1.3
        
        # Check action usage patterns
        if self.policy_tracker.uses_action_frequently("restart_service", threshold=0.40):
            # Agent over-uses restart_service, introduce cascade failure scenarios
            weights["cascade_failure"] = 1.8
        
        # Check first action patterns
        first_action_dist = self.policy_tracker.get_first_action_distribution()
        if first_action_dist.get("inspect_logs", 0.0) > 0.70:
            # Agent always starts with logs, boost misleading log scenarios
            weights["misleading_heavy"] = 1.5
        
        # Check escalation patterns
        escalation_steps = self.policy_tracker.get_escalation_step_distribution()
        if escalation_steps:
            avg_escalation_step = sum(escalation_steps) / len(escalation_steps)
            if avg_escalation_step < 4:
                # Agent escalates too early, penalize premature escalation
                weights["requires_deep_diagnostics"] = 1.6
        
        return dict(weights)
    
    def get_injection_config(self, task_level: TaskLevel) -> dict[str, Any]:
        """
        Get episode configuration overrides based on agent profile.
        
        Args:
            task_level: Current task difficulty
        
        Returns:
            Dictionary with configuration overrides:
            - misleading_log_count: Number of misleading logs to inject
            - contradictory_evidence: Whether to add contradictory evidence
            - cascade_failure_enabled: Whether to enable cascade failures
        """
        if self.policy_tracker.get_episode_count() < 5:
            return {
                "misleading_log_count": 0 if task_level == TaskLevel.EASY else 1,
                "contradictory_evidence": False,
                "cascade_failure_enabled": False,
            }
        
        config = {
            "misleading_log_count": 0,
            "contradictory_evidence": False,
            "cascade_failure_enabled": False,
        }
        
        # Adjust based on task level and agent performance
        avg_score = self.policy_tracker.get_average_score()
        
        if task_level == TaskLevel.MEDIUM:
            config["misleading_log_count"] = 1
            if avg_score > 0.75:
                config["misleading_log_count"] = 2
        
        elif task_level == TaskLevel.HARD:
            config["misleading_log_count"] = 2
            config["contradictory_evidence"] = avg_score > 0.70
        
        # Enable cascade failures if agent over-uses high-risk actions
        if self.policy_tracker.uses_action_frequently("restart_service", 0.40):
            config["cascade_failure_enabled"] = True
        
        return config
    
    def _infer_category_from_id(self, ticket_id: str) -> str:
        """
        Infer root cause category from ticket variant ID.
        
        Args:
            ticket_id: Ticket variant ID
        
        Returns:
            Category name
        """
        # Simple heuristic based on ID patterns
        if "dns" in ticket_id:
            return "dns"
        elif "vpn" in ticket_id:
            return "vpn"
        elif "auth" in ticket_id or "password" in ticket_id or "mfa" in ticket_id:
            return "auth"
        elif "sso" in ticket_id:
            return "sso"
        elif "wifi" in ticket_id or "network" in ticket_id or "dhcp" in ticket_id:
            return "network"
        elif "adapter" in ticket_id or "device" in ticket_id or "airplane" in ticket_id:
            return "hardware"
        else:
            return "unknown"
    
    def reset_tracker(self) -> None:
        """Reset policy tracker (useful for testing)."""
        self.policy_tracker = PolicyTracker(window_size=20)
    
    def get_statistics(self) -> dict[str, Any]:
        """
        Get adversary statistics for debugging/monitoring.
        
        Returns:
            Dictionary with statistics
        """
        return {
            "episodes_tracked": self.policy_tracker.get_episode_count(),
            "average_score": self.policy_tracker.get_average_score(),
            "action_frequencies": self.policy_tracker.get_action_frequencies(),
            "first_action_dist": self.policy_tracker.get_first_action_distribution(),
            "scores_by_category": self.policy_tracker.get_score_by_root_cause_category(),
            "weak_categories": self.policy_tracker.get_weakness_categories(),
        }
