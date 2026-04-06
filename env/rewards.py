"""
Reward calculation system for SupportOps Arena.
Implements dense reward shaping with detailed breakdown tracking.
"""

from typing import Any
from env.state import EnvState, TaskLevel
from env.actions import ActionType


# ─── Reward Events ────────────────────────────────────────────

REWARD_EVENTS = {
    # Positive rewards
    "DIAGNOSTIC_ACTION": +0.10,
    "CORRECT_TARGETED_DIAGNOSTIC": +0.20,
    "EVIDENCE_BEFORE_RISKY": +0.05,
    "CORRECT_REMEDIATION": +0.40,
    "OPTIMAL_RESOLUTION": +0.20,  # Only if steps_used <= min_required_steps
    "JUSTIFIED_ESCALATION": +0.30,
    "TERMINAL_CORRECT": +1.00,
    
    # Negative rewards
    "REDUNDANT_ACTION": -0.05,
    "RISKY_WITHOUT_DIAGNOSTIC": -0.15,
    "INCORRECT_REMEDIATION": -0.25,
    "HARMFUL_ACTION": -0.35,
    "PREMATURE_CLOSURE": -0.40,
    "BUDGET_EXCEEDED_PER_STEP": -0.02,
    "PREMATURE_ESCALATION": -0.20,  # Escalate before unlock
    "UNJUSTIFIED_ESCALATION": -0.20,  # Escalate without enough evidence
}


# ─── Configuration ────────────────────────────────────────────

MIN_STEPS_FOR_OPTIMAL = {
    TaskLevel.EASY: 4,
    TaskLevel.MEDIUM: 7,
    TaskLevel.HARD: 12,
}

MAX_POSSIBLE_REWARD = {
    TaskLevel.EASY: 2.05,
    TaskLevel.MEDIUM: 2.35,
    TaskLevel.HARD: 2.75,
}


class RewardCalculator:
    """
    Calculates rewards for agent actions with dense reward shaping.
    Tracks individual reward events for interpretability.
    """
    
    def calculate(
        self,
        events: list[str],
        state: EnvState,
        action: ActionType
    ) -> tuple[float, dict[str, Any]]:
        """
        Calculate total reward from a list of reward event keys.
        
        Args:
            events: List of reward event keys (e.g., ["DIAGNOSTIC_ACTION", "CORRECT_TARGETED_DIAGNOSTIC"])
            state: Current environment state
            action: Action that was taken
        
        Returns:
            Tuple of (total_reward, breakdown_dict)
            - total_reward: Sum of all event rewards
            - breakdown_dict: Dictionary mapping event keys to their reward values
        """
        breakdown = {}
        total_reward = 0.0
        
        for event_key in events:
            if event_key not in REWARD_EVENTS:
                raise ValueError(f"Unknown reward event: {event_key}")
            
            reward_value = REWARD_EVENTS[event_key]
            breakdown[event_key] = reward_value
            total_reward += reward_value
        
        return total_reward, breakdown
    
    def calculate_episode_score(
        self,
        cumulative_reward: float,
        task_level: TaskLevel
    ) -> float:
        """
        Normalize cumulative reward to episode score in [0, 1].
        
        Args:
            cumulative_reward: Total reward accumulated during episode
            task_level: Difficulty level of the task
        
        Returns:
            Normalized score clamped to [0, 1]
        """
        max_reward = MAX_POSSIBLE_REWARD[task_level]
        score = cumulative_reward / max_reward
        return max(0.0, min(1.0, score))
    
    def is_optimal_resolution(
        self,
        step_count: int,
        task_level: TaskLevel
    ) -> bool:
        """
        Check if resolution was achieved within optimal step count.
        
        Args:
            step_count: Number of steps taken
            task_level: Difficulty level
        
        Returns:
            True if step_count <= minimum optimal steps for this level
        """
        return step_count <= MIN_STEPS_FOR_OPTIMAL[task_level]
    
    def get_max_possible_reward(self, task_level: TaskLevel) -> float:
        """Get maximum possible reward for a task level."""
        return MAX_POSSIBLE_REWARD[task_level]
    
    def validate_events(self, events: list[str]) -> bool:
        """
        Validate that all event keys are recognized.
        
        Args:
            events: List of event keys to validate
        
        Returns:
            True if all events are valid
        
        Raises:
            ValueError: If any event key is not recognized
        """
        for event in events:
            if event not in REWARD_EVENTS:
                raise ValueError(f"Unknown reward event: {event}")
        return True


def get_reward_event_description(event_key: str) -> str:
    """Get human-readable description of a reward event."""
    descriptions = {
        "DIAGNOSTIC_ACTION": "Performed a diagnostic action to gather information",
        "CORRECT_TARGETED_DIAGNOSTIC": "Performed diagnostic action that targets the actual root cause",
        "EVIDENCE_BEFORE_RISKY": "Gathered evidence before taking risky action",
        "CORRECT_REMEDIATION": "Applied the correct fix for the root cause",
        "OPTIMAL_RESOLUTION": "Resolved issue within optimal step count",
        "JUSTIFIED_ESCALATION": "Escalated with sufficient evidence",
        "TERMINAL_CORRECT": "Episode terminated with correct resolution",
        "REDUNDANT_ACTION": "Repeated an action already performed",
        "RISKY_WITHOUT_DIAGNOSTIC": "Took risky action without prior diagnostics",
        "INCORRECT_REMEDIATION": "Applied wrong fix for the root cause",
        "HARMFUL_ACTION": "Action caused harm or disruption",
        "PREMATURE_CLOSURE": "Closed ticket without proper resolution",
        "BUDGET_EXCEEDED_PER_STEP": "Exceeded step budget (per step penalty)",
        "PREMATURE_ESCALATION": "Escalated before gathering minimum evidence",
        "UNJUSTIFIED_ESCALATION": "Escalated without sufficient diagnostic evidence",
    }
    return descriptions.get(event_key, f"Unknown event: {event_key}")


def get_all_positive_events() -> list[str]:
    """Get list of all positive reward event keys."""
    return [k for k, v in REWARD_EVENTS.items() if v > 0]


def get_all_negative_events() -> list[str]:
    """Get list of all negative reward event keys."""
    return [k for k, v in REWARD_EVENTS.items() if v < 0]
