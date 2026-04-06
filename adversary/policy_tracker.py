"""
Policy tracker for adaptive adversary system.
Tracks agent behavior across episodes to identify patterns and weaknesses.
"""

from collections import defaultdict, deque
from typing import Any
from env.state import ActionRecord


class PolicyTracker:
    """
    Tracks agent behavior across episodes.
    Maintains a sliding window of last N episodes for analysis.
    """
    
    def __init__(self, window_size: int = 20):
        """
        Initialize policy tracker.
        
        Args:
            window_size: Number of recent episodes to track
        """
        self.window_size = window_size
        self.episodes = deque(maxlen=window_size)
        self.action_counts = defaultdict(int)
        self.total_actions = 0
        self.root_cause_scores = defaultdict(list)
        self.escalation_steps = []
        self.first_actions = []
    
    def record_episode(
        self,
        episode_id: str,
        task_level: str,
        root_cause_category: str,
        root_cause_id: str,
        actions: list[ActionRecord],
        score: float
    ) -> None:
        """
        Record completed episode into tracker.
        
        Args:
            episode_id: Unique episode identifier
            task_level: Difficulty level
            root_cause_category: Category of root cause
            root_cause_id: Specific root cause ID
            actions: List of actions taken
            score: Episode score (0-1)
        """
        episode_data = {
            "episode_id": episode_id,
            "task_level": task_level,
            "root_cause_category": root_cause_category,
            "root_cause_id": root_cause_id,
            "actions": actions,
            "score": score,
            "num_actions": len(actions)
        }
        
        self.episodes.append(episode_data)
        
        # Update action counts
        for action in actions:
            self.action_counts[action.action_name] += 1
            self.total_actions += 1
        
        # Track root cause performance
        self.root_cause_scores[root_cause_category].append(score)
        
        # Track escalation patterns
        for i, action in enumerate(actions):
            if action.action_name == "escalate_ticket":
                self.escalation_steps.append(i + 1)  # 1-indexed
                break
        
        # Track first action
        if actions:
            self.first_actions.append(actions[0].action_name)
    
    def get_action_frequencies(self, task_level: str | None = None) -> dict[str, float]:
        """
        Get normalized frequency of each action type.
        
        Args:
            task_level: Optional filter by task level
        
        Returns:
            Dictionary mapping action names to frequencies (0-1)
        """
        if not self.total_actions:
            return {}
        
        # Filter episodes if task level specified
        if task_level:
            filtered_actions = defaultdict(int)
            total = 0
            for ep in self.episodes:
                if ep["task_level"] == task_level:
                    for action in ep["actions"]:
                        filtered_actions[action.action_name] += 1
                        total += 1
            
            if total == 0:
                return {}
            
            return {
                action: count / total
                for action, count in filtered_actions.items()
            }
        
        # Return overall frequencies
        return {
            action: count / self.total_actions
            for action, count in self.action_counts.items()
        }
    
    def get_first_action_distribution(self) -> dict[str, float]:
        """
        Analyze what agent does first.
        
        Returns:
            Dictionary mapping action names to probability of being first
        """
        if not self.first_actions:
            return {}
        
        counts = defaultdict(int)
        for action in self.first_actions:
            counts[action] += 1
        
        total = len(self.first_actions)
        return {
            action: count / total
            for action, count in counts.items()
        }
    
    def get_escalation_step_distribution(self) -> list[int]:
        """
        Get list of steps at which agent escalated.
        
        Returns:
            List of step numbers (useful for statistical analysis)
        """
        return self.escalation_steps.copy()
    
    def get_score_by_root_cause_category(self) -> dict[str, float]:
        """
        Get average score per root cause category.
        
        Returns:
            Dictionary mapping categories to average scores
        """
        return {
            category: sum(scores) / len(scores) if scores else 0.0
            for category, scores in self.root_cause_scores.items()
        }
    
    def get_average_score(self, task_level: str | None = None) -> float:
        """
        Get average score across episodes.
        
        Args:
            task_level: Optional filter by task level
        
        Returns:
            Average score (0-1)
        """
        if task_level:
            scores = [
                ep["score"] for ep in self.episodes
                if ep["task_level"] == task_level
            ]
        else:
            scores = [ep["score"] for ep in self.episodes]
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def get_episode_count(self) -> int:
        """Get number of recorded episodes."""
        return len(self.episodes)
    
    def get_weakness_categories(self, threshold: float = 0.40) -> list[str]:
        """
        Identify root cause categories where agent performs poorly.
        
        Args:
            threshold: Score threshold below which category is "weak"
        
        Returns:
            List of category names with average score < threshold
        """
        scores_by_category = self.get_score_by_root_cause_category()
        return [
            category for category, score in scores_by_category.items()
            if score < threshold
        ]
    
    def uses_action_frequently(self, action_name: str, threshold: float = 0.40) -> bool:
        """
        Check if agent uses a specific action frequently.
        
        Args:
            action_name: Name of action to check
            threshold: Frequency threshold (0-1)
        
        Returns:
            True if action frequency >= threshold
        """
        frequencies = self.get_action_frequencies()
        return frequencies.get(action_name, 0.0) >= threshold
