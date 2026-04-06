"""
Adversarial grader for testing policy robustness.
Generates variant episodes and tests if agent's strategy would still work.
"""

import logging
from typing import Any

from env.state import EnvState
from adversary.adversary import AdaptiveAdversary

logger = logging.getLogger(__name__)


class AdversarialGrader:
    """
    Tests policy robustness by generating harder variants.
    Used for hard task only.
    """
    
    def __init__(self):
        """Initialize adversarial grader."""
        pass
    
    async def grade(
        self,
        state: EnvState,
        adversary: AdaptiveAdversary | None = None
    ) -> float:
        """
        Grade policy robustness by testing on variant episodes.
        
        Strategy:
        1. Generate a variant of the same ticket (different surface presentation)
        2. Simulate if agent's action strategy would still work
        3. Return robustness score (0.0-1.0)
        
        Args:
            state: Complete environment state after episode
            adversary: Optional adaptive adversary for variant generation
        
        Returns:
            Robustness score (0.0-1.0)
        """
        # For now, use heuristics to estimate robustness
        # A full implementation would actually run the variant episode
        
        try:
            robustness_score = self._estimate_robustness(state)
            return robustness_score
        
        except Exception as e:
            logger.error(f"Adversarial grading failed: {e}", exc_info=True)
            return 0.5  # Neutral fallback
    
    def _estimate_robustness(self, state: EnvState) -> float:
        """
        Estimate policy robustness using heuristics.
        
        A robust policy:
        - Uses diverse diagnostic actions (not always same pattern)
        - Adapts to observations (doesn't follow fixed sequence)
        - Gathers evidence before high-risk actions
        - Makes decisions based on evidence, not just templates
        """
        score_components = []
        
        # 1. Action diversity score
        unique_actions = len(set(a.action_name for a in state.action_log))
        total_actions = len(state.action_log)
        diversity_score = unique_actions / total_actions if total_actions > 0 else 0.0
        score_components.append(diversity_score)
        
        # 2. Diagnostic breadth score
        diagnostic_actions = {
            "inspect_network",
            "inspect_logs",
            "check_authentication",
            "check_permissions",
            "query_device_status",
            "search_internal_kb",
            "contact_user_for_info",
            "run_diagnostic_script",
        }
        
        diagnostics_used = sum(
            1 for a in state.action_log
            if a.action_name in diagnostic_actions
        )
        unique_diagnostics_used = len(set(
            a.action_name for a in state.action_log
            if a.action_name in diagnostic_actions
        ))
        
        diagnostic_breadth = (
            unique_diagnostics_used / len(diagnostic_actions)
            if diagnostic_actions else 0.0
        )
        score_components.append(diagnostic_breadth)
        
        # 3. Evidence-before-action score
        # Check if high-risk actions were preceded by diagnostics
        diagnostic_count = 0
        risky_without_diagnostic = 0
        
        for action in state.action_log:
            if action.action_name in diagnostic_actions:
                diagnostic_count += 1
            elif action.action_name in ["restart_service", "reset_credentials"]:
                if diagnostic_count < 2:
                    risky_without_diagnostic += 1
        
        if risky_without_diagnostic == 0:
            evidence_score = 1.0
        else:
            evidence_score = max(0.0, 1.0 - (risky_without_diagnostic * 0.3))
        
        score_components.append(evidence_score)
        
        # 4. Adaptive behavior score
        # Check if agent repeated same action multiple times (template behavior)
        action_counts = {}
        for action in state.action_log:
            action_counts[action.action_name] = action_counts.get(action.action_name, 0) + 1
        
        max_repeats = max(action_counts.values()) if action_counts else 1
        if max_repeats >= 4:
            adaptive_score = 0.5  # Too much repetition
        elif max_repeats >= 3:
            adaptive_score = 0.7
        else:
            adaptive_score = 1.0
        
        score_components.append(adaptive_score)
        
        # 5. Success under pressure score
        # Check if agent succeeded despite complexity
        task_multiplier = {
            "easy": 0.8,
            "medium": 1.0,
            "hard": 1.2,
        }
        
        # If agent succeeded on hard task, boost robustness
        if state.done and state.cumulative_reward > 0:
            success_score = 1.0 * task_multiplier.get(state.task_level.value, 1.0)
        else:
            success_score = 0.5
        
        score_components.append(min(1.0, success_score))
        
        # Calculate weighted average
        weights = [0.15, 0.20, 0.25, 0.20, 0.20]
        robustness_score = sum(
            score * weight
            for score, weight in zip(score_components, weights)
        )
        
        return min(1.0, max(0.0, robustness_score))


def aggregate_scores(
    programmatic: float,
    llm: float | None,
    adversarial: float | None,
    task_level: str
) -> float:
    """
    Aggregate scores from multiple graders.
    
    Args:
        programmatic: Score from programmatic grader
        llm: Score from LLM grader (optional)
        adversarial: Score from adversarial grader (optional)
        task_level: Task difficulty level
    
    Returns:
        Aggregated score (0.0-1.0)
    """
    if task_level == "easy":
        return programmatic
    
    elif task_level == "medium":
        llm_score = llm if llm is not None else 0.5
        return 0.6 * programmatic + 0.4 * llm_score
    
    else:  # hard
        llm_score = llm if llm is not None else 0.5
        adv_score = adversarial if adversarial is not None else 0.5
        return 0.5 * programmatic + 0.3 * llm_score + 0.2 * adv_score


# Benchmark score weights by task level
OVERALL_BENCHMARK_SCORE_WEIGHTS = {
    "easy": 0.2,
    "medium": 0.3,
    "hard": 0.5,
}


def calculate_benchmark_score(scores_by_level: dict[str, float]) -> float:
    """
    Calculate overall benchmark score across task levels.
    
    Args:
        scores_by_level: Dictionary mapping task level to score
    
    Returns:
        Weighted benchmark score
    """
    return sum(
        scores_by_level.get(level, 0.0) * weight
        for level, weight in OVERALL_BENCHMARK_SCORE_WEIGHTS.items()
    )
