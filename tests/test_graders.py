"""
Tests for grading system.
"""

import pytest
from env.environment import SupportOpsArena
from env.actions import ActionType
from adversary.adversary import AdaptiveAdversary
from graders.programmatic import ProgrammaticGrader
from graders.adversarial_grader import aggregate_scores


class TestProgrammaticGrader:
    """Test programmatic grader functionality."""
    
    @pytest.mark.asyncio
    async def test_grader_initialization(self):
        """Test grader can be created."""
        grader = ProgrammaticGrader()
        assert grader is not None
    
    @pytest.mark.asyncio
    async def test_grade_complete_episode(self):
        """Test grading a complete episode."""
        adversary = AdaptiveAdversary()
        env = SupportOpsArena(adversary=adversary)
        await env.reset(task_level="easy", seed=42)
        
        # Run a simple episode
        await env.step(ActionType.INSPECT_LOGS)
        await env.step(ActionType.CHECK_AUTHENTICATION)
        await env.step(ActionType.INSPECT_NETWORK)
        await env.step(ActionType.RESOLVE_TICKET)
        
        # Grade episode
        state = await env.state()
        grader = ProgrammaticGrader()
        result = await grader.grade(state)
        
        assert result.score >= 0.0
        assert result.score <= 1.0
        assert result.action_sequence_safe is not None
        assert result.step_budget_respected is not None
    
    @pytest.mark.asyncio
    async def test_grade_score_range(self):
        """Test grade score is in valid range."""
        adversary = AdaptiveAdversary()
        env = SupportOpsArena(adversary=adversary)
        await env.reset(task_level="easy", seed=42)
        
        # Take minimal actions
        await env.step(ActionType.INSPECT_LOGS)
        await env.step(ActionType.RESOLVE_TICKET)
        
        state = await env.state()
        grader = ProgrammaticGrader()
        result = await grader.grade(state)
        
        assert 0.0 <= result.score <= 1.0


class TestScoreAggregation:
    """Test score aggregation logic."""
    
    def test_aggregate_easy(self):
        """Test aggregation for easy task (programmatic only)."""
        score = aggregate_scores(0.75, None, None, "easy")
        assert score == 0.75
    
    def test_aggregate_medium(self):
        """Test aggregation for medium task (programmatic + LLM)."""
        score = aggregate_scores(0.75, 0.60, None, "medium")
        assert 0.65 < score < 0.75  # Weighted average
    
    def test_aggregate_hard(self):
        """Test aggregation for hard task (all graders)."""
        score = aggregate_scores(0.75, 0.60, 0.50, "hard")
        assert 0.5 < score < 0.75  # Weighted average
