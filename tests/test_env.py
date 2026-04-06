"""
Tests for SupportOpsArena environment core functionality.
"""

import pytest
from env.environment import SupportOpsArena
from env.actions import ActionType
from env.state import TaskLevel
from adversary.adversary import AdaptiveAdversary


class TestEnvironmentBasics:
    """Test basic environment functionality."""
    
    @pytest.mark.asyncio
    async def test_environment_initialization(self):
        """Test environment can be created."""
        adversary = AdaptiveAdversary()
        env = SupportOpsArena(adversary=adversary)
        assert env is not None
    
    @pytest.mark.asyncio
    async def test_reset_easy(self):
        """Test reset for easy task."""
        adversary = AdaptiveAdversary()
        env = SupportOpsArena(adversary=adversary)
        obs = await env.reset(task_level="easy", seed=42)
        
        assert obs.task_level == TaskLevel.EASY
        assert obs.step_count == 0
        assert len(obs.system_logs) > 0
        assert obs.escalation_allowed == False
    
    @pytest.mark.asyncio
    async def test_reset_medium(self):
        """Test reset for medium task."""
        adversary = AdaptiveAdversary()
        env = SupportOpsArena(adversary=adversary)
        obs = await env.reset(task_level="medium", seed=42)
        
        assert obs.task_level == TaskLevel.MEDIUM
        assert obs.step_count == 0
    
    @pytest.mark.asyncio
    async def test_reset_hard(self):
        """Test reset for hard task."""
        adversary = AdaptiveAdversary()
        env = SupportOpsArena(adversary=adversary)
        obs = await env.reset(task_level="hard", seed=42)
        
        assert obs.task_level == TaskLevel.HARD
        assert obs.step_count == 0
    
    @pytest.mark.asyncio
    async def test_step_basic_action(self):
        """Test taking a basic step."""
        adversary = AdaptiveAdversary()
        env = SupportOpsArena(adversary=adversary)
        obs = await env.reset(task_level="easy", seed=42)
        
        result = await env.step(ActionType.INSPECT_LOGS)
        
        assert result.observation.step_count == 1
        assert result.reward != 0.0  # Should get diagnostic reward
        assert result.done == False
    
    @pytest.mark.asyncio
    async def test_invalid_task_level(self):
        """Test reset with invalid task level raises error."""
        adversary = AdaptiveAdversary()
        env = SupportOpsArena(adversary=adversary)
        
        with pytest.raises(ValueError):
            await env.reset(task_level="invalid")
    
    @pytest.mark.asyncio
    async def test_episode_done_on_resolve(self):
        """Test episode ends when ticket is resolved."""
        adversary = AdaptiveAdversary()
        env = SupportOpsArena(adversary=adversary)
        await env.reset(task_level="easy", seed=42)
        
        # Take a few diagnostic actions
        await env.step(ActionType.INSPECT_LOGS)
        await env.step(ActionType.CHECK_AUTHENTICATION)
        await env.step(ActionType.INSPECT_NETWORK)
        
        # Resolve ticket
        result = await env.step(ActionType.RESOLVE_TICKET)
        
        assert result.done == True
        assert "root_cause" in result.info
    
    @pytest.mark.asyncio
    async def test_state_method(self):
        """Test state() method returns full state."""
        adversary = AdaptiveAdversary()
        env = SupportOpsArena(adversary=adversary)
        await env.reset(task_level="easy", seed=42)
        
        state = await env.state()
        
        assert state.observation is not None
        assert state.hidden is not None
        assert state.episode_id is not None
    
    @pytest.mark.asyncio
    async def test_escalation_unlock(self):
        """Test escalation is unlocked after enough diagnostics."""
        adversary = AdaptiveAdversary()
        env = SupportOpsArena(adversary=adversary)
        obs = await env.reset(task_level="easy", seed=42)
        
        assert obs.escalation_allowed == False
        
        # Take 3 diagnostic actions (unlock threshold for easy)
        result1 = await env.step(ActionType.INSPECT_LOGS)
        result2 = await env.step(ActionType.CHECK_AUTHENTICATION)
        result3 = await env.step(ActionType.INSPECT_NETWORK)
        
        # Should now be unlocked
        assert result3.observation.escalation_allowed == True
    
    @pytest.mark.asyncio
    async def test_step_without_reset_raises_error(self):
        """Test that step without reset raises error."""
        adversary = AdaptiveAdversary()
        env = SupportOpsArena(adversary=adversary)
        
        with pytest.raises(RuntimeError):
            await env.step(ActionType.INSPECT_LOGS)


class TestRewardSystem:
    """Test reward calculation."""
    
    @pytest.mark.asyncio
    async def test_diagnostic_action_reward(self):
        """Test diagnostic actions give positive reward."""
        adversary = AdaptiveAdversary()
        env = SupportOpsArena(adversary=adversary)
        await env.reset(task_level="easy", seed=42)
        
        result = await env.step(ActionType.INSPECT_LOGS)
        
        assert result.reward > 0  # Should get diagnostic bonus
        assert "DIAGNOSTIC_ACTION" in str(result.info.get("reward_breakdown", {}))
    
    @pytest.mark.asyncio
    async def test_redundant_action_penalty(self):
        """Test redundant actions are penalized."""
        adversary = AdaptiveAdversary()
        env = SupportOpsArena(adversary=adversary)
        await env.reset(task_level="easy", seed=42)
        
        # Take same action twice
        result1 = await env.step(ActionType.INSPECT_LOGS)
        result2 = await env.step(ActionType.INSPECT_LOGS)
        
        # Second time should have penalty
        assert "REDUNDANT_ACTION" in str(result2.info.get("reward_breakdown", {}))
