"""
SupportOps Arena - Core OpenEnv-compliant environment.
Main environment class implementing the full OpenEnv interface.
"""

import uuid
import logging
from datetime import datetime
from typing import Any

from env.state import (
    EnvState, EnvObservation, StepResult, HiddenState,
    TaskLevel, ActionRecord, NetworkStatus, VPNStatus,
    AuthStatus, ServiceHealth
)
from env.actions import ActionType, ACTION_METADATA
from env.scenarios import TicketGenerator
from env.transitions import StateMachine
from env.rewards import RewardCalculator

# Setup logging
logger = logging.getLogger(__name__)


# ─── Configuration ────────────────────────────────────────────

MAX_STEPS = {
    TaskLevel.EASY: 10,
    TaskLevel.MEDIUM: 16,
    TaskLevel.HARD: 24,
}


class SupportOpsArena:
    """
    OpenEnv-compliant environment for enterprise IT incident triage.
    
    An RL environment simulating IT support operations where agents must:
    - Gather evidence through sequential tool use
    - Reason about hidden root causes
    - Apply risk-weighted remediation actions
    - Escalate when autonomous resolution is unsafe
    
    Implements full OpenEnv interface:
    - async reset(task_level: str, seed: int | None) -> EnvObservation
    - async step(action: ActionType) -> StepResult
    - async state() -> EnvState
    """
    
    def __init__(self, adversary: Any | None = None):
        """
        Initialize the environment.
        
        Args:
            adversary: Optional AdaptiveAdversary instance for difficulty adaptation
        """
        self.ticket_generator = TicketGenerator()
        self.state_machine = StateMachine()
        self.reward_calculator = RewardCalculator()
        self.adversary = adversary
        
        self._current_state: EnvState | None = None
        self._episode_active = False
        
        logger.info("SupportOpsArena environment initialized")
    
    async def reset(
        self,
        task_level: str = "easy",
        seed: int | None = None
    ) -> EnvObservation:
        """
        Reset environment and start new episode.
        
        Args:
            task_level: Difficulty level - "easy", "medium", or "hard"
            seed: Random seed for reproducibility
        
        Returns:
            Initial observation for the episode
        
        Raises:
            ValueError: If task_level is not valid
        """
        # Validate task level
        if task_level not in ["easy", "medium", "hard"]:
            raise ValueError(f"Invalid task_level: {task_level}. Must be 'easy', 'medium', or 'hard'")
        
        task_level_enum = TaskLevel(task_level)
        
        # Generate new episode ID
        episode_id = str(uuid.uuid4())
        
        logger.info(f"Starting episode {episode_id} at {task_level} difficulty (seed={seed})")
        
        # Apply adversary sampling weights if available
        if self.adversary:
            weights = self.adversary.get_sampling_weights(task_level_enum)
            self.ticket_generator.set_sampling_weights(weights)
        
        # Generate ticket scenario
        ticket_data = self.ticket_generator.generate_ticket(task_level_enum, seed)
        root_cause_data = ticket_data["root_cause"]
        obs_template = ticket_data["observation_template"]
        misleading_idx = ticket_data.get("misleading_log_index")
        
        # Create hidden state
        hidden = HiddenState(
            root_cause=root_cause_data["description"],
            root_cause_category=root_cause_data["category"],
            correct_remediation=root_cause_data["correct_remediation"].value,
            correct_remediation_alts=[
                alt.value if isinstance(alt, ActionType) else alt 
                for alt in root_cause_data.get("correct_remediation_alts", [])
            ],
            misleading_log_index=misleading_idx,
            severity=root_cause_data["severity"],
            affected_users_count=root_cause_data["affected_users"],
            ticket_variant_id=root_cause_data["id"]
        )
        
        # Create initial observation
        max_steps = MAX_STEPS[task_level_enum]
        observation = EnvObservation(
            ticket_id=f"INC-{episode_id[:8].upper()}",
            ticket_summary=obs_template["ticket_summary"],
            user_context=obs_template["user_context"],
            network_status=obs_template.get("network_status", NetworkStatus.UNKNOWN),
            vpn_status=obs_template.get("vpn_status", VPNStatus.NA),
            auth_status=obs_template.get("auth_status", AuthStatus.OK),
            service_health=obs_template.get("service_health", {}),
            system_logs=obs_template["system_logs"],
            action_history=[],
            step_count=0,
            escalation_allowed=False,
            task_level=task_level_enum,
            steps_remaining=max_steps
        )
        
        # Create full state
        self._current_state = EnvState(
            episode_id=episode_id,
            task_level=task_level_enum,
            observation=observation,
            hidden=hidden,
            cumulative_reward=0.0,
            done=False,
            created_at=datetime.utcnow().isoformat(),
            action_log=[],
            diagnostic_steps_taken=0
        )
        
        self._episode_active = True
        
        logger.info(f"Episode {episode_id} initialized: {root_cause_data['id']}")
        
        return observation
    
    async def step(self, action: ActionType) -> StepResult:
        """
        Execute one step in the environment.
        
        Args:
            action: Action to take (ActionType enum)
        
        Returns:
            StepResult containing (observation, reward, done, info)
        
        Raises:
            RuntimeError: If episode is not active or already terminated
            ValueError: If action is invalid
        """
        if not self._episode_active or self._current_state is None:
            raise RuntimeError("Episode not active. Call reset() first.")
        
        if self._current_state.done:
            raise RuntimeError("Episode already terminated. Call reset() to start new episode.")
        
        # Validate action
        if not isinstance(action, ActionType):
            logger.error(f"Invalid action type: {action}")
            # Return penalty for invalid action
            info = {
                "episode_id": self._current_state.episode_id,
                "step": self._current_state.observation.step_count,
                "root_cause": None,
                "reward_breakdown": {"HARMFUL_ACTION": -0.35},
                "correct": False
            }
            return StepResult(
                observation=self._current_state.observation,
                reward=-0.35,
                done=False,
                info=info
            )
        
        logger.debug(f"Step {self._current_state.observation.step_count + 1}: {action.value}")
        
        # Apply state transition
        next_state, reward_events = await self.state_machine.transition(
            self._current_state,
            action
        )
        
        # Check terminal conditions and add terminal rewards
        done, terminal_events = self._check_terminal_conditions(next_state, action)
        reward_events.extend(terminal_events)
        
        # Calculate reward
        reward, reward_breakdown = self.reward_calculator.calculate(
            reward_events,
            next_state,
            action
        )
        
        # Update state
        next_state.cumulative_reward += reward
        next_state.observation.step_count += 1
        next_state.observation.steps_remaining -= 1
        next_state.done = done
        
        # Record action in history
        action_record = ActionRecord(
            step=next_state.observation.step_count,
            action_name=action.value,
            rationale=None,
            reward_received=reward,
            timestamp=datetime.utcnow().isoformat()
        )
        next_state.observation.action_history.append(action_record)
        next_state.action_log.append(action_record)
        
        # Update adversary if available
        if self.adversary and done:
            await self.adversary.update_policy_tracker(
                next_state.hidden.ticket_variant_id,
                next_state.action_log,
                next_state.cumulative_reward
            )
        
        # Build info dict
        info: dict[str, Any] = {
            "episode_id": next_state.episode_id,
            "step": next_state.observation.step_count,
            "root_cause": None,  # Hidden until episode ends
            "reward_breakdown": reward_breakdown,
            "correct": False
        }
        
        # Reveal ground truth at episode end
        if done:
            info["root_cause"] = next_state.hidden.root_cause
            info["correct"] = self._check_correctness(next_state, action)
            info["episode_score"] = self.reward_calculator.calculate_episode_score(
                next_state.cumulative_reward,
                next_state.task_level
            )
            
            logger.info(
                f"Episode {next_state.episode_id} completed: "
                f"score={info['episode_score']:.3f}, "
                f"correct={info['correct']}, "
                f"steps={next_state.observation.step_count}"
            )
        
        # Update current state
        self._current_state = next_state
        
        if done:
            self._episode_active = False
        
        return StepResult(
            observation=next_state.observation,
            reward=reward,
            done=done,
            info=info
        )
    
    async def state(self) -> EnvState:
        """
        Return full internal state including hidden information.
        
        This method is for graders only and should not be called by agents
        during active episodes.
        
        Returns:
            Complete EnvState including hidden state
        
        Raises:
            RuntimeError: If no episode is active
        """
        if self._current_state is None:
            raise RuntimeError("No active episode. Call reset() first.")
        
        return self._current_state
    
    def _check_terminal_conditions(
        self,
        state: EnvState,
        action: ActionType
    ) -> tuple[bool, list[str]]:
        """
        Check if episode should terminate and return terminal reward events.
        
        Returns:
            Tuple of (done: bool, terminal_reward_events: list[str])
        """
        events = []
        
        # Check if action is terminal
        action_meta = ACTION_METADATA.get(action)
        if action_meta and action_meta.is_terminal:
            if action == ActionType.ESCALATE_TICKET:
                # Check if escalation was premature (already handled in transition)
                if not state.observation.escalation_allowed:
                    # Don't terminate if premature
                    return False, events
                
                # Check if escalation is justified
                if state.diagnostic_steps_taken >= 3:
                    events.append("JUSTIFIED_ESCALATION")
                else:
                    events.append("UNJUSTIFIED_ESCALATION")
                
                # Escalation is always "correct" if allowed and justified
                if state.diagnostic_steps_taken >= 3:
                    events.append("TERMINAL_CORRECT")
                
                return True, events
            
            elif action == ActionType.RESOLVE_TICKET:
                # Check if correct remediation
                correct_action = state.hidden.correct_remediation
                alt_actions = state.hidden.correct_remediation_alts
                
                # Check if we applied the correct fix earlier
                past_actions = [a.action_name for a in state.observation.action_history]
                
                if correct_action in past_actions:
                    events.append("CORRECT_REMEDIATION")
                    events.append("TERMINAL_CORRECT")
                    
                    # Check for optimal resolution
                    if self.reward_calculator.is_optimal_resolution(
                        state.observation.step_count,
                        state.task_level
                    ):
                        events.append("OPTIMAL_RESOLUTION")
                else:
                    # Check alternatives
                    alt_found = any(alt in past_actions for alt in alt_actions)
                    if alt_found:
                        events.append("CORRECT_REMEDIATION")
                        events.append("TERMINAL_CORRECT")
                    else:
                        events.append("INCORRECT_REMEDIATION")
                
                return True, events
            
            elif action == ActionType.CLOSE_WITHOUT_FIX:
                events.append("PREMATURE_CLOSURE")
                return True, events
        
        # Check step budget
        max_steps = MAX_STEPS[state.task_level]
        if state.observation.step_count >= max_steps:
            # Budget exceeded
            remaining_steps = max_steps - state.observation.step_count
            if remaining_steps < 0:
                for _ in range(abs(remaining_steps)):
                    events.append("BUDGET_EXCEEDED_PER_STEP")
            
            return True, events
        
        return False, events
    
    def _check_correctness(self, state: EnvState, action: ActionType) -> bool:
        """Check if episode was resolved correctly."""
        if action == ActionType.RESOLVE_TICKET:
            correct_action = state.hidden.correct_remediation
            alt_actions = state.hidden.correct_remediation_alts
            past_actions = [a.action_name for a in state.observation.action_history]
            
            return (
                correct_action in past_actions or
                any(alt in past_actions for alt in alt_actions)
            )
        elif action == ActionType.ESCALATE_TICKET:
            # Escalation with sufficient evidence is considered correct
            return state.diagnostic_steps_taken >= 3
        else:
            return False
