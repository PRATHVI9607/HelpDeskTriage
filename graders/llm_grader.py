"""
LLM-based grader for evaluating agent rationales and reasoning quality.
Uses OpenAI-compatible API to grade soft criteria.
"""

import os
import json
import logging
from typing import Any
from pydantic import BaseModel, Field

from openai import AsyncOpenAI

from env.state import EnvState

logger = logging.getLogger(__name__)


class LLMGradeResult(BaseModel):
    """Result from LLM grading."""
    diagnostic_coherence: float = Field(ge=0.0, le=1.0)
    evidence_sufficiency: float = Field(ge=0.0, le=1.0)
    root_cause_narration: float = Field(ge=0.0, le=1.0)
    overall: float = Field(ge=0.0, le=1.0)
    reasoning: str
    
    model_config = {"frozen": False}


class LLMGrader:
    """
    LLM judge using OpenAI-compatible API.
    Evaluates agent reasoning and rationales.
    """
    
    SYSTEM_PROMPT = """You are an expert IT operations grader evaluating an AI agent's performance on an incident triage task.

You will be given:
1. The hidden root cause (ground truth)
2. The agent's action history with observation updates
3. The final outcome (resolved/escalated/closed)

Grade the following dimensions on a scale of 0.0 to 1.0:
- diagnostic_coherence: Are the agent's actions logically consistent with the observations? Did it follow a coherent diagnostic strategy?
- evidence_sufficiency: Did the agent gather enough evidence before committing to a fix or escalation?
- root_cause_narration: Did the agent's actions demonstrate understanding of the actual root cause?

Return ONLY a valid JSON object with these exact keys:
{
  "diagnostic_coherence": <float 0.0-1.0>,
  "evidence_sufficiency": <float 0.0-1.0>,
  "root_cause_narration": <float 0.0-1.0>,
  "overall": <float 0.0-1.0>,
  "reasoning": "<brief explanation>"
}

Do not include any other text or markdown formatting."""
    
    def __init__(self):
        """Initialize LLM grader with API configuration."""
        self.api_base_url = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
        self.model_name = os.environ.get("MODEL_NAME", "gpt-4o-mini")
        self.api_key = os.environ.get("OPENAI_API_KEY", "")
        
        if not self.api_key:
            logger.warning("OPENAI_API_KEY not set. LLM grading will use fallback scores.")
        
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.api_base_url
        )
    
    async def grade(self, state: EnvState) -> LLMGradeResult:
        """
        Grade episode using LLM judge.
        
        Args:
            state: Complete environment state after episode
        
        Returns:
            LLMGradeResult with scores and reasoning
        """
        if not self.api_key:
            logger.warning("No API key available, returning fallback scores")
            return self._fallback_grade()
        
        try:
            # Build user prompt
            user_prompt = self._build_prompt(state)
            
            # Call LLM API
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            # Parse response
            content = response.choices[0].message.content
            if not content:
                logger.error("Empty response from LLM")
                return self._fallback_grade()
            
            # Try to parse JSON
            try:
                result_dict = json.loads(content)
            except json.JSONDecodeError:
                # Try to extract JSON from markdown code blocks
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()
                result_dict = json.loads(content)
            
            return LLMGradeResult(**result_dict)
        
        except Exception as e:
            logger.error(f"LLM grading failed: {e}", exc_info=True)
            return self._fallback_grade()
    
    def _build_prompt(self, state: EnvState) -> str:
        """Build grading prompt from episode state."""
        # Extract key information
        root_cause = state.hidden.root_cause
        category = state.hidden.root_cause_category
        severity = state.hidden.severity
        
        # Summarize action sequence
        action_summary = []
        for i, action in enumerate(state.action_log, 1):
            action_summary.append(
                f"{i}. {action.action_name} (reward: {action.reward_received:.2f})"
            )
        
        actions_text = "\n".join(action_summary)
        
        # Summarize observations
        final_obs = state.observation
        
        prompt = f"""## Episode Summary

**Ground Truth Root Cause:**
- Category: {category}
- Description: {root_cause}
- Severity: {severity}
- Affected Users: {state.hidden.affected_users_count}

**Correct Remediation:**
- Primary: {state.hidden.correct_remediation}
- Alternatives: {', '.join(state.hidden.correct_remediation_alts) if state.hidden.correct_remediation_alts else 'None'}

**Agent's Action Sequence:**
{actions_text}

**Final State:**
- Steps taken: {final_obs.step_count}
- Episode terminated: {state.done}
- Diagnostic steps: {state.diagnostic_steps_taken}
- Escalation allowed: {final_obs.escalation_allowed}

**Observations:**
- Network Status: {final_obs.network_status.value}
- VPN Status: {final_obs.vpn_status.value}
- Auth Status: {final_obs.auth_status.value}
- Service Health: {dict(final_obs.service_health)}

**Performance:**
- Cumulative Reward: {state.cumulative_reward:.2f}
- Task Level: {state.task_level.value}

Evaluate how well the agent diagnosed and resolved this incident."""
        
        return prompt
    
    def _fallback_grade(self) -> LLMGradeResult:
        """Return neutral fallback scores when LLM unavailable."""
        return LLMGradeResult(
            diagnostic_coherence=0.5,
            evidence_sufficiency=0.5,
            root_cause_narration=0.5,
            overall=0.5,
            reasoning="LLM grading unavailable, using fallback scores"
        )
