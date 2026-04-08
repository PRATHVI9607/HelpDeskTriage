#!/usr/bin/env python3
"""
SupportOps Arena — Baseline Inference Script
Runs baseline agent against all 3 task levels.
Produces [START]/[STEP]/[END] logs to stdout in JSON format.
Must complete in < 20 minutes total.
"""

import asyncio
import json
import os
import sys
import logging
from datetime import datetime

from openai import AsyncOpenAI

# Read environment variables
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:7860")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# Setup logging
logging.basicConfig(
    level=logging.WARNING,  # Only warnings/errors to stderr
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)


def create_openai_client() -> AsyncOpenAI:
    """
    Create OpenAI-compatible async client from required environment variables.
    HF_TOKEN is preferred per hackathon submission requirements.
    """
    api_key = HF_TOKEN or OPENAI_API_KEY or "local-baseline-no-key"
    return AsyncOpenAI(api_key=api_key, base_url=API_BASE_URL)


def validate_required_env() -> None:
    """Warn on missing required submission env vars (stderr only)."""
    missing = []
    if not API_BASE_URL:
        missing.append("API_BASE_URL")
    if not MODEL_NAME:
        missing.append("MODEL_NAME")
    if not HF_TOKEN:
        missing.append("HF_TOKEN")

    if missing:
        logging.warning(
            "Missing required env vars for submission: %s",
            ", ".join(missing)
        )


async def run_task(task_level: str, env, agent) -> dict:
    """
    Run one task level with baseline agent.
    
    Args:
        task_level: "easy", "medium", or "hard"
        env: SupportOpsArena environment instance
        agent: BaselineAgent instance
    
    Returns:
        Dictionary with task_level and score
    """
    # Reset environment
    obs = await env.reset(task_level=task_level)
    state = await env.state()
    episode_id = state.episode_id
    
    # [START] log
    print(json.dumps({
        "type": "[START]",
        "episode_id": episode_id,
        "task_level": task_level,
        "timestamp": datetime.utcnow().isoformat()
    }), flush=True)
    
    cumulative_reward = 0.0
    step = 0
    done = False
    
    # Run episode
    while not done and step < 50:  # Safety limit
        # Select action
        action = await agent.select_action(obs)
        
        # Take step
        result = await env.step(action)
        obs = result.observation
        reward = result.reward
        done = result.done
        info = result.info
        
        cumulative_reward += reward
        step += 1
        
        # [STEP] log
        print(json.dumps({
            "type": "[STEP]",
            "step": step,
            "action": action.value,
            "rationale": "",
            "reward": round(reward, 4),
            "cumulative_reward": round(cumulative_reward, 4),
            "done": done
        }), flush=True)
    
    # Get final state
    final_state = await env.state()
    
    # Calculate final score from programmatic grader
    from graders.programmatic import ProgrammaticGrader
    grader = ProgrammaticGrader()
    grade_result = await grader.grade(final_state)
    
    final_score = grade_result.score
    correct = grade_result.root_cause_identified
    
    # [END] log
    print(json.dumps({
        "type": "[END]",
        "episode_id": episode_id,
        "task_level": task_level,
        "final_score": round(final_score, 4),
        "steps_used": step,
        "correct": correct
    }), flush=True)
    
    return {"task_level": task_level, "score": final_score}


async def main():
    """Main entry point for inference script."""
    try:
        validate_required_env()
        llm_client = create_openai_client()
        _ = (llm_client, MODEL_NAME)

        # Import environment and agent
        from env.environment import SupportOpsArena
        from baseline.baseline_agent import BaselineAgent
        from adversary.adversary import AdaptiveAdversary
        
        # Create environment with adversary
        adversary = AdaptiveAdversary()
        env = SupportOpsArena(adversary=adversary)
        
        # Create baseline agent
        agent = BaselineAgent(seed=42)
        
        # Run all three task levels
        results = []
        for task_level in ["easy", "medium", "hard"]:
            result = await run_task(task_level, env, agent)
            results.append(result)
            
            # Reset agent for next episode
            agent.reset()
        
        # Calculate overall benchmark score
        # Weights: easy=0.2, medium=0.3, hard=0.5
        weights = [0.2, 0.3, 0.5]
        overall_score = sum(
            r["score"] * w 
            for r, w in zip(results, weights)
        )
        
        # [SUMMARY] log
        print(json.dumps({
            "type": "[SUMMARY]",
            "easy_score": round(results[0]["score"], 4),
            "medium_score": round(results[1]["score"], 4),
            "hard_score": round(results[2]["score"], 4),
            "overall_score": round(overall_score, 4),
            "timestamp": datetime.utcnow().isoformat()
        }), flush=True)
        
        return 0
    
    except Exception as e:
        print(json.dumps({
            "type": "[ERROR]",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
