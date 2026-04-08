"""
FastAPI server for SupportOps Arena.
Implements OpenEnv HTTP specification for remote environment access.
"""

import os
import uuid
import logging
from datetime import datetime, timedelta
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from env.environment import SupportOpsArena
from env.state import EnvObservation, StepResult, TaskLevel
from env.actions import ActionType
from adversary.adversary import AdaptiveAdversary
from baseline.baseline_agent import BaselineAgent
from graders.programmatic import ProgrammaticGrader
from graders.llm_grader import LLMGrader
from graders.adversarial_grader import AdversarialGrader, aggregate_scores

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="SupportOps Arena",
    description="OpenEnv-compliant IT incident triage RL environment",
    version="1.0.0"
)

STATIC_DIR = os.path.join("app", "static")
INDEX_HTML_PATH = os.path.join(STATIC_DIR, "index.html")

if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Session management
sessions: Dict[str, Dict[str, Any]] = {}
SESSION_TTL_SECONDS = 1800  # 30 minutes

# Baseline scores cache
baseline_scores: Dict[str, float] = {}


# ─── Request/Response Models ──────────────────────────────────

class ResetRequest(BaseModel):
    """Request to reset environment."""
    task_level: str = "easy"
    seed: int | None = None


class StepRequest(BaseModel):
    """Request to take a step."""
    action_name: str
    rationale: str | None = None


class GradeRequest(BaseModel):
    """Request to grade episode."""
    use_llm: bool = True
    use_adversarial: bool = False


# ─── Session Management ───────────────────────────────────────

def get_or_create_session(session_id: str | None = None) -> tuple[str, SupportOpsArena]:
    """Get existing session or create new one."""
    # Clean up expired sessions
    now = datetime.utcnow()
    expired = [
        sid for sid, data in sessions.items()
        if now - data["created_at"] > timedelta(seconds=SESSION_TTL_SECONDS)
    ]
    for sid in expired:
        del sessions[sid]
        logger.info(f"Expired session {sid}")
    
    # Create new session if needed
    if not session_id or session_id not in sessions:
        session_id = str(uuid.uuid4())
        adversary = AdaptiveAdversary()
        env = SupportOpsArena(adversary=adversary)
        
        sessions[session_id] = {
            "env": env,
            "adversary": adversary,
            "created_at": now,
            "last_accessed": now,
        }
        logger.info(f"Created new session {session_id}")
    else:
        sessions[session_id]["last_accessed"] = now
    
    return session_id, sessions[session_id]["env"]


# ─── API Endpoints ────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve dashboard UI."""
    try:
        with open(INDEX_HTML_PATH, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>SupportOps Arena</h1><p>Dashboard coming soon...</p>"
        )


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "version": "1.0.0",
        "active_sessions": len(sessions)
    }


@app.post("/reset")
async def reset(
    request: ResetRequest,
    x_session_id: str | None = Header(None)
) -> Dict[str, Any]:
    """
    Reset environment and start new episode.
    
    Returns:
        Initial observation and session ID
    """
    try:
        session_id, env = get_or_create_session(x_session_id)
        
        observation = await env.reset(
            task_level=request.task_level,
            seed=request.seed
        )
        
        logger.info(f"Reset session {session_id} to {request.task_level}")
        
        return {
            "session_id": session_id,
            "observation": observation.model_dump()
        }
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Reset failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")


@app.post("/step")
async def step(
    request: StepRequest,
    x_session_id: str | None = Header(None)
) -> Dict[str, Any]:
    """
    Take a step in the environment.
    
    Returns:
        Step result with observation, reward, done, info
    """
    try:
        if not x_session_id or x_session_id not in sessions:
            raise HTTPException(
                status_code=400,
                detail="No active session. Call /reset first."
            )
        
        session_id, env = get_or_create_session(x_session_id)
        
        # Parse action
        try:
            action = ActionType(request.action_name)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid action: {request.action_name}"
            )
        
        # Take step
        result = await env.step(action)
        
        logger.debug(
            f"Session {session_id} step: {action.value} -> "
            f"reward={result.reward:.2f}, done={result.done}"
        )
        
        return {
            "observation": result.observation.model_dump(),
            "reward": result.reward,
            "done": result.done,
            "info": result.info
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Step failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Step failed: {str(e)}")


@app.get("/state")
async def get_state(x_session_id: str | None = Header(None)) -> Dict[str, Any]:
    """
    Get full environment state (for graders).
    
    Returns:
        Complete EnvState including hidden information
    """
    try:
        if not x_session_id or x_session_id not in sessions:
            raise HTTPException(
                status_code=400,
                detail="No active session"
            )
        
        session_id, env = get_or_create_session(x_session_id)
        state = await env.state()
        
        return state.model_dump()
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get state failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Get state failed: {str(e)}")


@app.post("/grade")
async def grade_episode(
    request: GradeRequest,
    x_session_id: str | None = Header(None)
) -> Dict[str, Any]:
    """
    Grade current episode.
    
    Returns:
        Grading results from requested graders
    """
    try:
        if not x_session_id or x_session_id not in sessions:
            raise HTTPException(
                status_code=400,
                detail="No active session"
            )
        
        session_id, env = get_or_create_session(x_session_id)
        state = await env.state()
        
        if not state.done:
            raise HTTPException(
                status_code=400,
                detail="Episode not complete. Cannot grade active episode."
            )
        
        # Run programmatic grader
        prog_grader = ProgrammaticGrader()
        prog_result = await prog_grader.grade(state)
        
        results = {
            "programmatic": prog_result.model_dump()
        }
        
        # Run LLM grader if requested
        if request.use_llm:
            llm_grader = LLMGrader()
            llm_result = await llm_grader.grade(state)
            results["llm"] = llm_result.model_dump()
        
        # Run adversarial grader if requested
        if request.use_adversarial:
            adv_grader = AdversarialGrader()
            adversary = sessions[session_id].get("adversary")
            adv_score = await adv_grader.grade(state, adversary)
            results["adversarial"] = {"score": adv_score}
        
        # Calculate aggregate score
        aggregate = aggregate_scores(
            programmatic=prog_result.score,
            llm=results.get("llm", {}).get("overall"),
            adversarial=results.get("adversarial", {}).get("score"),
            task_level=state.task_level.value
        )
        
        results["aggregate_score"] = aggregate
        
        return results
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Grading failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Grading failed: {str(e)}")


@app.get("/tasks")
async def list_tasks() -> Dict[str, Any]:
    """List all available tasks with metadata."""
    return {
        "tasks": [
            {
                "id": "easy",
                "name": "Wi-Fi Connectivity Failure",
                "description": "Basic network connectivity issues",
                "max_steps": 10,
                "scenarios": 6
            },
            {
                "id": "medium",
                "name": "VPN Access Failure",
                "description": "Remote access and authentication issues",
                "max_steps": 16,
                "scenarios": 6
            },
            {
                "id": "hard",
                "name": "Cross-System Access Failure",
                "description": "Complex multi-service failures with shared root causes",
                "max_steps": 24,
                "scenarios": 6
            }
        ]
    }


@app.post("/baseline/run")
async def run_baseline() -> Dict[str, Any]:
    """
    Run baseline agent on all three task levels.
    
    Returns:
        Scores for each task level
    """
    try:
        logger.info("Starting baseline agent evaluation")
        
        agent = BaselineAgent()
        results = {}
        
        for task_level in ["easy", "medium", "hard"]:
            # Create fresh environment
            env = SupportOpsArena()
            
            # Run episode
            episode_result = await agent.run_episode(env, task_level)
            
            # Get final state for scoring
            state = await env.state()
            
            # Grade with programmatic grader
            grader = ProgrammaticGrader()
            grade_result = await grader.grade(state)
            
            results[task_level] = {
                "steps": episode_result["steps"],
                "reward": episode_result["cumulative_reward"],
                "grade_score": grade_result.score,
                "root_cause_identified": grade_result.root_cause_identified
            }
            
            # Cache score
            baseline_scores[task_level] = grade_result.score
            
            logger.info(
                f"Baseline {task_level}: "
                f"steps={episode_result['steps']}, "
                f"score={grade_result.score:.3f}"
            )
        
        return {
            "results": results,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Baseline run failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Baseline run failed: {str(e)}"
        )


@app.get("/scores")
async def get_scores() -> Dict[str, Any]:
    """Get cached baseline scores."""
    if not baseline_scores:
        return {
            "scores": {},
            "message": "No baseline scores available. Run /baseline/run first."
        }
    
    return {
        "scores": baseline_scores,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/openenv.yaml")
async def get_openenv_spec():
    """Serve openenv.yaml specification file."""
    try:
        return FileResponse("openenv.yaml", media_type="text/yaml")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="openenv.yaml not found")


# ─── Server Entry Point ───────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "7860"))
    
    logger.info(f"Starting SupportOps Arena server on {host}:{port}")
    
    uvicorn.run(
        "app.server:app",
        host=host,
        port=port,
        log_level="info"
    )
