#!/usr/bin/env python3
"""
Pre-submission validator for SupportOps Arena hackathon checklist.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent
REQUIRED_ENV_VARS = ("API_BASE_URL", "MODEL_NAME", "HF_TOKEN")
ALLOWED_LOG_TYPES = {"[START]", "[STEP]", "[END]", "[SUMMARY]"}


def run_command(cmd: list[str], timeout: int = 600) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def check_files() -> tuple[bool, str]:
    required = ("inference.py", "openenv.yaml", "Dockerfile", "app\\server.py")
    missing = [name for name in required if not (ROOT / name).exists()]
    if missing:
        return False, f"Missing required files: {', '.join(missing)}"
    return True, "Required files exist"


def check_env_config() -> tuple[bool, str]:
    env_file = ROOT / ".env.example"
    if not env_file.exists():
        return False, ".env.example not found"
    content = env_file.read_text(encoding="utf-8")
    missing = [key for key in REQUIRED_ENV_VARS if f"{key}=" not in content]
    if missing:
        return False, f".env.example missing keys: {', '.join(missing)}"
    return True, "Environment config keys present in .env.example"


def check_yaml_and_tasks() -> tuple[bool, str]:
    data = yaml.safe_load((ROOT / "openenv.yaml").read_text(encoding="utf-8"))
    tasks = data.get("tasks", [])
    if not isinstance(tasks, list) or len(tasks) < 3:
        return False, "openenv.yaml must define at least 3 tasks"
    for idx, task in enumerate(tasks):
        score_range = task.get("score_range")
        if not isinstance(score_range, list) or len(score_range) != 2:
            return False, f"Task[{idx}] missing score_range [min,max]"
    return True, f"openenv.yaml valid with {len(tasks)} tasks"


def run_tests() -> tuple[bool, str]:
    proc = run_command(["python", "-m", "pytest", "tests", "-v", "--asyncio-mode=auto"])
    if proc.returncode != 0:
        return False, proc.stdout + proc.stderr
    return True, "pytest tests passed"


def run_inference_and_validate_logs() -> tuple[bool, str]:
    proc = run_command(["python", "inference.py"], timeout=1200)
    if proc.returncode != 0:
        return False, proc.stdout + proc.stderr

    end_scores: list[float] = []
    for raw_line in proc.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            return False, f"Non-JSON stdout line: {line}\n{exc}"

        log_type = data.get("type")
        if log_type not in ALLOWED_LOG_TYPES:
            return False, f"Invalid log type: {log_type}"
        if log_type == "[END]":
            score = data.get("final_score")
            if not isinstance(score, (int, float)):
                return False, f"[END] missing numeric final_score: {data}"
            if not (0.0 <= float(score) <= 1.0):
                return False, f"final_score out of range [0,1]: {score}"
            end_scores.append(float(score))

    if len(end_scores) < 3:
        return False, f"Expected >=3 [END] logs, got {len(end_scores)}"
    return True, f"Inference/logs valid with {len(end_scores)} task scores"


async def check_grader_score_ranges() -> tuple[bool, str]:
    from env.environment import SupportOpsArena
    from baseline.baseline_agent import BaselineAgent
    from graders.programmatic import ProgrammaticGrader

    env = SupportOpsArena()
    agent = BaselineAgent(seed=42)
    grader = ProgrammaticGrader()
    details: list[str] = []

    for level in ("easy", "medium", "hard"):
        await agent.run_episode(env, level)
        state = await env.state()
        grade = await grader.grade(state)
        score = float(grade.score)
        reward = float(state.cumulative_reward)
        if not (0.0 <= score <= 1.0):
            return False, f"{level} score out of [0,1]: {score}"
        details.append(f"{level}:score={score:.4f},reward={reward:.4f}")
        agent.reset()

    return True, "; ".join(details)


def run_openenv_validate_if_available() -> tuple[bool, str]:
    if not shutil.which("openenv"):
        return True, "openenv CLI not found; skipped openenv validate"
    proc = run_command(["openenv", "validate", "openenv.yaml"])
    if proc.returncode != 0:
        return False, proc.stdout + proc.stderr
    return True, "openenv validate openenv.yaml passed"


def check_space_endpoints(space_url: str) -> tuple[bool, str]:
    if not space_url:
        return True, "No --space-url provided; skipped remote checks"

    health_url = f"{space_url.rstrip('/')}/health"
    reset_url = f"{space_url.rstrip('/')}/reset"

    try:
        with urllib.request.urlopen(health_url, timeout=20) as resp:
            if resp.status != 200:
                return False, f"GET /health returned {resp.status}"
    except urllib.error.URLError as exc:
        return False, f"Health check failed: {exc}"

    payload = json.dumps({"task_level": "easy"}).encode("utf-8")
    req = urllib.request.Request(
        reset_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            if resp.status != 200:
                return False, f"POST /reset returned {resp.status}"
    except urllib.error.URLError as exc:
        return False, f"Reset check failed: {exc}"

    return True, "Remote Space health/reset checks passed"


def run_docker_build() -> tuple[bool, str]:
    if not shutil.which("docker"):
        return True, "Docker not installed; skipped docker build"
    tag = f"supportops-arena-presub-{int(time.time())}"
    proc = run_command(["docker", "build", "-t", tag, "."], timeout=1800)
    if proc.returncode != 0:
        return False, proc.stdout + proc.stderr
    return True, f"Docker build passed ({tag})"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--space-url", default="", help="HF Space runtime URL (https://...hf.space)")
    parser.add_argument("--skip-docker", action="store_true")
    args = parser.parse_args()

    checks: list[tuple[str, bool, str]] = []
    checks.append(("required_files", *check_files()))
    checks.append(("env_config", *check_env_config()))
    checks.append(("yaml_tasks", *check_yaml_and_tasks()))
    checks.append(("tests", *run_tests()))
    checks.append(("inference_logs", *run_inference_and_validate_logs()))
    checks.append(("grader_ranges", *asyncio.run(check_grader_score_ranges())))
    checks.append(("openenv_validate", *run_openenv_validate_if_available()))
    if not args.skip_docker:
        checks.append(("docker_build", *run_docker_build()))
    checks.append(("remote_space", *check_space_endpoints(args.space_url)))

    failed = False
    for name, ok, message in checks:
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}: {message}")
        if not ok:
            failed = True

    if failed:
        print("Overall: FAIL")
        return 1

    print("Overall: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
