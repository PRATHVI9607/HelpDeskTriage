"""
Pytest fixtures and configuration for SupportOps Arena tests.
"""

import pytest
import asyncio

from env.environment import SupportOpsArena
from adversary.adversary import AdaptiveAdversary
from baseline.baseline_agent import BaselineAgent


@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def env():
    """Create test environment instance."""
    adversary = AdaptiveAdversary()
    environment = SupportOpsArena(adversary=adversary)
    return environment


@pytest.fixture
def agent():
    """Create baseline agent instance."""
    return BaselineAgent(seed=42)


@pytest.fixture
def task_levels():
    """Provide list of task levels."""
    return ["easy", "medium", "hard"]
