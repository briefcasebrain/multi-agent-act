"""Pytest configuration and shared fixtures."""

import pytest
import torch
import random
import numpy as np
from typing import List

from tests.fixtures.agents import (
    MockAgent, MockCompetitiveAgent, MockMentorAgent, MockStudentAgent,
    create_mock_agents, create_mentor_student_pair
)
from tests.fixtures.environments import (
    MockEnvironment, MockCompetitiveEnvironment, MockResearchEnvironment,
    create_mock_environment
)


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: mark test as unit test")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "scenario: mark test as scenario-specific")
    config.addinivalue_line("markers", "gpu: mark test as requiring GPU")


@pytest.fixture(scope="session", autouse=True)
def set_random_seeds():
    """Set random seeds for reproducible tests."""
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)


@pytest.fixture
def deterministic_environment():
    """Create a deterministic environment for reproducible tests."""
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    yield
    # Reset seeds after test
    torch.seed()


# Agent Fixtures
@pytest.fixture
def mock_agent():
    """Create a single mock agent."""
    return MockAgent("test_agent")


@pytest.fixture
def mock_agents():
    """Create a list of standard mock agents."""
    return create_mock_agents(count=4, agent_type="standard")


@pytest.fixture
def competitive_agents():
    """Create a list of competitive agents."""
    return create_mock_agents(count=4, agent_type="competitive")


@pytest.fixture
def mentor_student_agents():
    """Create mentor and student agents."""
    return create_mock_agents(count=4, agent_type="mentor")


@pytest.fixture
def mentor_student_pair():
    """Create a mentor-student pair."""
    return create_mentor_student_pair()


# Environment Fixtures
@pytest.fixture
def mock_environment():
    """Create a standard mock environment."""
    return create_mock_environment("standard", deterministic=True)


@pytest.fixture
def competitive_environment():
    """Create a competitive environment."""
    return create_mock_environment("competitive", deterministic=True)


@pytest.fixture
def research_environment():
    """Create a research environment."""
    return create_mock_environment("research", deterministic=True)


# Configuration Fixtures
@pytest.fixture
def basic_config():
    """Create a basic scenario configuration."""
    from multi_agent_collab_learning.core.config import ScenarioConfig, LearningScenarioType

    return ScenarioConfig(
        scenario_type=LearningScenarioType.COMPETITIVE_TOURNAMENT,
        duration_episodes=10,
        participants=4
    )


@pytest.fixture
def tournament_config():
    """Create a tournament-specific configuration."""
    from multi_agent_collab_learning.core.config import ScenarioConfig, LearningScenarioType

    return ScenarioConfig(
        scenario_type=LearningScenarioType.COMPETITIVE_TOURNAMENT,
        duration_episodes=20,
        participants=6,
        reward_structure="mixed",
        scenario_parameters={
            'tournament_format': 'bracket',
            'elo_k_factor': 32
        }
    )


@pytest.fixture
def mentor_student_config():
    """Create a mentor-student configuration."""
    from multi_agent_collab_learning.core.config import ScenarioConfig, LearningScenarioType

    return ScenarioConfig(
        scenario_type=LearningScenarioType.MENTOR_STUDENT,
        duration_episodes=30,
        participants=4,
        knowledge_sharing_rate=0.6,
        scenario_parameters={
            'mentor_ratio': 0.25,
            'teaching_effectiveness_threshold': 0.7
        }
    )


@pytest.fixture
def research_config():
    """Create a collaborative research configuration."""
    from multi_agent_collab_learning.core.config import ScenarioConfig, LearningScenarioType

    return ScenarioConfig(
        scenario_type=LearningScenarioType.COLLABORATIVE_RESEARCH,
        duration_episodes=40,
        participants=6,
        knowledge_sharing_rate=0.8,
        scenario_parameters={
            'research_topics': ['navigation', 'manipulation', 'communication'],
            'discovery_threshold': 0.7,
            'innovation_metric_weight': 0.6
        }
    )


# Utility Fixtures
@pytest.fixture
def temp_log_file(tmp_path):
    """Create a temporary log file for testing."""
    log_file = tmp_path / "test.log"
    return str(log_file)


@pytest.fixture
def sample_learning_data():
    """Provide sample learning data for visualization tests."""
    return {
        'agent_0': [0.1, 0.3, 0.5, 0.7, 0.8],
        'agent_1': [0.2, 0.4, 0.6, 0.75, 0.85],
        'agent_2': [0.15, 0.35, 0.55, 0.72, 0.82],
        'agent_3': [0.25, 0.45, 0.65, 0.78, 0.88]
    }


@pytest.fixture
def sample_collaboration_data():
    """Provide sample collaboration data for network tests."""
    return {
        'agent_0': {'agent_1': 0.8, 'agent_2': 0.6},
        'agent_1': {'agent_0': 0.8, 'agent_3': 0.7},
        'agent_2': {'agent_0': 0.6, 'agent_3': 0.9},
        'agent_3': {'agent_1': 0.7, 'agent_2': 0.9}
    }


# Scenario Result Fixtures
@pytest.fixture
def tournament_results():
    """Provide sample tournament results."""
    return {
        'tournament_winner': 'agent_0',
        'final_rankings': {
            'agent_0': 1850.0,
            'agent_1': 1750.0,
            'agent_2': 1650.0,
            'agent_3': 1550.0
        },
        'matches_played': 12,
        'competitive_behaviors': {
            'agent_0': 'aggressive',
            'agent_1': 'defensive',
            'agent_2': 'balanced',
            'agent_3': 'adaptive'
        }
    }


@pytest.fixture
def mentor_student_results():
    """Provide sample mentor-student results."""
    return {
        'knowledge_transfer_metrics': {
            'total_transfers': 45,
            'successful_transfers': 38,
            'transfer_efficiency': 0.844
        },
        'teaching_effectiveness': {
            'mentor_0': 0.85,
            'mentor_1': 0.72
        },
        'learning_progress': {
            'student_0': 0.78,
            'student_1': 0.65,
            'student_2': 0.82
        }
    }


@pytest.fixture
def research_results():
    """Provide sample research results."""
    return {
        'discoveries': [
            {'topic': 'navigation', 'significance': 0.85},
            {'topic': 'communication', 'significance': 0.72}
        ],
        'collaboration_events': [
            {'agents': ['agent_0', 'agent_1'], 'strength': 0.8},
            {'agents': ['agent_2', 'agent_3'], 'strength': 0.9}
        ],
        'innovation_metrics': {
            'breakthrough_count': 2,
            'cross_domain_transfers': 3,
            'collaboration_efficiency': 0.75
        }
    }