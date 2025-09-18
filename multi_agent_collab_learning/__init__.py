"""
Multi-Agent Collaborative Learning Library

A comprehensive Python library for implementing advanced multi-agent collaborative
learning scenarios including competitive tournaments, mentor-student networks,
collaborative research environments, and more.

Author: Aansh Shah <aansh@briefcasebrain.com>
License: MIT
"""

__version__ = "0.1.0"
__author__ = "Aansh Shah <aansh@briefcasebrain.com>"

# Core imports
from .core.config import (
    LearningScenarioType,
    ScenarioPhase,
    LearningPhase,
    ScenarioConfig,
    LearningOutcome
)

from .core.agents import CollaborativeAgent
from .core.environments import MultiAgentEnvironment
from .core.knowledge import KnowledgeDistillationEngine

# Scenario imports
from .scenarios.competitive import CompetitiveLearningTournament
from .scenarios.mentor_student import MentorStudentNetwork
from .scenarios.collaborative import CollaborativeResearchEnvironment
from .scenarios.orchestrator import ScenarioOrchestrator

# Utility imports
from .utils.logging import setup_logger
from .utils.visualization import plot_learning_curves

__all__ = [
    # Core classes
    'CollaborativeAgent',
    'MultiAgentEnvironment',
    'KnowledgeDistillationEngine',

    # Configuration
    'LearningScenarioType',
    'ScenarioPhase',
    'LearningPhase',
    'ScenarioConfig',
    'LearningOutcome',

    # Scenarios
    'CompetitiveLearningTournament',
    'MentorStudentNetwork',
    'CollaborativeResearchEnvironment',
    'ScenarioOrchestrator',

    # Utilities
    'setup_logger',
    'plot_learning_curves'
]