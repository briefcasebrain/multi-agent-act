"""
Core module for multi-agent collaborative learning.

Contains the fundamental classes and functionality for agents, environments,
knowledge systems, and configuration management.
"""

from .config import (
    LearningScenarioType,
    ScenarioPhase,
    ScenarioConfig,
    LearningOutcome
)

from .knowledge import KnowledgeDistillationEngine

__all__ = [
    'LearningScenarioType',
    'ScenarioPhase',
    'ScenarioConfig',
    'LearningOutcome',
    'KnowledgeDistillationEngine'
]