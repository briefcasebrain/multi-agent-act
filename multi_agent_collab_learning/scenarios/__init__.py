"""
Scenarios module for multi-agent collaborative learning.

Contains implementations of various collaborative learning scenarios including
competitive tournaments, mentor-student networks, collaborative research, and
scenario orchestration.
"""

from .competitive import CompetitiveLearningTournament
from .mentor_student import MentorStudentNetwork
from .collaborative import CollaborativeResearchEnvironment
from .orchestrator import ScenarioOrchestrator

__all__ = [
    'CompetitiveLearningTournament',
    'MentorStudentNetwork',
    'CollaborativeResearchEnvironment',
    'ScenarioOrchestrator'
]