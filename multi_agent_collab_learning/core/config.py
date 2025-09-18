"""
Configuration classes and enums for multi-agent collaborative learning scenarios.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Optional


class LearningScenarioType(Enum):
    """Types of collaborative learning scenarios."""
    COMPETITIVE_TOURNAMENT = "competitive_tournament"
    MENTOR_STUDENT = "mentor_student"
    COLLABORATIVE_RESEARCH = "collaborative_research"
    MULTI_TASK_COALITION = "multi_task_coalition"
    ADVERSARIAL_COLLABORATION = "adversarial_collaboration"
    FEDERATED_LEARNING = "federated_learning"
    CROSS_MODAL_TRANSFER = "cross_modal_transfer"
    DISTRIBUTED_PROBLEM_SOLVING = "distributed_problem_solving"
    COLLABORATIVE_CREATIVITY = "collaborative_creativity"
    MULTI_AGENT_TEACHING = "multi_agent_teaching"
    SWARM_LEARNING = "swarm_learning"
    HIERARCHICAL_ORGANIZATION = "hierarchical_organization"


class ScenarioPhase(Enum):
    """Phases of learning scenarios."""
    INITIALIZATION = "initialization"
    EXPLORATION = "exploration"
    COLLABORATION = "collaboration"
    COMPETITION = "competition"
    KNOWLEDGE_SHARING = "knowledge_sharing"
    EVALUATION = "evaluation"
    ADAPTATION = "adaptation"
    CONCLUSION = "conclusion"


class LearningPhase(Enum):
    """Phases of individual agent learning."""
    EXPLORATION = "exploration"
    ACTIVE = "active"
    REFLECTION = "reflection"
    CONSOLIDATION = "consolidation"


@dataclass(frozen=True)
class ScenarioConfig:
    """Configuration for learning scenarios."""
    scenario_type: LearningScenarioType
    duration_episodes: int = 100
    participants: int = 4
    success_criteria: Dict[str, float] = field(default_factory=dict)
    reward_structure: str = "mixed"  # individual, collective, mixed
    knowledge_sharing_rate: float = 0.3
    adaptation_frequency: int = 25
    evaluation_metrics: List[str] = field(default_factory=list)
    scenario_parameters: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.duration_episodes <= 0:
            raise ValueError("duration_episodes must be positive")
        if self.participants < 2:
            raise ValueError("participants must be at least 2")
        if self.adaptation_frequency <= 0:
            raise ValueError("adaptation_frequency must be positive")
        if self.reward_structure not in ["individual", "collective", "mixed"]:
            raise ValueError("reward_structure must be one of: individual, collective, mixed")
        if not 0 <= self.knowledge_sharing_rate <= 1:
            raise ValueError("knowledge_sharing_rate must be between 0 and 1")


@dataclass(frozen=True)
class LearningOutcome:
    """Represents learning outcomes from scenarios."""
    agent_id: str
    performance_improvement: float
    collaboration_effectiveness: float
    knowledge_acquired: float
    adaptation_speed: float = 0.5
    phase: LearningPhase = LearningPhase.ACTIVE
    confidence_level: float = 0.5
    learning_efficiency: float = 0.5
    adaptation_metrics: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        """Validate learning outcome data."""
        if not -1 <= self.performance_improvement <= 1:
            raise ValueError("performance_improvement must be between -1 and 1")
        if not 0 <= self.collaboration_effectiveness <= 1:
            raise ValueError("collaboration_effectiveness must be between 0 and 1")
        if not 0 <= self.knowledge_acquired <= 1:
            raise ValueError("knowledge_acquired must be between 0 and 1")