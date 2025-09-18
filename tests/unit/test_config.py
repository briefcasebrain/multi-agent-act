"""Unit tests for configuration classes."""

import pytest
from dataclasses import FrozenInstanceError

from multi_agent_collab_learning.core.config import (
    ScenarioConfig,
    LearningOutcome,
    LearningScenarioType,
    LearningPhase
)


class TestScenarioConfig:
    """Test ScenarioConfig class."""

    @pytest.mark.unit
    def test_valid_config_creation(self):
        """Test creating a valid configuration."""
        config = ScenarioConfig(
            scenario_type=LearningScenarioType.COMPETITIVE_TOURNAMENT,
            duration_episodes=50,
            participants=4
        )

        assert config.scenario_type == LearningScenarioType.COMPETITIVE_TOURNAMENT
        assert config.duration_episodes == 50
        assert config.participants == 4
        assert config.reward_structure == "mixed"  # default value
        assert config.knowledge_sharing_rate == 0.3  # default value

    @pytest.mark.unit
    def test_config_with_all_parameters(self):
        """Test configuration with all parameters specified."""
        success_criteria = {'min_performance': 0.7, 'collaboration_threshold': 0.6}
        evaluation_metrics = ['performance_improvement', 'collaboration_effectiveness']
        scenario_parameters = {'tournament_format': 'bracket', 'elo_k_factor': 32}

        config = ScenarioConfig(
            scenario_type=LearningScenarioType.MENTOR_STUDENT,
            duration_episodes=100,
            participants=6,
            reward_structure="collective",
            knowledge_sharing_rate=0.5,
            adaptation_frequency=20,
            success_criteria=success_criteria,
            evaluation_metrics=evaluation_metrics,
            scenario_parameters=scenario_parameters
        )

        assert config.scenario_type == LearningScenarioType.MENTOR_STUDENT
        assert config.duration_episodes == 100
        assert config.participants == 6
        assert config.reward_structure == "collective"
        assert config.knowledge_sharing_rate == 0.5
        assert config.adaptation_frequency == 20
        assert config.success_criteria == success_criteria
        assert config.evaluation_metrics == evaluation_metrics
        assert config.scenario_parameters == scenario_parameters

    @pytest.mark.unit
    def test_invalid_duration_episodes(self):
        """Test that invalid duration_episodes raises ValueError."""
        with pytest.raises(ValueError, match="duration_episodes must be positive"):
            ScenarioConfig(
                scenario_type=LearningScenarioType.COMPETITIVE_TOURNAMENT,
                duration_episodes=-1
            )

        with pytest.raises(ValueError, match="duration_episodes must be positive"):
            ScenarioConfig(
                scenario_type=LearningScenarioType.COMPETITIVE_TOURNAMENT,
                duration_episodes=0
            )

    @pytest.mark.unit
    def test_invalid_participants(self):
        """Test that invalid participants count raises ValueError."""
        with pytest.raises(ValueError, match="participants must be at least 2"):
            ScenarioConfig(
                scenario_type=LearningScenarioType.COMPETITIVE_TOURNAMENT,
                duration_episodes=50,
                participants=1
            )

        with pytest.raises(ValueError, match="participants must be at least 2"):
            ScenarioConfig(
                scenario_type=LearningScenarioType.COMPETITIVE_TOURNAMENT,
                duration_episodes=50,
                participants=0
            )

    @pytest.mark.unit
    def test_invalid_reward_structure(self):
        """Test that invalid reward_structure raises ValueError."""
        with pytest.raises(ValueError, match="reward_structure must be one of"):
            ScenarioConfig(
                scenario_type=LearningScenarioType.COMPETITIVE_TOURNAMENT,
                duration_episodes=50,
                participants=4,
                reward_structure="invalid"
            )

    @pytest.mark.unit
    def test_invalid_knowledge_sharing_rate(self):
        """Test that invalid knowledge_sharing_rate raises ValueError."""
        with pytest.raises(ValueError, match="knowledge_sharing_rate must be between 0 and 1"):
            ScenarioConfig(
                scenario_type=LearningScenarioType.COMPETITIVE_TOURNAMENT,
                duration_episodes=50,
                participants=4,
                knowledge_sharing_rate=-0.1
            )

        with pytest.raises(ValueError, match="knowledge_sharing_rate must be between 0 and 1"):
            ScenarioConfig(
                scenario_type=LearningScenarioType.COMPETITIVE_TOURNAMENT,
                duration_episodes=50,
                participants=4,
                knowledge_sharing_rate=1.1
            )

    @pytest.mark.unit
    def test_invalid_adaptation_frequency(self):
        """Test that invalid adaptation_frequency raises ValueError."""
        with pytest.raises(ValueError, match="adaptation_frequency must be positive"):
            ScenarioConfig(
                scenario_type=LearningScenarioType.COMPETITIVE_TOURNAMENT,
                duration_episodes=50,
                participants=4,
                adaptation_frequency=-5
            )

        with pytest.raises(ValueError, match="adaptation_frequency must be positive"):
            ScenarioConfig(
                scenario_type=LearningScenarioType.COMPETITIVE_TOURNAMENT,
                duration_episodes=50,
                participants=4,
                adaptation_frequency=0
            )

    @pytest.mark.unit
    def test_config_immutability(self):
        """Test that configuration is immutable after creation."""
        config = ScenarioConfig(
            scenario_type=LearningScenarioType.COMPETITIVE_TOURNAMENT,
            duration_episodes=50,
            participants=4
        )

        with pytest.raises(FrozenInstanceError):
            config.duration_episodes = 100

        with pytest.raises(FrozenInstanceError):
            config.participants = 6

    @pytest.mark.unit
    def test_config_defaults(self):
        """Test that default values are set correctly."""
        config = ScenarioConfig(
            scenario_type=LearningScenarioType.COMPETITIVE_TOURNAMENT,
            duration_episodes=50
        )

        assert config.participants == 4
        assert config.reward_structure == "mixed"
        assert config.knowledge_sharing_rate == 0.3
        assert config.adaptation_frequency == 25
        assert config.success_criteria == {}
        assert config.evaluation_metrics == []
        assert config.scenario_parameters == {}


class TestLearningOutcome:
    """Test LearningOutcome class."""

    @pytest.mark.unit
    def test_valid_outcome_creation(self):
        """Test creating a valid learning outcome."""
        outcome = LearningOutcome(
            agent_id="agent_1",
            performance_improvement=0.25,
            collaboration_effectiveness=0.8,
            knowledge_acquired=0.6
        )

        assert outcome.agent_id == "agent_1"
        assert outcome.performance_improvement == 0.25
        assert outcome.collaboration_effectiveness == 0.8
        assert outcome.knowledge_acquired == 0.6
        assert outcome.phase == LearningPhase.ACTIVE  # default value

    @pytest.mark.unit
    def test_outcome_with_all_parameters(self):
        """Test outcome with all parameters specified."""
        adaptation_metrics = {'flexibility': 0.7, 'speed': 0.5}

        outcome = LearningOutcome(
            agent_id="agent_2",
            performance_improvement=0.4,
            collaboration_effectiveness=0.9,
            knowledge_acquired=0.7,
            adaptation_speed=0.6,
            phase=LearningPhase.REFLECTION,
            confidence_level=0.85,
            learning_efficiency=0.75,
            adaptation_metrics=adaptation_metrics
        )

        assert outcome.agent_id == "agent_2"
        assert outcome.performance_improvement == 0.4
        assert outcome.collaboration_effectiveness == 0.9
        assert outcome.knowledge_acquired == 0.7
        assert outcome.adaptation_speed == 0.6
        assert outcome.phase == LearningPhase.REFLECTION
        assert outcome.confidence_level == 0.85
        assert outcome.learning_efficiency == 0.75
        assert outcome.adaptation_metrics == adaptation_metrics

    @pytest.mark.unit
    def test_invalid_performance_improvement(self):
        """Test that invalid performance_improvement raises ValueError."""
        with pytest.raises(ValueError, match="performance_improvement must be between -1 and 1"):
            LearningOutcome(
                agent_id="agent_1",
                performance_improvement=1.5,
                collaboration_effectiveness=0.8,
                knowledge_acquired=0.6
            )

        with pytest.raises(ValueError, match="performance_improvement must be between -1 and 1"):
            LearningOutcome(
                agent_id="agent_1",
                performance_improvement=-1.5,
                collaboration_effectiveness=0.8,
                knowledge_acquired=0.6
            )

    @pytest.mark.unit
    def test_invalid_collaboration_effectiveness(self):
        """Test that invalid collaboration_effectiveness raises ValueError."""
        with pytest.raises(ValueError, match="collaboration_effectiveness must be between 0 and 1"):
            LearningOutcome(
                agent_id="agent_1",
                performance_improvement=0.25,
                collaboration_effectiveness=1.5,
                knowledge_acquired=0.6
            )

        with pytest.raises(ValueError, match="collaboration_effectiveness must be between 0 and 1"):
            LearningOutcome(
                agent_id="agent_1",
                performance_improvement=0.25,
                collaboration_effectiveness=-0.1,
                knowledge_acquired=0.6
            )

    @pytest.mark.unit
    def test_invalid_knowledge_acquired(self):
        """Test that invalid knowledge_acquired raises ValueError."""
        with pytest.raises(ValueError, match="knowledge_acquired must be between 0 and 1"):
            LearningOutcome(
                agent_id="agent_1",
                performance_improvement=0.25,
                collaboration_effectiveness=0.8,
                knowledge_acquired=1.5
            )

        with pytest.raises(ValueError, match="knowledge_acquired must be between 0 and 1"):
            LearningOutcome(
                agent_id="agent_1",
                performance_improvement=0.25,
                collaboration_effectiveness=0.8,
                knowledge_acquired=-0.1
            )

    @pytest.mark.unit
    def test_outcome_immutability(self):
        """Test that learning outcome is immutable after creation."""
        outcome = LearningOutcome(
            agent_id="agent_1",
            performance_improvement=0.25,
            collaboration_effectiveness=0.8,
            knowledge_acquired=0.6
        )

        with pytest.raises(FrozenInstanceError):
            outcome.performance_improvement = 0.5

        with pytest.raises(FrozenInstanceError):
            outcome.agent_id = "agent_2"


class TestEnums:
    """Test enum classes."""

    @pytest.mark.unit
    def test_learning_scenario_type_values(self):
        """Test LearningScenarioType enum values."""
        assert LearningScenarioType.COMPETITIVE_TOURNAMENT.value == "competitive_tournament"
        assert LearningScenarioType.MENTOR_STUDENT.value == "mentor_student"
        assert LearningScenarioType.COLLABORATIVE_RESEARCH.value == "collaborative_research"

    @pytest.mark.unit
    def test_learning_phase_values(self):
        """Test LearningPhase enum values."""
        assert LearningPhase.EXPLORATION.value == "exploration"
        assert LearningPhase.ACTIVE.value == "active"
        assert LearningPhase.REFLECTION.value == "reflection"
        assert LearningPhase.CONSOLIDATION.value == "consolidation"

    @pytest.mark.unit
    def test_enum_iteration(self):
        """Test that enums can be iterated."""
        scenario_types = list(LearningScenarioType)
        assert len(scenario_types) >= 3  # At least the core 3 types
        assert LearningScenarioType.COMPETITIVE_TOURNAMENT in scenario_types
        assert LearningScenarioType.MENTOR_STUDENT in scenario_types
        assert LearningScenarioType.COLLABORATIVE_RESEARCH in scenario_types

        phases = list(LearningPhase)
        assert len(phases) == 4
        assert LearningPhase.ACTIVE in phases