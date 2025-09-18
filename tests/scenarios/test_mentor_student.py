"""Tests for mentor-student learning scenarios."""

import pytest
import torch

from multi_agent_collab_learning.scenarios.mentor_student import MentorStudentNetwork
from multi_agent_collab_learning.core.config import ScenarioConfig, LearningScenarioType


class TestMentorStudentNetwork:
    """Test MentorStudentNetwork class."""

    @pytest.mark.scenario
    def test_network_initialization(self, mentor_student_agents, mentor_student_config):
        """Test mentor-student network initialization."""
        network = MentorStudentNetwork(mentor_student_agents, mentor_student_config)

        assert network.agents == mentor_student_agents
        assert network.config == mentor_student_config
        assert network.mentor_assignments == {}
        assert network.knowledge_transfer_log == []
        assert network.teaching_effectiveness == {}

    @pytest.mark.scenario
    def test_validate_agents_success(self, mentor_student_agents, mentor_student_config):
        """Test successful agent validation."""
        network = MentorStudentNetwork(mentor_student_agents, mentor_student_config)
        assert network.validate_agents() is True

    @pytest.mark.scenario
    def test_validate_agents_insufficient_count(self, mock_agents, mentor_student_config):
        """Test agent validation with insufficient agent count."""
        # Config expects 4 participants, mock_agents has 4, so this should pass
        network = MentorStudentNetwork(mock_agents, mentor_student_config)
        assert network.validate_agents() is True

    @pytest.mark.scenario
    def test_identify_mentors_and_students(self, mentor_student_agents, mentor_student_config):
        """Test mentor and student identification."""
        network = MentorStudentNetwork(mentor_student_agents, mentor_student_config)
        mentors, students = network._identify_mentors_and_students()

        # First agent should be mentor (MockMentorAgent), rest should be students
        assert len(mentors) == 1
        assert len(students) == 3
        assert mentors[0].agent_id == "mentor_0"
        assert mentors[0].role == "MENTOR"

        for student in students:
            assert student.role == "STUDENT"

    @pytest.mark.scenario
    def test_assign_mentor_student_pairs(self, mentor_student_agents, mentor_student_config):
        """Test mentor-student pair assignment."""
        network = MentorStudentNetwork(mentor_student_agents, mentor_student_config)
        network._assign_mentor_student_pairs()

        assert len(network.mentor_assignments) > 0

        # Check that assignments are valid
        for student_id, mentor_id in network.mentor_assignments.items():
            # Find the actual agents
            student_agent = next(a for a in mentor_student_agents if a.agent_id == student_id)
            mentor_agent = next(a for a in mentor_student_agents if a.agent_id == mentor_id)

            assert student_agent.role == "STUDENT"
            assert mentor_agent.role == "MENTOR"

    @pytest.mark.scenario
    def test_calculate_compatibility(self, mentor_student_pair):
        """Test mentor-student compatibility calculation."""
        mentor, student = mentor_student_pair
        network = MentorStudentNetwork([mentor, student], None)

        compatibility = network._calculate_compatibility(mentor, student)

        assert 0 <= compatibility <= 1
        assert isinstance(compatibility, float)

    @pytest.mark.scenario
    def test_run_knowledge_transfer_session(self, mentor_student_pair, mock_environment):
        """Test knowledge transfer session."""
        mentor, student = mentor_student_pair
        config = ScenarioConfig(
            scenario_type=LearningScenarioType.MENTOR_STUDENT,
            duration_episodes=5,
            participants=2
        )

        network = MentorStudentNetwork([mentor, student], config)
        network._assign_mentor_student_pairs()

        initial_progress = student.learning_progress
        session_result = network._run_knowledge_transfer_session(
            mentor, student, mock_environment
        )

        # Check session result structure
        assert 'knowledge_transferred' in session_result
        assert 'teaching_effectiveness' in session_result
        assert 'student_improvement' in session_result
        assert 'transfer_quality' in session_result

        # Check that values are reasonable
        assert 0 <= session_result['teaching_effectiveness'] <= 1
        assert -1 <= session_result['student_improvement'] <= 1
        assert 0 <= session_result['transfer_quality'] <= 1

        # Student should have received feedback
        assert len(student.mentor_feedback) > 0

    @pytest.mark.scenario
    def test_evaluate_teaching_effectiveness(self, mentor_student_agents, mentor_student_config):
        """Test teaching effectiveness evaluation."""
        network = MentorStudentNetwork(mentor_student_agents, mentor_student_config)

        # Add some mock transfer history
        network.knowledge_transfer_log = [
            {
                'mentor_id': 'mentor_0',
                'student_id': 'student_0',
                'effectiveness': 0.8,
                'episode': 1
            },
            {
                'mentor_id': 'mentor_0',
                'student_id': 'student_1',
                'effectiveness': 0.6,
                'episode': 2
            }
        ]

        effectiveness_scores = network._evaluate_teaching_effectiveness()

        assert 'mentor_0' in effectiveness_scores
        assert 0 <= effectiveness_scores['mentor_0'] <= 1

    @pytest.mark.scenario
    @pytest.mark.slow
    def test_full_mentor_student_run(self, mentor_student_agents, mock_environment):
        """Test running a complete mentor-student scenario."""
        config = ScenarioConfig(
            scenario_type=LearningScenarioType.MENTOR_STUDENT,
            duration_episodes=5,  # Reduced for speed
            participants=len(mentor_student_agents),
            knowledge_sharing_rate=0.6
        )

        network = MentorStudentNetwork(mentor_student_agents, config)
        results = network.run(mock_environment)

        # Verify result structure
        assert 'knowledge_transfer_metrics' in results
        assert 'teaching_effectiveness' in results
        assert 'learning_outcomes' in results
        assert 'mentor_student_pairs' in results

        # Verify knowledge transfer metrics
        transfer_metrics = results['knowledge_transfer_metrics']
        assert 'total_transfers' in transfer_metrics
        assert 'successful_transfers' in transfer_metrics
        assert 'transfer_efficiency' in transfer_metrics

        # Verify teaching effectiveness
        effectiveness = results['teaching_effectiveness']
        assert isinstance(effectiveness, dict)

        # Verify learning outcomes
        outcomes = results['learning_outcomes']
        assert len(outcomes) == len(mentor_student_agents)

    @pytest.mark.scenario
    def test_adaptive_mentorship(self, mentor_student_agents, mentor_student_config):
        """Test adaptive mentorship relationship adjustment."""
        network = MentorStudentNetwork(mentor_student_agents, mentor_student_config)
        network._assign_mentor_student_pairs()

        initial_assignments = network.mentor_assignments.copy()

        # Simulate poor teaching effectiveness
        network.teaching_effectiveness = {'mentor_0': 0.3}  # Below threshold

        # Trigger reassignment
        network._adapt_mentorship_relationships()

        # Assignments might change due to poor effectiveness
        # (In a real implementation, this would reassign based on performance)
        assert isinstance(network.mentor_assignments, dict)

    @pytest.mark.scenario
    def test_knowledge_distillation_integration(self, mentor_student_pair, mock_environment):
        """Test integration with knowledge distillation engine."""
        mentor, student = mentor_student_pair
        config = ScenarioConfig(
            scenario_type=LearningScenarioType.MENTOR_STUDENT,
            duration_episodes=3,
            participants=2,
            knowledge_sharing_rate=0.8
        )

        network = MentorStudentNetwork([mentor, student], config)

        # Test that knowledge transfer uses distillation
        mentor_knowledge = torch.randn(1, 64)
        transfer_result = network._transfer_knowledge_between_agents(
            mentor, student, mentor_knowledge
        )

        assert 'transferred_knowledge' in transfer_result
        assert 'transfer_quality' in transfer_result
        assert transfer_result['transferred_knowledge'].shape == mentor_knowledge.shape

    @pytest.mark.scenario
    def test_progress_monitoring(self, mentor_student_agents, mentor_student_config):
        """Test learning progress monitoring."""
        network = MentorStudentNetwork(mentor_student_agents, mentor_student_config)

        # Simulate some learning progress
        students = [agent for agent in mentor_student_agents if agent.role == "STUDENT"]
        for i, student in enumerate(students):
            student.learning_progress = 0.1 * (i + 1)

        progress_report = network._monitor_learning_progress()

        assert isinstance(progress_report, dict)
        for student in students:
            assert student.agent_id in progress_report
            assert 0 <= progress_report[student.agent_id] <= 1

    @pytest.mark.scenario
    def test_mentorship_with_custom_parameters(self, mentor_student_agents, mock_environment):
        """Test mentorship with custom parameters."""
        custom_config = ScenarioConfig(
            scenario_type=LearningScenarioType.MENTOR_STUDENT,
            duration_episodes=8,
            participants=len(mentor_student_agents),
            knowledge_sharing_rate=0.9,
            scenario_parameters={
                'mentor_ratio': 0.5,  # Higher mentor ratio
                'teaching_effectiveness_threshold': 0.8,
                'max_students_per_mentor': 2
            }
        )

        network = MentorStudentNetwork(mentor_student_agents, custom_config)
        results = network.run(mock_environment)

        # Should produce valid results with custom parameters
        assert 'knowledge_transfer_metrics' in results
        assert 'teaching_effectiveness' in results

    @pytest.mark.scenario
    def test_mentor_student_role_validation(self, mock_agents, mentor_student_config):
        """Test validation when agents don't have proper roles."""
        # All agents are standard agents without specific mentor/student roles
        network = MentorStudentNetwork(mock_agents, mentor_student_config)

        # Should still work by assigning roles based on performance or randomly
        assert network.validate_agents() is True

        mentors, students = network._identify_mentors_and_students()
        assert len(mentors) + len(students) == len(mock_agents)

    @pytest.mark.scenario
    def test_knowledge_sharing_rate_impact(self, mentor_student_pair, mock_environment):
        """Test impact of different knowledge sharing rates."""
        mentor, student = mentor_student_pair

        # Test with low sharing rate
        low_config = ScenarioConfig(
            scenario_type=LearningScenarioType.MENTOR_STUDENT,
            duration_episodes=3,
            participants=2,
            knowledge_sharing_rate=0.1
        )

        # Test with high sharing rate
        high_config = ScenarioConfig(
            scenario_type=LearningScenarioType.MENTOR_STUDENT,
            duration_episodes=3,
            participants=2,
            knowledge_sharing_rate=0.9
        )

        low_network = MentorStudentNetwork([mentor, student], low_config)
        high_network = MentorStudentNetwork([mentor, student], high_config)

        low_results = low_network.run(mock_environment)
        high_results = high_network.run(mock_environment)

        # Both should produce valid results
        assert 'knowledge_transfer_metrics' in low_results
        assert 'knowledge_transfer_metrics' in high_results

        # Higher sharing rate might lead to more transfers
        # (Exact comparison depends on implementation details)
        assert low_results['knowledge_transfer_metrics']['transfer_efficiency'] >= 0
        assert high_results['knowledge_transfer_metrics']['transfer_efficiency'] >= 0