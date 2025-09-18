"""Integration tests for scenario orchestrator."""

import pytest
import torch

from multi_agent_collab_learning.scenarios.orchestrator import ScenarioOrchestrator
from multi_agent_collab_learning.scenarios.competitive import CompetitiveLearningTournament
from multi_agent_collab_learning.scenarios.mentor_student import MentorStudentNetwork
from multi_agent_collab_learning.scenarios.collaborative import CollaborativeResearchEnvironment
from multi_agent_collab_learning.core.config import ScenarioConfig, LearningScenarioType


class TestScenarioOrchestrator:
    """Test ScenarioOrchestrator integration."""

    @pytest.fixture
    def orchestrator(self, mock_agents):
        """Create a scenario orchestrator for testing."""
        return ScenarioOrchestrator(mock_agents)

    @pytest.fixture
    def sample_scenarios(self, mock_agents):
        """Create sample scenarios for testing."""
        tournament_config = ScenarioConfig(
            scenario_type=LearningScenarioType.COMPETITIVE_TOURNAMENT,
            duration_episodes=5,
            participants=len(mock_agents)
        )

        mentor_config = ScenarioConfig(
            scenario_type=LearningScenarioType.MENTOR_STUDENT,
            duration_episodes=5,
            participants=len(mock_agents)
        )

        research_config = ScenarioConfig(
            scenario_type=LearningScenarioType.COLLABORATIVE_RESEARCH,
            duration_episodes=5,
            participants=len(mock_agents)
        )

        tournament = CompetitiveLearningTournament(mock_agents, tournament_config)
        mentorship = MentorStudentNetwork(mock_agents, mentor_config)
        research = CollaborativeResearchEnvironment(mock_agents, research_config)

        return {
            'tournament': tournament,
            'mentorship': mentorship,
            'research': research
        }

    @pytest.mark.integration
    def test_orchestrator_initialization(self, mock_agents):
        """Test orchestrator initialization."""
        orchestrator = ScenarioOrchestrator(mock_agents)

        assert orchestrator.agents == mock_agents
        assert orchestrator.scenarios == {}
        assert orchestrator.execution_history == []
        assert orchestrator.cross_scenario_learning == []

    @pytest.mark.integration
    def test_register_scenario(self, orchestrator, sample_scenarios):
        """Test scenario registration."""
        tournament = sample_scenarios['tournament']

        orchestrator.register_scenario('test_tournament', tournament)

        assert 'test_tournament' in orchestrator.scenarios
        assert orchestrator.scenarios['test_tournament'] == tournament

    @pytest.mark.integration
    def test_register_multiple_scenarios(self, orchestrator, sample_scenarios):
        """Test registering multiple scenarios."""
        for name, scenario in sample_scenarios.items():
            orchestrator.register_scenario(name, scenario)

        assert len(orchestrator.scenarios) == 3
        for name in sample_scenarios.keys():
            assert name in orchestrator.scenarios

    @pytest.mark.integration
    def test_run_single_scenario(self, orchestrator, sample_scenarios, mock_environment):
        """Test running a single scenario."""
        tournament = sample_scenarios['tournament']
        orchestrator.register_scenario('tournament', tournament)

        results = orchestrator.run_scenario('tournament', mock_environment)

        # Check result structure
        assert 'scenario_name' in results
        assert 'scenario_results' in results
        assert 'execution_time' in results
        assert 'agents_involved' in results

        assert results['scenario_name'] == 'tournament'
        assert len(results['agents_involved']) == len(orchestrator.agents)

        # Check that history was updated
        assert len(orchestrator.execution_history) == 1
        assert orchestrator.execution_history[0]['scenario_name'] == 'tournament'

    @pytest.mark.integration
    def test_run_nonexistent_scenario(self, orchestrator, mock_environment):
        """Test running a scenario that doesn't exist."""
        with pytest.raises(ValueError, match="Scenario 'nonexistent' not found"):
            orchestrator.run_scenario('nonexistent', mock_environment)

    @pytest.mark.integration
    @pytest.mark.slow
    def test_run_scenario_suite(self, orchestrator, sample_scenarios, mock_environment):
        """Test running a suite of scenarios."""
        # Register all scenarios
        for name, scenario in sample_scenarios.items():
            orchestrator.register_scenario(name, scenario)

        scenario_sequence = ['tournament', 'mentorship', 'research']
        suite_results = orchestrator.run_scenario_suite(mock_environment, scenario_sequence)

        # Check suite result structure
        assert 'scenario_results' in suite_results
        assert 'cross_scenario_learning' in suite_results
        assert 'emergent_capabilities' in suite_results
        assert 'execution_summary' in suite_results

        # Check that all scenarios were executed
        scenario_results = suite_results['scenario_results']
        assert len(scenario_results) == 3

        for scenario_name in scenario_sequence:
            assert scenario_name in scenario_results
            result = scenario_results[scenario_name]
            assert 'scenario_results' in result
            assert 'execution_time' in result

        # Check execution history
        assert len(orchestrator.execution_history) == 3

    @pytest.mark.integration
    def test_analyze_cross_scenario_learning(self, orchestrator, sample_scenarios, mock_environment):
        """Test cross-scenario learning analysis."""
        # Register and run multiple scenarios
        for name, scenario in sample_scenarios.items():
            orchestrator.register_scenario(name, scenario)

        orchestrator.run_scenario('tournament', mock_environment)
        orchestrator.run_scenario('mentorship', mock_environment)

        # Analyze cross-scenario learning
        cross_learning = orchestrator._analyze_cross_scenario_learning()

        assert isinstance(cross_learning, list)
        # Should detect learning transfer between scenarios
        if len(cross_learning) > 0:
            for event in cross_learning:
                assert 'source_scenario' in event
                assert 'target_scenario' in event
                assert 'learning_type' in event
                assert 'agents_involved' in event

    @pytest.mark.integration
    def test_detect_emergent_capabilities(self, orchestrator, sample_scenarios, mock_environment):
        """Test emergent capability detection."""
        # Register and run scenarios
        for name, scenario in sample_scenarios.items():
            orchestrator.register_scenario(name, scenario)

        orchestrator.run_scenario('tournament', mock_environment)
        orchestrator.run_scenario('research', mock_environment)

        # Detect emergent capabilities
        emergent_caps = orchestrator._detect_emergent_capabilities()

        assert isinstance(emergent_caps, list)
        # May detect emergent capabilities across scenarios
        for capability in emergent_caps:
            assert 'capability_type' in capability
            assert 'evidence' in capability
            assert 'scenarios_involved' in capability

    @pytest.mark.integration
    def test_get_agent_performance_across_scenarios(self, orchestrator, sample_scenarios, mock_environment):
        """Test tracking agent performance across scenarios."""
        # Register and run multiple scenarios
        for name, scenario in sample_scenarios.items():
            orchestrator.register_scenario(name, scenario)

        orchestrator.run_scenario('tournament', mock_environment)
        orchestrator.run_scenario('mentorship', mock_environment)

        # Get performance summary
        performance_summary = orchestrator.get_agent_performance_across_scenarios()

        assert isinstance(performance_summary, dict)
        for agent in orchestrator.agents:
            assert agent.agent_id in performance_summary
            agent_perf = performance_summary[agent.agent_id]
            assert 'scenarios_participated' in agent_perf
            assert 'average_performance' in agent_perf
            assert 'performance_trend' in agent_perf

    @pytest.mark.integration
    def test_scenario_dependencies(self, orchestrator, sample_scenarios, mock_environment):
        """Test scenario execution with dependencies."""
        # Register scenarios
        for name, scenario in sample_scenarios.items():
            orchestrator.register_scenario(name, scenario)

        # Define scenario dependencies
        dependencies = {
            'research': ['tournament', 'mentorship']  # Research depends on tournament and mentorship
        }

        suite_results = orchestrator.run_scenario_suite(
            mock_environment,
            scenario_sequence=['research'],  # Only research, but should run dependencies first
            dependencies=dependencies
        )

        # Should have executed all dependent scenarios
        executed_scenarios = [h['scenario_name'] for h in orchestrator.execution_history]
        assert 'tournament' in executed_scenarios
        assert 'mentorship' in executed_scenarios
        assert 'research' in executed_scenarios

    @pytest.mark.integration
    def test_orchestrator_with_different_environments(self, orchestrator, sample_scenarios):
        """Test orchestrator with different environment types."""
        tournament = sample_scenarios['tournament']
        research = sample_scenarios['research']

        orchestrator.register_scenario('tournament', tournament)
        orchestrator.register_scenario('research', research)

        # Test with competitive environment
        from tests.fixtures.environments import create_mock_environment
        competitive_env = create_mock_environment("competitive", deterministic=True)
        tournament_results = orchestrator.run_scenario('tournament', competitive_env)

        # Test with research environment
        research_env = create_mock_environment("research", deterministic=True)
        research_results = orchestrator.run_scenario('research', research_env)

        # Both should produce valid results
        assert 'scenario_results' in tournament_results
        assert 'scenario_results' in research_results

    @pytest.mark.integration
    def test_concurrent_scenario_analysis(self, orchestrator, sample_scenarios, mock_environment):
        """Test analyzing scenarios that could run concurrently."""
        # Register scenarios
        for name, scenario in sample_scenarios.items():
            orchestrator.register_scenario(name, scenario)

        # Run scenarios
        orchestrator.run_scenario('tournament', mock_environment)
        orchestrator.run_scenario('mentorship', mock_environment)

        # Analyze potential for concurrent execution
        concurrent_analysis = orchestrator._analyze_concurrent_execution_potential()

        assert isinstance(concurrent_analysis, dict)
        assert 'compatible_pairs' in concurrent_analysis
        assert 'resource_conflicts' in concurrent_analysis

    @pytest.mark.integration
    def test_orchestrator_state_persistence(self, orchestrator, sample_scenarios, mock_environment):
        """Test that orchestrator maintains state across scenario executions."""
        tournament = sample_scenarios['tournament']
        orchestrator.register_scenario('tournament', tournament)

        # Run scenario multiple times
        orchestrator.run_scenario('tournament', mock_environment)
        orchestrator.run_scenario('tournament', mock_environment)

        # Should maintain history
        assert len(orchestrator.execution_history) == 2

        # Should track cumulative cross-scenario learning
        assert isinstance(orchestrator.cross_scenario_learning, list)

    @pytest.mark.integration
    def test_error_handling_in_suite(self, orchestrator, mock_environment):
        """Test error handling when scenario fails in suite."""
        # Create a scenario that will fail
        class FailingScenario:
            def __init__(self):
                self.agents = orchestrator.agents

            def validate_agents(self):
                return True

            def run(self, environment):
                raise RuntimeError("Intentional test failure")

        failing_scenario = FailingScenario()
        orchestrator.register_scenario('failing', failing_scenario)

        # Should handle failure gracefully
        with pytest.raises(RuntimeError, match="Intentional test failure"):
            orchestrator.run_scenario('failing', mock_environment)

    @pytest.mark.integration
    def test_orchestrator_reset(self, orchestrator, sample_scenarios, mock_environment):
        """Test orchestrator reset functionality."""
        # Register and run scenarios
        tournament = sample_scenarios['tournament']
        orchestrator.register_scenario('tournament', tournament)
        orchestrator.run_scenario('tournament', mock_environment)

        # Verify state exists
        assert len(orchestrator.execution_history) > 0

        # Reset orchestrator
        orchestrator.reset()

        # State should be cleared
        assert len(orchestrator.execution_history) == 0
        assert len(orchestrator.cross_scenario_learning) == 0
        assert len(orchestrator.scenarios) == 0  # Scenarios also cleared