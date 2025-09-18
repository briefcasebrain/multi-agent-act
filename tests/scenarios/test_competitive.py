"""Tests for competitive learning tournament scenarios."""

import pytest
import torch

from multi_agent_collab_learning.scenarios.competitive import CompetitiveLearningTournament
from multi_agent_collab_learning.core.config import ScenarioConfig, LearningScenarioType


class TestCompetitiveLearningTournament:
    """Test CompetitiveLearningTournament class."""

    @pytest.mark.scenario
    def test_tournament_initialization(self, competitive_agents, tournament_config):
        """Test tournament initialization."""
        tournament = CompetitiveLearningTournament(competitive_agents, tournament_config)

        assert tournament.agents == competitive_agents
        assert tournament.config == tournament_config
        assert tournament.elo_ratings == {}
        assert tournament.tournament_bracket == []
        assert tournament.match_history == []

    @pytest.mark.scenario
    def test_validate_agents_success(self, competitive_agents, tournament_config):
        """Test successful agent validation."""
        tournament = CompetitiveLearningTournament(competitive_agents, tournament_config)
        assert tournament.validate_agents() is True

    @pytest.mark.scenario
    def test_validate_agents_insufficient_count(self, mock_agents, tournament_config):
        """Test agent validation with insufficient agent count."""
        # Tournament config expects 6 participants, but we only have 4
        tournament = CompetitiveLearningTournament(mock_agents, tournament_config)
        assert tournament.validate_agents() is False

    @pytest.mark.scenario
    def test_validate_agents_missing_attributes(self, tournament_config):
        """Test agent validation with missing required attributes."""
        # Create agents missing required attributes
        incomplete_agents = []
        for i in range(6):
            agent = type('IncompleteAgent', (), {'agent_id': f'agent_{i}'})()
            incomplete_agents.append(agent)

        tournament = CompetitiveLearningTournament(incomplete_agents, tournament_config)
        assert tournament.validate_agents() is False

    @pytest.mark.scenario
    def test_initialize_elo_ratings(self, competitive_agents, tournament_config):
        """Test ELO rating initialization."""
        tournament = CompetitiveLearningTournament(competitive_agents, tournament_config)
        tournament._initialize_elo_ratings()

        assert len(tournament.elo_ratings) == len(competitive_agents)
        for agent in competitive_agents:
            assert agent.agent_id in tournament.elo_ratings
            assert tournament.elo_ratings[agent.agent_id] == 1500.0  # Default ELO

    @pytest.mark.scenario
    def test_generate_tournament_bracket(self, competitive_agents, tournament_config):
        """Test tournament bracket generation."""
        tournament = CompetitiveLearningTournament(competitive_agents, tournament_config)
        tournament._initialize_elo_ratings()
        tournament._generate_tournament_bracket()

        assert len(tournament.tournament_bracket) > 0

        # Verify all agents appear in bracket
        all_bracket_agents = set()
        for match in tournament.tournament_bracket:
            all_bracket_agents.update(match)

        agent_ids = {agent.agent_id for agent in competitive_agents}
        assert all_bracket_agents == agent_ids

    @pytest.mark.scenario
    def test_run_single_match(self, competitive_agents, tournament_config, competitive_environment):
        """Test running a single match."""
        tournament = CompetitiveLearningTournament(competitive_agents, tournament_config)
        tournament._initialize_elo_ratings()

        agent1 = competitive_agents[0]
        agent2 = competitive_agents[1]

        match_result = tournament._run_single_match(agent1, agent2, competitive_environment)

        assert 'winner' in match_result
        assert 'loser' in match_result
        assert 'scores' in match_result
        assert 'episode_count' in match_result

        # Winner should be one of the two agents
        assert match_result['winner'] in [agent1.agent_id, agent2.agent_id]
        assert match_result['loser'] in [agent1.agent_id, agent2.agent_id]
        assert match_result['winner'] != match_result['loser']

    @pytest.mark.scenario
    def test_update_elo_ratings(self, competitive_agents, tournament_config):
        """Test ELO rating updates."""
        tournament = CompetitiveLearningTournament(competitive_agents, tournament_config)
        tournament._initialize_elo_ratings()

        agent1 = competitive_agents[0]
        agent2 = competitive_agents[1]

        initial_rating1 = tournament.elo_ratings[agent1.agent_id]
        initial_rating2 = tournament.elo_ratings[agent2.agent_id]

        # Agent 1 wins
        tournament._update_elo_ratings(agent1.agent_id, agent2.agent_id)

        # Winner should gain rating, loser should lose rating
        assert tournament.elo_ratings[agent1.agent_id] > initial_rating1
        assert tournament.elo_ratings[agent2.agent_id] < initial_rating2

        # Total rating change should be zero (conservation)
        total_change = (
            (tournament.elo_ratings[agent1.agent_id] - initial_rating1) +
            (tournament.elo_ratings[agent2.agent_id] - initial_rating2)
        )
        assert abs(total_change) < 1e-10  # Should be zero within floating point precision

    @pytest.mark.scenario
    def test_detect_competitive_behavior(self, competitive_agents, tournament_config):
        """Test competitive behavior detection."""
        tournament = CompetitiveLearningTournament(competitive_agents, tournament_config)

        # Create mock match history
        tournament.match_history = [
            {'winner': 'agent_0', 'loser': 'agent_1', 'scores': {'agent_0': 0.8, 'agent_1': 0.2}},
            {'winner': 'agent_0', 'loser': 'agent_2', 'scores': {'agent_0': 0.9, 'agent_2': 0.1}},
            {'winner': 'agent_1', 'loser': 'agent_2', 'scores': {'agent_1': 0.6, 'agent_2': 0.4}},
        ]

        behaviors = tournament._detect_competitive_behavior()

        assert isinstance(behaviors, dict)
        assert len(behaviors) == len(competitive_agents)

        # All agents should have a behavior classification
        for agent in competitive_agents:
            assert agent.agent_id in behaviors
            assert behaviors[agent.agent_id] in ['aggressive', 'defensive', 'balanced', 'adaptive']

    @pytest.mark.scenario
    @pytest.mark.slow
    def test_full_tournament_run(self, competitive_agents, tournament_config, competitive_environment):
        """Test running a complete tournament."""
        # Use smaller tournament for faster testing
        quick_config = ScenarioConfig(
            scenario_type=LearningScenarioType.COMPETITIVE_TOURNAMENT,
            duration_episodes=5,  # Reduced for speed
            participants=len(competitive_agents),
            scenario_parameters={'tournament_format': 'bracket'}
        )

        tournament = CompetitiveLearningTournament(competitive_agents, quick_config)
        results = tournament.run(competitive_environment)

        # Verify result structure
        assert 'tournament_winner' in results
        assert 'final_rankings' in results
        assert 'matches_played' in results
        assert 'competitive_behaviors' in results
        assert 'learning_outcomes' in results

        # Verify winner is valid
        assert results['tournament_winner'] in [agent.agent_id for agent in competitive_agents]

        # Verify rankings
        rankings = results['final_rankings']
        assert len(rankings) == len(competitive_agents)
        for agent in competitive_agents:
            assert agent.agent_id in rankings
            assert isinstance(rankings[agent.agent_id], (int, float))

        # Verify match count
        assert results['matches_played'] > 0

        # Verify learning outcomes
        outcomes = results['learning_outcomes']
        assert len(outcomes) == len(competitive_agents)

    @pytest.mark.scenario
    def test_tournament_with_custom_parameters(self, competitive_agents, competitive_environment):
        """Test tournament with custom parameters."""
        custom_config = ScenarioConfig(
            scenario_type=LearningScenarioType.COMPETITIVE_TOURNAMENT,
            duration_episodes=10,
            participants=len(competitive_agents),
            scenario_parameters={
                'tournament_format': 'round_robin',
                'elo_k_factor': 16,  # Custom K-factor
                'min_matches_per_agent': 3
            }
        )

        tournament = CompetitiveLearningTournament(competitive_agents, custom_config)
        results = tournament.run(competitive_environment)

        # Should still produce valid results with custom parameters
        assert 'tournament_winner' in results
        assert 'final_rankings' in results
        assert results['matches_played'] > 0

    @pytest.mark.scenario
    def test_tournament_deterministic_behavior(self, competitive_agents, tournament_config, deterministic_environment):
        """Test that tournament produces deterministic results."""
        tournament1 = CompetitiveLearningTournament(competitive_agents, tournament_config)
        tournament2 = CompetitiveLearningTournament(competitive_agents, tournament_config)

        env1 = create_mock_environment("competitive", deterministic=True)
        env2 = create_mock_environment("competitive", deterministic=True)

        # Use reduced episodes for faster testing
        quick_config = ScenarioConfig(
            scenario_type=LearningScenarioType.COMPETITIVE_TOURNAMENT,
            duration_episodes=3,
            participants=len(competitive_agents)
        )

        tournament1.config = quick_config
        tournament2.config = quick_config

        results1 = tournament1.run(env1)
        results2 = tournament2.run(env2)

        # With deterministic environment and agents, results should be identical
        assert results1['tournament_winner'] == results2['tournament_winner']
        assert results1['matches_played'] == results2['matches_played']

    @pytest.mark.scenario
    def test_elo_rating_bounds(self, competitive_agents, tournament_config):
        """Test that ELO ratings stay within reasonable bounds."""
        tournament = CompetitiveLearningTournament(competitive_agents, tournament_config)
        tournament._initialize_elo_ratings()

        # Simulate many updates
        agent1_id = competitive_agents[0].agent_id
        agent2_id = competitive_agents[1].agent_id

        for _ in range(100):
            tournament._update_elo_ratings(agent1_id, agent2_id)

        # Ratings should stay within reasonable bounds
        for rating in tournament.elo_ratings.values():
            assert 0 < rating < 5000  # Reasonable ELO range

    @pytest.mark.scenario
    def test_bracket_format_validation(self, competitive_agents):
        """Test validation of tournament bracket formats."""
        valid_config = ScenarioConfig(
            scenario_type=LearningScenarioType.COMPETITIVE_TOURNAMENT,
            duration_episodes=10,
            participants=len(competitive_agents),
            scenario_parameters={'tournament_format': 'bracket'}
        )

        tournament = CompetitiveLearningTournament(competitive_agents, valid_config)
        assert tournament.validate_agents() is True

        # Test with invalid format (should still work with default handling)
        invalid_config = ScenarioConfig(
            scenario_type=LearningScenarioType.COMPETITIVE_TOURNAMENT,
            duration_episodes=10,
            participants=len(competitive_agents),
            scenario_parameters={'tournament_format': 'invalid_format'}
        )

        tournament_invalid = CompetitiveLearningTournament(competitive_agents, invalid_config)
        # Should not crash, should use default format
        assert tournament_invalid.validate_agents() is True


# Import required for deterministic environment test
from tests.fixtures.environments import create_mock_environment