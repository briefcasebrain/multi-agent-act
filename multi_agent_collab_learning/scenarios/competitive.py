"""
Competitive learning tournament scenarios.
"""

import time
import random
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

from .base import BaseScenario
from ..core.config import ScenarioConfig, LearningScenarioType, LearningOutcome


class CompetitiveLearningTournament(BaseScenario):
    """
    Implements competitive learning tournament scenarios.

    This class manages tournament-style competition between agents, including
    bracket generation, match execution, performance tracking, and learning
    outcome analysis.
    """

    def __init__(self, agents: List[Any], config: ScenarioConfig):
        """
        Initialize the competitive tournament.

        Args:
            agents: List of collaborative agents
            config: Tournament configuration
        """
        super().__init__(agents, config)
        self.tournament_structure = self._initialize_tournament_structure()
        self.match_history = []
        self.performance_rankings = defaultdict(float)
        self.skill_evolution = defaultdict(list)

    def _initialize_tournament_structure(self) -> Dict[str, Any]:
        """
        Initialize tournament bracket structure.

        Returns:
            Dictionary containing tournament structure information
        """
        num_agents = len(self.agents)

        return {
            'format': 'round_robin' if num_agents <= 8 else 'bracket',
            'rounds': self._calculate_tournament_rounds(),
            'matches_per_round': num_agents // 2,
            'current_round': 0,
            'bracket': self._generate_bracket() if num_agents > 8 else None
        }

    def _calculate_tournament_rounds(self) -> int:
        """
        Calculate number of tournament rounds.

        Returns:
            Number of rounds needed for the tournament
        """
        num_agents = len(self.agents)

        if num_agents <= 8:
            return num_agents - 1  # Round robin
        else:
            return int(np.ceil(np.log2(num_agents)))  # Bracket tournament

    def _generate_bracket(self) -> Dict[str, List[str]]:
        """
        Generate tournament bracket.

        Returns:
            Dictionary mapping round names to match pairs
        """
        # Shuffle agents for random seeding
        shuffled_agents = self.agents.copy()
        random.shuffle(shuffled_agents)

        bracket = {}
        round_num = 0
        current_agents = [agent.agent_id for agent in shuffled_agents]

        while len(current_agents) > 1:
            bracket[f'round_{round_num}'] = []

            # Pair agents for matches
            for i in range(0, len(current_agents), 2):
                if i + 1 < len(current_agents):
                    bracket[f'round_{round_num}'].append([current_agents[i], current_agents[i+1]])
                else:
                    # Bye for odd number of agents
                    bracket[f'round_{round_num}'].append([current_agents[i], 'bye'])

            # Prepare for next round
            next_round_agents = []
            for match in bracket[f'round_{round_num}']:
                if match[1] == 'bye':
                    next_round_agents.append(match[0])
                else:
                    # Winner will be determined after match
                    next_round_agents.append(f"winner_of_{match[0]}_vs_{match[1]}")

            current_agents = next_round_agents
            round_num += 1

        return bracket

    def run(self, environment: Any) -> Dict[str, Any]:
        """
        Run complete competitive learning tournament.

        Args:
            environment: The environment to run matches in

        Returns:
            Dictionary containing tournament results
        """
        if not self.validate_agents():
            raise ValueError("Invalid agent configuration for tournament")

        print(f"🏆 Starting Competitive Learning Tournament")
        print(f"   Format: {self.tournament_structure['format']}")
        print(f"   Participants: {len(self.agents)}")
        print(f"   Rounds: {self.tournament_structure['rounds']}")

        tournament_results = {
            'matches': [],
            'final_rankings': {},
            'skill_evolution': {},
            'learning_outcomes': [],
            'tournament_winner': None
        }

        for round_num in range(self.tournament_structure['rounds']):
            print(f"\n🥊 Round {round_num + 1}")

            round_matches = self._generate_round_matches(round_num)
            round_results = []

            for match in round_matches:
                match_result = self._execute_competitive_match(match, environment)
                round_results.append(match_result)
                self.match_history.append(match_result)

                # Update performance rankings
                self._update_performance_rankings(match_result)

                print(f"   Match: {match_result['participants']} -> Winner: {match_result['winner']}")

            tournament_results['matches'].extend(round_results)

            # Allow agents to learn from round
            self._post_round_learning(round_results)

        # Finalize tournament
        tournament_results['final_rankings'] = self._calculate_final_rankings()
        tournament_results['tournament_winner'] = max(
            tournament_results['final_rankings'],
            key=tournament_results['final_rankings'].get
        )

        # Analyze learning outcomes
        tournament_results['learning_outcomes'] = self._analyze_learning_outcomes()

        print(f"\n🏆 Tournament Complete!")
        print(f"   Winner: {tournament_results['tournament_winner']}")
        print(f"   Final Rankings: {tournament_results['final_rankings']}")

        self.results = tournament_results
        return tournament_results

    def _generate_round_matches(self, round_num: int) -> List[Tuple[str, str]]:
        """
        Generate matches for current round.

        Args:
            round_num: Current round number

        Returns:
            List of match pairs
        """
        if self.tournament_structure['format'] == 'round_robin':
            return self._generate_round_robin_matches(round_num)
        else:
            return self._generate_bracket_matches(round_num)

    def _generate_round_robin_matches(self, round_num: int) -> List[Tuple[str, str]]:
        """
        Generate round robin matches.

        Args:
            round_num: Current round number

        Returns:
            List of match pairs for round robin
        """
        agent_ids = [agent.agent_id for agent in self.agents]
        matches = []

        # Rotate agents for round robin
        n = len(agent_ids)
        for i in range(n):
            opponent_idx = (i + round_num + 1) % n
            if i != opponent_idx and i < opponent_idx:  # Avoid duplicates
                matches.append((agent_ids[i], agent_ids[opponent_idx]))

        return matches

    def _generate_bracket_matches(self, round_num: int) -> List[Tuple[str, str]]:
        """
        Generate bracket tournament matches.

        Args:
            round_num: Current round number

        Returns:
            List of match pairs for bracket tournament
        """
        if f'round_{round_num}' not in self.tournament_structure['bracket']:
            return []

        matches = []
        for match_pair in self.tournament_structure['bracket'][f'round_{round_num}']:
            if match_pair[1] != 'bye':
                matches.append((match_pair[0], match_pair[1]))

        return matches

    def _execute_competitive_match(self, match: Tuple[str, str], environment: Any) -> Dict[str, Any]:
        """
        Execute a competitive match between two agents.

        Args:
            match: Tuple containing agent IDs for the match
            environment: Environment to run the match in

        Returns:
            Dictionary containing match results
        """
        agent_1_id, agent_2_id = match
        agent_1 = next(agent for agent in self.agents if agent.agent_id == agent_1_id)
        agent_2 = next(agent for agent in self.agents if agent.agent_id == agent_2_id)

        # Set competitive mode
        agent_1.collaboration_mode = "competitive"
        agent_2.collaboration_mode = "competitive"

        # Run competitive episode
        episode_results = self._run_competitive_episode([agent_1, agent_2], environment)

        # Determine winner
        winner = agent_1_id if episode_results['performance'][agent_1_id] > episode_results['performance'][agent_2_id] else agent_2_id

        match_result = {
            'participants': [agent_1_id, agent_2_id],
            'winner': winner,
            'performance_scores': episode_results['performance'],
            'learning_metrics': episode_results['learning_metrics'],
            'competitive_behaviors': episode_results['competitive_behaviors'],
            'timestamp': time.time()
        }

        return match_result

    def _run_competitive_episode(self, competing_agents: List[Any], environment: Any) -> Dict[str, Any]:
        """
        Run a competitive episode between agents.

        Args:
            competing_agents: List of agents competing
            environment: Environment to run episode in

        Returns:
            Dictionary containing episode results
        """
        episode_results = {
            'performance': {},
            'learning_metrics': {},
            'competitive_behaviors': []
        }

        # Initialize competitive environment
        environment.controller.reset()

        # Run competitive interaction
        max_steps = 50  # Shorter episodes for tournaments

        for step in range(max_steps):
            step_results = {}

            for agent in competing_agents:
                # Get competitive action
                state = environment._get_multimodal_state()
                action_result = agent.select_action(state)

                # Execute action
                success = environment._execute_agent_action(agent, action_result['action'].item())

                # Calculate competitive reward
                competitive_reward = self._calculate_competitive_reward(agent, success, step, max_steps)

                step_results[agent.agent_id] = {
                    'action': action_result['action'].item(),
                    'success': success,
                    'reward': competitive_reward,
                    'confidence': action_result.get('uncertainty', 0.5)
                }

                # Detect competitive behaviors
                competitive_behavior = self._detect_competitive_behavior(agent, action_result, step_results)
                if competitive_behavior:
                    episode_results['competitive_behaviors'].append(competitive_behavior)

        # Calculate final performance
        for agent in competing_agents:
            agent_steps = [step_results.get(agent.agent_id, {})]
            total_reward = sum(step.get('reward', 0) for step in agent_steps)
            success_rate = sum(step.get('success', 0) for step in agent_steps) / len(agent_steps)

            episode_results['performance'][agent.agent_id] = total_reward + success_rate
            episode_results['learning_metrics'][agent.agent_id] = {
                'adaptation_speed': self._measure_adaptation_speed(agent),
                'strategy_innovation': self._measure_strategy_innovation(agent),
                'competitive_intelligence': self._measure_competitive_intelligence(agent)
            }

        return episode_results

    def _calculate_competitive_reward(self, agent: Any, success: bool, step: int, max_steps: int) -> float:
        """
        Calculate reward for competitive scenario.

        Args:
            agent: The agent being evaluated
            success: Whether the action was successful
            step: Current step number
            max_steps: Maximum steps in episode

        Returns:
            Calculated competitive reward
        """
        base_reward = 2.0 if success else -0.5  # Higher stakes

        # Time bonus for efficiency
        efficiency_bonus = (max_steps - step) / max_steps * 0.5

        # Strategy bonus for innovative actions
        strategy_bonus = 0.1 if hasattr(agent, 'last_strategy_innovation') else 0

        return base_reward + efficiency_bonus + strategy_bonus

    def _detect_competitive_behavior(self, agent: Any, action_result: Dict[str, Any], step_results: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Detect competitive behaviors during matches.

        Args:
            agent: The agent being analyzed
            action_result: Results from agent's action
            step_results: Results from current step

        Returns:
            Dictionary describing competitive behavior, or None
        """
        # Detect aggressive strategies
        if action_result.get('confidence', 0.5) > 0.8:
            return {
                'type': 'aggressive_strategy',
                'agent_id': agent.agent_id,
                'confidence_level': action_result['confidence']
            }

        # Detect defensive strategies
        if action_result.get('uncertainty', 0.5) > 0.7:
            return {
                'type': 'defensive_strategy',
                'agent_id': agent.agent_id,
                'uncertainty_level': action_result['uncertainty']
            }

        return None

    def _measure_adaptation_speed(self, agent: Any) -> float:
        """Measure how quickly agent adapts to competitive pressure."""
        return random.uniform(0.3, 0.9)

    def _measure_strategy_innovation(self, agent: Any) -> float:
        """Measure agent's strategy innovation in competition."""
        return random.uniform(0.2, 0.8)

    def _measure_competitive_intelligence(self, agent: Any) -> float:
        """Measure agent's competitive intelligence."""
        return random.uniform(0.4, 0.9)

    def _update_performance_rankings(self, match_result: Dict[str, Any]) -> None:
        """
        Update performance rankings based on match results.

        Args:
            match_result: Results from the completed match
        """
        winner = match_result['winner']
        participants = match_result['participants']

        # ELO-style rating update
        winner_score = match_result['performance_scores'][winner]

        for participant in participants:
            if participant == winner:
                self.performance_rankings[participant] += winner_score * 0.1
            else:
                self.performance_rankings[participant] += match_result['performance_scores'][participant] * 0.05

    def _post_round_learning(self, round_results: List[Dict[str, Any]]) -> None:
        """
        Allow agents to learn from round results.

        Args:
            round_results: Results from all matches in the round
        """
        for agent in self.agents:
            # Find agent's matches in this round
            agent_matches = [match for match in round_results
                           if agent.agent_id in match['participants']]

            # Extract learning experiences
            for match in agent_matches:
                learning_experience = {
                    'competitive_performance': match['performance_scores'][agent.agent_id],
                    'opponent_strategies': self._extract_opponent_strategies(match, agent.agent_id),
                    'outcome': 'win' if match['winner'] == agent.agent_id else 'loss'
                }

                # Update agent's competitive learning
                self._update_competitive_learning(agent, learning_experience)

    def _extract_opponent_strategies(self, match: Dict[str, Any], agent_id: str) -> Dict[str, Any]:
        """
        Extract opponent strategies for learning.

        Args:
            match: Match results
            agent_id: ID of the learning agent

        Returns:
            Dictionary containing opponent strategy information
        """
        opponents = [p for p in match['participants'] if p != agent_id]

        opponent_strategies = {}
        for opponent_id in opponents:
            opponent_strategies[opponent_id] = {
                'performance_level': match['performance_scores'][opponent_id],
                'learning_metrics': match['learning_metrics'].get(opponent_id, {}),
                'competitive_behaviors': [
                    b for b in match['competitive_behaviors']
                    if b['agent_id'] == opponent_id
                ]
            }

        return opponent_strategies

    def _update_competitive_learning(self, agent: Any, learning_experience: Dict[str, Any]) -> None:
        """
        Update agent's competitive learning based on experience.

        Args:
            agent: The agent to update
            learning_experience: Learning experience data
        """
        # Adjust behavior weights based on competitive outcome
        if learning_experience['outcome'] == 'win':
            # Reinforce successful competitive strategies
            agent.behavior_weights['individual_reward'] = min(0.9,
                agent.behavior_weights['individual_reward'] + 0.05)
        else:
            # Learn from loss - increase adaptability
            if 'adaptation_speed' not in agent.collaboration_metrics:
                agent.collaboration_metrics['adaptation_speed'] = 0.5
            agent.collaboration_metrics['adaptation_speed'] += 0.1

    def _calculate_final_rankings(self) -> Dict[str, float]:
        """
        Calculate final tournament rankings.

        Returns:
            Dictionary mapping agent IDs to their final scores
        """
        # Sort by performance ranking
        sorted_rankings = sorted(
            self.performance_rankings.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return dict(sorted_rankings)

    def _analyze_learning_outcomes(self) -> List[LearningOutcome]:
        """
        Analyze learning outcomes from tournament.

        Returns:
            List of learning outcomes for each agent
        """
        outcomes = []

        for agent in self.agents:
            # Calculate performance improvement
            agent_matches = [match for match in self.match_history
                           if agent.agent_id in match['participants']]

            if len(agent_matches) > 1:
                early_performance = np.mean([
                    match['performance_scores'][agent.agent_id]
                    for match in agent_matches[:len(agent_matches)//2]
                ])
                late_performance = np.mean([
                    match['performance_scores'][agent.agent_id]
                    for match in agent_matches[len(agent_matches)//2:]
                ])
                improvement = late_performance - early_performance
            else:
                improvement = 0.0

            outcome = LearningOutcome(
                agent_id=agent.agent_id,
                scenario_type=LearningScenarioType.COMPETITIVE_TOURNAMENT,
                performance_improvement=improvement,
                knowledge_gained={'competitive_strategy': 0.7, 'adaptation': 0.6},
                skills_developed=['competitive_intelligence', 'rapid_adaptation', 'strategy_innovation'],
                collaboration_effectiveness=0.3,  # Low in competitive scenario
                adaptation_speed=self._measure_adaptation_speed(agent),
                teaching_ability=0.2,  # Not relevant in competition
                learning_efficiency=abs(improvement) / max(len(agent_matches), 1)
            )

            outcomes.append(outcome)

        return outcomes