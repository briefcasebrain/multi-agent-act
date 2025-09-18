"""
Scenario orchestrator for managing multiple collaborative learning scenarios.
"""

import time
import random
import numpy as np
from typing import Dict, List, Any, Optional, Union

from .base import BaseScenario
from ..core.config import ScenarioConfig, LearningScenarioType
from ..utils.logging import get_logger

logger = get_logger(__name__)


class ScenarioOrchestrator:
    """
    Orchestrates multiple collaborative learning scenarios.

    This class manages the execution of multiple learning scenarios in sequence,
    tracks cross-scenario learning, and analyzes agent development over time.
    """

    def __init__(self, agents: List[Any]):
        """
        Initialize the scenario orchestrator.

        Args:
            agents: List of collaborative agents
        """
        self.agents = agents
        self.scenarios = {}
        self.scenario_results = {}
        self.cross_scenario_learning = {}

    def register_scenario(self, scenario_name: str, scenario_instance: BaseScenario) -> None:
        """
        Register a learning scenario.

        Args:
            scenario_name: Name identifier for the scenario
            scenario_instance: Instance of a scenario class
        """
        if not isinstance(scenario_instance, BaseScenario):
            raise TypeError("Scenario instance must inherit from BaseScenario")

        self.scenarios[scenario_name] = scenario_instance
        logger.info(f"Registered scenario: {scenario_name}")

    def unregister_scenario(self, scenario_name: str) -> None:
        """
        Unregister a learning scenario.

        Args:
            scenario_name: Name of the scenario to remove
        """
        if scenario_name in self.scenarios:
            del self.scenarios[scenario_name]
            logger.info(f"Unregistered scenario: {scenario_name}")

    def list_scenarios(self) -> List[str]:
        """
        Get list of registered scenario names.

        Returns:
            List of scenario names
        """
        return list(self.scenarios.keys())

    def run_single_scenario(
        self,
        scenario_name: str,
        environment: Any,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run a single scenario.

        Args:
            scenario_name: Name of the scenario to run
            environment: Environment to run the scenario in
            **kwargs: Additional arguments passed to scenario

        Returns:
            Scenario results dictionary
        """
        if scenario_name not in self.scenarios:
            raise ValueError(f"Scenario '{scenario_name}' not found")

        scenario = self.scenarios[scenario_name]
        logger.info(f"Running single scenario: {scenario_name}")

        # Run the scenario
        results = scenario.run(environment, **kwargs)

        # Store results
        self.scenario_results[scenario_name] = {
            'results': results,
            'timestamp': time.time(),
            'agents_count': len(self.agents)
        }

        logger.info(f"Completed scenario: {scenario_name}")
        return results

    def run_scenario_suite(
        self,
        environment: Any,
        scenario_sequence: Optional[List[str]] = None,
        inter_scenario_adaptation: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run a suite of collaborative learning scenarios.

        Args:
            environment: Environment to run scenarios in
            scenario_sequence: Ordered list of scenarios to run (optional)
            inter_scenario_adaptation: Whether to allow agent adaptation between scenarios
            **kwargs: Additional arguments passed to scenarios

        Returns:
            Suite results dictionary
        """
        if scenario_sequence is None:
            scenario_sequence = list(self.scenarios.keys())

        logger.info("Running Collaborative Learning Scenario Suite")
        logger.info(f"Scenarios: {len(scenario_sequence)}")
        logger.info(f"Agents: {len(self.agents)}")

        suite_results = {
            'scenario_results': {},
            'cross_scenario_learning': {},
            'agent_development': {},
            'emergent_capabilities': [],
            'execution_timeline': []
        }

        start_time = time.time()

        # Run scenarios in sequence
        for i, scenario_name in enumerate(scenario_sequence):
            if scenario_name not in self.scenarios:
                logger.warning(f"Scenario '{scenario_name}' not found, skipping...")
                continue

            logger.info(f"Running Scenario {i+1}/{len(scenario_sequence)}: {scenario_name}")

            scenario_start_time = time.time()

            # Run scenario
            scenario_result = self._run_single_scenario_internal(
                scenario_name, environment, **kwargs
            )

            scenario_end_time = time.time()

            # Store results
            suite_results['scenario_results'][scenario_name] = scenario_result
            suite_results['execution_timeline'].append({
                'scenario': scenario_name,
                'start_time': scenario_start_time,
                'end_time': scenario_end_time,
                'duration': scenario_end_time - scenario_start_time
            })

            # Analyze cross-scenario learning
            if i > 0:
                cross_learning = self._analyze_cross_scenario_learning(scenario_name, i)
                suite_results['cross_scenario_learning'][scenario_name] = cross_learning

            # Allow agents to adapt between scenarios
            if inter_scenario_adaptation and i < len(scenario_sequence) - 1:
                self._inter_scenario_adaptation(scenario_name, scenario_result)

        end_time = time.time()

        # Final analysis
        suite_results['agent_development'] = self._analyze_agent_development()
        suite_results['emergent_capabilities'] = self._identify_emergent_capabilities()
        suite_results['total_duration'] = end_time - start_time

        logger.info("Scenario Suite Complete!")
        self._log_suite_summary(suite_results)

        return suite_results

    def _run_single_scenario_internal(
        self,
        scenario_name: str,
        environment: Any,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Internal method to run a single scenario.

        Args:
            scenario_name: Name of the scenario
            environment: Environment to run in
            **kwargs: Additional arguments

        Returns:
            Scenario results
        """
        scenario = self.scenarios[scenario_name]

        # Validate scenario can run
        if not scenario.validate_agents():
            logger.warning(f"Scenario {scenario_name} validation failed")
            return {}

        # Run the scenario
        results = scenario.run(environment, **kwargs)

        # Store in orchestrator's results
        self.scenario_results[scenario_name] = {
            'results': results,
            'timestamp': time.time(),
            'agents_count': len(self.agents)
        }

        return results

    def _analyze_cross_scenario_learning(
        self,
        current_scenario: str,
        scenario_index: int
    ) -> Dict[str, Any]:
        """
        Analyze learning transfer between scenarios.

        Args:
            current_scenario: Name of current scenario
            scenario_index: Index of current scenario in sequence

        Returns:
            Cross-scenario learning analysis
        """
        cross_learning = {
            'skill_transfer': {},
            'adaptation_speed': {},
            'learning_efficiency': {},
            'capability_emergence': []
        }

        # Analyze skill transfer
        for agent in self.agents:
            agent_id = agent.agent_id

            # Compare performance metrics across scenarios
            previous_scenarios = list(self.scenario_results.keys())

            if len(previous_scenarios) > 0:
                # Calculate skill transfer
                skill_transfer = self._calculate_skill_transfer(
                    agent_id, previous_scenarios, current_scenario
                )
                cross_learning['skill_transfer'][agent_id] = skill_transfer

                # Measure adaptation speed
                adaptation_speed = self._measure_adaptation_speed_cross_scenario(
                    agent_id, current_scenario
                )
                cross_learning['adaptation_speed'][agent_id] = adaptation_speed

                # Measure learning efficiency improvement
                learning_efficiency = self._measure_learning_efficiency_improvement(
                    agent_id, previous_scenarios
                )
                cross_learning['learning_efficiency'][agent_id] = learning_efficiency

        return cross_learning

    def _calculate_skill_transfer(
        self,
        agent_id: str,
        previous_scenarios: List[str],
        current_scenario: str
    ) -> float:
        """
        Calculate skill transfer between scenarios for an agent.

        Args:
            agent_id: Agent identifier
            previous_scenarios: List of previously completed scenarios
            current_scenario: Current scenario name

        Returns:
            Skill transfer score between 0 and 1
        """
        # Simplified implementation - would analyze specific skill metrics
        base_transfer = 0.3  # Baseline transfer

        # Scenario similarity affects transfer
        similarity_bonuses = {
            ('competitive_tournament', 'mentor_student'): 0.2,
            ('mentor_student', 'collaborative_research'): 0.4,
            ('collaborative_research', 'competitive_tournament'): 0.1,
            ('competitive', 'mentor_student'): 0.2,
            ('mentor_student', 'collaborative'): 0.4,
            ('collaborative', 'competitive'): 0.1
        }

        total_transfer = base_transfer
        for prev_scenario in previous_scenarios:
            # Create scenario pair key
            pair_key = tuple(sorted([prev_scenario, current_scenario]))
            transfer_bonus = similarity_bonuses.get(pair_key, 0.1)
            total_transfer += transfer_bonus

        return min(1.0, total_transfer)

    def _measure_adaptation_speed_cross_scenario(
        self,
        agent_id: str,
        current_scenario: str
    ) -> float:
        """
        Measure how quickly agent adapts to new scenario.

        Args:
            agent_id: Agent identifier
            current_scenario: Current scenario name

        Returns:
            Adaptation speed score
        """
        # Simplified - would measure actual adaptation metrics
        base_speed = 0.5

        # Agents get better at adapting over time
        num_previous_scenarios = len(self.scenario_results)
        experience_bonus = min(0.3, num_previous_scenarios * 0.1)

        # Add some variability based on agent characteristics
        agent_variability = random.uniform(-0.1, 0.1)

        return min(1.0, max(0.0, base_speed + experience_bonus + agent_variability))

    def _measure_learning_efficiency_improvement(
        self,
        agent_id: str,
        previous_scenarios: List[str]
    ) -> float:
        """
        Measure improvement in learning efficiency over scenarios.

        Args:
            agent_id: Agent identifier
            previous_scenarios: List of previous scenario names

        Returns:
            Learning efficiency improvement score
        """
        if len(previous_scenarios) < 2:
            return 0.0

        # Simplified calculation based on scenario count
        # In reality, would track actual learning metrics
        improvement_rate = len(previous_scenarios) * 0.05
        return min(0.5, improvement_rate)

    def _inter_scenario_adaptation(
        self,
        completed_scenario: str,
        scenario_results: Dict[str, Any]
    ) -> None:
        """
        Allow agents to adapt between scenarios.

        Args:
            completed_scenario: Name of just completed scenario
            scenario_results: Results from the completed scenario
        """
        logger.info("Inter-scenario adaptation...")

        for agent in self.agents:
            # Extract agent performance indicators from scenario results
            agent_performed_well = self._evaluate_agent_performance_in_scenario(
                agent, completed_scenario, scenario_results
            )

            # Update agent parameters based on performance
            self._update_agent_for_adaptation(agent, agent_performed_well)

    def _evaluate_agent_performance_in_scenario(
        self,
        agent: Any,
        scenario_name: str,
        scenario_results: Dict[str, Any]
    ) -> bool:
        """
        Evaluate if an agent performed well in the scenario.

        Args:
            agent: Agent to evaluate
            scenario_name: Name of the scenario
            scenario_results: Results from the scenario

        Returns:
            True if agent performed well
        """
        # Simplified evaluation - would extract actual performance metrics
        # from scenario results based on scenario type

        # Check if agent appears in any positive outcomes
        agent_id = agent.agent_id

        # Look for agent in various result structures
        if 'final_rankings' in scenario_results:
            rankings = scenario_results['final_rankings']
            if agent_id in rankings:
                # Check if in top half of performers
                sorted_agents = sorted(rankings.items(), key=lambda x: x[1], reverse=True)
                agent_rank = next(i for i, (aid, _) in enumerate(sorted_agents) if aid == agent_id)
                return agent_rank < len(sorted_agents) / 2

        if 'learning_outcomes' in scenario_results:
            outcomes = scenario_results['learning_outcomes']
            for outcome in outcomes:
                if hasattr(outcome, 'agent_id') and outcome.agent_id == agent_id:
                    return outcome.performance_improvement > 0

        # Default evaluation
        return random.random() > 0.5

    def _update_agent_for_adaptation(self, agent: Any, performed_well: bool) -> None:
        """
        Update agent parameters for inter-scenario adaptation.

        Args:
            agent: Agent to update
            performed_well: Whether agent performed well in last scenario
        """
        # Get or initialize behavior weights
        behavior_weights = getattr(agent, 'behavior_weights', {})

        if performed_well:
            # Reinforce successful strategies
            learning_rate = behavior_weights.get('learning_rate', 0.1)
            behavior_weights['learning_rate'] = min(0.8, learning_rate + 0.05)

            # Increase confidence
            confidence = getattr(agent, 'confidence_level', 0.5)
            agent.confidence_level = min(1.0, confidence + 0.1)
        else:
            # Increase exploration for next scenario
            exploration = behavior_weights.get('exploration_tendency', 0.3)
            behavior_weights['exploration_tendency'] = min(0.9, exploration + 0.1)

            # Encourage more collaborative behavior
            team_reward = behavior_weights.get('team_reward', 0.3)
            behavior_weights['team_reward'] = min(0.8, team_reward + 0.05)

        agent.behavior_weights = behavior_weights

    def _analyze_agent_development(self) -> Dict[str, Any]:
        """
        Analyze how agents developed across scenarios.

        Returns:
            Agent development analysis
        """
        development = {}

        for agent in self.agents:
            agent_id = agent.agent_id

            # Track capability growth
            development[agent_id] = {
                'scenarios_participated': len(self.scenario_results),
                'skill_diversification': self._measure_skill_diversification(agent),
                'adaptability_improvement': self._measure_adaptability_improvement(agent),
                'collaboration_evolution': self._measure_collaboration_evolution(agent),
                'leadership_development': self._measure_leadership_development(agent),
                'learning_trajectory': self._trace_learning_trajectory(agent)
            }

        return development

    def _measure_skill_diversification(self, agent: Any) -> float:
        """
        Measure how agent's skills diversified across scenarios.

        Args:
            agent: Agent to analyze

        Returns:
            Skill diversification score
        """
        # Count different skill areas agent engaged with
        skill_areas = set()

        # Extract from collaboration metrics (simplified)
        collaboration_metrics = getattr(agent, 'collaboration_metrics', {})
        for metric_key in collaboration_metrics:
            if 'success' in metric_key or 'effectiveness' in metric_key:
                skill_areas.add(metric_key.split('_')[0])

        return len(skill_areas) / 8.0  # Normalize by max possible skills

    def _measure_adaptability_improvement(self, agent: Any) -> float:
        """
        Measure improvement in agent's adaptability.

        Args:
            agent: Agent to analyze

        Returns:
            Adaptability improvement score
        """
        # Simplified - would track actual adaptation metrics over time
        collaboration_metrics = getattr(agent, 'collaboration_metrics', {})
        return collaboration_metrics.get('adaptation_speed', 0.5)

    def _measure_collaboration_evolution(self, agent: Any) -> float:
        """
        Measure evolution of agent's collaboration abilities.

        Args:
            agent: Agent to analyze

        Returns:
            Collaboration evolution score
        """
        initial_collaboration = 0.3  # Assumed starting point
        behavior_weights = getattr(agent, 'behavior_weights', {})
        current_collaboration = behavior_weights.get('team_reward', 0.3)

        return current_collaboration - initial_collaboration

    def _measure_leadership_development(self, agent: Any) -> float:
        """
        Measure development of leadership capabilities.

        Args:
            agent: Agent to analyze

        Returns:
            Leadership development score
        """
        collaboration_metrics = getattr(agent, 'collaboration_metrics', {})

        leadership_indicators = [
            collaboration_metrics.get('leadership_actions', 0) / max(
                collaboration_metrics.get('messages_sent', 1), 1
            ),
            1.0 if getattr(agent, 'role', None) and 'LEADER' in str(agent.role) else 0.0,
            collaboration_metrics.get('trust_received', 0.5)
        ]

        return np.mean(leadership_indicators)

    def _trace_learning_trajectory(self, agent: Any) -> Dict[str, List[float]]:
        """
        Trace agent's learning trajectory across scenarios.

        Args:
            agent: Agent to analyze

        Returns:
            Learning trajectory data
        """
        # Simplified trajectory tracking
        trajectory = {
            'performance': [],
            'collaboration': [],
            'adaptation': []
        }

        # In a real implementation, this would extract actual metrics
        # from each scenario's results for this specific agent
        for scenario_name in self.scenario_results:
            # Placeholder values - would extract from actual results
            trajectory['performance'].append(random.uniform(0.3, 0.9))
            trajectory['collaboration'].append(random.uniform(0.2, 0.8))
            trajectory['adaptation'].append(random.uniform(0.4, 0.9))

        return trajectory

    def _identify_emergent_capabilities(self) -> List[Dict[str, Any]]:
        """
        Identify capabilities that emerged across scenarios.

        Returns:
            List of emergent capabilities
        """
        emergent_capabilities = []

        # Analyze agent behaviors that emerged through scenario progression
        for agent in self.agents:
            agent_id = agent.agent_id
            initial_role = getattr(agent, 'role', 'COOPERATOR')
            behavior_weights = getattr(agent, 'behavior_weights', {})

            # Check for role evolution
            if ('FOLLOWER' in str(initial_role) or 'COOPERATOR' in str(initial_role)) and \
               behavior_weights.get('leadership_tendency', 0.0) > 0.6:
                emergent_capabilities.append({
                    'type': 'leadership_emergence',
                    'agent_id': agent_id,
                    'description': f"Agent developed leadership capabilities from {initial_role}",
                    'strength': behavior_weights.get('leadership_tendency', 0.0)
                })

            # Check for specialization emergence
            collaboration_metrics = getattr(agent, 'collaboration_metrics', {})
            specialized_skills = [k for k, v in collaboration_metrics.items() if v > 0.8]
            if len(specialized_skills) > 2:
                emergent_capabilities.append({
                    'type': 'specialization_emergence',
                    'agent_id': agent_id,
                    'description': f"Agent developed expertise in {specialized_skills}",
                    'skills': specialized_skills
                })

            # Check for cross-scenario adaptation
            if len(self.scenario_results) > 2 and \
               behavior_weights.get('adaptation_rate', 0.0) > 0.7:
                emergent_capabilities.append({
                    'type': 'meta_learning_emergence',
                    'agent_id': agent_id,
                    'description': "Agent developed meta-learning capabilities",
                    'evidence': 'High adaptation rate across multiple scenarios'
                })

        return emergent_capabilities

    def _log_suite_summary(self, suite_results: Dict[str, Any]) -> None:
        """
        Log summary of scenario suite results.

        Args:
            suite_results: Complete suite results
        """
        logger.info(f"Scenarios Completed: {len(suite_results['scenario_results'])}")
        logger.info(f"Agents Developed: {len(suite_results['agent_development'])}")
        logger.info(f"Emergent Capabilities: {len(suite_results['emergent_capabilities'])}")
        logger.info(f"Total Duration: {suite_results['total_duration']:.2f} seconds")

        # Show emergent capabilities
        for capability in suite_results['emergent_capabilities'][:3]:
            logger.info(f"- {capability['type']}: {capability['description']}")

    def get_scenario_results(self, scenario_name: Optional[str] = None) -> Union[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        """
        Get results from scenarios.

        Args:
            scenario_name: Specific scenario name (optional)

        Returns:
            Scenario results or all results if no name specified
        """
        if scenario_name:
            return self.scenario_results.get(scenario_name, {})
        return self.scenario_results.copy()

    def clear_results(self) -> None:
        """Clear all stored scenario results."""
        self.scenario_results.clear()
        self.cross_scenario_learning.clear()
        logger.info("Cleared all scenario results")

    def export_results(self, filepath: str, format: str = 'json') -> None:
        """
        Export scenario results to file.

        Args:
            filepath: Path to save file
            format: Export format ('json' or 'csv')
        """
        import json
        from pathlib import Path

        if format.lower() == 'json':
            with open(filepath, 'w') as f:
                json.dump({
                    'scenario_results': self.scenario_results,
                    'cross_scenario_learning': self.cross_scenario_learning
                }, f, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format}")

        logger.info(f"Exported results to {filepath}")

    def __len__(self) -> int:
        """Return number of registered scenarios."""
        return len(self.scenarios)

    def __contains__(self, scenario_name: str) -> bool:
        """Check if scenario is registered."""
        return scenario_name in self.scenarios

    def __getitem__(self, scenario_name: str) -> BaseScenario:
        """Get scenario by name."""
        return self.scenarios[scenario_name]