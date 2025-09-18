"""
Demonstration script for the multi-agent collaborative learning library.
"""

import sys
import time
from pathlib import Path

# Add the library to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from multi_agent_collab_learning import (
    ScenarioOrchestrator,
    CompetitiveLearningTournament,
    MentorStudentNetwork,
    CollaborativeResearchEnvironment,
    ScenarioConfig,
    LearningScenarioType,
    setup_logger,
    plot_learning_curves
)

# Set up logging
logger = setup_logger("demo", log_file="demo.log")


class MockAgent:
    """Mock agent class for demonstration purposes."""

    def __init__(self, agent_id: str, role: str = "COOPERATOR"):
        self.agent_id = agent_id
        self.role = role
        self.collaboration_mode = "cooperative"
        self.collaboration_metrics = {
            'navigation_success': 0.3,
            'communication_effectiveness': 0.4,
            'problem_solving': 0.2,
            'adaptation_speed': 0.3
        }
        self.behavior_weights = {
            'individual_reward': 0.4,
            'team_reward': 0.3,
            'learning_rate': 0.1,
            'exploration_tendency': 0.3
        }
        self.current_research_topic = None
        self.confidence_level = 0.5
        self.motivation_level = 0.5

    def select_action(self, state):
        """Mock action selection."""
        import torch
        return {
            'action': torch.tensor([1]),
            'confidence': 0.7,
            'uncertainty': 0.3
        }


class MockEnvironment:
    """Mock environment class for demonstration purposes."""

    def __init__(self):
        self.controller = MockController()

    def _get_multimodal_state(self):
        """Mock state getter."""
        import torch
        return torch.randn(1, 64)

    def _execute_agent_action(self, agent, action):
        """Mock action execution."""
        import random
        return random.random() > 0.3  # 70% success rate


class MockController:
    """Mock controller for environment."""

    def reset(self):
        """Mock reset."""
        pass

    def stop(self):
        """Mock stop."""
        pass


def create_mock_agents(num_agents: int = 6):
    """
    Create mock agents for demonstration.

    Args:
        num_agents: Number of agents to create

    Returns:
        List of mock agents
    """
    roles = ["LEADER", "COORDINATOR", "SPECIALIST", "EXPLORER", "COLLECTOR", "COOPERATOR"]
    agents = []

    for i in range(num_agents):
        role = roles[i % len(roles)]
        agent = MockAgent(f"agent_{i}", role)
        agents.append(agent)

    return agents


def demo_competitive_tournament():
    """Demonstrate competitive learning tournament."""
    logger.info("=== Competitive Tournament Demo ===")

    # Create agents
    agents = create_mock_agents(4)
    environment = MockEnvironment()

    # Configure tournament
    config = ScenarioConfig(
        scenario_type=LearningScenarioType.COMPETITIVE_TOURNAMENT,
        duration_episodes=10,  # Shorter for demo
        participants=len(agents)
    )

    # Create and run tournament
    tournament = CompetitiveLearningTournament(agents, config)
    results = tournament.run(environment)

    logger.info(f"Tournament Winner: {results['tournament_winner']}")
    logger.info(f"Final Rankings: {results['final_rankings']}")
    logger.info(f"Total Matches: {len(results['matches'])}")

    return results


def demo_mentor_student_network():
    """Demonstrate mentor-student learning network."""
    logger.info("=== Mentor-Student Network Demo ===")

    # Create agents
    agents = create_mock_agents(4)
    environment = MockEnvironment()

    # Configure mentorship
    config = ScenarioConfig(
        scenario_type=LearningScenarioType.MENTOR_STUDENT,
        duration_episodes=15,  # Shorter for demo
        participants=len(agents)
    )

    # Create and run mentorship network
    mentor_network = MentorStudentNetwork(agents, config)
    results = mentor_network.run(environment)

    logger.info(f"Knowledge Transfer Metrics: {results['knowledge_transfer_metrics']}")
    logger.info(f"Teaching Effectiveness: {results['teaching_effectiveness']}")
    logger.info(f"Emergent Strategies: {len(results['emergent_teaching_strategies'])}")

    return results


def demo_collaborative_research():
    """Demonstrate collaborative research environment."""
    logger.info("=== Collaborative Research Demo ===")

    # Create agents
    agents = create_mock_agents(6)
    environment = MockEnvironment()

    # Configure research
    config = ScenarioConfig(
        scenario_type=LearningScenarioType.COLLABORATIVE_RESEARCH,
        duration_episodes=20,  # Shorter for demo
        participants=len(agents)
    )

    # Create and run research environment
    research_env = CollaborativeResearchEnvironment(agents, config)
    results = research_env.run(environment)

    logger.info(f"Total Discoveries: {len(results['discoveries'])}")
    logger.info(f"Research Outcomes: {results['research_outcomes']}")
    logger.info(f"Innovation Metrics: {results['innovation_metrics']}")

    return results


def demo_scenario_orchestrator():
    """Demonstrate scenario orchestration."""
    logger.info("=== Scenario Orchestration Demo ===")

    # Create agents
    agents = create_mock_agents(6)
    environment = MockEnvironment()

    # Create orchestrator
    orchestrator = ScenarioOrchestrator(agents)

    # Register scenarios
    tournament_config = ScenarioConfig(
        scenario_type=LearningScenarioType.COMPETITIVE_TOURNAMENT,
        duration_episodes=8,
        participants=len(agents)
    )
    tournament = CompetitiveLearningTournament(agents, tournament_config)
    orchestrator.register_scenario('competitive', tournament)

    mentor_config = ScenarioConfig(
        scenario_type=LearningScenarioType.MENTOR_STUDENT,
        duration_episodes=10,
        participants=len(agents)
    )
    mentor_network = MentorStudentNetwork(agents, mentor_config)
    orchestrator.register_scenario('mentorship', mentor_network)

    research_config = ScenarioConfig(
        scenario_type=LearningScenarioType.COLLABORATIVE_RESEARCH,
        duration_episodes=12,
        participants=len(agents)
    )
    research_env = CollaborativeResearchEnvironment(agents, research_config)
    orchestrator.register_scenario('research', research_env)

    # Run scenario suite
    scenario_sequence = ['competitive', 'mentorship', 'research']
    suite_results = orchestrator.run_scenario_suite(environment, scenario_sequence)

    logger.info(f"Scenarios Completed: {len(suite_results['scenario_results'])}")
    logger.info(f"Cross-Scenario Learning Events: {len(suite_results['cross_scenario_learning'])}")
    logger.info(f"Agent Development Tracked: {len(suite_results['agent_development'])}")
    logger.info(f"Emergent Capabilities: {len(suite_results['emergent_capabilities'])}")

    return suite_results


def demo_visualization():
    """Demonstrate visualization capabilities."""
    logger.info("=== Visualization Demo ===")

    # Generate sample learning data
    learning_data = {
        'agent_0': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        'agent_1': [0.1, 0.25, 0.35, 0.55, 0.65, 0.75, 0.85],
        'agent_2': [0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7],
        'agent_3': [0.15, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9]
    }

    try:
        # Create learning curves plot
        fig = plot_learning_curves(
            learning_data,
            title="Agent Learning Progress Demo",
            save_path="demo_learning_curves.png"
        )

        logger.info("Created learning curves visualization: demo_learning_curves.png")

        # Close the figure to free memory
        import matplotlib.pyplot as plt
        plt.close(fig)

    except ImportError as e:
        logger.warning(f"Visualization skipped due to missing dependency: {e}")


def main():
    """Run all demonstration scenarios."""
    logger.info("🎭 Multi-Agent Collaborative Learning Library Demo")
    logger.info("=" * 60)

    start_time = time.time()

    try:
        # Run individual scenario demos
        competitive_results = demo_competitive_tournament()
        mentor_results = demo_mentor_student_network()
        research_results = demo_collaborative_research()

        # Run orchestration demo
        suite_results = demo_scenario_orchestrator()

        # Run visualization demo
        demo_visualization()

        end_time = time.time()

        logger.info("=" * 60)
        logger.info(f"🎉 Demo completed successfully in {end_time - start_time:.2f} seconds")
        logger.info("Key Results:")
        logger.info(f"  • Competitive Tournament: {len(competitive_results.get('matches', []))} matches")
        logger.info(f"  • Mentor-Student: {len(mentor_results.get('learning_progress', {}))} students trained")
        logger.info(f"  • Research: {len(research_results.get('discoveries', []))} discoveries made")
        logger.info(f"  • Suite: {len(suite_results.get('scenario_results', {}))} scenarios orchestrated")

    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)