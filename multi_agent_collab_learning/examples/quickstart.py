"""
Quick start example for the multi-agent collaborative learning library.
"""

import sys
from pathlib import Path

# Add the library to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from multi_agent_collab_learning import (
    CompetitiveLearningTournament,
    ScenarioConfig,
    LearningScenarioType,
    setup_logger
)

# Simple mock classes for demonstration
class SimpleAgent:
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.role = "COOPERATOR"
        self.collaboration_mode = "cooperative"
        self.collaboration_metrics = {}
        self.behavior_weights = {'individual_reward': 0.5}

    def select_action(self, state):
        import torch
        return {'action': torch.tensor([1]), 'confidence': 0.7}

class SimpleEnvironment:
    def __init__(self):
        self.controller = SimpleController()

    def _get_multimodal_state(self):
        import torch
        return torch.randn(1, 32)

    def _execute_agent_action(self, agent, action):
        import random
        return random.random() > 0.4

class SimpleController:
    def reset(self): pass
    def stop(self): pass


def quickstart_example():
    """
    Quick start example showing basic library usage.
    """
    # Set up logging
    logger = setup_logger("quickstart")
    logger.info("🚀 Quick Start Example")

    # 1. Create agents
    agents = [SimpleAgent(f"agent_{i}") for i in range(4)]
    logger.info(f"Created {len(agents)} agents")

    # 2. Create environment
    environment = SimpleEnvironment()
    logger.info("Created environment")

    # 3. Configure scenario
    config = ScenarioConfig(
        scenario_type=LearningScenarioType.COMPETITIVE_TOURNAMENT,
        duration_episodes=5,  # Very short for quick demo
        participants=len(agents)
    )
    logger.info("Configured tournament scenario")

    # 4. Create and run tournament
    tournament = CompetitiveLearningTournament(agents, config)
    logger.info("Running tournament...")

    results = tournament.run(environment)

    # 5. Display results
    logger.info("✅ Tournament completed!")
    logger.info(f"Winner: {results['tournament_winner']}")
    logger.info(f"Rankings: {results['final_rankings']}")
    logger.info(f"Matches played: {len(results['matches'])}")

    return results


if __name__ == "__main__":
    quickstart_example()