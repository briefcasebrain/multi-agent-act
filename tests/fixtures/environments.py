"""Test environment fixtures and utilities."""

import torch
import random
from typing import Any, Dict, List


class MockEnvironment:
    """Mock environment for testing scenarios."""

    def __init__(self, state_dim: int = 64, deterministic: bool = False):
        self.state_dim = state_dim
        self.deterministic = deterministic
        self.step_count = 0
        self.max_steps = 100
        self.controller = MockController()
        self._episode_history = []

    def _get_multimodal_state(self) -> torch.Tensor:
        """Return mock environment state."""
        if self.deterministic:
            # Deterministic state for reproducible tests
            torch.manual_seed(self.step_count)
            state = torch.randn(1, self.state_dim)
            torch.seed()  # Reset seed
        else:
            state = torch.randn(1, self.state_dim)

        return state

    def _execute_agent_action(self, agent: Any, action: torch.Tensor) -> bool:
        """Execute agent action and return success."""
        self.step_count += 1

        if self.deterministic:
            # Deterministic success based on agent and action
            success = (hash(agent.agent_id) + int(action.sum().item())) % 2 == 0
        else:
            success = random.random() > 0.3  # 70% success rate

        # Log action execution
        self._episode_history.append({
            'step': self.step_count,
            'agent': agent.agent_id,
            'action': action.clone(),
            'success': success
        })

        return success

    def reset(self) -> torch.Tensor:
        """Reset environment to initial state."""
        self.step_count = 0
        self._episode_history = []
        self.controller.reset()
        return self._get_multimodal_state()

    def is_done(self) -> bool:
        """Check if episode is complete."""
        return self.step_count >= self.max_steps

    def get_episode_history(self) -> List[Dict[str, Any]]:
        """Get history of actions executed in this episode."""
        return self._episode_history.copy()


class MockController:
    """Mock environment controller."""

    def __init__(self):
        self.is_running = False
        self.reset_count = 0

    def reset(self) -> None:
        """Reset the controller."""
        self.reset_count += 1
        self.is_running = True

    def stop(self) -> None:
        """Stop the controller."""
        self.is_running = False


class MockCompetitiveEnvironment(MockEnvironment):
    """Mock environment optimized for competitive scenarios."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.competition_metrics = {
            'rounds_completed': 0,
            'winner_history': [],
            'performance_scores': {}
        }

    def evaluate_competition_round(self, agents: List[Any]) -> Dict[str, Any]:
        """Evaluate a competition round between agents."""
        self.competition_metrics['rounds_completed'] += 1

        # Mock performance scoring
        scores = {}
        for agent in agents:
            if self.deterministic:
                score = hash(agent.agent_id) % 100 / 100.0
            else:
                score = random.random()
            scores[agent.agent_id] = score

        winner = max(scores, key=scores.get)
        self.competition_metrics['winner_history'].append(winner)
        self.competition_metrics['performance_scores'].update(scores)

        return {
            'winner': winner,
            'scores': scores,
            'round_number': self.competition_metrics['rounds_completed']
        }


class MockResearchEnvironment(MockEnvironment):
    """Mock environment for collaborative research scenarios."""

    def __init__(self, research_topics: List[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.research_topics = research_topics or ["navigation", "manipulation", "communication"]
        self.discoveries = []
        self.collaboration_events = []

    def simulate_research_session(self, agents: List[Any]) -> Dict[str, Any]:
        """Simulate a collaborative research session."""
        session_results = {
            'discoveries': [],
            'collaborations': [],
            'innovation_score': 0.0
        }

        # Mock discovery generation
        for topic in self.research_topics:
            if self.deterministic:
                discovery_chance = (hash(topic) % 100) / 100.0
            else:
                discovery_chance = random.random()

            if discovery_chance > 0.7:  # 30% chance of discovery
                discovery = {
                    'topic': topic,
                    'significance': discovery_chance,
                    'contributing_agents': [agent.agent_id for agent in agents[:2]]
                }
                session_results['discoveries'].append(discovery)
                self.discoveries.append(discovery)

        # Mock collaboration events
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                if self.deterministic:
                    collab_strength = ((hash(agents[i].agent_id) + hash(agents[j].agent_id)) % 100) / 100.0
                else:
                    collab_strength = random.random()

                if collab_strength > 0.5:  # Collaboration threshold
                    collaboration = {
                        'agents': [agents[i].agent_id, agents[j].agent_id],
                        'strength': collab_strength,
                        'topic': random.choice(self.research_topics)
                    }
                    session_results['collaborations'].append(collaboration)
                    self.collaboration_events.append(collaboration)

        # Calculate innovation score
        session_results['innovation_score'] = (
            len(session_results['discoveries']) * 0.6 +
            len(session_results['collaborations']) * 0.4
        )

        return session_results


def create_mock_environment(env_type: str = "standard", **kwargs) -> MockEnvironment:
    """Create a mock environment for testing."""
    if env_type == "competitive":
        return MockCompetitiveEnvironment(**kwargs)
    elif env_type == "research":
        return MockResearchEnvironment(**kwargs)
    else:
        return MockEnvironment(**kwargs)