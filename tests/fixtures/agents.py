"""Test agent fixtures and utilities."""

import torch
from typing import Dict, Any, List


class MockAgent:
    """Mock agent for testing scenarios."""

    def __init__(self, agent_id: str, role: str = "COOPERATOR"):
        self.agent_id = agent_id
        self.role = role
        self.collaboration_mode = "cooperative"

        # Initialize collaboration metrics
        self.collaboration_metrics = {
            'navigation_success': 0.5,
            'communication_effectiveness': 0.6,
            'problem_solving': 0.4,
            'adaptation_speed': 0.7
        }

        # Initialize behavior weights
        self.behavior_weights = {
            'individual_reward': 0.4,
            'team_reward': 0.3,
            'learning_rate': 0.1,
            'exploration_tendency': 0.3
        }

        # Optional attributes
        self.current_research_topic = None
        self.confidence_level = 0.5
        self._action_history = []

    def select_action(self, state: torch.Tensor) -> Dict[str, Any]:
        """Mock action selection."""
        action = torch.tensor([0.5])  # Simple deterministic action
        self._action_history.append(action)

        return {
            'action': action,
            'confidence': 0.7,
            'uncertainty': 0.3
        }

    def update_metrics(self, **kwargs):
        """Update agent metrics."""
        for key, value in kwargs.items():
            if key in self.collaboration_metrics:
                self.collaboration_metrics[key] = value
            elif key in self.behavior_weights:
                self.behavior_weights[key] = value


class MockCompetitiveAgent(MockAgent):
    """Mock agent with competitive behavior."""

    def __init__(self, agent_id: str):
        super().__init__(agent_id, role="COMPETITOR")
        self.collaboration_mode = "competitive"
        self.behavior_weights['individual_reward'] = 0.8
        self.behavior_weights['team_reward'] = 0.1


class MockMentorAgent(MockAgent):
    """Mock mentor agent for mentor-student scenarios."""

    def __init__(self, agent_id: str):
        super().__init__(agent_id, role="MENTOR")
        self.collaboration_metrics['communication_effectiveness'] = 0.9
        self.teaching_effectiveness = 0.8
        self.mentorship_history = []

    def provide_guidance(self, student_agent: 'MockAgent') -> Dict[str, Any]:
        """Provide guidance to student agent."""
        guidance = {
            'knowledge_transfer': torch.randn(32),
            'advice': "Focus on exploration",
            'confidence_boost': 0.1
        }
        self.mentorship_history.append({
            'student': student_agent.agent_id,
            'guidance': guidance
        })
        return guidance


class MockStudentAgent(MockAgent):
    """Mock student agent for mentor-student scenarios."""

    def __init__(self, agent_id: str):
        super().__init__(agent_id, role="STUDENT")
        self.learning_progress = 0.0
        self.mentor_feedback = []

    def receive_guidance(self, guidance: Dict[str, Any]) -> None:
        """Receive guidance from mentor."""
        self.mentor_feedback.append(guidance)
        if 'confidence_boost' in guidance:
            self.confidence_level += guidance['confidence_boost']
            self.confidence_level = min(1.0, self.confidence_level)


def create_mock_agents(count: int = 4, agent_type: str = "standard") -> List[MockAgent]:
    """Create a list of mock agents for testing."""
    agents = []

    for i in range(count):
        agent_id = f"agent_{i}"

        if agent_type == "competitive":
            agent = MockCompetitiveAgent(agent_id)
        elif agent_type == "mentor" and i == 0:
            agent = MockMentorAgent(agent_id)
        elif agent_type == "mentor" and i > 0:
            agent = MockStudentAgent(agent_id)
        else:
            agent = MockAgent(agent_id)

        agents.append(agent)

    return agents


def create_mentor_student_pair() -> tuple[MockMentorAgent, MockStudentAgent]:
    """Create a mentor-student pair for testing."""
    mentor = MockMentorAgent("mentor_0")
    student = MockStudentAgent("student_0")
    return mentor, student