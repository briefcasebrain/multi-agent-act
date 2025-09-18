"""
Base agent classes for multi-agent collaborative learning.
"""

from typing import Dict, Any, List, Optional, Tuple
import torch
import torch.nn as nn


class CollaborativeAgent:
    """Base class for collaborative learning agents."""

    def __init__(self, agent_id: str, role: str = "COOPERATOR"):
        self.agent_id = agent_id
        self.role = role
        self.collaboration_mode = "cooperative"

        # Collaboration metrics
        self.collaboration_metrics = {
            'navigation_success': 0.0,
            'communication_effectiveness': 0.0,
            'problem_solving': 0.0,
            'adaptation_speed': 0.0
        }

        # Behavior weights
        self.behavior_weights = {
            'individual_reward': 0.5,
            'team_reward': 0.3,
            'learning_rate': 0.1,
            'exploration_tendency': 0.2
        }

        # Learning state
        self.learning_history = []
        self.knowledge_base = {}
        self.confidence_level = 0.5

    def select_action(self, state: torch.Tensor) -> Dict[str, Any]:
        """Select action given current state."""
        raise NotImplementedError("Subclasses must implement select_action")

    def update_collaboration_metrics(self, metrics: Dict[str, float]) -> None:
        """Update collaboration metrics."""
        for key, value in metrics.items():
            if key in self.collaboration_metrics:
                self.collaboration_metrics[key] = value

    def update_behavior_weights(self, weights: Dict[str, float]) -> None:
        """Update behavior weights."""
        for key, value in weights.items():
            if key in self.behavior_weights:
                self.behavior_weights[key] = value

    def get_collaboration_score(self) -> float:
        """Calculate overall collaboration score."""
        return sum(self.collaboration_metrics.values()) / len(self.collaboration_metrics)

    def share_knowledge(self, knowledge: Dict[str, Any]) -> None:
        """Receive knowledge from other agents."""
        self.knowledge_base.update(knowledge)

    def __repr__(self) -> str:
        return f"CollaborativeAgent(id={self.agent_id}, role={self.role})"


class NeuralCollaborativeAgent(CollaborativeAgent):
    """Neural network-based collaborative agent."""

    def __init__(self, agent_id: str, state_dim: int = 64, action_dim: int = 32, **kwargs):
        super().__init__(agent_id, **kwargs)
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Neural network components
        self.policy_network = self._build_policy_network()
        self.value_network = self._build_value_network()

    def _build_policy_network(self) -> nn.Module:
        """Build policy network."""
        return nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_dim),
            nn.Tanh()
        )

    def _build_value_network(self) -> nn.Module:
        """Build value network."""
        return nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def select_action(self, state: torch.Tensor) -> Dict[str, Any]:
        """Select action using neural policy."""
        with torch.no_grad():
            action = self.policy_network(state)
            value = self.value_network(state)

            # Add exploration noise
            noise = torch.randn_like(action) * self.behavior_weights.get('exploration_tendency', 0.1)
            action = action + noise

            # Calculate confidence based on value and exploration
            confidence = torch.sigmoid(value).item()
            uncertainty = 1.0 - confidence

        return {
            'action': action,
            'confidence': confidence,
            'uncertainty': uncertainty,
            'value': value.item()
        }

    def update_policy(self, states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor) -> Dict[str, float]:
        """Update policy networks."""
        # Simplified policy update (would be more sophisticated in practice)
        policy_loss = nn.MSELoss()(self.policy_network(states), actions)
        value_loss = nn.MSELoss()(self.value_network(states).squeeze(), rewards)

        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item()
        }