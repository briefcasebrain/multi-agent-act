"""
Base environment classes for multi-agent collaborative learning.
"""

from typing import Dict, Any, List, Optional, Tuple
import torch
import numpy as np


class MultiAgentEnvironment:
    """Base class for multi-agent environments."""

    def __init__(self, state_dim: int = 64, action_dim: int = 32, max_agents: int = 10):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_agents = max_agents
        self.current_step = 0
        self.max_steps = 1000
        self.agents_in_environment = []

    def reset(self) -> torch.Tensor:
        """Reset environment to initial state."""
        self.current_step = 0
        self.agents_in_environment = []
        return self._get_initial_state()

    def step(self, actions: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float], bool, Dict[str, Any]]:
        """Execute actions and return next state, rewards, done, info."""
        self.current_step += 1

        # Process actions
        next_state = self._process_actions(actions)
        rewards = self._calculate_rewards(actions)
        done = self._is_done()
        info = self._get_info()

        return next_state, rewards, done, info

    def add_agent(self, agent_id: str) -> bool:
        """Add agent to environment."""
        if len(self.agents_in_environment) >= self.max_agents:
            return False

        if agent_id not in self.agents_in_environment:
            self.agents_in_environment.append(agent_id)
            return True
        return False

    def remove_agent(self, agent_id: str) -> bool:
        """Remove agent from environment."""
        if agent_id in self.agents_in_environment:
            self.agents_in_environment.remove(agent_id)
            return True
        return False

    def get_state(self) -> torch.Tensor:
        """Get current environment state."""
        return self._get_multimodal_state()

    def _get_initial_state(self) -> torch.Tensor:
        """Get initial environment state."""
        return torch.zeros(1, self.state_dim)

    def _get_multimodal_state(self) -> torch.Tensor:
        """Get multimodal state representation."""
        # Base implementation returns random state
        return torch.randn(1, self.state_dim)

    def _process_actions(self, actions: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Process agent actions and return next state."""
        # Base implementation returns next random state
        return torch.randn(1, self.state_dim)

    def _calculate_rewards(self, actions: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Calculate rewards for each agent."""
        rewards = {}
        for agent_id in actions:
            # Base implementation gives random rewards
            rewards[agent_id] = np.random.uniform(-1, 1)
        return rewards

    def _is_done(self) -> bool:
        """Check if episode is complete."""
        return self.current_step >= self.max_steps

    def _get_info(self) -> Dict[str, Any]:
        """Get additional environment information."""
        return {
            'step': self.current_step,
            'agents_count': len(self.agents_in_environment),
            'environment_state': 'active'
        }

    def _execute_agent_action(self, agent: Any, action: torch.Tensor) -> bool:
        """Execute single agent action (compatibility method)."""
        # Simple success based on action magnitude
        action_magnitude = torch.norm(action).item()
        success_probability = min(action_magnitude / 2.0, 0.9)
        return np.random.random() < success_probability


class CollaborativeEnvironment(MultiAgentEnvironment):
    """Environment designed for collaborative learning scenarios."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.collaboration_bonus = 0.5
        self.team_objectives = []
        self.individual_objectives = []

    def _calculate_rewards(self, actions: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Calculate rewards with collaboration bonuses."""
        base_rewards = super()._calculate_rewards(actions)

        # Add collaboration bonus if multiple agents act
        if len(actions) > 1:
            collaboration_bonus = self.collaboration_bonus / len(actions)
            for agent_id in base_rewards:
                base_rewards[agent_id] += collaboration_bonus

        return base_rewards

    def add_team_objective(self, objective: Dict[str, Any]) -> None:
        """Add team-based objective."""
        self.team_objectives.append(objective)

    def add_individual_objective(self, agent_id: str, objective: Dict[str, Any]) -> None:
        """Add individual objective for specific agent."""
        self.individual_objectives.append({
            'agent_id': agent_id,
            'objective': objective
        })


class CompetitiveEnvironment(MultiAgentEnvironment):
    """Environment designed for competitive learning scenarios."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.competition_metrics = {}
        self.leaderboard = {}

    def _calculate_rewards(self, actions: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Calculate competitive rewards with relative scoring."""
        base_rewards = super()._calculate_rewards(actions)

        # Implement competitive scoring (zero-sum style)
        if len(base_rewards) > 1:
            total_reward = sum(base_rewards.values())
            avg_reward = total_reward / len(base_rewards)

            # Adjust rewards relative to average
            for agent_id in base_rewards:
                base_rewards[agent_id] = base_rewards[agent_id] - avg_reward

        return base_rewards

    def update_leaderboard(self, agent_scores: Dict[str, float]) -> None:
        """Update competitive leaderboard."""
        for agent_id, score in agent_scores.items():
            if agent_id not in self.leaderboard:
                self.leaderboard[agent_id] = 0.0
            self.leaderboard[agent_id] += score

    def get_rankings(self) -> List[Tuple[str, float]]:
        """Get current agent rankings."""
        return sorted(self.leaderboard.items(), key=lambda x: x[1], reverse=True)