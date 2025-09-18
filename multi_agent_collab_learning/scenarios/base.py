"""
Base classes for learning scenarios.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
from ..core.config import ScenarioConfig


class BaseScenario(ABC):
    """
    Abstract base class for all learning scenarios.

    All scenario implementations should inherit from this class and implement
    the required abstract methods.
    """

    def __init__(self, agents: List[Any], config: ScenarioConfig):
        """
        Initialize the scenario.

        Args:
            agents: List of collaborative agents
            config: Scenario configuration
        """
        self.agents = agents
        self.config = config
        self.results = {}

    @abstractmethod
    def run(self, environment: Any) -> Dict[str, Any]:
        """
        Run the learning scenario.

        Args:
            environment: The environment to run the scenario in

        Returns:
            Dictionary containing scenario results
        """
        pass

    def validate_agents(self) -> bool:
        """
        Validate that agents are properly configured for this scenario.

        Returns:
            True if agents are valid, False otherwise
        """
        if not self.agents:
            return False
        if len(self.agents) < self.config.participants:
            return False
        return True

    def get_results(self) -> Dict[str, Any]:
        """
        Get the results from the last scenario run.

        Returns:
            Dictionary containing scenario results
        """
        return self.results.copy()