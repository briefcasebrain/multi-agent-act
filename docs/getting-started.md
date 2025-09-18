---
layout: default
title: 🚀 Getting Started
nav_order: 2
---

# <i class="fas fa-rocket"></i> Getting Started
{: .no_toc }

Complete guide to setting up and using the Multi-Agent Collaborative Learning library.

## Table of Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch 1.12+ (for neural components)
- NumPy 1.21+
- NetworkX 2.8+ (for collaboration networks)

### Quick Install

```bash
# Basic installation
pip install multi-agent-collab-learning

# With visualization support
pip install multi-agent-collab-learning[visualization]

# Full installation with all optional dependencies
pip install multi-agent-collab-learning[all]
```

### Development Installation

```bash
# Clone the repository
git clone https://github.com/briefcasebrain/multi-agent-collab-learning.git
cd multi-agent-collab-learning

# Install in development mode
pip install -e .[dev]

# Run tests to verify installation
pytest
```

## Basic Usage

### 1. Create Your First Tournament

```python
from multi_agent_collab_learning import (
    CompetitiveLearningTournament,
    ScenarioConfig,
    LearningScenarioType,
    setup_logger
)

# Set up logging
logger = setup_logger("my_tournament")

# Create mock agents (replace with your agents)
class SimpleAgent:
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.collaboration_metrics = {}
        self.behavior_weights = {'individual_reward': 0.5}

agents = [SimpleAgent(f"agent_{i}") for i in range(4)]

# Configure tournament
config = ScenarioConfig(
    scenario_type=LearningScenarioType.COMPETITIVE_TOURNAMENT,
    duration_episodes=20,
    participants=len(agents)
)

# Create and run tournament
tournament = CompetitiveLearningTournament(agents, config)

# Mock environment (replace with your environment)
class SimpleEnvironment:
    def __init__(self):
        self.controller = type('Controller', (), {'reset': lambda: None})()

    def _get_multimodal_state(self):
        import torch
        return torch.randn(1, 32)

    def _execute_agent_action(self, agent, action):
        import random
        return random.random() > 0.4

environment = SimpleEnvironment()

# Run the tournament
results = tournament.run(environment)

print(f"Tournament winner: {results['tournament_winner']}")
print(f"Final rankings: {results['final_rankings']}")
```

### 2. Set Up Mentor-Student Learning

```python
from multi_agent_collab_learning import (
    MentorStudentNetwork,
    ScenarioConfig,
    LearningScenarioType
)

# Configure mentorship scenario
config = ScenarioConfig(
    scenario_type=LearningScenarioType.MENTOR_STUDENT,
    duration_episodes=30,
    participants=len(agents),
    knowledge_sharing_rate=0.4  # Higher sharing for mentorship
)

# Create mentor-student network
mentor_network = MentorStudentNetwork(agents, config)
results = mentor_network.run(environment)

print(f"Knowledge transfer metrics: {results['knowledge_transfer_metrics']}")
print(f"Teaching effectiveness: {results['teaching_effectiveness']}")
```

### 3. Orchestrate Multiple Scenarios

```python
from multi_agent_collab_learning import (
    ScenarioOrchestrator,
    CollaborativeResearchEnvironment
)

# Create orchestrator
orchestrator = ScenarioOrchestrator(agents)

# Register multiple scenarios
orchestrator.register_scenario('tournament', tournament)
orchestrator.register_scenario('mentorship', mentor_network)

# Add research scenario
research_config = ScenarioConfig(
    scenario_type=LearningScenarioType.COLLABORATIVE_RESEARCH,
    duration_episodes=40
)
research_env = CollaborativeResearchEnvironment(agents, research_config)
orchestrator.register_scenario('research', research_env)

# Run scenario suite
suite_results = orchestrator.run_scenario_suite(
    environment,
    scenario_sequence=['tournament', 'mentorship', 'research']
)

print(f"Cross-scenario learning events: {len(suite_results['cross_scenario_learning'])}")
print(f"Emergent capabilities: {suite_results['emergent_capabilities']}")
```

## Agent Implementation Guide

### Creating Compatible Agents

Your agents need to implement a minimal interface to work with the library:

```python
class MyAgent:
    def __init__(self, agent_id: str, role: str = "COOPERATOR"):
        # Required attributes
        self.agent_id = agent_id
        self.role = role
        self.collaboration_mode = "cooperative"

        # Collaboration metrics (updated by scenarios)
        self.collaboration_metrics = {
            'navigation_success': 0.3,
            'communication_effectiveness': 0.4,
            'problem_solving': 0.2,
            'adaptation_speed': 0.3
        }

        # Behavior weights (evolved during learning)
        self.behavior_weights = {
            'individual_reward': 0.4,
            'team_reward': 0.3,
            'learning_rate': 0.1,
            'exploration_tendency': 0.3
        }

        # Optional attributes for advanced scenarios
        self.current_research_topic = None
        self.confidence_level = 0.5

    def select_action(self, state):
        """Select action given current state."""
        # Implement your agent's policy here
        # Must return dict with 'action', 'confidence', 'uncertainty'

        import torch
        action = torch.tensor([self.choose_action(state)])

        return {
            'action': action,
            'confidence': 0.7,  # Action confidence [0, 1]
            'uncertainty': 0.3   # Action uncertainty [0, 1]
        }

    def choose_action(self, state):
        """Your custom action selection logic."""
        # Implement based on your agent architecture
        # Neural network, rule-based, etc.
        return 0  # Placeholder
```

### Environment Interface

Environments need to provide these methods:

```python
class MyEnvironment:
    def __init__(self):
        self.controller = self.create_controller()

    def create_controller(self):
        """Create environment controller."""
        class Controller:
            def reset(self): pass
            def stop(self): pass
        return Controller()

    def _get_multimodal_state(self):
        """Return current environment state."""
        import torch
        # Return state tensor or dict
        return torch.randn(1, 64)  # Example state

    def _execute_agent_action(self, agent, action):
        """Execute agent action in environment."""
        # Process the action and return success boolean
        return True  # or False based on action success
```

## Configuration Options

### Scenario Configuration

```python
from multi_agent_collab_learning import ScenarioConfig, LearningScenarioType

config = ScenarioConfig(
    # Required
    scenario_type=LearningScenarioType.COMPETITIVE_TOURNAMENT,

    # Episode settings
    duration_episodes=100,          # Number of episodes to run
    participants=6,                 # Number of participating agents

    # Reward structure
    reward_structure="mixed",       # "individual", "collective", "mixed"

    # Learning parameters
    knowledge_sharing_rate=0.3,     # Rate of knowledge exchange [0, 1]
    adaptation_frequency=25,        # Episodes between adaptations

    # Success criteria (scenario-specific)
    success_criteria={
        'min_performance': 0.7,
        'collaboration_threshold': 0.6
    },

    # Evaluation metrics to track
    evaluation_metrics=[
        'performance_improvement',
        'collaboration_effectiveness',
        'adaptation_speed'
    ],

    # Custom scenario parameters
    scenario_parameters={
        'tournament_format': 'bracket',  # For tournaments
        'research_topics': ['navigation', 'manipulation']  # For research
    }
)
```

### Logging Configuration

```python
from multi_agent_collab_learning import setup_logger

# Basic logging
logger = setup_logger("my_experiment")

# Advanced logging with file output
logger = setup_logger(
    name="my_experiment",
    level="DEBUG",
    log_file="experiments/run_001.log",
    format_string="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
```

## Visualization

### Learning Curves

```python
from multi_agent_collab_learning import plot_learning_curves

# Example performance data
learning_data = {
    'agent_0': [0.2, 0.35, 0.5, 0.65, 0.8],
    'agent_1': [0.1, 0.3, 0.45, 0.7, 0.85],
    'agent_2': [0.3, 0.4, 0.5, 0.6, 0.7]
}

# Create and save plot
fig = plot_learning_curves(
    learning_data,
    title="Agent Performance Over Time",
    xlabel="Episodes",
    ylabel="Success Rate",
    save_path="results/learning_curves.png"
)
```

### Collaboration Networks

```python
from multi_agent_collab_learning.utils.visualization import plot_collaboration_network

# Collaboration strength data
collaboration_data = {
    'agent_0': {'agent_1': 0.8, 'agent_2': 0.6},
    'agent_1': {'agent_0': 0.8, 'agent_3': 0.7},
    'agent_2': {'agent_0': 0.6, 'agent_3': 0.9}
}

fig = plot_collaboration_network(
    collaboration_data,
    title="Agent Collaboration Network",
    save_path="results/collaboration_network.png"
)
```

## Common Patterns

### 1. Experimental Setup

```python
import numpy as np
from pathlib import Path

# Create experiment directory
exp_dir = Path("experiments/run_001")
exp_dir.mkdir(parents=True, exist_ok=True)

# Set random seeds for reproducibility
import random
import torch
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Configure logging with experiment directory
logger = setup_logger("experiment", log_file=exp_dir / "experiment.log")
```

### 2. Result Analysis

```python
def analyze_results(scenario_results):
    """Analyze and summarize scenario results."""

    analysis = {
        'performance_summary': {},
        'learning_trends': {},
        'collaboration_metrics': {}
    }

    # Extract performance metrics
    if 'final_rankings' in scenario_results:
        rankings = scenario_results['final_rankings']
        analysis['performance_summary'] = {
            'winner': max(rankings, key=rankings.get),
            'avg_performance': np.mean(list(rankings.values())),
            'performance_std': np.std(list(rankings.values()))
        }

    # Extract learning outcomes
    if 'learning_outcomes' in scenario_results:
        outcomes = scenario_results['learning_outcomes']
        improvements = [o.performance_improvement for o in outcomes]
        analysis['learning_trends'] = {
            'avg_improvement': np.mean(improvements),
            'improvement_consistency': 1 - np.std(improvements)
        }

    return analysis

# Use in your experiments
results = tournament.run(environment)
analysis = analyze_results(results)
print(f"Analysis: {analysis}")
```

### 3. Custom Scenario Development

```python
from multi_agent_collab_learning.scenarios.base import BaseScenario

class CustomCooperationScenario(BaseScenario):
    """Custom scenario for testing cooperation strategies."""

    def run(self, environment):
        """Implement custom scenario logic."""

        if not self.validate_agents():
            raise ValueError("Invalid agents for cooperation scenario")

        results = {
            'cooperation_events': [],
            'individual_contributions': {},
            'team_performance': 0.0
        }

        for episode in range(self.config.duration_episodes):
            episode_result = self._run_cooperation_episode(environment)
            results['cooperation_events'].extend(episode_result['events'])

            # Update team performance
            results['team_performance'] += episode_result['team_score']

        # Calculate final metrics
        results['avg_team_performance'] = (
            results['team_performance'] / self.config.duration_episodes
        )

        self.results = results
        return results

    def _run_cooperation_episode(self, environment):
        """Run one cooperation episode."""
        # Implement episode logic
        return {'events': [], 'team_score': 0.5}
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
   ```bash
   pip install -e .[all]
   ```

2. **Memory Issues**: Use smaller batch sizes or fewer agents
   ```python
   config.participants = 4  # Reduce agent count
   ```

3. **Slow Performance**: Enable GPU acceleration
   ```python
   import torch
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   ```

### Debug Mode

Enable detailed logging for debugging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
logger = setup_logger("debug", level=logging.DEBUG)
```

## Next Steps

- Explore [Architecture Guide](architecture.html) for system design details
- Review [API Reference](api-reference.html) for complete documentation
- Check [Examples](examples.html) for advanced usage patterns
- Read [Data Flow Guide](data-flow.html) for understanding system internals

---

Ready to build sophisticated multi-agent learning systems! 🚀