# Multi-Agent Collaborative Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://briefcasebrain.github.io/multi-agent-collab-learning/)

A comprehensive Python library for implementing advanced multi-agent collaborative learning scenarios. Developed by [Briefcase Brain](https://github.com/briefcasebrain), this library provides robust frameworks for competitive tournaments, mentor-student networks, collaborative research environments, and sophisticated scenario orchestration.

## Features

- **🏆 Competitive Learning Tournaments**: ELO-ranked tournaments with adaptive strategies
- **👨‍🏫 Mentor-Student Networks**: Knowledge distillation and teaching effectiveness tracking
- **🔬 Collaborative Research**: Discovery systems with cross-domain knowledge transfer
- **🎭 Scenario Orchestration**: Sequential scenario execution with cross-learning analysis
- **📊 Rich Visualization**: Learning curves, collaboration networks, and performance analytics
- **🔧 Extensible Architecture**: Modular design for custom scenarios and learning algorithms

## Installation

### From PyPI (coming soon)
```bash
pip install multi-agent-collab-learning
```

### From Source
```bash
git clone https://github.com/briefcasebrain/multi-agent-collab-learning.git
cd multi-agent-collab-learning
pip install -e .
```

### With Optional Dependencies
```bash
# For AI2Thor integration
pip install multi-agent-collab-learning[ai2thor]

# For development
pip install multi-agent-collab-learning[dev]

# All dependencies
pip install multi-agent-collab-learning[all]
```

## Quick Start

```python
from multi_agent_collab_learning import (
    ScenarioOrchestrator,
    CompetitiveLearningTournament,
    MentorStudentNetwork,
    CollaborativeResearchEnvironment,
    ScenarioConfig,
    LearningScenarioType
)

# Create agents (your agent implementation)
agents = create_your_agents()

# Setup competitive tournament
tournament_config = ScenarioConfig(
    scenario_type=LearningScenarioType.COMPETITIVE_TOURNAMENT,
    duration_episodes=50,
    participants=len(agents)
)
tournament = CompetitiveLearningTournament(agents, tournament_config)

# Setup scenario orchestrator
orchestrator = ScenarioOrchestrator(agents)
orchestrator.register_scenario('tournament', tournament)

# Run scenarios
results = orchestrator.run_scenario_suite(environment, ['tournament'])
```

## Architecture

```
multi_agent_collab_learning/
├── core/                    # Core functionality
│   ├── config.py           # Configuration classes
│   ├── knowledge.py        # Knowledge distillation engine
│   └── agents.py           # Base agent classes
├── scenarios/              # Learning scenarios
│   ├── competitive.py      # Tournament scenarios
│   ├── mentor_student.py   # Mentorship scenarios
│   ├── collaborative.py    # Research scenarios
│   └── orchestrator.py     # Scenario management
├── learning/               # Learning algorithms
│   ├── algorithms.py       # Core learning algorithms
│   ├── metrics.py         # Performance metrics
│   └── transfer.py        # Knowledge transfer
└── utils/                  # Utilities
    ├── logging.py          # Logging utilities
    ├── visualization.py    # Plotting functions
    └── helpers.py          # Common utilities
```

## Scenarios

### Competitive Learning Tournaments
- Round-robin and bracket tournaments
- ELO-style performance ranking
- Competitive behavior detection
- Strategy innovation metrics

### Mentor-Student Networks
- Knowledge distillation between agents
- Teaching effectiveness tracking
- Adaptive mentorship relationships
- Learning progress monitoring

### Collaborative Research Environments
- Multi-team research coordination
- Discovery and breakthrough detection
- Cross-domain knowledge sharing
- Innovation metrics calculation

## Configuration

All scenarios use the `ScenarioConfig` class for configuration:

```python
config = ScenarioConfig(
    scenario_type=LearningScenarioType.COMPETITIVE_TOURNAMENT,
    duration_episodes=100,
    participants=6,
    reward_structure="mixed",  # "individual", "collective", "mixed"
    knowledge_sharing_rate=0.3,
    adaptation_frequency=25
)
```

## Visualization

The library includes rich visualization capabilities:

```python
from multi_agent_collab_learning.utils import plot_learning_curves

# Plot agent learning progress
plot_learning_curves(
    learning_data={
        "agent_1": [0.1, 0.3, 0.6, 0.8],
        "agent_2": [0.2, 0.4, 0.5, 0.7]
    },
    title="Agent Learning Progress",
    save_path="learning_curves.png"
)
```

## Requirements

- Python 3.8+
- PyTorch 1.12+
- NumPy 1.21+
- NetworkX 2.8+
- Matplotlib 3.5+

## Development

```bash
# Clone repository
git clone https://github.com/briefcasebrain/multi-agent-collab-learning.git
cd multi-agent-collab-learning

# Install development dependencies
pip install -e .[dev]

# Run tests
pytest

# Run linting
black .
flake8 .
mypy .
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

We welcome contributions from the community! Whether you're fixing bugs, adding features, improving documentation, or suggesting new scenarios, your contributions help make this library better for everyone.

- 📖 Read our [Contributing Guidelines](CONTRIBUTING.md)
- 🐛 Report bugs via [GitHub Issues](https://github.com/briefcasebrain/multi-agent-collab-learning/issues)
- 💡 Suggest features through [GitHub Discussions](https://github.com/briefcasebrain/multi-agent-collab-learning/discussions)
- 🔄 Submit pull requests following our guidelines

## Citation

If you use this library in your research, please cite:

```bibtex
@software{shah2024multiagentcollab,
  title={Multi-Agent Collaborative Learning},
  author={Shah, Aansh},
  year={2024},
  url={https://github.com/briefcasebrain/multi-agent-collab-learning}
}
```

## About Briefcase Brain

[Briefcase Brain](https://github.com/briefcasebrain) develops cutting-edge AI research tools and frameworks. Our mission is to democratize access to advanced artificial intelligence research capabilities through open-source libraries and educational resources.

### Contact

- **Primary Maintainer**: Aansh Shah - [aansh@briefcasebrain.com](mailto:aansh@briefcasebrain.com)
- **Organization**: [github.com/briefcasebrain](https://github.com/briefcasebrain)
- **Issues & Support**: [GitHub Issues](https://github.com/briefcasebrain/multi-agent-collab-learning/issues)

## Acknowledgments

- Inspired by advances in multi-agent reinforcement learning
- Built on PyTorch for neural network components
- Uses NetworkX for collaboration network analysis