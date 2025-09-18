# Contributing to Multi-Agent Collaborative Learning

Thank you for your interest in contributing to the Multi-Agent Collaborative Learning library! We welcome contributions from the community and are excited to work with you.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Issue Guidelines](#issue-guidelines)
- [Community](#community)

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Basic understanding of multi-agent systems and reinforcement learning

### Quick Setup

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/briefcasebrain/multi-agent-collab-learning.git
   cd multi-agent-collab-learning
   ```

3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install development dependencies:
   ```bash
   pip install -e .[dev]
   ```

5. Run tests to verify setup:
   ```bash
   pytest
   ```

## Development Setup

### Development Dependencies

The development environment includes:
- **pytest**: Testing framework
- **black**: Code formatting
- **flake8**: Linting
- **mypy**: Type checking
- **sphinx**: Documentation generation

### Environment Configuration

Create a `.env` file for local development (optional):
```env
# Example configuration
LOG_LEVEL=DEBUG
ENABLE_VISUALIZATION=true
```

### IDE Setup

We recommend using an IDE with Python support. For VS Code, useful extensions include:
- Python
- Pylance
- Black Formatter
- GitLens

## How to Contribute

### Types of Contributions

We welcome several types of contributions:

1. **Bug Reports**: Help us identify and fix issues
2. **Feature Requests**: Suggest new functionality
3. **Code Contributions**: Implement new features or fix bugs
4. **Documentation**: Improve or add documentation
5. **Examples**: Add usage examples or tutorials
6. **Performance Improvements**: Optimize existing code

### Areas of Interest

We're particularly interested in contributions to:
- New collaborative learning scenarios
- Performance optimizations
- Integration with popular RL frameworks
- Advanced visualization features
- Additional knowledge transfer mechanisms
- Multi-modal learning support

## Coding Standards

### Code Style

- **Formatting**: Use Black with default settings
- **Line Length**: 100 characters maximum
- **Imports**: Use isort for import organization
- **Type Hints**: Include type hints for all public APIs

### Naming Conventions

- **Classes**: PascalCase (e.g., `ScenarioOrchestrator`)
- **Functions/Methods**: snake_case (e.g., `run_scenario`)
- **Variables**: snake_case (e.g., `learning_rate`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `MAX_EPISODES`)

### Code Organization

- Keep functions focused and single-purpose
- Use descriptive variable and function names
- Add docstrings to all public classes and methods
- Follow the existing architecture patterns

### Example Code Style

```python
from typing import Dict, List, Optional
import torch
import numpy as np


class ExampleScenario:
    """Example scenario demonstrating coding standards.

    Args:
        agents: List of participating agents
        config: Scenario configuration
    """

    def __init__(self, agents: List['Agent'], config: 'ScenarioConfig') -> None:
        self.agents = agents
        self.config = config
        self._results: Dict[str, float] = {}

    def run_episode(self, environment: 'Environment') -> Dict[str, Any]:
        """Run a single episode of the scenario.

        Args:
            environment: The environment to run in

        Returns:
            Dictionary containing episode results

        Raises:
            ValueError: If agents are not properly configured
        """
        if not self._validate_setup():
            raise ValueError("Invalid scenario setup")

        # Implementation here
        return self._results
```

## Testing

### Test Structure

Tests are organized in the `tests/` directory:
```
tests/
├── unit/           # Unit tests
├── integration/    # Integration tests
├── scenarios/      # Scenario-specific tests
└── fixtures/       # Test fixtures and utilities
```

### Writing Tests

- Write unit tests for all new functionality
- Include integration tests for complex features
- Use descriptive test names
- Test both success and failure cases

### Test Example

```python
import pytest
from multi_agent_collab_learning import ScenarioConfig, LearningScenarioType


class TestScenarioConfig:
    """Test ScenarioConfig class."""

    def test_valid_config_creation(self):
        """Test creating a valid configuration."""
        config = ScenarioConfig(
            scenario_type=LearningScenarioType.COMPETITIVE_TOURNAMENT,
            duration_episodes=50,
            participants=4
        )

        assert config.scenario_type == LearningScenarioType.COMPETITIVE_TOURNAMENT
        assert config.duration_episodes == 50
        assert config.participants == 4

    def test_invalid_duration_raises_error(self):
        """Test that invalid duration raises ValueError."""
        with pytest.raises(ValueError, match="duration_episodes must be positive"):
            ScenarioConfig(
                scenario_type=LearningScenarioType.COMPETITIVE_TOURNAMENT,
                duration_episodes=-1
            )
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=multi_agent_collab_learning

# Run specific test file
pytest tests/unit/test_config.py

# Run with verbose output
pytest -v
```

## Documentation

### Docstring Format

Use Google-style docstrings:

```python
def example_function(param1: str, param2: Optional[int] = None) -> bool:
    """Example function with proper docstring.

    Args:
        param1: Description of param1
        param2: Description of param2, defaults to None

    Returns:
        Description of return value

    Raises:
        ValueError: Description of when this is raised

    Example:
        >>> result = example_function("test")
        >>> print(result)
        True
    """
    pass
```

### Documentation Building

Documentation is built automatically, but you can build locally:

```bash
# Build documentation
cd docs
bundle exec jekyll serve

# View at http://localhost:4000
```

## Pull Request Process

### Before Submitting

1. **Update Documentation**: Ensure documentation reflects your changes
2. **Add Tests**: Include appropriate tests for new functionality
3. **Run Quality Checks**:
   ```bash
   # Format code
   black .

   # Check linting
   flake8 .

   # Type checking
   mypy multi_agent_collab_learning

   # Run tests
   pytest
   ```

### Pull Request Guidelines

1. **Create Feature Branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Commit Messages**: Use clear, descriptive commit messages:
   ```
   Add competitive behavior detection to tournaments

   - Implement aggressive/defensive strategy detection
   - Add confidence-based behavior classification
   - Include tests for behavior detection logic

   Fixes #123
   ```

3. **Keep PRs Focused**: One feature or fix per pull request

4. **Update CHANGELOG**: Add entry describing your changes

### PR Template

When creating a pull request, include:

- **Description**: What does this PR do?
- **Motivation**: Why is this change needed?
- **Testing**: How was this tested?
- **Checklist**: Confirm all requirements are met

## Issue Guidelines

### Bug Reports

Include:
- **Environment**: Python version, OS, library version
- **Reproduction Steps**: Minimal code to reproduce the issue
- **Expected Behavior**: What should happen
- **Actual Behavior**: What actually happens
- **Error Messages**: Full error traceback if applicable

### Feature Requests

Include:
- **Use Case**: Why is this feature needed?
- **Proposed Solution**: How should it work?
- **Alternatives**: Other approaches considered
- **Examples**: Code examples of desired usage

### Issue Labels

We use labels to categorize issues:
- `bug`: Something isn't working
- `enhancement`: New feature or request
- `documentation`: Improvements or additions to documentation
- `good first issue`: Good for newcomers
- `help wanted`: Extra attention is needed

## Community

### Communication Channels

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Email**: For private inquiries: aansh@briefcasebrain.com

### Getting Help

- Check existing issues and documentation first
- Provide minimal reproducible examples
- Be respectful and patient

### Recognition

Contributors are recognized in:
- CHANGELOG.md for each release
- GitHub contributors page
- Special recognition for significant contributions

## Development Workflow

### Typical Workflow

1. **Check Issues**: Look for existing issues or create one
2. **Fork & Clone**: Get the code locally
3. **Create Branch**: Feature branch from main
4. **Develop**: Write code following standards
5. **Test**: Ensure all tests pass
6. **Document**: Update documentation as needed
7. **Submit PR**: Create pull request with description
8. **Review**: Address feedback from maintainers
9. **Merge**: PR gets merged after approval

### Release Process

Releases follow semantic versioning (MAJOR.MINOR.PATCH):
- **PATCH**: Bug fixes
- **MINOR**: New features (backward compatible)
- **MAJOR**: Breaking changes

### Maintenance

The project maintainers will:
- Review PRs within 1-2 weeks
- Provide feedback and guidance
- Help with technical questions
- Coordinate releases

Thank you for contributing to Multi-Agent Collaborative Learning! Your contributions help make this library better for everyone.