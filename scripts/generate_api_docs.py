#!/usr/bin/env python3
"""
Generate comprehensive API documentation for the multi-agent collaborative learning library.
"""

import os
import sys
import inspect
from pathlib import Path
from typing import Dict, List, Any

# Add the library to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the library
import multi_agent_collab_learning
from multi_agent_collab_learning.core import config, knowledge
from multi_agent_collab_learning.scenarios import competitive, mentor_student, collaborative, orchestrator
from multi_agent_collab_learning.utils import logging, visualization


def generate_module_docs(module, module_name: str, output_path: Path):
    """Generate documentation for a Python module."""

    content = f"""---
layout: default
title: {module_name}
parent: API Reference
nav_order: {hash(module_name) % 100}
---

# {module_name}
{{: .no_toc }}

{module.__doc__ or f"API documentation for {module_name} module."}

## Table of Contents
{{: .no_toc .text-delta }}

1. TOC
{{:toc}}

---

"""

    # Get all classes in the module
    classes = [obj for name, obj in inspect.getmembers(module, inspect.isclass)
               if obj.__module__ == module.__name__]

    # Get all functions in the module
    functions = [obj for name, obj in inspect.getmembers(module, inspect.isfunction)
                 if obj.__module__ == module.__name__]

    # Document classes
    if classes:
        content += "## Classes\n\n"
        for cls in classes:
            content += generate_class_docs(cls)

    # Document functions
    if functions:
        content += "## Functions\n\n"
        for func in functions:
            content += generate_function_docs(func)

    # Write to file
    output_file = output_path / f"{module_name.replace('.', '_')}.md"
    with open(output_file, 'w') as f:
        f.write(content)

    print(f"Generated documentation for {module_name}")


def generate_class_docs(cls) -> str:
    """Generate documentation for a class."""
    content = f"### {cls.__name__}\n\n"

    if cls.__doc__:
        content += f"{cls.__doc__}\n\n"

    # Constructor
    try:
        sig = inspect.signature(cls.__init__)
        content += f"#### Constructor\n\n```python\n{cls.__name__}{sig}\n```\n\n"

        if cls.__init__.__doc__:
            content += f"{cls.__init__.__doc__}\n\n"
    except (ValueError, TypeError):
        pass

    # Methods
    methods = [method for name, method in inspect.getmembers(cls, inspect.ismethod)
               if not name.startswith('_') or name in ['__call__', '__str__', '__repr__']]

    if methods:
        content += "#### Methods\n\n"
        for method in methods:
            content += generate_method_docs(method)

    # Public attributes/properties
    properties = [prop for name, prop in inspect.getmembers(cls, inspect.isdatadescriptor)
                  if not name.startswith('_')]

    if properties:
        content += "#### Properties\n\n"
        for prop in properties:
            content += f"- **{prop.fget.__name__ if hasattr(prop, 'fget') else 'property'}**"
            if hasattr(prop, 'fget') and prop.fget.__doc__:
                content += f": {prop.fget.__doc__}"
            content += "\n"

    content += "\n---\n\n"
    return content


def generate_method_docs(method) -> str:
    """Generate documentation for a method."""
    content = f"##### {method.__name__}\n\n"

    try:
        sig = inspect.signature(method)
        content += f"```python\n{method.__name__}{sig}\n```\n\n"
    except (ValueError, TypeError):
        pass

    if method.__doc__:
        content += f"{method.__doc__}\n\n"

    return content


def generate_function_docs(func) -> str:
    """Generate documentation for a function."""
    content = f"### {func.__name__}\n\n"

    try:
        sig = inspect.signature(func)
        content += f"```python\n{func.__name__}{sig}\n```\n\n"
    except (ValueError, TypeError):
        pass

    if func.__doc__:
        content += f"{func.__doc__}\n\n"

    content += "\n---\n\n"
    return content


def main():
    """Generate API documentation."""
    print("Generating API documentation...")

    # Create output directory
    docs_dir = Path("docs")
    api_dir = docs_dir / "api"
    api_dir.mkdir(parents=True, exist_ok=True)

    # Create API index page
    api_index = """---
layout: default
title: API Reference
nav_order: 4
has_children: true
---

# API Reference

Complete API documentation for the Multi-Agent Collaborative Learning library.

## Modules

- [Core Configuration](core_config.html) - Configuration classes and enums
- [Knowledge Distillation](core_knowledge.html) - Knowledge transfer mechanisms
- [Competitive Scenarios](scenarios_competitive.html) - Tournament implementations
- [Mentor-Student Networks](scenarios_mentor_student.html) - Mentorship scenarios
- [Collaborative Research](scenarios_collaborative.html) - Research environments
- [Scenario Orchestrator](scenarios_orchestrator.html) - Scenario management
- [Utilities](utils_logging.html) - Logging and visualization utilities

## Quick Reference

### Core Classes

- `ScenarioConfig` - Configuration for learning scenarios
- `LearningScenarioType` - Enum of available scenario types
- `KnowledgeDistillationEngine` - Neural knowledge transfer

### Scenario Classes

- `CompetitiveLearningTournament` - Tournament-style competition
- `MentorStudentNetwork` - Mentor-student relationships
- `CollaborativeResearchEnvironment` - Multi-team research
- `ScenarioOrchestrator` - Multi-scenario management

### Utility Functions

- `setup_logger()` - Configure logging
- `plot_learning_curves()` - Visualize learning progress
"""

    with open(api_dir / "index.md", 'w') as f:
        f.write(api_index)

    # Generate documentation for each module
    modules = [
        (config, "core.config"),
        (knowledge, "core.knowledge"),
        (competitive, "scenarios.competitive"),
        (mentor_student, "scenarios.mentor_student"),
        (collaborative, "scenarios.collaborative"),
        (orchestrator, "scenarios.orchestrator"),
        (logging, "utils.logging"),
        (visualization, "utils.visualization"),
    ]

    for module, name in modules:
        try:
            generate_module_docs(module, name, api_dir)
        except Exception as e:
            print(f"Warning: Failed to generate docs for {name}: {e}")

    print(f"API documentation generated in {api_dir}")


if __name__ == "__main__":
    main()