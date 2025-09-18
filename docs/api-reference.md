---
layout: default
title: 📚 API Reference
nav_order: 6
---

# API Reference
{: .no_toc }

Complete API documentation for the Multi-Agent Collaborative Learning library.

## Table of Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Core Module

### Configuration Classes

#### `ScenarioConfig`

Central configuration class for all learning scenarios.

```python
@dataclass
class ScenarioConfig:
    scenario_type: LearningScenarioType
    duration_episodes: int = 100
    participants: int = 6
    success_criteria: Dict[str, float] = field(default_factory=dict)
    reward_structure: str = "mixed"
    knowledge_sharing_rate: float = 0.3
    adaptation_frequency: int = 25
    evaluation_metrics: List[str] = field(default_factory=list)
    scenario_parameters: Dict[str, Any] = field(default_factory=dict)
```

**Parameters:**
- `scenario_type`: Type of learning scenario to run
- `duration_episodes`: Number of episodes to execute
- `participants`: Number of participating agents
- `success_criteria`: Scenario-specific success thresholds
- `reward_structure`: One of "individual", "collective", or "mixed"
- `knowledge_sharing_rate`: Rate of knowledge exchange [0, 1]
- `adaptation_frequency`: Episodes between adaptation cycles
- `evaluation_metrics`: List of metrics to track
- `scenario_parameters`: Custom scenario-specific parameters

#### `LearningScenarioType`

Enumeration of available learning scenarios.

```python
class LearningScenarioType(Enum):
    COMPETITIVE_TOURNAMENT = "competitive_tournament"
    MENTOR_STUDENT = "mentor_student"
    COLLABORATIVE_RESEARCH = "collaborative_research"
    MULTI_TASK_COALITION = "multi_task_coalition"
    ADVERSARIAL_COLLABORATION = "adversarial_collaboration"
    FEDERATED_LEARNING = "federated_learning"
    CROSS_MODAL_TRANSFER = "cross_modal_transfer"
    DISTRIBUTED_PROBLEM_SOLVING = "distributed_problem_solving"
    COLLABORATIVE_CREATIVITY = "collaborative_creativity"
    MULTI_AGENT_TEACHING = "multi_agent_teaching"
    SWARM_LEARNING = "swarm_learning"
    HIERARCHICAL_ORGANIZATION = "hierarchical_organization"
```

#### `LearningOutcome`

Data class representing learning outcomes from scenarios.

```python
@dataclass
class LearningOutcome:
    agent_id: str
    scenario_type: LearningScenarioType
    performance_improvement: float
    knowledge_gained: Dict[str, float]
    skills_developed: List[str]
    collaboration_effectiveness: float
    adaptation_speed: float
    teaching_ability: float
    learning_efficiency: float
    timestamp: float = field(default_factory=time.time)
```

### Knowledge Systems

#### `KnowledgeDistillationEngine`

Neural knowledge distillation and transfer engine.

```python
class KnowledgeDistillationEngine(nn.Module):
    def __init__(self, feature_dim: int = 512, distillation_temperature: float = 4.0)
```

**Methods:**

##### `compress_knowledge(teacher_knowledge: torch.Tensor) -> torch.Tensor`

Compress teacher knowledge for efficient transfer.

**Parameters:**
- `teacher_knowledge`: High-dimensional knowledge representation

**Returns:**
- Compressed knowledge tensor

##### `decompress_knowledge(compressed_knowledge: torch.Tensor) -> torch.Tensor`

Decompress knowledge for student agent.

**Parameters:**
- `compressed_knowledge`: Compressed knowledge representation

**Returns:**
- Decompressed knowledge tensor

##### `distill_knowledge(teacher_outputs: torch.Tensor, student_outputs: torch.Tensor) -> torch.Tensor`

Perform knowledge distillation between teacher and student.

**Parameters:**
- `teacher_outputs`: Teacher model outputs (logits)
- `student_outputs`: Student model outputs (logits)

**Returns:**
- Knowledge distillation loss

##### `assess_knowledge_quality(original_knowledge: torch.Tensor, transferred_knowledge: torch.Tensor) -> torch.Tensor`

Assess quality of knowledge transfer.

**Parameters:**
- `original_knowledge`: Original knowledge representation
- `transferred_knowledge`: Transferred knowledge representation

**Returns:**
- Quality score between 0 and 1

---

## Scenarios Module

### Base Scenario

#### `BaseScenario`

Abstract base class for all learning scenarios.

```python
class BaseScenario(ABC):
    def __init__(self, agents: List[Any], config: ScenarioConfig)
```

**Abstract Methods:**

##### `run(environment: Any) -> Dict[str, Any]`

Run the learning scenario.

**Parameters:**
- `environment`: Environment to run the scenario in

**Returns:**
- Dictionary containing scenario results

**Methods:**

##### `validate_agents() -> bool`

Validate that agents are properly configured for this scenario.

**Returns:**
- True if agents are valid, False otherwise

##### `get_results() -> Dict[str, Any]`

Get the results from the last scenario run.

**Returns:**
- Copy of scenario results dictionary

### Competitive Learning

#### `CompetitiveLearningTournament`

Implements competitive learning tournament scenarios.

```python
class CompetitiveLearningTournament(BaseScenario):
    def __init__(self, agents: List[Any], config: ScenarioConfig)
```

**Methods:**

##### `run(environment: Any) -> Dict[str, Any]`

Run complete competitive learning tournament.

**Returns:**
- Dictionary with keys:
  - `matches`: List of all match results
  - `final_rankings`: Agent ID to final score mapping
  - `tournament_winner`: ID of winning agent
  - `learning_outcomes`: List of LearningOutcome objects

**Private Methods:**

##### `_execute_competitive_match(match: Tuple[str, str], environment: Any) -> Dict[str, Any]`

Execute a competitive match between two agents.

##### `_update_performance_rankings(match_result: Dict[str, Any]) -> None`

Update ELO-style performance rankings based on match results.

### Mentor-Student Learning

#### `MentorStudentNetwork`

Implements mentor-student learning relationships.

```python
class MentorStudentNetwork(BaseScenario):
    def __init__(self, agents: List[Any], config: ScenarioConfig)
```

**Attributes:**
- `knowledge_distillation`: KnowledgeDistillationEngine instance
- `mentorship_graph`: NetworkX DiGraph of mentorship relationships
- `learning_progress`: Dict tracking student learning progress
- `teaching_effectiveness`: Dict tracking mentor effectiveness

**Methods:**

##### `run(environment: Any) -> Dict[str, Any]`

Run mentor-student learning scenario.

**Returns:**
- Dictionary with keys:
  - `mentorship_outcomes`: List of mentorship interaction results
  - `knowledge_transfer_metrics`: Overall transfer effectiveness metrics
  - `teaching_effectiveness`: Per-mentor teaching metrics
  - `learning_progress`: Per-student learning trajectories
  - `emergent_teaching_strategies`: Identified teaching patterns

**Private Methods:**

##### `_transfer_knowledge(mentor: Any, student: Any, teaching_content: Dict[str, Any]) -> Dict[str, Any]`

Transfer knowledge from mentor to student using neural distillation.

##### `_adapt_mentorship_relationships() -> None`

Dynamically adapt mentorship relationships based on effectiveness.

### Collaborative Research

#### `CollaborativeResearchEnvironment`

Implements collaborative research scenario with discovery systems.

```python
class CollaborativeResearchEnvironment(BaseScenario):
    def __init__(self, agents: List[Any], config: ScenarioConfig)
```

**Attributes:**
- `research_topics`: List of available research topics
- `knowledge_base`: Global knowledge repository
- `research_teams`: Dict of formed research teams
- `discovery_history`: List of all discoveries made
- `collaboration_network`: NetworkX Graph of team collaborations

**Methods:**

##### `run(environment: Any) -> Dict[str, Any]`

Run collaborative research scenario.

**Returns:**
- Dictionary with keys:
  - `discoveries`: List of all discoveries made
  - `research_outcomes`: Per-team research results
  - `collaboration_patterns`: Network analysis of collaborations
  - `knowledge_evolution`: Timeline of knowledge growth
  - `innovation_metrics`: Innovation and creativity measurements

**Private Methods:**

##### `_form_research_teams() -> None`

Form research teams based on complementary skills.

##### `_attempt_discovery(research_result: Dict[str, Any], topic: Dict[str, Any]) -> Dict[str, Any]`

Determine if research results constitute a discovery.

### Scenario Orchestration

#### `ScenarioOrchestrator`

Orchestrates multiple collaborative learning scenarios.

```python
class ScenarioOrchestrator:
    def __init__(self, agents: List[Any])
```

**Attributes:**
- `agents`: List of participating agents
- `scenarios`: Dict of registered scenarios
- `scenario_results`: Historical results from all runs
- `cross_scenario_learning`: Cross-scenario learning analysis

**Methods:**

##### `register_scenario(scenario_name: str, scenario_instance: BaseScenario) -> None`

Register a learning scenario.

**Parameters:**
- `scenario_name`: Name identifier for the scenario
- `scenario_instance`: Instance of a scenario class

##### `unregister_scenario(scenario_name: str) -> None`

Unregister a learning scenario.

**Parameters:**
- `scenario_name`: Name of the scenario to remove

##### `run_scenario_suite(environment: Any, scenario_sequence: Optional[List[str]] = None, inter_scenario_adaptation: bool = True, **kwargs) -> Dict[str, Any]`

Run a suite of collaborative learning scenarios.

**Parameters:**
- `environment`: Environment to run scenarios in
- `scenario_sequence`: Ordered list of scenarios to run
- `inter_scenario_adaptation`: Whether to allow agent adaptation between scenarios
- `**kwargs`: Additional arguments passed to scenarios

**Returns:**
- Dictionary with keys:
  - `scenario_results`: Results from each scenario
  - `cross_scenario_learning`: Learning transfer analysis
  - `agent_development`: Long-term agent development metrics
  - `emergent_capabilities`: Identified emergent behaviors
  - `execution_timeline`: Timing information for each scenario

##### `run_single_scenario(scenario_name: str, environment: Any, **kwargs) -> Dict[str, Any]`

Run a single scenario.

**Parameters:**
- `scenario_name`: Name of the scenario to run
- `environment`: Environment to run the scenario in
- `**kwargs`: Additional arguments passed to scenario

**Returns:**
- Scenario results dictionary

##### `get_scenario_results(scenario_name: Optional[str] = None) -> Union[Dict[str, Any], Dict[str, Dict[str, Any]]]`

Get results from scenarios.

**Parameters:**
- `scenario_name`: Specific scenario name (optional)

**Returns:**
- Scenario results or all results if no name specified

##### `export_results(filepath: str, format: str = 'json') -> None`

Export scenario results to file.

**Parameters:**
- `filepath`: Path to save file
- `format`: Export format ('json' or 'csv')

---

## Utilities Module

### Logging

#### `setup_logger(name: str = "multi_agent_collab_learning", level: Union[str, int] = logging.INFO, log_file: Optional[Union[str, Path]] = None, format_string: Optional[str] = None) -> logging.Logger`

Set up a logger for the multi-agent collaborative learning library.

**Parameters:**
- `name`: Name of the logger
- `level`: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `log_file`: Optional file path to write logs to
- `format_string`: Custom format string for log messages

**Returns:**
- Configured logger instance

#### `get_logger(name: str = "multi_agent_collab_learning") -> logging.Logger`

Get an existing logger or create a new one with default settings.

**Parameters:**
- `name`: Name of the logger

**Returns:**
- Logger instance

### Visualization

#### `plot_learning_curves(learning_data: Dict[str, List[float]], title: str = "Learning Curves", xlabel: str = "Episodes", ylabel: str = "Performance", save_path: Optional[Union[str, Path]] = None, figsize: tuple = (10, 6), show_legend: bool = True) -> plt.Figure`

Plot learning curves for multiple agents.

**Parameters:**
- `learning_data`: Dictionary mapping agent IDs to performance values
- `title`: Plot title
- `xlabel`: X-axis label
- `ylabel`: Y-axis label
- `save_path`: Optional path to save the plot
- `figsize`: Figure size tuple
- `show_legend`: Whether to show legend

**Returns:**
- Matplotlib figure object

#### `plot_scenario_comparison(scenario_results: Dict[str, Dict[str, float]], metrics: List[str], title: str = "Scenario Performance Comparison", save_path: Optional[Union[str, Path]] = None, figsize: tuple = (12, 8)) -> plt.Figure`

Plot comparison of performance across different scenarios.

**Parameters:**
- `scenario_results`: Nested dict of scenario -> agent -> metric values
- `metrics`: List of metrics to plot
- `title`: Plot title
- `save_path`: Optional path to save the plot
- `figsize`: Figure size tuple

**Returns:**
- Matplotlib figure object

#### `plot_collaboration_network(collaboration_data: Dict[str, Dict[str, float]], title: str = "Agent Collaboration Network", save_path: Optional[Union[str, Path]] = None, figsize: tuple = (10, 10)) -> plt.Figure`

Plot agent collaboration network as a graph.

**Parameters:**
- `collaboration_data`: Nested dict of agent -> agent -> collaboration strength
- `title`: Plot title
- `save_path`: Optional path to save the plot
- `figsize`: Figure size tuple

**Returns:**
- Matplotlib figure object

#### `plot_knowledge_evolution(knowledge_timeline: List[Dict[str, Any]], title: str = "Knowledge Evolution Timeline", save_path: Optional[Union[str, Path]] = None, figsize: tuple = (12, 6)) -> plt.Figure`

Plot knowledge evolution over time.

**Parameters:**
- `knowledge_timeline`: List of knowledge states over time
- `title`: Plot title
- `save_path`: Optional path to save the plot
- `figsize`: Figure size tuple

**Returns:**
- Matplotlib figure object

---

## Agent Interface

While not part of the core library, agents should implement this interface for full compatibility:

### Required Attributes

```python
class AgentInterface:
    agent_id: str                              # Unique agent identifier
    role: str                                  # Agent role (e.g., "LEADER", "FOLLOWER")
    collaboration_mode: str                    # Current collaboration mode
    collaboration_metrics: Dict[str, float]    # Performance metrics
    behavior_weights: Dict[str, float]         # Behavior parameters
```

### Required Methods

```python
def select_action(self, state: Any) -> Dict[str, Any]:
    """
    Select action given current state.

    Returns:
        Dictionary with keys:
        - 'action': Action tensor or value
        - 'confidence': Action confidence [0, 1]
        - 'uncertainty': Action uncertainty [0, 1]
    """
    pass
```

### Optional Attributes

```python
current_research_topic: Optional[str]       # For research scenarios
confidence_level: float                     # Overall agent confidence
motivation_level: float                     # For mentor-student scenarios
```

---

## Environment Interface

Environments should implement this minimal interface:

### Required Attributes

```python
class EnvironmentInterface:
    controller: Any  # Environment controller with reset() method
```

### Required Methods

```python
def _get_multimodal_state(self) -> Any:
    """Return current environment state."""
    pass

def _execute_agent_action(self, agent: Any, action: Any) -> bool:
    """
    Execute agent action in environment.

    Returns:
        Success boolean
    """
    pass
```

---

## Error Handling

### Common Exceptions

The library defines several custom exceptions for better error handling:

```python
class ScenarioValidationError(ValueError):
    """Raised when scenario validation fails."""
    pass

class AgentCompatibilityError(TypeError):
    """Raised when agent doesn't implement required interface."""
    pass

class KnowledgeTransferError(RuntimeError):
    """Raised when knowledge transfer fails."""
    pass
```

### Error Recovery

Most methods include graceful error recovery:

- Invalid configurations are validated at initialization
- Missing agent methods are detected and handled gracefully
- Network errors in knowledge transfer are retried automatically
- Scenario failures are logged but don't crash the orchestrator

This API reference provides comprehensive documentation for all public interfaces in the Multi-Agent Collaborative Learning library. For implementation examples, see the [Examples](examples.html) section.