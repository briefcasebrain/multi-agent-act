---
layout: default
title: 💡 Examples
nav_order: 5
---

# <i class="fas fa-code"></i> Examples
{: .no_toc }

<div class="content-section" style="background: linear-gradient(135deg, #fff3e0, #fce4ec); text-align: center;">
  <h2 style="margin-top: 0; color: #1e293b;"><i class="fas fa-play"></i> Interactive Code Examples</h2>
  <p style="color: #475569;">Learn through hands-on examples with copy-paste ready code and detailed explanations</p>
  <div style="display: flex; gap: 1rem; justify-content: center; flex-wrap: wrap; margin-top: 1.5rem;">
    <span class="badge" style="background: #10b981; color: white; padding: 0.5rem 1rem; border-radius: 0.5rem;">
      <i class="fas fa-copy"></i> Copy-Paste Ready
    </span>
    <span class="badge" style="background: #3b82f6; color: white; padding: 0.5rem 1rem; border-radius: 0.5rem;">
      <i class="fas fa-graduation-cap"></i> Educational
    </span>
    <span class="badge" style="background: #8b5cf6; color: white; padding: 0.5rem 1rem; border-radius: 0.5rem;">
      <i class="fas fa-rocket"></i> Production Ready
    </span>
  </div>
</div>

{: .important }
> **🎯 Learning Path**: Start with Basic Examples, then progress to Advanced Scenarios. Each example builds upon previous concepts.

<div class="feature-grid">
  <div class="feature-card">
    <div class="feature-icon">🏆</div>
    <h4>Tournament Examples</h4>
    <p>Competitive learning scenarios</p>
  </div>
  <div class="feature-card">
    <div class="feature-icon">👨‍🏫</div>
    <h4>Mentorship Networks</h4>
    <p>Knowledge transfer examples</p>
  </div>
  <div class="feature-card">
    <div class="feature-icon">🔬</div>
    <h4>Research Collaboration</h4>
    <p>Discovery-based scenarios</p>
  </div>
  <div class="feature-card">
    <div class="feature-icon">🎭</div>
    <h4>Custom Scenarios</h4>
    <p>Build your own scenarios</p>
  </div>
</div>

## Table of Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## <i class="fas fa-play-circle"></i> Basic Examples

{: .note }
**💡 Quick Start**: These examples require minimal setup and demonstrate core concepts. Perfect for getting familiar with the library.

### 1. <i class="fas fa-trophy"></i> Simple Tournament

<div class="content-section">

{: .highlight }
**🎯 Goal**: Create a basic competitive tournament with 4 agents and understand the core concepts.

**📋 What You'll Learn**:
- Agent creation and configuration
- Tournament setup and execution
- Result analysis and interpretation

{: .warning }
**⚠️ Prerequisites**: Make sure you have the library installed: `pip install multi-agent-collab-learning`

</div>

**Step 1: Import Required Components**
```python
from multi_agent_collab_learning import (
    CompetitiveLearningTournament,
    ScenarioConfig,
    LearningScenarioType
)
import torch
import random
```

**Step 2: Create a Simple Agent Class**
```python
class SimpleAgent:
    """A basic agent with skill-based decision making."""

    def __init__(self, agent_id: str, skill_level: float = 0.5):
        self.agent_id = agent_id
        self.skill_level = skill_level  # Agent's base skill (0.0 to 1.0)
        self.collaboration_metrics = {}
        self.behavior_weights = {'individual_reward': skill_level}

        # Track performance over time
        self.wins = 0
        self.losses = 0
        self.games_played = 0

    def select_action(self, state):
        """Select action based on skill level with some randomness."""

        # Add some variability to make it interesting
        performance_variance = random.uniform(-0.1, 0.1)
        success_probability = max(0, min(1, self.skill_level + performance_variance))

        # Make decision (1 = aggressive, 0 = defensive)
        action = 1 if random.random() < success_probability else 0

        return {
            'action': torch.tensor([action]),
            'confidence': self.skill_level,
            'uncertainty': 1 - self.skill_level,
            'strategy': 'aggressive' if action == 1 else 'defensive'
        }

    def update_performance(self, won: bool):
        """Update agent's performance statistics."""
        self.games_played += 1
        if won:
            self.wins += 1
        else:
            self.losses += 1

    @property
    def win_rate(self) -> float:
        """Calculate current win rate."""
        return self.wins / max(1, self.games_played)
```

{: .note }
**🔍 Code Explanation**:
- `skill_level`: Determines base performance (higher = better)
- `select_action()`: Makes decisions with controlled randomness
- Performance tracking helps analyze improvement over time

# Create agents with different skill levels
agents = [
    SimpleAgent("beginner", 0.3),
    SimpleAgent("intermediate", 0.6),
    SimpleAgent("advanced", 0.8),
    SimpleAgent("expert", 0.9)
]

# Configure tournament
config = ScenarioConfig(
    scenario_type=LearningScenarioType.COMPETITIVE_TOURNAMENT,
    duration_episodes=10,
    participants=len(agents)
)

# Simple environment
class SimpleEnvironment:
    def __init__(self):
        self.controller = type('Controller', (), {'reset': lambda: None})()

    def _get_multimodal_state(self):
        import torch
        return torch.randn(1, 16)

    def _execute_agent_action(self, agent, action):
        # Success based on agent skill and some randomness
        return action == 1 and agent.skill_level > 0.5

# Run tournament
tournament = CompetitiveLearningTournament(agents, config)
results = tournament.run(SimpleEnvironment())

print(f"Winner: {results['tournament_winner']}")
for agent_id, score in results['final_rankings'].items():
    print(f"{agent_id}: {score:.2f}")
```

### 2. Mentor-Student Learning

Implementing knowledge transfer between agents:

```python
from multi_agent_collab_learning import MentorStudentNetwork

# Create agents with mentor/student roles
class LearningAgent(SimpleAgent):
    def __init__(self, agent_id: str, role: str, initial_skill: float):
        super().__init__(agent_id, initial_skill)
        self.role = role
        self.learning_history = []

    def learn_from_mentor(self, knowledge):
        """Simulate learning from mentor."""
        if self.role == "STUDENT":
            # Improve skill based on mentor knowledge
            improvement = knowledge * 0.1
            self.skill_level = min(1.0, self.skill_level + improvement)
            self.learning_history.append(improvement)

# Create mentor-student pairs
agents = [
    LearningAgent("mentor_1", "LEADER", 0.9),      # High skill mentor
    LearningAgent("student_1", "FOLLOWER", 0.3),   # Low skill student
    LearningAgent("mentor_2", "COORDINATOR", 0.8),
    LearningAgent("student_2", "COOPERATOR", 0.4),
]

config = ScenarioConfig(
    scenario_type=LearningScenarioType.MENTOR_STUDENT,
    duration_episodes=20,
    knowledge_sharing_rate=0.5
)

# Run mentorship scenario
mentor_network = MentorStudentNetwork(agents, config)
results = mentor_network.run(SimpleEnvironment())

print("Teaching Effectiveness:")
for mentor, effectiveness in results['teaching_effectiveness'].items():
    print(f"{mentor}: {effectiveness['average_effectiveness']:.2f}")
```

## Advanced Examples

### 3. Multi-Modal Learning Environment

Integrating with AI2Thor for realistic environments:

```python
try:
    import ai2thor.controller

    class ThorEnvironment:
        def __init__(self, scene: str = "FloorPlan1"):
            self.controller = ai2thor.controller.Controller(
                agentMode="default",
                scene=scene,
                gridSize=0.25,
                width=300,
                height=300
            )

        def _get_multimodal_state(self):
            """Get visual and semantic state from AI2Thor."""
            import torch
            import numpy as np

            # Get current event from AI2Thor
            event = self.controller.last_event

            # Extract visual features (simplified)
            frame = event.frame
            visual_features = torch.from_numpy(
                np.mean(frame, axis=2).flatten()[:64]
            ).float().unsqueeze(0)

            # Add semantic information
            objects = len(event.metadata['objects'])
            semantic_features = torch.tensor([
                event.metadata['agent']['position']['x'],
                event.metadata['agent']['position']['z'],
                event.metadata['agent']['rotation']['y'],
                objects
            ]).float().unsqueeze(0)

            # Combine features
            return torch.cat([visual_features, semantic_features], dim=1)

        def _execute_agent_action(self, agent, action):
            """Execute action in AI2Thor environment."""
            action_mapping = {
                0: 'MoveAhead',
                1: 'RotateLeft',
                2: 'RotateRight',
                3: 'LookUp',
                4: 'LookDown'
            }

            thor_action = action_mapping.get(int(action), 'Pass')
            event = self.controller.step(thor_action)

            return event.metadata['lastActionSuccess']

    # Use with navigation-focused agents
    class NavigationAgent(SimpleAgent):
        def __init__(self, agent_id: str):
            super().__init__(agent_id, 0.5)
            self.visited_positions = set()

        def select_action(self, state):
            import torch

            # Extract position info from state
            pos_x, pos_z = state[0, -4], state[0, -3]
            current_pos = (round(pos_x.item(), 1), round(pos_z.item(), 1))

            # Exploration bonus for new positions
            if current_pos not in self.visited_positions:
                self.visited_positions.add(current_pos)
                exploration_bonus = 0.2
            else:
                exploration_bonus = 0.0

            # Simple navigation policy
            action = self._choose_navigation_action(state)
            confidence = self.skill_level + exploration_bonus

            return {
                'action': torch.tensor([action]),
                'confidence': min(1.0, confidence),
                'uncertainty': 1 - confidence
            }

        def _choose_navigation_action(self, state):
            import random
            # Simplified: random exploration with slight forward bias
            if random.random() < 0.5:
                return 0  # MoveAhead
            else:
                return random.choice([1, 2])  # Turn left or right

    # Run collaborative navigation
    nav_agents = [NavigationAgent(f"nav_agent_{i}") for i in range(3)]

    # Use research scenario for exploration
    config = ScenarioConfig(
        scenario_type=LearningScenarioType.COLLABORATIVE_RESEARCH,
        duration_episodes=15,
        participants=len(nav_agents)
    )

    from multi_agent_collab_learning import CollaborativeResearchEnvironment
    research_env = CollaborativeResearchEnvironment(nav_agents, config)

    thor_environment = ThorEnvironment()
    results = research_env.run(thor_environment)

    print(f"Discoveries made: {len(results['discoveries'])}")
    thor_environment.controller.stop()

except ImportError:
    print("AI2Thor not available. Install with: pip install ai2thor")
```

### 4. Custom Knowledge Distillation

Implementing custom knowledge transfer mechanisms:

```python
from multi_agent_collab_learning.core.knowledge import KnowledgeDistillationEngine
import torch
import torch.nn as nn

class AdvancedKnowledgeEngine(KnowledgeDistillationEngine):
    """Extended knowledge distillation with custom transfer methods."""

    def __init__(self, feature_dim: int = 512):
        super().__init__(feature_dim)

        # Add custom components
        self.skill_classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 8)  # 8 skill categories
        )

        self.adaptation_network = nn.LSTM(
            input_size=feature_dim,
            hidden_size=128,
            num_layers=2,
            batch_first=True
        )

    def transfer_specific_skill(self, teacher_knowledge: torch.Tensor,
                              skill_type: int) -> torch.Tensor:
        """Transfer knowledge for a specific skill."""

        # Classify teacher's skills
        skill_distribution = self.skill_classifier(teacher_knowledge)
        skill_mask = torch.zeros_like(skill_distribution)
        skill_mask[:, skill_type] = 1.0

        # Focus on specific skill
        focused_knowledge = teacher_knowledge * skill_mask.sum(dim=1, keepdim=True)

        # Compress and transfer
        compressed = self.compress_knowledge(focused_knowledge)
        transferred = self.decompress_knowledge(compressed)

        return transferred

    def adaptive_knowledge_transfer(self, knowledge_sequence: List[torch.Tensor]) -> torch.Tensor:
        """Transfer knowledge considering temporal adaptation."""

        if len(knowledge_sequence) < 2:
            return knowledge_sequence[0] if knowledge_sequence else torch.zeros(1, self.feature_dim)

        # Stack sequence
        sequence_tensor = torch.stack(knowledge_sequence).unsqueeze(0)

        # Process through LSTM for temporal patterns
        output, (hidden, cell) = self.adaptation_network(sequence_tensor)

        # Return final adapted knowledge
        return output[0, -1, :].unsqueeze(0)

# Use custom knowledge engine
class SkillSpecificAgent(SimpleAgent):
    def __init__(self, agent_id: str, specialized_skill: int):
        super().__init__(agent_id)
        self.specialized_skill = specialized_skill
        self.knowledge_history = []

    def receive_knowledge(self, knowledge_engine: AdvancedKnowledgeEngine,
                         teacher_knowledge: torch.Tensor):
        """Receive and process knowledge from teacher."""

        # Get skill-specific knowledge
        transferred = knowledge_engine.transfer_specific_skill(
            teacher_knowledge, self.specialized_skill
        )

        # Store in history
        self.knowledge_history.append(transferred)

        # Adaptive processing if we have history
        if len(self.knowledge_history) > 1:
            adapted = knowledge_engine.adaptive_knowledge_transfer(
                self.knowledge_history[-3:]  # Last 3 knowledge transfers
            )

            # Update agent capabilities based on transferred knowledge
            knowledge_strength = torch.norm(adapted).item()
            self.skill_level = min(1.0, self.skill_level + knowledge_strength * 0.01)

# Example usage
advanced_engine = AdvancedKnowledgeEngine()

# Create specialized agents
agents = [
    SkillSpecificAgent("navigation_expert", 0),
    SkillSpecificAgent("manipulation_expert", 1),
    SkillSpecificAgent("communication_expert", 2),
    SkillSpecificAgent("learning_novice", 0)
]

# Simulate knowledge transfer session
teacher_knowledge = torch.randn(1, 512)  # Simulated teacher knowledge

for agent in agents:
    agent.receive_knowledge(advanced_engine, teacher_knowledge)
    print(f"{agent.agent_id} skill level: {agent.skill_level:.2f}")
```

### 5. Custom Scenario Development

Creating domain-specific learning scenarios:

```python
from multi_agent_collab_learning.scenarios.base import BaseScenario
import numpy as np

class SwarmCoordinationScenario(BaseScenario):
    """Custom scenario for swarm coordination tasks."""

    def __init__(self, agents, config):
        super().__init__(agents, config)
        self.swarm_positions = {}
        self.coordination_metrics = {}

    def run(self, environment):
        """Run swarm coordination scenario."""

        print(f"🐝 Starting Swarm Coordination Scenario")

        results = {
            'coordination_events': [],
            'swarm_coherence': [],
            'formation_maintenance': [],
            'task_completion_rate': 0.0
        }

        for episode in range(self.config.duration_episodes):
            episode_result = self._run_swarm_episode(environment)

            results['coordination_events'].extend(episode_result['events'])
            results['swarm_coherence'].append(episode_result['coherence'])
            results['formation_maintenance'].append(episode_result['formation_quality'])

            if episode % 10 == 0:
                avg_coherence = np.mean(results['swarm_coherence'][-10:])
                print(f"Episode {episode}: Swarm coherence = {avg_coherence:.2f}")

        # Calculate final metrics
        results['avg_coherence'] = np.mean(results['swarm_coherence'])
        results['task_completion_rate'] = self._calculate_completion_rate(results)

        self.results = results
        return results

    def _run_swarm_episode(self, environment):
        """Run one episode of swarm coordination."""

        environment.controller.reset()

        # Initialize swarm formation
        self._initialize_swarm_formation()

        episode_events = []
        coherence_scores = []

        for step in range(50):  # 50 steps per episode
            # Get swarm state
            swarm_state = self._get_swarm_state()

            # Coordinate agent actions
            coordinated_actions = self._coordinate_agent_actions(swarm_state, environment)

            # Execute actions and measure coordination
            coordination_quality = self._execute_coordinated_actions(
                coordinated_actions, environment
            )

            coherence_scores.append(coordination_quality)

            # Detect coordination events
            if coordination_quality > 0.8:
                episode_events.append({
                    'type': 'high_coordination',
                    'step': step,
                    'quality': coordination_quality,
                    'participants': list(coordinated_actions.keys())
                })

        # Calculate formation quality
        formation_quality = self._assess_formation_quality()

        return {
            'events': episode_events,
            'coherence': np.mean(coherence_scores),
            'formation_quality': formation_quality
        }

    def _initialize_swarm_formation(self):
        """Initialize swarm in formation."""
        # Create circular formation
        n_agents = len(self.agents)
        radius = 2.0

        for i, agent in enumerate(self.agents):
            angle = 2 * np.pi * i / n_agents
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)

            self.swarm_positions[agent.agent_id] = {'x': x, 'y': y}

    def _get_swarm_state(self):
        """Get current state of the swarm."""
        return {
            'positions': self.swarm_positions.copy(),
            'center_of_mass': self._calculate_center_of_mass(),
            'spread': self._calculate_swarm_spread(),
            'avg_velocity': self._estimate_avg_velocity()
        }

    def _coordinate_agent_actions(self, swarm_state, environment):
        """Coordinate actions across all agents."""
        coordinated_actions = {}

        for agent in self.agents:
            # Get individual agent's preferred action
            individual_state = environment._get_multimodal_state()
            preferred_action = agent.select_action(individual_state)

            # Modify based on swarm coordination requirements
            coordinated_action = self._apply_swarm_coordination(
                agent, preferred_action, swarm_state
            )

            coordinated_actions[agent.agent_id] = coordinated_action

        return coordinated_actions

    def _apply_swarm_coordination(self, agent, preferred_action, swarm_state):
        """Apply swarm coordination rules to individual agent actions."""

        agent_pos = self.swarm_positions[agent.agent_id]
        center_of_mass = swarm_state['center_of_mass']

        # Calculate coordination factors
        distance_to_center = np.sqrt(
            (agent_pos['x'] - center_of_mass['x'])**2 +
            (agent_pos['y'] - center_of_mass['y'])**2
        )

        # Adjust action based on swarm coherence requirements
        if distance_to_center > 3.0:  # Too far from swarm
            # Bias toward center
            coordination_bias = 0.7
        else:
            # Allow more individual freedom
            coordination_bias = 0.3

        # Blend individual preference with swarm coordination
        coordinated_action = {
            'action': preferred_action['action'],
            'confidence': preferred_action['confidence'] * (1 - coordination_bias) + coordination_bias * 0.8,
            'swarm_coordination': coordination_bias
        }

        return coordinated_action

    def _execute_coordinated_actions(self, coordinated_actions, environment):
        """Execute coordinated actions and measure quality."""

        coordination_quality = 0.0
        successful_actions = 0

        for agent_id, action in coordinated_actions.items():
            agent = next(a for a in self.agents if a.agent_id == agent_id)

            # Execute action
            success = environment._execute_agent_action(agent, action['action'])

            if success:
                successful_actions += 1
                # Add to coordination quality
                coordination_quality += action.get('swarm_coordination', 0.5)

        # Normalize coordination quality
        coordination_quality /= len(coordinated_actions)

        # Bonus for high success rate
        success_rate = successful_actions / len(coordinated_actions)
        coordination_quality *= (0.5 + 0.5 * success_rate)

        return coordination_quality

    def _calculate_center_of_mass(self):
        """Calculate center of mass of swarm."""
        if not self.swarm_positions:
            return {'x': 0, 'y': 0}

        total_x = sum(pos['x'] for pos in self.swarm_positions.values())
        total_y = sum(pos['y'] for pos in self.swarm_positions.values())
        n = len(self.swarm_positions)

        return {'x': total_x / n, 'y': total_y / n}

    def _calculate_swarm_spread(self):
        """Calculate spread/dispersion of swarm."""
        center = self._calculate_center_of_mass()

        distances = []
        for pos in self.swarm_positions.values():
            dist = np.sqrt((pos['x'] - center['x'])**2 + (pos['y'] - center['y'])**2)
            distances.append(dist)

        return np.std(distances) if distances else 0.0

    def _estimate_avg_velocity(self):
        """Estimate average velocity of swarm (simplified)."""
        return np.random.uniform(0.5, 1.5)  # Placeholder

    def _assess_formation_quality(self):
        """Assess quality of maintained formation."""
        spread = self._calculate_swarm_spread()

        # Good formation has moderate, consistent spread
        ideal_spread = 1.5
        spread_quality = max(0, 1 - abs(spread - ideal_spread) / ideal_spread)

        return spread_quality

    def _calculate_completion_rate(self, results):
        """Calculate task completion rate."""
        # Count episodes with good coordination
        good_episodes = sum(1 for coherence in results['swarm_coherence'] if coherence > 0.7)
        return good_episodes / len(results['swarm_coherence'])

# Example usage of custom scenario
swarm_agents = [SimpleAgent(f"swarm_agent_{i}") for i in range(6)]

swarm_config = ScenarioConfig(
    scenario_type=LearningScenarioType.SWARM_LEARNING,  # Custom type
    duration_episodes=25,
    participants=len(swarm_agents)
)

swarm_scenario = SwarmCoordinationScenario(swarm_agents, swarm_config)
swarm_results = swarm_scenario.run(SimpleEnvironment())

print(f"Swarm coordination results:")
print(f"  Average coherence: {swarm_results['avg_coherence']:.2f}")
print(f"  Task completion rate: {swarm_results['task_completion_rate']:.2f}")
print(f"  Coordination events: {len(swarm_results['coordination_events'])}")
```

## Performance Analysis Examples

### 6. Comprehensive Experiment Analysis

```python
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

class ExperimentAnalyzer:
    """Comprehensive analysis of multi-agent learning experiments."""

    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        self.experiment_data = {}

    def analyze_scenario_suite(self, suite_results):
        """Analyze complete scenario suite results."""

        analysis = {
            'scenario_performance': {},
            'agent_development': {},
            'cross_scenario_learning': {},
            'emergent_behaviors': []
        }

        # Analyze each scenario
        for scenario_name, results in suite_results['scenario_results'].items():
            analysis['scenario_performance'][scenario_name] = self._analyze_scenario_performance(results)

        # Analyze agent development across scenarios
        for agent_id, development in suite_results['agent_development'].items():
            analysis['agent_development'][agent_id] = self._analyze_agent_development(development)

        # Analyze cross-scenario learning
        for scenario, learning_data in suite_results['cross_scenario_learning'].items():
            analysis['cross_scenario_learning'][scenario] = self._analyze_cross_learning(learning_data)

        # Identify emergent behaviors
        analysis['emergent_behaviors'] = self._identify_emergent_behaviors(suite_results)

        return analysis

    def _analyze_scenario_performance(self, scenario_results):
        """Analyze performance for a single scenario."""

        performance_metrics = {}

        # Extract performance indicators based on scenario type
        if 'final_rankings' in scenario_results:
            rankings = scenario_results['final_rankings']
            performance_metrics['winner'] = max(rankings, key=rankings.get)
            performance_metrics['performance_spread'] = max(rankings.values()) - min(rankings.values())
            performance_metrics['avg_performance'] = sum(rankings.values()) / len(rankings)

        if 'learning_outcomes' in scenario_results:
            outcomes = scenario_results['learning_outcomes']
            improvements = [getattr(o, 'performance_improvement', 0) for o in outcomes]
            performance_metrics['avg_improvement'] = sum(improvements) / len(improvements)
            performance_metrics['improvement_consistency'] = 1 - np.std(improvements) / max(np.mean(improvements), 0.01)

        return performance_metrics

    def _analyze_agent_development(self, development_data):
        """Analyze individual agent development."""

        development_analysis = {
            'skill_growth': development_data.get('skill_diversification', 0),
            'adaptability': development_data.get('adaptability_improvement', 0),
            'collaboration_improvement': development_data.get('collaboration_evolution', 0),
            'leadership_emergence': development_data.get('leadership_development', 0)
        }

        # Calculate overall development score
        development_analysis['overall_development'] = np.mean(list(development_analysis.values()))

        return development_analysis

    def create_performance_report(self, analysis, output_path: Path):
        """Create comprehensive performance report."""

        report = []
        report.append("# Multi-Agent Learning Experiment Report\n")
        report.append(f"Generated on: {pd.Timestamp.now()}\n\n")

        # Scenario Performance Summary
        report.append("## Scenario Performance Summary\n")
        for scenario, perf in analysis['scenario_performance'].items():
            report.append(f"### {scenario}\n")
            for metric, value in perf.items():
                if isinstance(value, float):
                    report.append(f"- {metric}: {value:.3f}\n")
                else:
                    report.append(f"- {metric}: {value}\n")
            report.append("\n")

        # Agent Development Analysis
        report.append("## Agent Development Analysis\n")
        for agent_id, dev in analysis['agent_development'].items():
            report.append(f"### {agent_id}\n")
            for metric, value in dev.items():
                report.append(f"- {metric}: {value:.3f}\n")
            report.append("\n")

        # Cross-Scenario Learning
        report.append("## Cross-Scenario Learning\n")
        for scenario, learning in analysis['cross_scenario_learning'].items():
            report.append(f"- {scenario}: Effective skill transfer observed\n")

        # Write report
        with open(output_path / "experiment_report.md", 'w') as f:
            f.writelines(report)

        print(f"Report saved to {output_path / 'experiment_report.md'}")

# Example usage
def run_comprehensive_experiment():
    """Run comprehensive multi-scenario experiment with analysis."""

    # Create experiment directory
    exp_dir = Path("experiments/comprehensive_run")
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logger("comprehensive_experiment",
                         log_file=exp_dir / "experiment.log")

    # Create diverse agent pool
    agents = [
        SimpleAgent("speed_focused", 0.7),
        SimpleAgent("accuracy_focused", 0.8),
        SimpleAgent("balanced", 0.6),
        SimpleAgent("adaptive", 0.5)
    ]

    # Create orchestrator and register scenarios
    orchestrator = ScenarioOrchestrator(agents)

    # Tournament
    tournament_config = ScenarioConfig(
        scenario_type=LearningScenarioType.COMPETITIVE_TOURNAMENT,
        duration_episodes=15
    )
    orchestrator.register_scenario('tournament',
                                 CompetitiveLearningTournament(agents, tournament_config))

    # Mentorship
    mentor_config = ScenarioConfig(
        scenario_type=LearningScenarioType.MENTOR_STUDENT,
        duration_episodes=20
    )
    orchestrator.register_scenario('mentorship',
                                 MentorStudentNetwork(agents, mentor_config))

    # Run experiment
    suite_results = orchestrator.run_scenario_suite(
        SimpleEnvironment(),
        ['tournament', 'mentorship']
    )

    # Analyze results
    analyzer = ExperimentAnalyzer(exp_dir)
    analysis = analyzer.analyze_scenario_suite(suite_results)
    analyzer.create_performance_report(analysis, exp_dir)

    return analysis

# Run the comprehensive experiment
if __name__ == "__main__":
    analysis = run_comprehensive_experiment()
    print("Comprehensive experiment completed!")
```

These examples demonstrate the full range of capabilities in the Multi-Agent Collaborative Learning library, from basic tournament setups to advanced custom scenarios and comprehensive experimental analysis. Each example builds on the previous ones, showing how to create increasingly sophisticated multi-agent learning systems.