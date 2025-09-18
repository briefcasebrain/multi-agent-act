"""
Visualization utilities for multi-agent collaborative learning.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any, Optional, Union
from pathlib import Path


def plot_learning_curves(
    learning_data: Dict[str, List[float]],
    title: str = "Learning Curves",
    xlabel: str = "Episodes",
    ylabel: str = "Performance",
    save_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (10, 6),
    show_legend: bool = True
) -> plt.Figure:
    """
    Plot learning curves for multiple agents.

    Args:
        learning_data: Dictionary mapping agent IDs to performance values
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        save_path: Optional path to save the plot
        figsize: Figure size tuple
        show_legend: Whether to show legend

    Returns:
        Matplotlib figure object
    """
    if not learning_data:
        raise ValueError("Learning data cannot be empty")

    fig, ax = plt.subplots(figsize=figsize)

    for agent_id, performance_values in learning_data.items():
        episodes = range(len(performance_values))
        ax.plot(episodes, performance_values, label=f'Agent {agent_id}', marker='o', alpha=0.7)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

    if show_legend:
        ax.legend()

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_scenario_comparison(
    scenario_results: Dict[str, Dict[str, float]],
    metrics: List[str],
    title: str = "Scenario Performance Comparison",
    save_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (12, 8)
) -> plt.Figure:
    """
    Plot comparison of performance across different scenarios.

    Args:
        scenario_results: Nested dict of scenario -> agent -> metric values
        metrics: List of metrics to plot
        title: Plot title
        save_path: Optional path to save the plot
        figsize: Figure size tuple

    Returns:
        Matplotlib figure object
    """
    num_metrics = len(metrics)
    fig, axes = plt.subplots(2, (num_metrics + 1) // 2, figsize=figsize)

    if num_metrics == 1:
        axes = [axes]
    elif num_metrics <= 2:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    scenarios = list(scenario_results.keys())
    agents = list(next(iter(scenario_results.values())).keys())

    for i, metric in enumerate(metrics):
        ax = axes[i] if i < len(axes) else axes[0]

        # Prepare data for each agent across scenarios
        agent_data = {}
        for agent in agents:
            agent_data[agent] = [scenario_results[scenario].get(agent, {}).get(metric, 0)
                               for scenario in scenarios]

        # Create bar plot
        x = np.arange(len(scenarios))
        width = 0.8 / len(agents)

        for j, agent in enumerate(agents):
            offset = (j - len(agents)/2 + 0.5) * width
            ax.bar(x + offset, agent_data[agent], width, label=f'Agent {agent}', alpha=0.7)

        ax.set_title(f'{metric.replace("_", " ").title()}')
        ax.set_xlabel('Scenarios')
        ax.set_ylabel('Score')
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace('_', ' ').title() for s in scenarios], rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_collaboration_network(
    collaboration_data: Dict[str, Dict[str, float]],
    title: str = "Agent Collaboration Network",
    save_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (10, 10)
) -> plt.Figure:
    """
    Plot agent collaboration network as a graph.

    Args:
        collaboration_data: Nested dict of agent -> agent -> collaboration strength
        title: Plot title
        save_path: Optional path to save the plot
        figsize: Figure size tuple

    Returns:
        Matplotlib figure object
    """
    try:
        import networkx as nx
    except ImportError:
        raise ImportError("NetworkX is required for network visualization. Install with: pip install networkx")

    fig, ax = plt.subplots(figsize=figsize)

    # Create graph
    G = nx.Graph()

    # Add nodes
    all_agents = set()
    for agent1, connections in collaboration_data.items():
        all_agents.add(agent1)
        for agent2 in connections:
            all_agents.add(agent2)

    G.add_nodes_from(all_agents)

    # Add edges with weights
    for agent1, connections in collaboration_data.items():
        for agent2, strength in connections.items():
            if strength > 0.1:  # Only show significant collaborations
                G.add_edge(agent1, agent2, weight=strength)

    # Calculate layout
    pos = nx.spring_layout(G, k=3, iterations=50)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightblue',
                          node_size=1000, alpha=0.7, ax=ax)

    # Draw edges with thickness based on collaboration strength
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]

    nx.draw_networkx_edges(G, pos, width=[w*5 for w in weights],
                          alpha=0.6, edge_color='gray', ax=ax)

    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax)

    ax.set_title(title)
    ax.axis('off')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_knowledge_evolution(
    knowledge_timeline: List[Dict[str, Any]],
    title: str = "Knowledge Evolution Timeline",
    save_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (12, 6)
) -> plt.Figure:
    """
    Plot knowledge evolution over time.

    Args:
        knowledge_timeline: List of knowledge states over time
        title: Plot title
        save_path: Optional path to save the plot
        figsize: Figure size tuple

    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Extract time points and knowledge types
    timestamps = [entry['timestamp'] for entry in knowledge_timeline]
    knowledge_types = set()
    for entry in knowledge_timeline:
        knowledge_types.update(entry.get('knowledge_areas', []))

    knowledge_types = list(knowledge_types)

    # Plot knowledge accumulation for each type
    for knowledge_type in knowledge_types:
        accumulation = []
        count = 0

        for entry in knowledge_timeline:
            if knowledge_type in entry.get('knowledge_areas', []):
                count += 1
            accumulation.append(count)

        ax.plot(timestamps, accumulation, label=knowledge_type.replace('_', ' ').title(),
               marker='o', alpha=0.7)

    ax.set_title(title)
    ax.set_xlabel('Time')
    ax.set_ylabel('Cumulative Knowledge Items')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig