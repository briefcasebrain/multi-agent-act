"""
Collaborative research environment scenarios.
"""

import time
import random
import numpy as np
from typing import Dict, List, Any, Optional
from collections import defaultdict

try:
    import networkx as nx
except ImportError:
    nx = None

from .base import BaseScenario
from ..core.config import ScenarioConfig, LearningScenarioType, LearningOutcome
from ..utils.logging import get_logger

logger = get_logger(__name__)


class CollaborativeResearchEnvironment(BaseScenario):
    """
    Implements collaborative research scenario where agents work together on discovery.

    This class manages research teams working on different topics, facilitating
    knowledge sharing, discovery tracking, and innovation measurement.
    """

    def __init__(self, agents: List[Any], config: ScenarioConfig):
        """
        Initialize the collaborative research environment.

        Args:
            agents: List of collaborative agents
            config: Scenario configuration
        """
        super().__init__(agents, config)

        if nx is None:
            raise ImportError("NetworkX is required for collaborative research. Install with: pip install networkx")

        self.research_topics = self._initialize_research_topics()
        self.knowledge_base = {}
        self.research_teams = {}
        self.discovery_history = []
        self.collaboration_network = nx.Graph()

    def _initialize_research_topics(self) -> List[Dict[str, Any]]:
        """
        Initialize research topics for collaborative exploration.

        Returns:
            List of research topic definitions
        """
        return [
            {
                'topic_id': 'spatial_navigation',
                'complexity': 0.4,
                'required_skills': ['navigation', 'spatial_reasoning'],
                'knowledge_areas': ['pathfinding', 'obstacle_avoidance', 'route_optimization'],
                'collaboration_benefit': 0.8  # High benefit from collaboration
            },
            {
                'topic_id': 'object_manipulation',
                'complexity': 0.6,
                'required_skills': ['dexterity', 'planning', 'physics_understanding'],
                'knowledge_areas': ['grasping', 'stacking', 'tool_use'],
                'collaboration_benefit': 0.7
            },
            {
                'topic_id': 'communication_protocols',
                'complexity': 0.8,
                'required_skills': ['communication', 'protocol_design', 'social_intelligence'],
                'knowledge_areas': ['language_evolution', 'efficient_encoding', 'context_awareness'],
                'collaboration_benefit': 0.9
            },
            {
                'topic_id': 'emergent_behavior',
                'complexity': 1.0,
                'required_skills': ['systems_thinking', 'pattern_recognition', 'complexity_science'],
                'knowledge_areas': ['self_organization', 'collective_intelligence', 'phase_transitions'],
                'collaboration_benefit': 0.95
            }
        ]

    def run(self, environment: Any) -> Dict[str, Any]:
        """
        Run collaborative research scenario.

        Args:
            environment: Environment to run the scenario in

        Returns:
            Dictionary containing scenario results
        """
        if not self.validate_agents():
            raise ValueError("Invalid agent configuration for collaborative research")

        logger.info("Starting Collaborative Research Environment")
        logger.info(f"Researchers: {len(self.agents)}")
        logger.info(f"Research Topics: {len(self.research_topics)}")

        results = {
            'discoveries': [],
            'research_outcomes': {},
            'collaboration_patterns': {},
            'knowledge_evolution': {},
            'innovation_metrics': {}
        }

        # Form research teams
        self._form_research_teams()

        # Run research episodes
        for episode in range(self.config.duration_episodes):
            episode_results = self._run_research_episode(environment)

            # Process discoveries
            self._process_discoveries(episode_results)

            # Update collaboration network
            self._update_collaboration_network(episode_results)

            if episode % 20 == 0:
                self._log_research_progress(episode)

        # Analyze final results
        results['discoveries'] = self.discovery_history
        results['research_outcomes'] = self._analyze_research_outcomes()
        results['collaboration_patterns'] = self._analyze_collaboration_patterns()
        results['knowledge_evolution'] = self._trace_knowledge_evolution()
        results['innovation_metrics'] = self._calculate_innovation_metrics()

        logger.info("Collaborative Research Complete!")
        logger.info(f"Total Discoveries: {len(self.discovery_history)}")

        self.results = results
        return results

    def _form_research_teams(self) -> None:
        """Form research teams based on complementary skills."""
        # Assign agents to research topics based on skill compatibility
        for topic in self.research_topics:
            topic_id = topic['topic_id']
            required_skills = topic['required_skills']

            # Find agents with relevant skills
            suitable_agents = []
            for agent in self.agents:
                skill_match = self._calculate_skill_match(agent, required_skills)
                if skill_match > 0.4:  # Threshold for participation
                    suitable_agents.append((agent, skill_match))

            # Select top agents for this research team
            suitable_agents.sort(key=lambda x: x[1], reverse=True)
            team_members = [agent for agent, _ in suitable_agents[:3]]  # Max 3 per team

            if team_members:
                self.research_teams[topic_id] = {
                    'members': [agent.agent_id for agent in team_members],
                    'topic': topic,
                    'research_progress': 0.0,
                    'discoveries': [],
                    'collaboration_score': 0.0
                }

                # Update agent team assignments
                for agent in team_members:
                    agent.current_research_topic = topic_id

        logger.info(f"Formed {len(self.research_teams)} research teams")

    def _calculate_skill_match(self, agent: Any, required_skills: List[str]) -> float:
        """
        Calculate how well agent's skills match research requirements.

        Args:
            agent: Agent to evaluate
            required_skills: Required skills for research

        Returns:
            Skill match score between 0 and 1
        """
        # Map agent properties to skill scores
        role_name = getattr(agent, 'role', 'COOPERATOR')
        if hasattr(role_name, 'name'):
            role_name = role_name.name

        agent_skills = {
            'navigation': 0.7 if role_name in ['EXPLORER', 'COORDINATOR'] else 0.4,
            'spatial_reasoning': 0.8 if role_name == 'SPECIALIST' else 0.5,
            'dexterity': 0.6 if role_name == 'COLLECTOR' else 0.3,
            'planning': 0.8 if role_name in ['LEADER', 'COORDINATOR'] else 0.5,
            'physics_understanding': 0.7 if role_name == 'SPECIALIST' else 0.4,
            'communication': 0.9 if role_name == 'LEADER' else 0.6,
            'protocol_design': 0.7 if role_name == 'COORDINATOR' else 0.3,
            'social_intelligence': 0.8 if role_name == 'COOPERATOR' else 0.5,
            'systems_thinking': 0.8 if role_name in ['LEADER', 'SPECIALIST'] else 0.4,
            'pattern_recognition': 0.7 if role_name in ['SPECIALIST', 'EXPLORER'] else 0.5,
            'complexity_science': 0.6 if role_name == 'SPECIALIST' else 0.3
        }

        # Calculate match score
        skill_scores = [agent_skills.get(skill, 0.3) for skill in required_skills]
        return np.mean(skill_scores)

    def _run_research_episode(self, environment: Any) -> Dict[str, Any]:
        """
        Run one episode of collaborative research.

        Args:
            environment: Research environment

        Returns:
            Episode results dictionary
        """
        environment.controller.reset()

        episode_results = {
            'team_activities': {},
            'knowledge_sharing_events': [],
            'discovery_attempts': [],
            'collaboration_interactions': []
        }

        max_steps = 100

        for step in range(max_steps):
            # Research phase - teams work on their topics
            self._execute_research_phase(environment, episode_results)

            # Collaboration phase - inter-team knowledge sharing
            if step % 25 == 0:  # Periodic collaboration
                self._execute_collaboration_phase(episode_results)

            # Discovery evaluation phase
            self._evaluate_discovery_attempts(episode_results)

        return episode_results

    def _execute_research_phase(self, environment: Any, episode_results: Dict[str, Any]) -> None:
        """
        Execute research activities for each team.

        Args:
            environment: Research environment
            episode_results: Episode results to update
        """
        for topic_id, team_info in self.research_teams.items():
            team_members = [
                next(a for a in self.agents if a.agent_id == aid)
                for aid in team_info['members']
            ]

            # Collaborative research activity
            research_result = self._conduct_team_research(team_members, team_info['topic'], environment)

            episode_results['team_activities'][topic_id] = research_result

            # Update team progress
            team_info['research_progress'] += research_result['progress_made']
            team_info['collaboration_score'] = research_result['collaboration_effectiveness']

    def _execute_collaboration_phase(self, episode_results: Dict[str, Any]) -> None:
        """
        Execute inter-team collaboration and knowledge sharing.

        Args:
            episode_results: Episode results to update
        """
        for topic_id_1, team_info_1 in self.research_teams.items():
            for topic_id_2, team_info_2 in self.research_teams.items():
                if topic_id_1 < topic_id_2:  # Avoid duplicate pairs

                    # Assess collaboration potential
                    collaboration_potential = self._assess_collaboration_potential(
                        team_info_1['topic'], team_info_2['topic']
                    )

                    if collaboration_potential > 0.5:
                        # Execute knowledge sharing
                        sharing_result = self._execute_knowledge_sharing(
                            team_info_1, team_info_2
                        )

                        episode_results['knowledge_sharing_events'].append({
                            'teams': [topic_id_1, topic_id_2],
                            'knowledge_exchanged': sharing_result['knowledge_exchanged'],
                            'collaboration_benefit': sharing_result['benefit'],
                            'new_insights': sharing_result['insights_generated']
                        })

    def _conduct_team_research(self, team_members: List[Any], topic: Dict[str, Any], environment: Any) -> Dict[str, Any]:
        """
        Conduct collaborative research for a team.

        Args:
            team_members: List of team member agents
            topic: Research topic information
            environment: Research environment

        Returns:
            Research results dictionary
        """
        # Simulate collaborative research process
        individual_contributions = []

        for agent in team_members:
            # Individual research contribution
            contribution = self._individual_research_contribution(agent, topic, environment)
            individual_contributions.append(contribution)

        # Combine individual contributions collaboratively
        collaborative_result = self._combine_research_contributions(
            individual_contributions, topic['collaboration_benefit']
        )

        # Check for potential discoveries
        discovery_attempt = self._attempt_discovery(collaborative_result, topic)

        if discovery_attempt['discovered']:
            logger.info(f"Discovery in {topic['topic_id']}: {discovery_attempt['discovery_type']}")

        return {
            'individual_contributions': individual_contributions,
            'collaborative_result': collaborative_result,
            'discovery_attempt': discovery_attempt,
            'progress_made': collaborative_result['progress'],
            'collaboration_effectiveness': self._calculate_collaboration_effectiveness(individual_contributions)
        }

    def _individual_research_contribution(self, agent: Any, topic: Dict[str, Any], environment: Any) -> Dict[str, Any]:
        """
        Calculate individual agent's research contribution.

        Args:
            agent: Contributing agent
            topic: Research topic
            environment: Research environment

        Returns:
            Individual contribution metrics
        """
        # Skill match affects contribution quality
        skill_match = self._calculate_skill_match(agent, topic['required_skills'])

        # Agent's research ability
        collaboration_metrics = getattr(agent, 'collaboration_metrics', {})
        research_ability = skill_match * (0.5 + collaboration_metrics.get('problem_solving', 0.3))

        # Environmental factors
        state = environment._get_multimodal_state()

        # Generate research contribution
        contribution = {
            'agent_id': agent.agent_id,
            'contribution_quality': research_ability + random.uniform(-0.1, 0.1),
            'novel_insights': random.randint(0, 3) if research_ability > 0.6 else 0,
            'methodology_innovation': research_ability > 0.7,
            'data_quality': min(1.0, research_ability * 1.2),
            'collaboration_readiness': getattr(agent, 'behavior_weights', {}).get('team_reward', 0.3)
        }

        return contribution

    def _combine_research_contributions(self, contributions: List[Dict[str, Any]], collaboration_benefit: float) -> Dict[str, Any]:
        """
        Combine individual research contributions collaboratively.

        Args:
            contributions: List of individual contributions
            collaboration_benefit: Benefit multiplier for collaboration

        Returns:
            Combined research results
        """
        if not contributions:
            return {'progress': 0, 'quality': 0, 'innovation': 0}

        # Base combination
        avg_quality = np.mean([c['contribution_quality'] for c in contributions])
        total_insights = sum(c['novel_insights'] for c in contributions)
        any_methodology_innovation = any(c['methodology_innovation'] for c in contributions)
        avg_data_quality = np.mean([c['data_quality'] for c in contributions])

        # Collaboration multiplier
        collaboration_multiplier = 1.0 + (collaboration_benefit * len(contributions) * 0.1)

        # Synergy bonus for diverse contributions
        quality_diversity = np.std([c['contribution_quality'] for c in contributions])
        synergy_bonus = min(0.3, quality_diversity * 0.5)  # Diversity creates synergy

        combined_result = {
            'progress': (avg_quality + synergy_bonus) * collaboration_multiplier,
            'quality': avg_data_quality * collaboration_multiplier,
            'innovation': (total_insights + (1 if any_methodology_innovation else 0)) * collaboration_multiplier,
            'synergy_achieved': synergy_bonus > 0.1,
            'collaboration_effectiveness': collaboration_multiplier - 1.0
        }

        return combined_result

    def _attempt_discovery(self, research_result: Dict[str, Any], topic: Dict[str, Any]) -> Dict[str, Any]:
        """
        Attempt to make a discovery based on research results.

        Args:
            research_result: Combined research results
            topic: Research topic information

        Returns:
            Discovery attempt results
        """
        progress = research_result['progress']
        quality = research_result['quality']
        innovation = research_result['innovation']

        # Discovery threshold varies by topic complexity
        discovery_threshold = topic['complexity'] * 0.8

        # Combined score for discovery
        discovery_score = (progress * 0.4 + quality * 0.3 + innovation * 0.3)

        discovery_attempt = {
            'attempted': True,
            'discovered': discovery_score > discovery_threshold,
            'discovery_score': discovery_score,
            'threshold': discovery_threshold
        }

        if discovery_attempt['discovered']:
            # Determine discovery type
            if innovation > 2:
                discovery_type = 'breakthrough'
            elif quality > 0.8:
                discovery_type = 'fundamental_insight'
            elif progress > 0.7:
                discovery_type = 'incremental_advance'
            else:
                discovery_type = 'methodological_improvement'

            discovery_attempt.update({
                'discovery_type': discovery_type,
                'topic_area': topic['topic_id'],
                'knowledge_area': random.choice(topic['knowledge_areas']),
                'significance': discovery_score - discovery_threshold,
                'reproducibility': quality,
                'novelty': min(1.0, innovation / 3.0)
            })

        return discovery_attempt

    def _calculate_collaboration_effectiveness(self, contributions: List[Dict[str, Any]]) -> float:
        """
        Calculate how effectively the team collaborated.

        Args:
            contributions: List of individual contributions

        Returns:
            Collaboration effectiveness score
        """
        if len(contributions) < 2:
            return 0.5

        # Measure collaboration readiness
        avg_collaboration_readiness = np.mean([c['collaboration_readiness'] for c in contributions])

        # Measure contribution balance (not dominated by one person)
        qualities = [c['contribution_quality'] for c in contributions]
        contribution_balance = 1.0 - np.std(qualities) / max(np.mean(qualities), 0.1)

        # Overall collaboration effectiveness
        effectiveness = (avg_collaboration_readiness + contribution_balance) / 2.0

        return min(1.0, effectiveness)

    def _assess_collaboration_potential(self, topic1: Dict[str, Any], topic2: Dict[str, Any]) -> float:
        """
        Assess potential for collaboration between two research topics.

        Args:
            topic1: First research topic
            topic2: Second research topic

        Returns:
            Collaboration potential score
        """
        # Skill overlap creates collaboration potential
        skills1 = set(topic1['required_skills'])
        skills2 = set(topic2['required_skills'])

        skill_overlap = len(skills1.intersection(skills2)) / len(skills1.union(skills2))

        # Knowledge area synergy
        areas1 = set(topic1['knowledge_areas'])
        areas2 = set(topic2['knowledge_areas'])

        # Some knowledge areas naturally synergize
        synergy_pairs = {
            ('pathfinding', 'grasping'): 0.6,
            ('communication', 'self_organization'): 0.8,
            ('pattern_recognition', 'collective_intelligence'): 0.7
        }

        area_synergy = 0
        for area1 in areas1:
            for area2 in areas2:
                pair = tuple(sorted([area1, area2]))
                area_synergy = max(area_synergy, synergy_pairs.get(pair, 0))

        # Complexity complementarity (different complexity levels can be beneficial)
        complexity_diff = abs(topic1['complexity'] - topic2['complexity'])
        complexity_complement = min(0.5, complexity_diff * 0.8)  # Moderate difference is good

        collaboration_potential = (skill_overlap + area_synergy + complexity_complement) / 3.0

        return collaboration_potential

    def _execute_knowledge_sharing(self, team1: Dict[str, Any], team2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute knowledge sharing between two research teams.

        Args:
            team1: First research team
            team2: Second research team

        Returns:
            Knowledge sharing results
        """
        # Extract knowledge from each team
        team1_knowledge = self._extract_team_knowledge(team1)
        team2_knowledge = self._extract_team_knowledge(team2)

        # Calculate knowledge exchange value
        knowledge_exchanged = {
            'team1_to_team2': self._calculate_knowledge_value(team1_knowledge, team2['topic']),
            'team2_to_team1': self._calculate_knowledge_value(team2_knowledge, team1['topic'])
        }

        # Generate insights from knowledge exchange
        insights_generated = self._generate_cross_domain_insights(team1_knowledge, team2_knowledge)

        # Calculate collaboration benefit
        benefit = (knowledge_exchanged['team1_to_team2'] + knowledge_exchanged['team2_to_team1']) / 2.0

        # Update team knowledge bases
        self._update_team_knowledge(team1, team2_knowledge, knowledge_exchanged['team2_to_team1'])
        self._update_team_knowledge(team2, team1_knowledge, knowledge_exchanged['team1_to_team2'])

        return {
            'knowledge_exchanged': knowledge_exchanged,
            'benefit': benefit,
            'insights_generated': insights_generated
        }

    def _extract_team_knowledge(self, team: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract knowledge accumulated by research team.

        Args:
            team: Research team information

        Returns:
            Team knowledge representation
        """
        return {
            'research_progress': team['research_progress'],
            'discoveries': len(team['discoveries']),
            'collaboration_experience': team['collaboration_score'],
            'topic_expertise': team['topic']['complexity'] * team['research_progress'],
            'methodological_knowledge': random.uniform(0.2, 0.8),  # Simplified
            'domain_insights': team['topic']['knowledge_areas']
        }

    def _calculate_knowledge_value(self, knowledge: Dict[str, Any], recipient_topic: Dict[str, Any]) -> float:
        """
        Calculate value of knowledge for recipient topic.

        Args:
            knowledge: Knowledge to be transferred
            recipient_topic: Topic receiving the knowledge

        Returns:
            Knowledge value score
        """
        # Higher value if knowledge matches recipient's needs
        expertise_match = knowledge['topic_expertise'] * recipient_topic['complexity']
        methodological_value = knowledge['methodological_knowledge'] * 0.5
        collaboration_value = knowledge['collaboration_experience'] * recipient_topic['collaboration_benefit']

        total_value = (expertise_match + methodological_value + collaboration_value) / 3.0

        return min(1.0, total_value)

    def _generate_cross_domain_insights(self, knowledge1: Dict[str, Any], knowledge2: Dict[str, Any]) -> List[str]:
        """
        Generate insights from cross-domain knowledge combination.

        Args:
            knowledge1: First knowledge source
            knowledge2: Second knowledge source

        Returns:
            List of generated insights
        """
        insights = []

        # Combine domain insights
        combined_domains = set(knowledge1['domain_insights'] + knowledge2['domain_insights'])

        # Generate insights based on domain combinations
        insight_generators = {
            ('pathfinding', 'communication'): "Distributed pathfinding algorithms",
            ('grasping', 'self_organization'): "Emergent manipulation strategies",
            ('pattern_recognition', 'tool_use'): "Adaptive tool selection",
            ('collective_intelligence', 'route_optimization'): "Swarm navigation systems"
        }

        for domain1 in knowledge1['domain_insights']:
            for domain2 in knowledge2['domain_insights']:
                pair = tuple(sorted([domain1, domain2]))
                if pair in insight_generators:
                    insights.append(insight_generators[pair])

        return insights[:3]  # Limit to top 3 insights

    def _update_team_knowledge(self, team: Dict[str, Any], new_knowledge: Dict[str, Any], value: float) -> None:
        """
        Update team's knowledge base with new information.

        Args:
            team: Team to update
            new_knowledge: New knowledge to integrate
            value: Value of the new knowledge
        """
        # Improve research progress based on knowledge value
        progress_boost = value * 0.1
        team['research_progress'] += progress_boost

        # Improve collaboration score
        collaboration_boost = value * 0.05
        team['collaboration_score'] = min(1.0, team['collaboration_score'] + collaboration_boost)

    def _evaluate_discovery_attempts(self, episode_results: Dict[str, Any]) -> None:
        """
        Evaluate and record discovery attempts.

        Args:
            episode_results: Episode results containing discovery attempts
        """
        for topic_id, team_activity in episode_results['team_activities'].items():
            discovery_attempt = team_activity['discovery_attempt']

            if discovery_attempt['attempted']:
                episode_results['discovery_attempts'].append({
                    'team': topic_id,
                    'attempt': discovery_attempt,
                    'timestamp': time.time()
                })

                if discovery_attempt['discovered']:
                    self.discovery_history.append({
                        'discovery_id': len(self.discovery_history),
                        'team': topic_id,
                        'discovery_type': discovery_attempt['discovery_type'],
                        'topic_area': discovery_attempt['topic_area'],
                        'significance': discovery_attempt['significance'],
                        'timestamp': time.time()
                    })

    def _process_discoveries(self, episode_results: Dict[str, Any]) -> None:
        """
        Process discoveries and update global knowledge base.

        Args:
            episode_results: Episode results containing discoveries
        """
        for discovery_attempt in episode_results['discovery_attempts']:
            if discovery_attempt['attempt']['discovered']:
                discovery = discovery_attempt['attempt']

                # Add to global knowledge base
                topic_area = discovery['topic_area']
                if topic_area not in self.knowledge_base:
                    self.knowledge_base[topic_area] = []

                self.knowledge_base[topic_area].append({
                    'discovery_type': discovery['discovery_type'],
                    'knowledge_area': discovery['knowledge_area'],
                    'significance': discovery['significance'],
                    'team': discovery_attempt['team'],
                    'timestamp': time.time()
                })

    def _update_collaboration_network(self, episode_results: Dict[str, Any]) -> None:
        """
        Update collaboration network based on knowledge sharing.

        Args:
            episode_results: Episode results containing collaboration events
        """
        for sharing_event in episode_results['knowledge_sharing_events']:
            team1, team2 = sharing_event['teams']
            benefit = sharing_event['collaboration_benefit']

            # Add or update edge in collaboration network
            if self.collaboration_network.has_edge(team1, team2):
                # Strengthen existing collaboration
                current_weight = self.collaboration_network[team1][team2]['weight']
                new_weight = min(1.0, current_weight + benefit * 0.1)
                self.collaboration_network[team1][team2]['weight'] = new_weight
            else:
                # Create new collaboration edge
                self.collaboration_network.add_edge(team1, team2, weight=benefit * 0.5)

    def _log_research_progress(self, episode: int) -> None:
        """
        Log research progress across all teams.

        Args:
            episode: Current episode number
        """
        logger.info(f"Research Episode {episode}:")

        for topic_id, team_info in self.research_teams.items():
            logger.info(f"Team {topic_id}: {team_info['research_progress']:.2f} progress, "
                      f"{team_info['collaboration_score']:.2f} collaboration")

        logger.info(f"Total Discoveries: {len(self.discovery_history)}")

        # Show recent discovery types
        recent_discoveries = self.discovery_history[-3:]
        for discovery in recent_discoveries:
            logger.info(f"- {discovery['discovery_type']} in {discovery['topic_area']}")

    def _analyze_research_outcomes(self) -> Dict[str, Any]:
        """
        Analyze overall research outcomes.

        Returns:
            Research outcomes analysis
        """
        outcomes = {}

        for topic_id, team_info in self.research_teams.items():
            topic_discoveries = [d for d in self.discovery_history if d['team'] == topic_id]

            outcomes[topic_id] = {
                'total_progress': team_info['research_progress'],
                'discoveries_made': len(topic_discoveries),
                'avg_discovery_significance': np.mean([d['significance'] for d in topic_discoveries]) if topic_discoveries else 0,
                'collaboration_effectiveness': team_info['collaboration_score'],
                'research_efficiency': len(topic_discoveries) / max(team_info['research_progress'], 0.1)
            }

        return outcomes

    def _analyze_collaboration_patterns(self) -> Dict[str, Any]:
        """
        Analyze collaboration patterns between research teams.

        Returns:
            Collaboration patterns analysis
        """
        patterns = {
            'network_density': nx.density(self.collaboration_network),
            'strongest_collaborations': [],
            'collaboration_clusters': [],
            'knowledge_flow': {}
        }

        # Find strongest collaborations
        if self.collaboration_network.edges():
            sorted_edges = sorted(
                self.collaboration_network.edges(data=True),
                key=lambda x: x[2]['weight'], reverse=True
            )
            patterns['strongest_collaborations'] = [
                {'teams': (edge[0], edge[1]), 'strength': edge[2]['weight']}
                for edge in sorted_edges[:3]
            ]

        # Identify collaboration clusters
        try:
            clusters = list(nx.connected_components(self.collaboration_network))
            patterns['collaboration_clusters'] = [list(cluster) for cluster in clusters]
        except:
            patterns['collaboration_clusters'] = []

        return patterns

    def _trace_knowledge_evolution(self) -> Dict[str, Any]:
        """
        Trace how knowledge evolved throughout the research process.

        Returns:
            Knowledge evolution timeline
        """
        evolution = {
            'knowledge_growth': {},
            'discovery_timeline': [],
            'cross_domain_fertilization': []
        }

        # Track knowledge growth per topic
        for topic_area, discoveries in self.knowledge_base.items():
            evolution['knowledge_growth'][topic_area] = {
                'total_discoveries': len(discoveries),
                'breakthrough_discoveries': len([d for d in discoveries if d['discovery_type'] == 'breakthrough']),
                'knowledge_depth': np.mean([d['significance'] for d in discoveries]) if discoveries else 0
            }

        # Create discovery timeline
        sorted_discoveries = sorted(self.discovery_history, key=lambda x: x['timestamp'])
        evolution['discovery_timeline'] = [
            {'time': d['timestamp'], 'type': d['discovery_type'], 'area': d['topic_area']}
            for d in sorted_discoveries
        ]

        return evolution

    def _calculate_innovation_metrics(self) -> Dict[str, float]:
        """
        Calculate various innovation metrics.

        Returns:
            Innovation metrics dictionary
        """
        metrics = {}

        # Discovery rate
        if self.config.duration_episodes > 0:
            metrics['discovery_rate'] = len(self.discovery_history) / self.config.duration_episodes

        # Innovation diversity
        discovery_types = [d['discovery_type'] for d in self.discovery_history]
        if discovery_types:
            type_counts = {dt: discovery_types.count(dt) for dt in set(discovery_types)}
            metrics['innovation_diversity'] = len(type_counts) / len(discovery_types)
        else:
            metrics['innovation_diversity'] = 0

        # Collaboration impact
        collaborative_discoveries = [d for d in self.discovery_history if d['significance'] > 0.5]
        metrics['collaboration_impact'] = len(collaborative_discoveries) / max(len(self.discovery_history), 1)

        # Knowledge integration
        total_knowledge_areas = sum(len(discoveries) for discoveries in self.knowledge_base.values())
        metrics['knowledge_integration'] = total_knowledge_areas / max(len(self.knowledge_base), 1)

        return metrics