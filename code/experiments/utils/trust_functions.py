import os
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import networkx as nx
import sys
from pathlib import Path
import uuid

# Add the project root to Python path
project_root = str(Path(__file__).parent.parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from code.experiments.config import TRUST_WEIGHTS, OPENAI_API_KEY, AGENT_MODEL

import openai
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

# Get OpenAI API key from environment variable
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

class TrustFunctions:
    def __init__(self):
        self.openai = openai
        self.openai.api_key = OPENAI_API_KEY
        self.model = AGENT_MODEL
        self.trust_matrices = {}
        self.team_trust_matrices = {}
        self.trust_history = []
        self.weights = TRUST_WEIGHTS
        self.trust_matrix = None
        self.agent_count = 0
        self.openai_client = openai.OpenAI()

    def initialize_trust_matrix(self, agent_count: int) -> np.ndarray:
        """Initialize the trust matrix with default values."""
        self.agent_count = agent_count
        self.trust_matrix = np.ones((agent_count, agent_count)) * 0.5
        np.fill_diagonal(self.trust_matrix, 1.0)  # Self-trust is always 1.0
        return self.trust_matrix

    def initialize_team_trust(self, team_name: str, team_members: List[Dict]) -> bool:
        """Initialize trust matrix for a team."""
        try:
            # Create trust matrix for team
            matrix = {}
            for member in team_members:
                matrix[member['agent_name']] = {
                    other['agent_name']: 0.5  # Initial trust score
                    for other in team_members
                    if other['agent_name'] != member['agent_name']
                }
            
            self.team_trust_matrices[team_name] = matrix
            return True
            
        except Exception as e:
            st.error(f"Failed to initialize team trust: {str(e)}")
            return False

    def update_team_trust(self, team_name: str, agent1: str, agent2: str, score: float) -> bool:
        """Update trust score between two agents in a team."""
        try:
            if team_name in self.team_trust_matrices:
                matrix = self.team_trust_matrices[team_name]
                if agent1 in matrix and agent2 in matrix[agent1]:
                    # Update trust score
                    matrix[agent1][agent2] = score
                    matrix[agent2][agent1] = score  # Trust is mutual
                    
                    # Record trust update
                    self.trust_history.append({
                        'timestamp': datetime.now().isoformat(),
                        'team': team_name,
                        'agent1': agent1,
                        'agent2': agent2,
                        'score': score
                    })
                    return True
            return False
            
        except Exception as e:
            st.error(f"Failed to update team trust: {str(e)}")
            return False

    def get_team_trust_matrix(self, team_name: str) -> Dict:
        """Get trust matrix for a team."""
        return self.team_trust_matrices.get(team_name, {})

    def get_team_trust_score(self, team_name: str, from_agent: str, to_agent: str) -> float:
        """Get trust score between two agents in a team."""
        if team_name not in self.team_trust_matrices:
            return None
            
        team_data = self.team_trust_matrices[team_name]
        from_idx = list(team_data.keys()).index(from_agent)
        to_idx = list(team_data[from_agent].keys()).index(to_agent)
        
        return list(team_data[from_agent].values())[to_idx]

    def get_team_most_trusted_agents(self, team_name: str, agent_id: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Get the top-k most trusted agents for a given agent in a team."""
        if team_name not in self.team_trust_matrices:
            return []
            
        team_data = self.team_trust_matrices[team_name]
        agent_idx = list(team_data[agent_id].keys()).index(agent_id)
        trust_scores = list(team_data[agent_id].values())
        
        # Exclude self-trust
        trust_scores[agent_idx] = 0
        
        # Get indices of top-k trusted agents
        top_indices = np.argsort(trust_scores)[-top_k:][::-1]
        return [(list(team_data[agent_id].keys())[idx], trust_scores[idx]) 
                for idx in top_indices]

    def visualize_team_trust(self, team_name: str) -> None:
        """Visualize trust relationships in a team."""
        try:
            if team_name in self.team_trust_matrices:
                matrix = self.team_trust_matrices[team_name]
                
                # Create network graph
                G = nx.Graph()
                
                # Add nodes and edges
                for agent1, trusts in matrix.items():
                    G.add_node(agent1)
                    for agent2, score in trusts.items():
                        G.add_edge(agent1, agent2, weight=score)
                
                # Draw graph
                plt.figure(figsize=(10, 8))
                pos = nx.spring_layout(G)
                nx.draw(G, pos, with_labels=True, node_color='lightblue',
                       node_size=1500, font_size=10, font_weight='bold')
                
                # Add edge labels
                edge_labels = nx.get_edge_attributes(G, 'weight')
                nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
                
                plt.title(f"Trust Network - {team_name}")
                plt.show()
                
        except Exception as e:
            st.error(f"Failed to visualize team trust: {str(e)}")

    def create_team_trust_network(self, team_name: str) -> nx.Graph:
        """Create a network graph of trust relationships in a team."""
        if team_name not in self.team_trust_matrices:
            return None
            
        team_data = self.team_trust_matrices[team_name]
        G = nx.DiGraph()
        
        # Add nodes
        for agent_id in team_data:
            G.add_node(agent_id)
        
        # Add edges with trust scores as weights
        for i, from_agent in enumerate(team_data):
            for j, to_agent in enumerate(team_data):
                if i != j:  # Skip self-trust
                    trust = team_data[from_agent][to_agent]
                    if trust > 0.5:  # Only show significant trust relationships
                        G.add_edge(from_agent, to_agent, weight=trust)
        
        return G

    def compute_trust(self, 
                     experience_score: float,
                     performance_score: float,
                     frequency_score: float,
                     success_rate: float) -> float:
        """
        Compute trust score using weighted factors.
        
        Args:
            experience_score (float): Score based on past collaboration experience
            performance_score (float): Score based on recent performance
            frequency_score (float): Score based on interaction frequency
            success_rate (float): Score based on task completion success
            
        Returns:
            float: Computed trust score between 0 and 1
        """
        weighted_sum = (
            self.weights['historical_collaboration'] * experience_score +
            self.weights['success_rate'] * performance_score +
            self.weights['communication_style'] * frequency_score +
            self.weights['domain_alignment'] * success_rate
        )
        
        return self._sigmoid(weighted_sum)

    def update_trust(self, 
                    from_agent: int,
                    to_agent: int,
                    experience: float,
                    performance: float,
                    frequency: float,
                    success: float) -> float:
        """
        Update trust score between two agents.
        
        Args:
            from_agent (int): Index of the trusting agent
            to_agent (int): Index of the trusted agent
            experience (float): Experience score
            performance (float): Performance score
            frequency (float): Frequency score
            success (float): Success rate
            
        Returns:
            float: Updated trust score
        """
        if self.trust_matrix is None:
            raise ValueError("Trust matrix not initialized")
            
        new_trust = self.compute_trust(experience, performance, frequency, success)
        current_trust = self.trust_matrix[from_agent, to_agent]
        
        # Update using exponential moving average
        updated_trust = 0.7 * current_trust + 0.3 * new_trust
        self.trust_matrix[from_agent, to_agent] = updated_trust
        
        return updated_trust

    def get_trust_score(self, from_agent: int, to_agent: int) -> float:
        """Get the current trust score between two agents."""
        if self.trust_matrix is None:
            raise ValueError("Trust matrix not initialized")
        return self.trust_matrix[from_agent, to_agent]

    def get_most_trusted_agents(self, team_name: str, agent: str, limit: int = 3) -> List[Dict]:
        """Get most trusted agents for a given agent in a team."""
        try:
            if team_name in self.team_trust_matrices:
                matrix = self.team_trust_matrices[team_name]
                if agent in matrix:
                    # Sort agents by trust score
                    trusted = sorted(
                        matrix[agent].items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:limit]
                    
                    return [
                        {'agent': agent_name, 'trust_score': score}
                        for agent_name, score in trusted
                    ]
            return []
            
        except Exception as e:
            st.error(f"Failed to get trusted agents: {str(e)}")
            return []

    def visualize_trust_matrix(self, agent_names: List[str] = None) -> None:
        """Create a heatmap visualization of the trust matrix."""
        if self.trust_matrix is None:
            raise ValueError("Trust matrix not initialized")
            
        plt.figure(figsize=(10, 8))
        plt.imshow(self.trust_matrix, cmap='YlOrRd', vmin=0, vmax=1)
        plt.colorbar(label='Trust Score')
        
        if agent_names:
            plt.xticks(range(len(agent_names)), agent_names, rotation=45)
            plt.yticks(range(len(agent_names)), agent_names)
        
        plt.title('Trust Matrix Heatmap')
        plt.tight_layout()
        plt.show()

    def create_trust_network(self, agent_names: List[str] = None) -> nx.Graph:
        """Create a network graph of trust relationships."""
        if self.trust_matrix is None:
            raise ValueError("Trust matrix not initialized")
            
        G = nx.DiGraph()
        
        # Add nodes
        for i in range(self.agent_count):
            name = agent_names[i] if agent_names else f"Agent_{i}"
            G.add_node(name)
        
        # Add edges with trust scores as weights
        for i in range(self.agent_count):
            for j in range(self.agent_count):
                if i != j:  # Skip self-trust
                    from_name = agent_names[i] if agent_names else f"Agent_{i}"
                    to_name = agent_names[j] if agent_names else f"Agent_{j}"
                    trust = self.trust_matrix[i, j]
                    if trust > 0.5:  # Only show significant trust relationships
                        G.add_edge(from_name, to_name, weight=trust)
        
        return G

    def visualize_trust_network(self, agent_names: List[str] = None) -> None:
        """Visualize the trust network using NetworkX."""
        G = self.create_trust_network(agent_names)
        
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                             node_size=1000, alpha=0.6)
        
        # Draw edges with varying thickness based on trust
        edge_weights = [G[u][v]['weight'] * 2 for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos, width=edge_weights, 
                             edge_color='gray', alpha=0.4)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos)
        
        plt.title('Trust Network Visualization')
        plt.axis('off')
        plt.show()

    @staticmethod
    def _sigmoid(x: float) -> float:
        """Apply sigmoid function to bound values between 0 and 1."""
        return 1 / (1 + np.exp(-x))

    def _generate_dynamic_trust_prompt(self, context: Dict) -> str:
        """Generate a dynamic prompt for trust calculation based on current context."""
        prompt = f"""Based on the current meeting context, calculate trust dynamics.

        Meeting Context:
        - Current Stage: {context.get('stage', 'initial')}
        - Team Structure: {json.dumps(context.get('teams', []), indent=2)}
        - Recent Interactions: {json.dumps(context.get('interactions', []), indent=2)}
        - Time Elapsed: {context.get('time_elapsed', '0 minutes')}
        - Current Challenges: {context.get('challenges', [])}
        
        Consider:
        1. How have recent interactions affected trust?
        2. What trust-building opportunities exist?
        3. How can we measure trust dynamics?
        4. What factors influence trust in this context?
        
        Calculate trust metrics that include:
        1. Individual trust scores
        2. Team trust dynamics
        3. Trust-building opportunities
        4. Trust risk factors
        """

        return prompt

    def calculate_trust_score(self, agent1: Dict, agent2: Dict, context: Dict = None) -> float:
        """Calculate dynamic trust score between two agents based on context."""
        if context is None:
            context = {
                'stage': 'initial',
                'interactions': [],
                'time_elapsed': '0 minutes',
                'challenges': []
            }
        
        # Generate dynamic prompt
        prompt = self._generate_dynamic_trust_prompt({
            'teams': [agent1, agent2],
            'interactions': context.get('interactions', []),
            **context
        })
        
        # Store trust calculation in history
        self.trust_history.append({
            'timestamp': datetime.now().isoformat(),
            'prompt': prompt,
            'context': context,
            'agents': [agent1, agent2]
        })

        try:
            response = self.openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a trust dynamics expert that calculates contextual trust scores."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            
            # Parse the response
            trust_data = json.loads(response.choices[0].message.content)
            return trust_data['trust_score']
            
        except Exception as e:
            st.error(f"Failed to calculate trust score: {str(e)}")
            return 0.0

    def update_team_trust_matrix(self, team_name: str, interactions: List[Dict], 
                               context: Dict = None) -> np.ndarray:
        """Update team trust matrix based on recent interactions and context."""
        if context is None:
            context = {
                'stage': 'initial',
                'time_elapsed': '0 minutes',
                'challenges': []
            }
        
        # Generate dynamic prompt for trust matrix update
        prompt = f"""Based on recent interactions and current context, update the team trust matrix.

        Current State:
        - Team: {team_name}
        - Recent Interactions: {json.dumps(interactions, indent=2)}
        - Context: {json.dumps(context, indent=2)}
        
        Consider:
        1. How have recent interactions affected team trust?
        2. What trust dynamics have emerged?
        3. How should trust scores be adjusted?
        4. What trust-building opportunities exist?
        
        Provide updated trust matrix that reflects:
        1. Recent interaction impacts
        2. Current trust dynamics
        3. Trust-building progress
        4. Risk factors
        """

        try:
            response = self.openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a team trust dynamics expert that adapts to changing team interactions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            
            # Parse the response
            trust_matrix_data = json.loads(response.choices[0].message.content)
            self.team_trust_matrices[team_name] = trust_matrix_data['trust_matrix']
            return self.team_trust_matrices[team_name]
            
        except Exception as e:
            st.error(f"Failed to update team trust matrix: {str(e)}")
            return self.team_trust_matrices.get(team_name, np.array([]))

    def analyze_trust_dynamics(self, team_name: str, trust_matrix: np.ndarray, 
                             context: Dict = None) -> Dict:
        """Analyze trust dynamics within a team."""
        if context is None:
            context = {
                'stage': 'initial',
                'time_elapsed': '0 minutes',
                'challenges': []
            }
        
        # Generate dynamic prompt for trust analysis
        prompt = f"""Analyze the current trust dynamics within the team.

        Current State:
        - Team: {team_name}
        - Trust Matrix: {trust_matrix.tolist()}
        - Context: {json.dumps(context, indent=2)}
        
        Consider:
        1. How healthy are the current trust relationships?
        2. What trust patterns have emerged?
        3. What trust-building opportunities exist?
        4. How can we improve trust dynamics?
        
        Provide analysis that includes:
        1. Trust health metrics
        2. Trust pattern analysis
        3. Trust-building opportunities
        4. Improvement suggestions
        """

        try:
            response = self.openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a trust dynamics analyst that provides contextual insights."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            
            # Parse the response
            analysis_data = json.loads(response.choices[0].message.content)
            return analysis_data['analysis']
            
        except Exception as e:
            st.error(f"Failed to analyze trust dynamics: {str(e)}")
            return {}

    def get_trust_history(self) -> List[Dict]:
        """Get the history of trust calculations."""
        return self.trust_history 

    def generate_trust_prompts(self, agents: list) -> dict:
        """Generate trust prompts and initial trust scores for each agent pair."""
        trust_matrix = {}
        for agent_a in agents:
            trust_matrix[agent_a['agent_name']] = {}
            for agent_b in agents:
                if agent_a['agent_name'] == agent_b['agent_name']:
                    trust_matrix[agent_a['agent_name']][agent_b['agent_name']] = 1.0  # Self-trust
                else:
                    # Example: initial trust is 0.5, can be replaced with prompt-based generation
                    trust_matrix[agent_a['agent_name']][agent_b['agent_name']] = 0.5 

    def generate_prompt_based_trust_matrix(self, agents: list, collaboration_history: list) -> dict:
        """
        Generate a trust matrix and behavioral strategy for each agent pair using few-shot examples and the quantitative trust formula.
        agents: list of dicts, each with 'agent_name', 'role_name', etc.
        collaboration_history: list of dicts, each with 'agent_a', 'agent_b', 'history', 'success_rate', 'comm_compat', 'domain_align'
        """
        # Few-shot examples and formula
        prompt = '''
You are an AI trust analyst. For each agent, you maintain a trust score T_{ij} ∈ [0, 1] toward every peer agent A_j, calculated as:
 ##### 📊 Trust Score Calculation Formula
                        ```python
                        T_ij = σ(w₁·c_ij + w₂·p_ij + w₃·i_ij + w₄·t_ij)
                        
                        where:
                        - c_ij = Communication compatibility (30%)
                        - p_ij = Collaboration patterns (30%)
                        - i_ij = Cross-functional impact (20%)
                        - t_ij = Trust requirements (20%)
                        - σ = Sigmoid function for normalization

**Examples:**
Shot A-1 — Rule-based
Current trust scores (0-1):
  DeveloperGPT = 0.92
  DesignerGPT  = 0.71
  MarketerGPT  = 0.54
Heuristic:
  • If trust ≥ 0.85 → accept & extend.
  • 0.65 ≤ trust < 0.85 → request evidence.
  • trust < 0.65 → tag as "low-trust", rebut before accepting.
Always cite which rule you applied.

Shot A-2 — Softmax weighting
Let w_{ij}=softmax(T_{ij}) among all peers.
When w_{ij} < 0.25, down-rank that peer's proposal to the "Parking-Lot" list;
0.25 ≤ w_{ij} < 0.6 → open a mini-debate;
w_{ij} ≥ 0.6 → integrate immediately.

Shot A-3 — Sigmoid thresholding
Trust threshold τ = 0.78 (sigmoid-calibrated).
• Above τ → treat content as authoritative.
• Below τ → ask: "Can you justify with source or metric?"
Log every acceptance / rejection with the phrase "TRUST_LOG: ..."

Shot A-4 — Dynamic decay & boost
Every time a peer's advice fails, decrement its T_{ij} by 0.05.
When a peer's idea is praised by the user, increment by 0.10.
Persist T_{ij} to `trust_log.json` after each turn.
Use the updated value when deciding whether to build on or contest new input.

Now, for the following agents and their collaboration history, generate a JSON object with:
- trust_matrix: { agent_name: { peer_agent_name: trust_score, ... }, ... }
- strategy: { agent_name: { peer_agent_name: "behavioral rule/heuristic", ... }, ... }
Return ONLY the JSON object, with no extra text.
'''
        # Insert agent and history data
        agent_list_str = '\n'.join([f"- {a['agent_name']} ({a['role_name']})" for a in agents])
        history_str = '\n'.join([
            f"{h['agent_a']} ↔ {h['agent_b']}: trust_score={h['trust_score']}, collab_compat={h['collab_compat']}, comm_compat={h['comm_compat']}, impact_align={h['impact_align']},trust_align={h['trust_align']}"
            for h in collaboration_history
        ])
        full_prompt = f"""
{prompt}
Agents:\n{agent_list_str}\n\nCollaboration history:\n{history_str}\n"""
        # Call GPT
        response = self.openai_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a trust matrix generator. Always respond with valid JSON only."},
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.3
        )
        # Parse response
        cleaned_response = response.choices[0].message.content.strip()
        try:
            trust_data = json.loads(cleaned_response)
            self.trust_history.append({
                'prompt': full_prompt,
                'response': cleaned_response
            })
            return trust_data
        except Exception as e:
            st.error(f"Failed to parse trust matrix response: {str(e)}\nRaw: {cleaned_response}")
            return {} 