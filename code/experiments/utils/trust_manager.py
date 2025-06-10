from typing import Dict, List
import numpy as np
from config import TRUST_WEIGHTS

class TrustManager:
    def __init__(self):
        self.trust_matrix: Dict[str, Dict[str, float]] = {}
        self.weights = TRUST_WEIGHTS

    def initialize_trust(self, agent_id: str) -> None:
        """Initialize trust scores for a new agent."""
        if agent_id not in self.trust_matrix:
            self.trust_matrix[agent_id] = {}

    def update_trust(self, agent_id: str, quality_score: float) -> None:
        """
        Update trust scores based on interaction quality.
        
        Args:
            agent_id (str): ID of the agent whose trust is being updated
            quality_score (float): Quality score of the last interaction (0-1)
        """
        for other_agent in self.trust_matrix:
            if other_agent != agent_id:
                current_trust = self.trust_matrix[other_agent].get(agent_id, 0.5)
                # Update trust using weighted moving average
                new_trust = (0.7 * current_trust) + (0.3 * quality_score)
                self.trust_matrix[other_agent][agent_id] = new_trust

    def get_trust_score(self, from_agent: str, to_agent: str) -> float:
        """Get the trust score from one agent to another."""
        return self.trust_matrix.get(from_agent, {}).get(to_agent, 0.5)

    def get_all_trust_scores(self) -> Dict[str, Dict[str, float]]:
        """Get the complete trust matrix."""
        return self.trust_matrix

    def calculate_team_trust(self) -> float:
        """Calculate the average trust score across the entire team."""
        if not self.trust_matrix:
            return 0.0
        
        trust_scores = []
        for from_agent in self.trust_matrix:
            for to_agent, score in self.trust_matrix[from_agent].items():
                trust_scores.append(score)
        
        return np.mean(trust_scores) if trust_scores else 0.0

    def get_most_trusted_agent(self, agent_id: str) -> str:
        """Get the ID of the agent most trusted by the given agent."""
        if agent_id not in self.trust_matrix:
            return None
        
        trust_scores = self.trust_matrix[agent_id]
        return max(trust_scores.items(), key=lambda x: x[1])[0] if trust_scores else None 