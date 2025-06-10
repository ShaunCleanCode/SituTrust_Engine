import os
import json
import uuid
from typing import Dict, List
import openai
from config import OPENAI_API_KEY, AGENT_MODEL

class AgentLauncher:
    def __init__(self):
        self.openai = openai
        self.openai.api_key = OPENAI_API_KEY
        self.model = AGENT_MODEL

    def launch_agent(self, role_dict: Dict, room: Dict = None, trust_manager = None) -> Dict:
        """
        Create and initialize a new agent with the given role configuration.
        
        Args:
            role_dict (Dict): Role configuration including name, description, etc.
            room (Dict): Optional room configuration for spatial context
            trust_manager: Optional trust manager for trust-based interactions
            
        Returns:
            Dict: Initialized agent configuration
        """
        # Generate unique agent ID
        agent_id = str(uuid.uuid4())
        
        # Create agent directory structure
        agent_dir = self._create_agent_directory(role_dict['role_name'])
        
        # Generate and save agent prompt
        prompt = self._generate_agent_prompt(role_dict, room)
        self._save_agent_prompt(agent_dir, prompt)
        
        # Save agent metadata
        metadata = {
            'id': agent_id,
            'role': role_dict,
            'room': room,
            'created_at': str(datetime.now()),
            'status': 'active'
        }
        self._save_agent_metadata(agent_dir, metadata)
        
        # Initialize agent state
        state = {
            'id': agent_id,
            'name': role_dict['role_name'],
            'path': agent_dir,
            'schema': role_dict['input_schema'],
            'memory': [],
            'trust_scores': {},
            'current_room': room['id'] if room else None
        }
        
        # Initialize trust if trust manager is provided
        if trust_manager:
            trust_manager.initialize_trust(agent_id)
            state['trust_manager'] = trust_manager
        
        return state

    def _create_agent_directory(self, role_name: str) -> str:
        """Create directory structure for the agent."""
        base_dir = "agents"
        agent_dir = os.path.join(base_dir, role_name)
        
        # Create main directory
        os.makedirs(agent_dir, exist_ok=True)
        
        # Create subdirectories
        os.makedirs(os.path.join(agent_dir, "memory"), exist_ok=True)
        os.makedirs(os.path.join(agent_dir, "artifacts"), exist_ok=True)
        os.makedirs(os.path.join(agent_dir, "logs"), exist_ok=True)
        
        return agent_dir

    def _generate_agent_prompt(self, role_dict: Dict, room: Dict = None) -> str:
        """Generate the agent's system prompt."""
        # Base prompt with role information
        prompt = f"""You are {role_dict['role_name']}.

ROLE DESCRIPTION:
{role_dict['description']}

EXPERTISE:
{', '.join(role_dict['expertise'])}

COMMUNICATION STYLE:
{role_dict['communication_style']}

TRUST APPROACH:
{role_dict['trust_approach']}

INPUT SCHEMA:
Your inputs will follow this structure: {role_dict['input_schema']}
"""

        # Add spatial context if room is provided
        if room:
            prompt += f"""

SPATIAL CONTEXT:
You are currently in: {room['type']}
Room Description: {room['description']}
Available Features: {', '.join(room['features'])}
"""

        # Add collaboration guidelines
        prompt += """

COLLABORATION GUIDELINES:
1. Always consider the trust levels between you and other agents
2. Build on others' contributions while maintaining your expertise
3. Provide clear reasoning for your decisions
4. Acknowledge limitations and uncertainties
5. Focus on the shared goal while maintaining your role's perspective
"""

        return prompt

    def _save_agent_prompt(self, agent_dir: str, prompt: str) -> None:
        """Save the agent's system prompt to a file."""
        prompt_path = os.path.join(agent_dir, "prompt.txt")
        with open(prompt_path, "w") as f:
            f.write(prompt)

    def _save_agent_metadata(self, agent_dir: str, metadata: Dict) -> None:
        """Save agent metadata to a JSON file."""
        metadata_path = os.path.join(agent_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def update_agent_state(self, agent_state: Dict, new_memory: List = None, new_trust: Dict = None) -> Dict:
        """Update the agent's state with new memory or trust information."""
        if new_memory:
            agent_state['memory'].extend(new_memory)
            
        if new_trust:
            agent_state['trust_scores'].update(new_trust)
            
        return agent_state 