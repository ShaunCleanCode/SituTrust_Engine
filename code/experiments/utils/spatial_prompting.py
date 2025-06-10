from typing import Dict, List
import openai
from config import OPENAI_API_KEY, AGENT_MODEL, ROOM_TEMPLATES

class SpatialPrompting:
    def __init__(self):
        self.openai = openai
        self.openai.api_key = OPENAI_API_KEY
        self.model = AGENT_MODEL
        self.room_templates = ROOM_TEMPLATES

    def generate_space_prompt(self, goal: str, constraints: List[str], team_structure: List[Dict]) -> Dict:
        """
        Generate an optimal collaboration space based on goal, constraints, and team structure.
        
        Args:
            goal (str): The main objective
            constraints (List[str]): List of constraints
            team_structure (List[Dict]): List of agent roles and their relationships
            
        Returns:
            Dict: Space configuration including name, description, and features
        """
        # First, try to match with existing templates
        template_space = self._match_template_space(goal, team_structure)
        if template_space:
            return template_space

        # If no template matches, generate a custom space
        return self._generate_custom_space(goal, constraints, team_structure)

    def _match_template_space(self, goal: str, team_structure: List[Dict]) -> Dict:
        """Match the goal and team structure with existing room templates."""
        # Analyze goal keywords
        goal_keywords = set(goal.lower().split())
        
        # Score each template based on keyword matching and team size
        best_match = None
        best_score = 0
        
        for template_name, template in self.room_templates.items():
            score = 0
            
            # Match keywords in description
            desc_keywords = set(template['description'].lower().split())
            score += len(goal_keywords.intersection(desc_keywords))
            
            # Check if team size fits capacity
            if len(team_structure) <= template['capacity']:
                score += 2
            
            if score > best_score:
                best_score = score
                best_match = {
                    'name': template_name,
                    'description': template['description'],
                    'capacity': template['capacity'],
                    'features': template['features']
                }
        
        return best_match if best_score > 0 else None

    def _generate_custom_space(self, goal: str, constraints: List[str], team_structure: List[Dict]) -> Dict:
        """Generate a custom space using GPT."""
        prompt = self._create_space_generation_prompt(goal, constraints, team_structure)
        
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a spatial design expert for AI collaboration."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Parse the response into a space configuration
            space_config = self._parse_space_suggestion(response.choices[0].message.content)
            return space_config
        except Exception as e:
            print(f"Error generating custom space: {e}")
            # Fallback to a default space
            return {
                'name': 'default_collaboration_space',
                'description': 'A flexible space for general collaboration',
                'capacity': len(team_structure),
                'features': ['whiteboard', 'meeting_table', 'projector']
            }

    def _create_space_generation_prompt(self, goal: str, constraints: List[str], team_structure: List[Dict]) -> str:
        """Create a prompt for space generation."""
        team_info = "\n".join([f"- {role['role_name']}: {role['description']}" for role in team_structure])
        constraints_text = "\n".join([f"- {c}" for c in constraints])
        
        return f"""
        Design an optimal collaboration space for the following scenario:

        Goal: {goal}

        Team Structure:
        {team_info}

        Constraints:
        {constraints_text}

        Provide a space configuration in JSON format with:
        {{
            "name": "Unique name for the space",
            "description": "Detailed description of the space and its purpose",
            "capacity": number of agents it can accommodate,
            "features": ["list", "of", "space", "features"],
            "collaboration_rules": ["list", "of", "guidelines", "for", "collaboration"]
        }}

        Focus on creating a space that enhances the team's ability to achieve the goal.
        """

    def _parse_space_suggestion(self, response: str) -> Dict:
        """Parse the GPT response into a space configuration."""
        try:
            import json
            # Extract JSON object from the response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                space_config = json.loads(json_match.group())
            else:
                raise ValueError("No valid JSON found in response")
        except:
            # Fallback to a basic space configuration
            space_config = {
                'name': 'basic_collaboration_space',
                'description': 'A standard space for team collaboration',
                'capacity': 5,
                'features': ['whiteboard', 'meeting_table'],
                'collaboration_rules': [
                    'Respect turn-taking',
                    'Document all decisions',
                    'Maintain clear communication'
                ]
            }
        
        return space_config

    def get_space_prompt(self, space_config: Dict) -> str:
        """Generate a prompt that describes the space and its rules."""
        prompt = f"""You are currently in: {space_config['name']}

SPACE DESCRIPTION:
{space_config['description']}

AVAILABLE FEATURES:
{', '.join(space_config['features'])}

COLLABORATION RULES:
{chr(10).join(['- ' + rule for rule in space_config['collaboration_rules']])}

Please adapt your communication and collaboration style to this space.
"""
        return prompt 