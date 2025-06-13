from typing import Dict, List
import uuid
import sys
from pathlib import Path

# Add the project root to Python path
project_root = str(Path(__file__).parent.parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from code.experiments.config import ROOM_TEMPLATES, AGENT_MODEL, OPENAI_API_KEY

import openai
import json
from datetime import datetime
import streamlit as st

class SpatialManager:
    def __init__(self):
        self.spaces = {}
        self.space_history = []
        self.openai_client = openai.OpenAI()
        self.rooms: Dict[str, Dict] = {}
        self.templates = ROOM_TEMPLATES

    def create_room(self, room_type: str, capacity: int) -> Dict:
        """
        Create a new virtual room based on a template.
        
        Args:
            room_type (str): Type of room to create (must match a template)
            capacity (int): Maximum number of agents allowed in the room
            
        Returns:
            Dict: Room configuration and metadata
        """
        if room_type not in self.templates:
            raise ValueError(f"Unknown room type: {room_type}")
            
        room_id = str(uuid.uuid4())
        template = self.templates[room_type]
        
        room = {
            'id': room_id,
            'type': room_type,
            'description': template['description'],
            'capacity': min(capacity, template['capacity']),
            'features': template['features'],
            'agents': [],
            'artifacts': []
        }
        
        self.rooms[room_id] = room
        return room

    def add_agent_to_room(self, room_id: str, agent_id: str) -> bool:
        """
        Add an agent to a room if there's capacity.
        
        Args:
            room_id (str): ID of the room
            agent_id (str): ID of the agent to add
            
        Returns:
            bool: True if agent was added successfully
        """
        if room_id not in self.rooms:
            return False
            
        room = self.rooms[room_id]
        if len(room['agents']) >= room['capacity']:
            return False
            
        if agent_id not in room['agents']:
            room['agents'].append(agent_id)
            return True
            
        return False

    def remove_agent_from_room(self, room_id: str, agent_id: str) -> bool:
        """Remove an agent from a room."""
        if room_id not in self.rooms:
            return False
            
        room = self.rooms[room_id]
        if agent_id in room['agents']:
            room['agents'].remove(agent_id)
            return True
            
        return False

    def add_artifact_to_room(self, room_id: str, artifact: Dict) -> bool:
        """Add an artifact (e.g., code, text, image) to a room."""
        if room_id not in self.rooms:
            return False
            
        self.rooms[room_id]['artifacts'].append(artifact)
        return True

    def get_room_state(self, room_id: str) -> Dict:
        """Get the current state of a room."""
        return self.rooms.get(room_id, {})

    def get_agent_room(self, agent_id: str) -> str:
        """Get the room ID where an agent is currently located."""
        for room_id, room in self.rooms.items():
            if agent_id in room['agents']:
                return room_id
        return None

    def create_meeting_space(self, meeting_type: str, team_name: str = None) -> Dict:
        """Create a meeting space based on type and team."""
        try:
            # Generate space prompt based on meeting type
            if meeting_type == 'c_level':
                prompt = """
                Create a professional meeting space for C-level executives:
                - Location: Palo Alto Blue Bottle 3rd Floor Strategy Room
                - Structure: C-shaped discussion table, 3 digital boards, central holographic projector
                - Context: Horizontal structure + responsibility sharing + real-time log storage
                - Atmosphere: Professional, strategic, collaborative
                - Tools: Digital whiteboards, real-time analytics, decision support systems
                """
            else:
                prompt = f"""
                Create a collaborative workspace for {team_name}:
                - Location: Team-specific meeting room
                - Structure: Collaborative workspace with digital tools
                - Context: Team-specific goals and collaboration patterns
                - Atmosphere: Dynamic, creative, focused
                - Tools: Team-specific tools and resources
                """
            
            # Get space configuration from GPT
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a spatial design expert that creates optimal collaboration spaces."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            
            # Parse and store space configuration
            space_config = json.loads(response.choices[0].message.content)
            space_id = f"{meeting_type}_{team_name if team_name else 'c_level'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            space = {
                'id': space_id,
                'type': meeting_type,
                'team_name': team_name,
                'config': space_config,
                'created_at': datetime.now().isoformat(),
                'status': 'active'
            }
            
            self.spaces[space_id] = space
            self.space_history.append(space)
            
            return space
            
        except Exception as e:
            st.error(f"Failed to create meeting space: {str(e)}")
            return None

    def get_space(self, space_id: str) -> Dict:
        """Get space configuration by ID."""
        return self.spaces.get(space_id)

    def update_space(self, space_id: str, updates: Dict) -> bool:
        """Update space configuration."""
        try:
            if space_id in self.spaces:
                self.spaces[space_id].update(updates)
                return True
            return False
        except Exception as e:
            st.error(f"Failed to update space: {str(e)}")
            return False

    def get_space_history(self) -> List[Dict]:
        """Get space creation history."""
        return self.space_history 

class SpatialPrompting:
    def __init__(self):
        self.openai_client = openai.OpenAI()
        self.model = AGENT_MODEL

    def generate_space_prompt(self, goal: str, participants: list, tasks: list, mood: str = "") -> str:
        """
        Generate a creative space prompt using few-shot examples and the current meeting context.
        """
        # Few-shot examples
        few_shot = '''
[Example 1]
You are entering the **Quantum Dev Hub**, an ultra-modern engineering floor designed for maximum focus, collaboration, and rapid iteration.

🖥  Layout  
•  Location : 38th floor of "Velocity Tower," overlooking San Francisco Bay  
•  Shape    : Hexagonal open-plan room (≈120 m²) with six work-stations on the perimeter and a shared core console in the center  
•  Lighting : Circadian LED panels (auto-adjust to daylight) + low-glare desk lamps  
•  Acoustics : −35 dB acoustic baffles, white-noise floor at 200 Hz for deep concentration       

🛠  Infrastructure  
1. **Central Holo-Table** – 270° holographic display of real-time CI / CD status, API latency charts, and test coverage heat-map  
2. **CodeSphere Pods** (6) – curved 49" ultrawide monitors, silent mechanical keyboards, and instant VM spin-up buttons  
3. **Live Pair-Programming Wall** – glass touch-screen for ad-hoc peer reviews; supports simultaneous multi-cursor 

[Example 2]
You are entering the **Solvay 전략 회의실**, a strategic roundtable for high-stakes decision-making.

🌐 공간 Prompting
• 위치: 팔로알토 블루보틀 3층 전략실
• 구조: C자형 토론형 원형 책상, 디지털 보드 3개, 중앙 Holographic Projector
• 컨텍스트: 수평 구조 + 책임 분담 + 실시간 로그 저장
'''
        # Build meeting context string
        participants_str = ', '.join(participants)
        tasks_str = ', '.join(tasks)
        context = f"""
Now, given the following meeting context, generate a new space prompt in the same style:
- Goal: {goal}
- Participants: {participants_str}
- Tasks: {tasks_str}
- Mood: {mood}
"""
        full_prompt = f"""
You are a spatial architect AI. Given a meeting's goal, participants, and context, design a creative, optimal collaboration space.
{few_shot}
{context}
"""
        # Call GPT
        response = self.openai_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a spatial architect. Always respond with a creative space prompt in the same style as the examples."},
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content.strip() 