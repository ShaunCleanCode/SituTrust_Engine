import unittest
import json
import os
from datetime import datetime
from pathlib import Path
import sys
import logging
import openai
from typing import List, Dict, Optional

# Add the project root to Python path
project_root = str(Path(__file__).parent.parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from code.experiments.config import OPENAI_API_KEY, AGENT_MODEL

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(project_root, 'workspace/tests/test.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TestIntegratedCollaboration(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test."""
        self.openai_client = openai.OpenAI()
        
        # Create test output directories
        self.test_output_dir = Path(project_root) / "workspace" / "tests" / "integrated_collaboration"
        self.test_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for different stages
        self.stages = {
            'c_level': self.test_output_dir / "c_level",
            'spatial': self.test_output_dir / "spatial",
            'trust': self.test_output_dir / "trust",
            'meeting': self.test_output_dir / "meeting",
            'expert_agents': self.test_output_dir / "expert_agents"
        }
        
        for stage_dir in self.stages.values():
            stage_dir.mkdir(exist_ok=True)
        
        logger.info("Test environment setup completed")

    def _save_test_output(self, stage: str, test_name: str, data: dict):
        """Save test output to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.stages[stage] / f"{test_name}_{timestamp}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {stage} output to {output_file}")

    def generate_c_level_roles(self, goal: str, constraints: List[str]) -> List[Dict]:
        """Generate C-level roles based on goal and constraints."""
        prompt = f"""
        Based on the following project context, generate appropriate C-level roles:

        Goal: {goal}
        Constraints: {json.dumps(constraints, indent=2)}

        Generate a JSON array of C-level roles with the following structure:
        [
            {{
                "role_name": "string",  // e.g., "CTO", "CPO", "CMO"
                "role_type": "C-Level",
                "team_name": "string",  // e.g., "Technical Team", "Product Team"
                "agent_name": "string", // e.g., "cto_gpt", "cpo_gpt"
                "responsibilities": "string",
                "required_skills": "string",
                "team_size": "number",
                "key_metrics": "string",
                "collaboration_patterns": "string",
                "trust_requirements": "string",
                "decision_authority": "string",
                "communication_channels": "string",
                "success_criteria": "string"
            }}
        ]

        Return ONLY the JSON array, with no additional text.
        """
        
        response = self.openai_client.chat.completions.create(
            model=AGENT_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert in organizational design. Always respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        
        return json.loads(response.choices[0].message.content)

    def generate_space_prompt(self, goal: str, participants: List[str], tasks: List[str], mood: str) -> str:
        """Generate a space prompt for the meeting."""
        prompt = f"""
        Create a detailed meeting space description for:
        Goal: {goal}
        Participants: {', '.join(participants)}
        Tasks: {json.dumps(tasks, indent=2)}
        Mood: {mood}

        Describe the physical and psychological environment that would best facilitate this meeting.
        Focus on creating an atmosphere that promotes trust, collaboration, and effective communication.
        """
        
        response = self.openai_client.chat.completions.create(
            model=AGENT_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert in creating optimal meeting environments."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()

    def generate_trust_matrix(self, roles: List[Dict], collaboration_history: List[Dict]) -> Dict:
        """Generate a trust matrix based on roles and collaboration history."""
        prompt = f"""
        Generate a trust matrix for the following roles and collaboration history:

        Roles: {json.dumps(roles, indent=2)}
        Collaboration History: {json.dumps(collaboration_history, indent=2)}

        Return a JSON object with:
        1. trust_matrix: A matrix of trust scores between roles
        2. strategy: Trust-building strategies for each role pair
        """
        
        response = self.openai_client.chat.completions.create(
            model=AGENT_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert in team dynamics and trust building."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        
        return json.loads(response.choices[0].message.content)

    def generate_team_expert_agents(self, c_role: Dict, phase_logs: List[Dict], project_context: Dict) -> List[Dict]:
        """Generate expert agents for a specific C-level role's team."""
        prompt = f"""
        Generate expert agents for {c_role['role_name']}'s team based on:

        C-Level Role: {json.dumps(c_role, indent=2)}
        Meeting Decisions: {json.dumps(phase_logs, indent=2)}
        Project Context: {json.dumps(project_context, indent=2)}

        Generate a JSON array of expert agents with:
        [
            {{
                "role_name": "string",
                "role_type": "Team Expert",
                "team_name": "{c_role['team_name']}",
                "agent_name": "string",
                "expertise_level": "string",
                "responsibilities": "string",
                "required_skills": ["string"],
                "key_deliverables": ["string"],
                "collaboration_patterns": "string",
                "trust_requirements": "string",
                "decision_authority": "string",
                "communication_channels": "string",
                "success_criteria": "string",
                "specialization_areas": ["string"],
                "reasoning": "string"
            }}
        ]

        Return ONLY the JSON array, with no additional text.
        """
        
        response = self.openai_client.chat.completions.create(
            model=AGENT_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert in team composition and organizational design."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        
        return json.loads(response.choices[0].message.content)

    def test_tech_startup_collaboration(self):
        """Test the entire collaboration process for a tech startup scenario."""
        logger.info("Starting tech startup collaboration test")
        
        # 1. Initialize project context
        project_context = {
            'goal': """
            Develop and launch an AI-powered project management platform that helps teams 
            collaborate more effectively using natural language processing and machine learning.
            """,
            'constraints': [
                "Must be launched within 6 months",
                "Need to ensure data privacy and security compliance",
                "Must be scalable to handle enterprise-level usage",
                "Should integrate with popular project management tools"
            ]
        }
        
        # 2. Generate C-Level roles
        logger.info("Generating C-Level roles...")
        c_level_roles = self.generate_c_level_roles(
            project_context['goal'],
            project_context['constraints']
        )
        self.assertIsNotNone(c_level_roles)
        self.assertTrue(len(c_level_roles) > 0)
        logger.info(f"Generated {len(c_level_roles)} C-Level roles")
        
        # Save C-Level roles
        self._save_test_output('c_level', 'tech_startup', {
            'context': project_context,
            'roles': c_level_roles
        })
        
        # 3. Generate space prompt
        logger.info("Generating space prompt...")
        participants = [role['agent_name'] for role in c_level_roles]
        tasks = [role['responsibilities'] for role in c_level_roles]
        mood = "High trust, strategic collaboration, rapid iteration"
        
        space_prompt = self.generate_space_prompt(
            project_context['goal'],
            participants,
            tasks,
            mood
        )
        self.assertIsNotNone(space_prompt)
        logger.info("Space prompt generated successfully")
        
        # Save spatial prompt
        self._save_test_output('spatial', 'tech_startup', {
            'prompt': space_prompt,
            'participants': participants,
            'tasks': tasks,
            'mood': mood
        })
        
        # 4. Generate trust matrix
        logger.info("Generating trust matrix...")
        collaboration_history = []
        for i, a in enumerate(c_level_roles):
            for j, b in enumerate(c_level_roles):
                if i != j:
                    collaboration_history.append({
                        'agent_a': a['agent_name'],
                        'agent_b': b['agent_name'],
                        'history': 2,
                        'success_rate': 0.8,
                        'comm_compat': 0.9,
                        'domain_align': 0.85
                    })
        
        trust_data = self.generate_trust_matrix(
            c_level_roles,
            collaboration_history
        )
        self.assertIsNotNone(trust_data)
        logger.info("Trust matrix generated successfully")
        
        # Save trust matrix
        self._save_test_output('trust', 'tech_startup', {
            'trust_matrix': trust_data,
            'collaboration_history': collaboration_history
        })
        
        # 5. Simulate C-Level meeting
        logger.info("Starting C-Level meeting simulation...")
        phase_titles = [
            "PHASE 1 – Problem Definition and Structural Analysis",
            "PHASE 2 – Role Allocation and Responsibility Assignment",
            "PHASE 3 – Tactical Strategy Conference"
        ]
        
        phase_prompts = [
            "Each C-Level role defines the problem from their own perspective, analyzes the structure, and conducts one round each of discussion, rebuttal, and consensus. All responses and decisions—whether to accept, refute, or request justification—must be based on trust-weighted reasoning.",
            "Each C-Level discusses detailed role allocation and responsibility assignment. They must reach consensus on the responsibilities of each team and agent, including how collaboration will occur. Each leader must define the total number of members on their team and assign specific roles to individual agents.",
            "C-Levels engage in tactical strategy planning, including execution strategies, timelines, and final agreements. Each team must produce a detailed task line for their agents, specifying what each agent is responsible for and by when. The final outcome should be a consensus-driven execution plan across all teams."
        ]
        
        phase_logs = []
        for title, prompt in zip(phase_titles, phase_prompts):
            logger.info(f"Processing {title}...")
            phase_context = f"""
Meeting Objective: {project_context['goal']}
Participants: {', '.join(participants)}
Environment: {space_prompt}
Trust Matrix: {json.dumps(trust_data.get('trust_matrix', {}), ensure_ascii=False)}
Behavioral Strategy: {json.dumps(trust_data.get('strategy', {}), ensure_ascii=False)}

{prompt}
Have each role engage in 2–3 turns of realistic dialogue. In the conversation, each participant must clearly demonstrate trust-based behaviors such as acceptance, rebuttal, or request for justification at least once. Conclude with a brief summary of the discussion.
"""
            response = self.openai_client.chat.completions.create(
                model=AGENT_MODEL,
                messages=[
                    {"role": "system", "content": "You are a C-Level meeting simulator. Always respond in markdown, with dialogue and summary."},
                    {"role": "user", "content": phase_context}
                ],
                temperature=0.7
            )
            
            phase_dialogue = response.choices[0].message.content.strip()
            phase_logs.append({
                'phase': title,
                'dialogue': phase_dialogue
            })
            logger.info(f"Completed {title}")
        
        # Save meeting decisions
        self._save_test_output('meeting', 'tech_startup', {
            'phases': phase_logs,
            'space_prompt': space_prompt,
            'trust_matrix': trust_data
        })
        
        # 6. Generate expert agents
        logger.info("Starting expert agent generation...")
        agent_results = {}
        agent_artifacts = {}
        
        for c_role in c_level_roles:
            logger.info(f"Generating expert agents for {c_role['role_name']}'s team")
            expert_agents = self.generate_team_expert_agents(
                c_role,
                phase_logs,
                project_context
            )
            
            if expert_agents:
                # Save expert agents
                with open(os.path.join(self.stages['expert_agents'], f"{c_role['team_name']}_expert_agents.json"), "w", encoding="utf-8") as f:
                    json.dump(expert_agents, f, indent=2, ensure_ascii=False)
                
                for agent in expert_agents:
                    task_prompt = f"You are {agent['agent_name']} ({agent['role_name']}) in {c_role['team_name']}. Your main responsibilities: {agent['responsibilities']}. Please generate a sample output for your main responsibility."
                    
                    try:
                        response = self.openai_client.chat.completions.create(
                            model=AGENT_MODEL,
                            messages=[
                                {"role": "system", "content": "You are an expert agent. Always respond with a relevant output for your role."},
                                {"role": "user", "content": task_prompt}
                            ],
                            temperature=0.7
                        )
                        
                        output = response.choices[0].message.content.strip()
                        
                        # Save agent output
                        agent_dir = self.stages['expert_agents'] / c_role['team_name'] / agent['agent_name']
                        agent_dir.mkdir(parents=True, exist_ok=True)
                        
                        output_path = agent_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_output.txt"
                        with open(output_path, 'w', encoding='utf-8') as f:
                            f.write(output)
                        
                        agent['artifact_path'] = str(output_path)
                        agent_artifacts[agent['agent_name']] = str(output_path)
                        logger.info(f"Generated output for agent {agent['agent_name']}")
                        
                    except Exception as e:
                        logger.error(f"Error generating output for agent {agent['agent_name']}: {str(e)}")
                
                agent_results[c_role['team_name']] = expert_agents
        
        # Save expert agents results
        self._save_test_output('expert_agents', 'tech_startup', {
            'agent_results': agent_results,
            'agent_artifacts': agent_artifacts
        })
        
        # Verify results
        self.assertIsNotNone(agent_results)
        self.assertTrue(len(agent_results) > 0)
        self.assertTrue(len(agent_artifacts) > 0)
        
        logger.info("Test completed successfully!")
        logger.info(f"Results saved in: {self.test_output_dir}")

if __name__ == '__main__':
    unittest.main() 