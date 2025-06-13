from typing import List, Dict, Optional
import json
import os
from datetime import datetime
import streamlit as st
import openai
from code.experiments.config import OPENAI_API_KEY, AGENT_MODEL
import re
import uuid
import sys
from pathlib import Path

# Add the project root to Python path
project_root = str(Path(__file__).parent.parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

class RoleGenerator:
    def __init__(self):
        self.openai_client = openai.OpenAI()
        self.prompt_history = []
        self.response_history = []
        self.model = AGENT_MODEL

    def _clean_json_response(self, response: str) -> str:
        """Clean and validate JSON response."""
        try:
            # Remove any markdown code block markers
            response = re.sub(r'```json\s*|\s*```', '', response)
            response = re.sub(r'```\s*|\s*```', '', response)
            
            # Remove any leading/trailing whitespace
            response = response.strip()
            
            # Try to parse and re-serialize to ensure valid JSON
            parsed = json.loads(response)
            return json.dumps(parsed)
        except json.JSONDecodeError as e:
            st.error(f"Failed to clean JSON response: {str(e)}")
            return response

    def _validate_role_structure(self, role: Dict) -> bool:
        """Validate that a role has all required fields with correct types."""
        required_fields = {
            'role_name': str,
            'role_type': str,
            'team_name': str,
            'agent_name': str,
            'responsibilities': str,
            'required_skills': str,
            'team_size': int,
            'key_metrics': str,
            'collaboration_patterns': str,
            'trust_requirements': str,
            'decision_authority': str,
            'communication_channels': str,
            'success_criteria': str,
            'initial_team_requirements': dict
        }
        
        try:
            for field, field_type in required_fields.items():
                if field not in role:
                    st.error(f"Missing required field: {field}")
                    return False
                if not isinstance(role[field], field_type):
                    st.error(f"Invalid type for field {field}: expected {field_type}, got {type(role[field])}")
                    return False
            
            # Validate initial_team_requirements structure
            team_req = role['initial_team_requirements']
            if not all(key in team_req for key in ['required_roles', 'required_skills', 'team_structure', 'collaboration_model']):
                st.error("Missing required fields in initial_team_requirements")
                return False
            
            return True
        except Exception as e:
            st.error(f"Error validating role structure: {str(e)}")
            return False
        

    def generate_c_level_roles(self, goal: str, constraints: str) -> Optional[List[Dict]]:
        """Generate C-level roles based on goal and constraints with enhanced context awareness."""
        try:
            # Generate prompt
            prompt = self._generate_c_level_prompt({
                'goal': goal,
                'constraints': constraints
            })
            
            # Log the prompt
            st.write("🤖 Sending C-level prompt to SituTrust Engine:")
            
            # Get response from GPT
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": """You are an expert in organizational design and leadership structure. 
                        Your task is to analyze project requirements and determine the most appropriate C-level roles.
                        Think step by step about what roles are truly necessary for the project's success.
                        Consider both traditional and specialized roles based on the specific context."""
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            
            # Get the response content
            response_content = response.choices[0].message.content
            
            # Log the raw response
            st.write("📝 Raw GPT Response:")
            st.code(response_content)
            
            # Clean and parse response
            try:
                cleaned_response = self._clean_json_response(response_content)
                roles = json.loads(cleaned_response)
                
                # Validate roles structure
                if not isinstance(roles, list):
                    st.error("❌ GPT response is not a list of roles")
                    return None
                
                # Validate each role
                valid_roles = []
                for role in roles:
                    if self._validate_role_structure(role) and role['role_type'] == 'C-Level':
                        # Ensure reasoning field exists
                        if 'reasoning' not in role:
                            role['reasoning'] = "Role necessity reasoning not provided"
                        valid_roles.append(role)
                    else:
                        st.warning(f"Skipping invalid or non-C-level role: {role.get('role_name', 'Unknown')}")
                
                if not valid_roles:
                    st.error("❌ No valid C-level roles were generated")
                    return None
                
                # Log parsed roles with reasoning
                st.write("✅ Successfully parsed C-level roles:")
                for role in valid_roles:
                    st.write(f"### {role['role_name']}")
                    st.write(f"**Reasoning:** {role['reasoning']}")
                    st.json(role)
                
                # Store in history
                self.prompt_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'prompt': prompt
                })
                self.response_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'response': cleaned_response
                })
                
                return valid_roles
                
            except json.JSONDecodeError as e:
                st.error(f"❌ Failed to parse GPT response as JSON: {str(e)}")
                st.write("Raw response that failed to parse:")
                st.code(response_content)
                return None
                
        except Exception as e:
            st.error(f"❌ Error generating C-level roles: {str(e)}")
            return None    

    def _generate_c_level_prompt(self, context: Dict) -> str:
        """Generate a dynamic prompt for C-level roles with Chain of Thought reasoning."""
        return f"""
        You are an expert in organizational design and leadership structure. Your task is to analyze the project context and determine the most appropriate C-level roles needed.

        Let's think through this step by step:

        1. First, analyze the project goal and constraints:
        Goal: {context['goal']}
        Constraints: {context['constraints']}

        2. Consider the following aspects:
           - What are the key functional areas needed for this project?
           - What specialized expertise is required?
           - What strategic leadership roles are essential?
           - What cross-functional coordination is needed?
           - What unique challenges or opportunities exist?
           - What are the critical success factors for this project?
           - What are the potential risks and how can they be mitigated?

        3. For each potential C-level role, consider:
           - Is this role truly necessary for the project's success?
           - What unique value does this role bring?
           - How does this role interact with other C-level positions?
           - What specific challenges will this role address?
           - What are the key decision-making authorities?
           - What are the critical dependencies with other roles?
           - How will this role contribute to building trust across teams?
           - What are the specific KPIs for measuring success?
        
        4. ROLE FIELD GUIDELINES (CRITICAL)
           For each executive, include a field `"role"` that goes beyond the titleto describe:

-  **Team Influence**: How do they inspire and direct teams? What kinds of cross-functional impact do they create?
-  **Core Mission**: What strategic objectives do they own? How does this tie to the company’s success?
-  **Accountability**: What are their personal success metrics and failure responsibilities?
-  **Behavioral Patterns**: How do they communicate, escalate issues, negotiate alignment, and handle conflict?
-  **Structural Role**: Where do they sit in the hierarchy? Whom do they delegate to?

Make this field **rich, narrative, and actionable**, written like an onboarding briefing for a human executive. This field will be used to train aligned role-specific AI agents, so clarity is critical.

        5. Based on this analysis, generate a JSON array of C-level roles with the following structure:
        [
            {{
                "role_name": "string",  // e.g., "CTO", "CPO", "CMO", "CRO", "CDO", etc.
                "role_type": "C-Level",
                "role": "string",  // Fully articulated narrative as described above
                "team_name": "string",  // e.g., "Technical Team", "Product Team", "Marketing Team"
                "agent_name": "string", // e.g., "cto_gpt", "cpo_gpt", "cmo_gpt"
                "responsibilities": "string",  // Detailed list of responsibilities
                "required_skills": "string",   // Comprehensive list of required skills
                "team_size": "number",
                "key_metrics": "string",       // Specific KPIs and success metrics
                "collaboration_patterns": "string",  // Detailed collaboration strategies
                "trust_requirements": "string",      // Trust-building strategies
                "decision_authority": "string",      // Clear decision-making boundaries
                "communication_channels": "string",  // Communication protocols
                "success_criteria": "string",        // Specific success criteria
                "initial_team_requirements": {{
                    "required_roles": ["string"],
                    "required_skills": ["string"],
                    "team_structure": "string",
                    "collaboration_model": "string"
                }},
                "cross_functional_impact": "string",  // Impact on other teams
                "risk_management": "string",          // Risk management strategies
                "innovation_focus": "string",         // Innovation and growth areas
                "stakeholder_management": "string",   // Stakeholder engagement strategy
                "reasoning": "string"  // Explain why this role is necessary and how it contributes to the project
            }}
        ]

        Ensure the response includes:
        1. ONLY C-level executives with 'role_type' set to 'C-Level'
        2. A clearly written 'role' field for each executive, describing their mission, leadership style, behavioral signature, and organizational influence. THIS FIELD IS MANDATORY.
        3. Clear role definitions and responsibilities
        4. Initial team structure requirements
        5. Collaboration patterns and communication channels
        6. Detailed trust-building strategies
        7. Specific KPIs and success metrics
        8. Cross-functional impact and dependencies
        9. Risk management approaches
        10. Innovation focus areas
        11. Stakeholder management strategies
        12. Reasoning for each role's necessity

        IMPORTANT:
        - Return ONLY the JSON array, with no additional text or explanation.
        - The 'role' field is REQUIRED for each entry. Do not skip or leave this field vague or generic.   
        - The response must be valid JSON that can be parsed directly.
        - Include a 'reasoning' field for each role explaining its necessity.
        - Consider both traditional C-level roles and any specialized roles needed for this specific project.
        - Ensure each role has clear boundaries and responsibilities to avoid overlap.
        - Focus on creating a cohesive leadership team that can work together effectively."""

    def _generate_team_roles_prompt(self, c_level_role: Dict, context: Dict) -> str:
        """Generate a dynamic prompt for team roles under a specific C-level executive."""
        return f"""
        Based on the following C-level role and project context, generate a comprehensive set of team roles.
        Focus on the specific team members needed under this C-level executive.

        C-Level Role: {json.dumps(c_level_role, indent=2)}
        Project Goal: {context['goal']}
        Project Constraints: {context['constraints']}

        Generate a JSON array of team roles with the following structure:
        [
            {{
                "role_name": "string",  // e.g., "Senior Developer", "Product Manager", "Marketing Specialist"
                "role_type": "Team",
                "team_name": "{c_level_role['team_name']}",
                "agent_name": "string", // e.g., "senior_dev_gpt", "pm_gpt", "marketing_spec_gpt"
                "responsibilities": "string",
                "required_skills": "string",
                "team_size": "number",
                "key_metrics": "string",
                "collaboration_patterns": "string",
                "trust_requirements": "string",
                "decision_authority": "string",
                "communication_channels": "string",
                "success_criteria": "string",
                "initial_team_requirements": {{
                    "required_roles": ["string"],
                    "required_skills": ["string"],
                    "team_structure": "string",
                    "collaboration_model": "string"
                }}
            }}
        ]

        Ensure the response includes:
        1. ONLY team-level roles with 'role_type' set to 'Team'
        2. Roles that align with the C-level executive's team requirements
        3. Clear role definitions and responsibilities
        4. Specific collaboration patterns and communication channels

        IMPORTANT: Return ONLY the JSON array, with no additional text or explanation.
        The response must be valid JSON that can be parsed directly.
        """

    

    def generate_team_roles(self, c_level_role: Dict, goal: str, constraints: str) -> Optional[List[Dict]]:
        """Generate team roles for a specific C-level executive."""
        try:
            # Generate prompt
            prompt = self._generate_team_roles_prompt(c_level_role, {
                'goal': goal,
                'constraints': constraints
            })
            
            # Log the prompt
            st.write(f"🤖 Sending team roles prompt for {c_level_role['role_name']} to GPT:")
            st.code(prompt)
            
            # Get response from GPT
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a team role generation expert. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            
            # Get the response content
            response_content = response.choices[0].message.content
            
            # Log the raw response
            st.write("📝 Raw GPT Response:")
            st.code(response_content)
            
            # Clean and parse response
            try:
                cleaned_response = self._clean_json_response(response_content)
                roles = json.loads(cleaned_response)
                
                # Validate roles structure
                if not isinstance(roles, list):
                    st.error("❌ GPT response is not a list of roles")
                    return None
                
                # Validate each role
                valid_roles = []
                for role in roles:
                    if self._validate_role_structure(role) and role['role_type'] == 'Team':
                        valid_roles.append(role)
                    else:
                        st.warning(f"Skipping invalid or non-team role: {role.get('role_name', 'Unknown')}")
                
                if not valid_roles:
                    st.error("❌ No valid team roles were generated")
                    return None
                
                # Log parsed roles
                st.write("✅ Successfully parsed team roles:")
                st.json(valid_roles)
                
                # Store in history
                self.prompt_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'prompt': prompt
                })
                self.response_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'response': cleaned_response
                })
                
                return valid_roles
                
            except json.JSONDecodeError as e:
                st.error(f"❌ Failed to parse GPT response as JSON: {str(e)}")
                st.write("Raw response that failed to parse:")
                st.code(response_content)
                return None
                
        except Exception as e:
            st.error(f"❌ Error generating team roles: {str(e)}")
            return None

    def get_prompt_history(self) -> List[Dict]:
        """Get the history of prompts and responses."""
        return self.prompt_history

    def get_response_history(self) -> List[Dict]:
        """Get the history of responses."""
        return self.response_history

    def generate_team_agents(self, team_requirements: Dict, c_level_decisions: Dict) -> List[Dict]:
        """Generate team agents based on requirements and C-level decisions."""
        try:
            # Generate prompt for team agent creation
            prompt = f"""
            Create specialized AI agents for the team based on:
            Team Requirements: {json.dumps(team_requirements, indent=2)}
            C-Level Decisions: {json.dumps(c_level_decisions, indent=2)}
            
            Generate a JSON array of team members with:
            1. Role name
            2. Agent name
            3. Responsibilities
            4. Required skills
            5. Collaboration patterns
            
            Return ONLY the JSON array, with no additional text or explanation.
            """
            
            # Get team agents from GPT
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a team structure designer. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            
            # Clean and parse response
            cleaned_response = self._clean_json_response(response.choices[0].message.content)
            return json.loads(cleaned_response)
            
        except Exception as e:
            st.error(f"Failed to generate team agents: {str(e)}")
            return []

    def update_roles(self, current_roles: List[Dict], goal: str, constraints: List[str], 
                    meeting_progress: Dict = None) -> List[Dict]:
        """Update roles based on meeting progress and changing requirements."""
        # Generate dynamic prompt for role updates
        prompt = f"""Based on the current meeting state, determine necessary role adjustments.

        Current State:
        - Roles: {json.dumps(current_roles, indent=2)}
        - Progress: {json.dumps(meeting_progress, indent=2) if meeting_progress else 'No progress data'}
        - Goal: {goal}
        - Constraints: {constraints}
        
        Consider:
        1. What roles are no longer needed?
        2. What new roles are required?
        3. How should existing roles be modified?
        4. What team restructuring is needed?
        
        Provide a response that includes:
        1. Role modifications
        2. Team structure changes
        3. New collaboration patterns
        4. Updated success criteria
        """

        try:
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a role management expert that adapts to changing meeting dynamics."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            
            # Parse the response
            updated_roles_data = json.loads(response.choices[0].message.content)
            return updated_roles_data['roles']
            
        except Exception as e:
            st.error(f"Failed to update roles: {str(e)}")
            return current_roles

    def analyze_role_effectiveness(self, roles: List[Dict], meeting_progress: Dict) -> Dict:
        """Analyze the effectiveness of current roles and suggest improvements."""
        # Generate dynamic prompt for analysis
        prompt = f"""Analyze the current team structure and role effectiveness.

        Current State:
        - Roles: {json.dumps(roles, indent=2)}
        - Progress: {json.dumps(meeting_progress, indent=2)}
        
        Consider:
        1. How well are roles performing?
        2. What are the current bottlenecks?
        3. What opportunities exist for improvement?
        4. How can we optimize the team structure?
        
        Provide analysis that includes:
        1. Role performance metrics
        2. Team effectiveness indicators
        3. Improvement opportunities
        4. Optimization suggestions
        """

        try:
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a team effectiveness analyst that provides dynamic insights."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            
            # Parse the response
            analysis_data = json.loads(response.choices[0].message.content)
            return analysis_data['analysis']
            
        except Exception as e:
            st.error(f"Failed to analyze role effectiveness: {str(e)}")
            return {}

    

    

    

    

    def generate_team_expert_agents(self, c_level_role: Dict, meeting_decisions: Dict, project_context: Dict) -> Optional[List[Dict]]:
        """
        Generate team expert agents based on C-level meeting decisions and project context.
        
        Args:
            c_level_role: The C-level role (e.g., CTO, CPO) for which to generate team experts
            meeting_decisions: Decisions made during the C-level meeting
            project_context: Project context including goal and constraints
            
        Returns:
            List of team expert agent configurations
        """
        try:
            # Create a detailed logging section
            st.write("## 🎯 Team Expert Agent Generation Process")
            st.write(f"### 📊 Generating experts for {c_level_role['role_name']}'s team")
            
            
            
            
            # Generate prompt for team expert agent creation
            prompt = f"""
            You are an expert in team composition and organizational design. Your task is to create specialized expert agents
            for the {c_level_role['role_name']}'s team based on the C-level meeting decisions and project requirements.

            Let's think through this step by step:

            1. Analyze the C-level role and its requirements:
            C-Level Role: {json.dumps(c_level_role, indent=2)}

            2. Consider the meeting decisions:
            Meeting Decisions: {json.dumps(meeting_decisions, indent=2)}

            3. Review project context:
            Project Goal: {project_context['goal']}
            Project Constraints: {project_context['constraints']}

            4. For each potential expert role, consider:
               - What specific expertise is needed?
               - How does this role contribute to the team's objectives?
               - What unique skills and knowledge are required?
               - How will this role collaborate with other team members?
               - What specific deliverables or outcomes are expected?
            6. For each expert agent, include a detailed "role" field that describes:

                - **Mission Focus**: What is the agent’s strategic mission within the team? What higher-order goal do they own?
                - **Operational Influence**: How do they affect workflows, decisions, and other team members?
                - **Behavioral Profile**: How do they communicate, negotiate, escalate, or resolve conflicts?
                - **Collaboration Mode**: Do they lead, advise, validate, or execute? What rituals (e.g., syncs, code reviews) do they drive?
                - **Alignment Role**: How do they ensure coherence with the C-level’s objective? What “guardrails” do they enforce?
                - **Knowledge Anchor**: What specific subject-matter knowledge do they bring? What past experience backs it?

                The "role" field should be rich, written like an onboarding manual for a real human agent. It will be used to simulate realistic autonomous behavior in multi-agent workflows.   

            7. Generate a JSON array of expert agents with the following structure:
            [
                {{
                    "role_name": "string",  // e.g., "Senior Developer", "Data Scientist", "UX Designer"
                    "role_type": "Team Expert",
                    "team_name": "{c_level_role['team_name']}",
                    "agent_name": "string", // e.g., "senior_dev_gpt", "data_scientist_gpt"
                    "expertise_level": "string", // e.g., "Senior", "Lead", "Principal"
                    "role": "string",  // This field is mandatory and must be described in detail.
                    "responsibilities": "string",
                    "required_skills": ["string"],
                    "key_deliverables": ["string"],
                    "collaboration_patterns": "string",
                    "trust_requirements": "string",
                    "decision_authority": "string",
                    "communication_channels": "string",
                    "success_criteria": "string",
                    "specialization_areas": ["string"],
                    "reasoning": "string"  // Explain why this expert role is necessary
                }}
            ]

            Ensure the response includes:
            1. ONLY team expert roles with 'role_type' set to 'Team Expert'
            2. Clear role definitions and responsibilities
            3. Specific expertise areas and required skills
            4. Collaboration patterns and communication channels
            5. Reasoning for each role's necessity
            6. Key deliverables and success criteria

            IMPORTANT:
            - Return ONLY the JSON array, with no additional text or explanation.
            - The response must be valid JSON that can be parsed directly.
            - Include a 'reasoning' field for each role explaining its necessity.
            - Consider both technical and non-technical expert roles as needed.
            - Ensure roles align with the C-level meeting decisions and project requirements.
            """
            
            
            
            # Get response from GPT
            with st.spinner(f"🔄 Generating expert agents for {c_level_role['role_name']}'s team..."):
                response = self.openai_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": """You are an expert in team composition and organizational design.
                            Your task is to create specialized expert agents for teams based on C-level decisions and project requirements.
                            Think step by step about what expert roles are truly necessary for the team's success.
                            Consider both technical and non-technical expertise as needed."""
                        },
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7
                )
            
            # Get the response content
            response_content = response.choices[0].message.content
            
            
            # Clean and parse response
            try:
                cleaned_response = self._clean_json_response(response_content)
                expert_agents = json.loads(cleaned_response)
                
                # Validate expert agents structure
                if not isinstance(expert_agents, list):
                    st.error("❌ GPT response is not a list of expert agents")
                    return None
                
                # Validate each expert agent
                valid_expert_agents = []
                for agent in expert_agents:
                    if self._validate_expert_agent_structure(agent):
                        # Ensure reasoning field exists
                        if 'reasoning' not in agent:
                            agent['reasoning'] = "Role necessity reasoning not provided"
                        valid_expert_agents.append(agent)
                    else:
                        st.warning(f"Skipping invalid expert agent: {agent.get('role_name', 'Unknown')}")
                
                if not valid_expert_agents:
                    st.error("❌ No valid expert agents were generated")
                    return None
                
                # Display generated expert agents
                st.write("### ✅ Generated Expert Agents")
                
                # Create a summary table
                summary_data = []
                for agent in valid_expert_agents:
                    summary_data.append({
                        "Role": agent['role_name'],
                        "Expertise Level": agent['expertise_level'],
                        "Specialization": ", ".join(agent['specialization_areas']),
                        "Key Skills": ", ".join(agent['required_skills'][:3]) + "..." if len(agent['required_skills']) > 3 else ", ".join(agent['required_skills'])
                    })
                
                st.table(summary_data)
                
                # Display detailed information for each agent
                for agent in valid_expert_agents:
                    with st.expander(f"🔍 {agent['role_name']} - {agent['expertise_level']}", expanded=True):
                        st.write("#### Role Details")
                        st.json({
                            "Role Name": agent['role_name'],
                            "Expertise Level": agent['expertise_level'],
                            "Team": agent['team_name'],
                            "Agent Name": agent['agent_name']
                        })
                        
                        st.write("#### Responsibilities & Requirements")
                        st.json({
                            "Responsibilities": agent['responsibilities'],
                            "Required Skills": agent['required_skills'],
                            "Specialization Areas": agent['specialization_areas'],
                            "Key Deliverables": agent['key_deliverables']
                        })
                        
                        st.write("#### Collaboration & Success")
                        st.json({
                            "Collaboration Patterns": agent['collaboration_patterns'],
                            "Communication Channels": agent['communication_channels'],
                            "Trust Requirements": agent['trust_requirements'],
                            "Decision Authority": agent['decision_authority'],
                            "Success Criteria": agent['success_criteria']
                        })
                        
                        st.write("#### Reasoning")
                        st.info(agent['reasoning'])
                
                # Store in history
                self.prompt_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'prompt': prompt
                })
                self.response_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'response': cleaned_response
                })
                
                return valid_expert_agents
                
            except json.JSONDecodeError as e:
                st.error(f"❌ Failed to parse GPT response as JSON: {str(e)}")
                st.write("Raw response that failed to parse:")
                st.code(response_content)
                return None
                
        except Exception as e:
            st.error(f"❌ Error generating team expert agents: {str(e)}")
            return None

    def _validate_expert_agent_structure(self, agent: Dict) -> bool:
        """Validate that an expert agent has all required fields with correct types."""
        required_fields = {
            'role_name': str,
            'role_type': str,
            'team_name': str,
            'agent_name': str,
            'expertise_level': str,
            'responsibilities': str,
            'required_skills': list,
            'key_deliverables': list,
            'collaboration_patterns': str,
            'trust_requirements': str,
            'decision_authority': str,
            'communication_channels': str,
            'success_criteria': str,
            'specialization_areas': list,
            'reasoning': str
        }
        
        try:
            for field, field_type in required_fields.items():
                if field not in agent:
                    st.error(f"Missing required field: {field}")
                    return False
                if not isinstance(agent[field], field_type):
                    st.error(f"Invalid type for field {field}: expected {field_type}, got {type(agent[field])}")
                    return False
            
            # Additional validation for expert agent specific fields
            if agent['role_type'] != 'Team Expert':
                st.error("Invalid role_type for expert agent")
                return False
            
            return True
        except Exception as e:
            st.error(f"Error validating expert agent structure: {str(e)}")
            return False 
        
    def summarize_meeting_logs_with_gpt(self,phase_logs: List[Dict], model="gpt-4", temperature=0.3) -> Dict[str, Dict[str, str]]:
        """
        Uses GPT to summarize each phase's meeting dialogue to extract essential decisions.

        Args:
            phase_logs (List[Dict]): List of meeting phases containing dialogues
            model (str): Model to use (default: "gpt-4")
            temperature (float): Sampling temperature

        Returns:
            Dict[str, Dict[str, str]]: Summary info per phase
        """
        summarized_results = {}

        for i, phase in enumerate(phase_logs):
            phase_name = phase.get("phase", f"Phase {i+1}")
            dialogue = phase.get("dialogue", "")

            prompt = f"""
    You are a project analyst AI agent. Your task is to extract a concise summary of the meeting below.
    Focus on: 
    - The key goals, outcomes, and decisions made
    - Any agreed task responsibilities by team
    - Keep it brief (around 100–150 words)

    ### Meeting Transcript:
    {dialogue}

    ### Response Format:
    Summary: <summary text>
    Team Tasks:
    - Team A: <responsibility>
    - Team B: <responsibility>
    - ...
            """

            try:
                response = self.openai_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a summarization assistant for AI meetings."},
                        {"role": "user", "content": prompt.strip()}
                    ],
                    temperature=temperature
                )
                content = response.choices[0].message.content.strip()
                summarized_results[phase_name] = {"summary": content}

            except Exception as e:
                summarized_results[phase_name] = {"summary": f"[ERROR] {str(e)}"}

        return summarized_results