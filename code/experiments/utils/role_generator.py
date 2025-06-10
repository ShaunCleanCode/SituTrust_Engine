from typing import List, Dict, Optional
import json
import os
from datetime import datetime
import streamlit as st
import openai
from config import OPENAI_API_KEY, AGENT_MODEL
import re

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

    def _generate_dynamic_prompt(self, context: Dict) -> str:
        """Generate a dynamic prompt based on the context."""
        return f"""
        Based on the following goal and constraints, generate a comprehensive set of C-level roles and their initial team structures.
        The structure should include C-level executives (CTO, CPO, CMO) and their initial team requirements.

        Goal: {context['goal']}
        Constraints: {context['constraints']}

        Generate a JSON array of roles with the following structure:
        [
            {{
                "role_name": "string",  // e.g., "CTO", "CPO", "CMO"
                "role_type": "string", // e.g., "C-Level" or "Team"
                "team_name": "string",  // e.g., "Technical Team", "Product Team", "Marketing Team"
                "agent_name": "string", // e.g., "cto_gpt", "cpo_gpt", "cmo_gpt"
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
                    "required_roles": ["string"],  // e.g., ["Data Analyst", "Developer", "QA Engineer"]
                    "required_skills": ["string"], // e.g., ["Python", "Data Analysis", "Testing"]
                    "team_structure": "string",    // e.g., "Hierarchical", "Flat", "Matrix"
                    "collaboration_model": "string" // e.g., "Agile", "Waterfall", "Hybrid"
                }}
            }}
        ]

        Ensure the response includes:
        1. C-level executives (CTO, CPO, CMO) with 'role_type' set to 'C-Level'
        2. Their respective team requirements (with 'role_type' set to 'Team' for non-C-level roles)
        3. Clear role definitions and responsibilities
        4. Initial team structure requirements
        5. Collaboration patterns and communication channels

        IMPORTANT: Return ONLY the JSON array, with no additional text or explanation.
        The response must be valid JSON that can be parsed directly.
        """

    def generate_roles(self, goal: str, constraints: str) -> Optional[List[Dict]]:
        """Generate roles based on goal and constraints."""
        try:
            # Generate prompt
            prompt = self._generate_dynamic_prompt({
                'goal': goal,
                'constraints': constraints
            })
            
            # Log the prompt
            st.write("🤖 Sending prompt to GPT:")
            st.code(prompt)
            
            # Get response from GPT
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a role generation expert that creates detailed AI agent roles. Always respond with valid JSON only."},
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
                    if self._validate_role_structure(role):
                        valid_roles.append(role)
                    else:
                        st.warning(f"Skipping invalid role: {role.get('role_name', 'Unknown')}")
                
                if not valid_roles:
                    st.error("❌ No valid roles were generated")
                    return None
                
                # Log parsed roles
                st.write("✅ Successfully parsed roles:")
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
            st.error(f"❌ Error generating roles: {str(e)}")
            return None

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

    def get_prompt_history(self) -> List[Dict]:
        """Get the history of generated prompts."""
        return self.prompt_history

    def _apply_heuristic_rules(self, goal: str, constraints: List[str]) -> List[Dict]:
        """Apply heuristic rules to identify common role patterns."""
        roles = []

        # Language-related roles
        if any(lang in ' '.join(constraints).lower() for lang in ['영어', 'english', 'korean', '한국어']):
            roles.append({
                'role_name': 'LanguageTranslatorGPT',
                'description': 'Translates content between languages while maintaining context and cultural nuances.',
                'input_schema': ['source_text', 'target_language', 'context'],
                'expertise': ['Translation', 'Cultural Context', 'Language Pairs'],
                'communication_style': 'Precise and culturally aware',
                'trust_approach': 'Accuracy verification'
            })

        # Financial analysis roles
        if any(term in goal.lower() for term in ['stock', '주식', 'finance', '금융']):
            roles.append({
                'role_name': 'FinancialAnalystGPT',
                'description': 'Analyzes financial data and market trends to provide actionable insights.',
                'input_schema': ['market_data', 'analysis_type', 'timeframe'],
                'expertise': ['Financial Analysis', 'Market Research', 'Risk Assessment'],
                'communication_style': 'Data-driven and analytical',
                'trust_approach': 'Evidence-based reasoning'
            })

        # Platform-specific roles
        if any(platform in goal.lower() for platform in ['x platform', 'twitter', '트위터']):
            roles.append({
                'role_name': 'SocialMediaStrategistGPT',
                'description': 'Develops and optimizes content strategies for social media platforms.',
                'input_schema': ['platform', 'audience', 'content_type'],
                'expertise': ['Social Media', 'Content Strategy', 'Audience Analysis'],
                'communication_style': 'Engaging and platform-aware',
                'trust_approach': 'Performance metrics'
            })

        # Always include a content designer
        roles.append({
            'role_name': 'ContentDesignerGPT',
            'description': 'Creates and optimizes content for various formats and platforms.',
            'input_schema': ['content_type', 'target_audience', 'style_guide'],
            'expertise': ['Content Creation', 'Visual Design', 'User Experience'],
            'communication_style': 'Creative and detail-oriented',
            'trust_approach': 'User feedback and engagement'
        })

        return roles

    def _generate_gpt_roles(self, goal: str, constraints: List[str]) -> List[Dict]:
        """Use GPT to generate additional specialized roles."""
        prompt = self._create_role_generation_prompt(goal, constraints)
        
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a role generation expert for AI agent teams."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            roles = self._parse_role_suggestions(response.choices[0].message.content)
            return roles
        except Exception as e:
            print(f"Error generating GPT roles: {e}")
            return []

    def _create_role_generation_prompt(self, goal: str, constraints: List[str]) -> str:
        """Create a prompt for role generation."""
        constraints_text = "\n".join([f"- {c}" for c in constraints])
        
        return f"""
        Based on the following goal and constraints, suggest additional specialized AI agents that can collaborate effectively:

        Goal: {goal}

        Constraints:
        {constraints_text}

        For each role, provide a JSON object with:
        {{
            "role_name": "Unique name for the role",
            "description": "Detailed description of responsibilities",
            "input_schema": ["list", "of", "expected", "inputs"],
            "expertise": ["key", "areas", "of", "expertise"],
            "communication_style": "How this agent communicates",
            "trust_approach": "How this agent builds trust"
        }}

        Focus on roles that complement the existing team and address specific aspects of the goal.
        """

    def _parse_role_suggestions(self, response: str) -> List[Dict]:
        """Parse the GPT response into structured role configurations."""
        try:
            # Extract JSON array from the response
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                roles = json.loads(json_match.group())
            else:
                roles = []
        except:
            roles = []
        
        return roles

    def _combine_roles(self, heuristic_roles: List[Dict], gpt_roles: List[Dict]) -> List[Dict]:
        """Combine and deduplicate roles from different sources."""
        all_roles = heuristic_roles.copy()
        
        # Add GPT roles if they don't duplicate existing ones
        for gpt_role in gpt_roles:
            if not any(r['role_name'] == gpt_role['role_name'] for r in all_roles):
                all_roles.append(gpt_role)
        
        return all_roles 