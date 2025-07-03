from typing import Dict, List, Optional
import json
import os
from datetime import datetime
import streamlit as st
import openai
import sys
from pathlib import Path

# Add the project root to Python path
project_root = str(Path(__file__).parent.parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from code.experiments.config import OPENAI_API_KEY, AGENT_MODEL

import numpy as np
import uuid
import re

class CollaborationFlow:
    def __init__(self):
        self.teams = {}
        self.team_tasks = {}
        self.team_artifacts = {}
        self.team_logs = {}
        self.meeting_log = []
        self.task_assignments = {}
        self.flow_history = []
        self.openai_client = openai.OpenAI()
        self.model = AGENT_MODEL

    def _generate_dynamic_flow_prompt(self, context: Dict) -> str:
        """Generate a dynamic prompt for meeting flow based on current context."""
        prompt = f"""Based on the current meeting context, determine the optimal flow and next steps.

        Meeting Context:
        - Current Stage: {context.get('stage', 'initial')}
        - Goal: {context.get('goal', '')}
        - Constraints: {context.get('constraints', [])}
        - Team Structure: {json.dumps(context.get('teams', []), indent=2)}
        - Progress: {context.get('progress', '0%')}
        - Current Challenges: {context.get('challenges', [])}
        - Time Elapsed: {context.get('time_elapsed', '0 minutes')}
        
        Consider:
        1. What is the most effective next step?
        2. How can we optimize the current flow?
        3. What adjustments are needed?
        4. How can we ensure goal achievement?
        
        Provide flow guidance that includes:
        1. Next steps
        2. Required actions
        3. Team coordination
        4. Success criteria
        """

        return prompt

    def launch_meeting(self, goal: str, constraints: List[str], roles: List[Dict], 
                      space_config: Dict, context: Dict = None) -> Dict:
        """Launch a new meeting with dynamic flow configuration."""
        if context is None:
            context = {
                'stage': 'initial',
                'progress': '0%',
                'challenges': [],
                'time_elapsed': '0 minutes'
            }
        
        # Generate dynamic prompt
        prompt = self._generate_dynamic_flow_prompt({
            'goal': goal,
            'constraints': constraints,
            'teams': roles,
            'space_config': space_config,
            **context
        })
        
        # Store flow configuration in history
        self.flow_history.append({
            'timestamp': datetime.now().isoformat(),
            'prompt': prompt,
            'context': context
        })

        try:
            st.write("🤖 Requesting GPT for meeting flow configuration...")
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a meeting flow expert that creates dynamic collaboration processes. Respond with a valid JSON object containing flow configuration."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            
            # Extract the response content
            content = response.choices[0].message.content.strip()
            st.write("📥 Received GPT response:")
            st.code(content, language="json")
            
            # Try to parse the JSON response
            try:
                flow_config = json.loads(content)
                st.write("✅ Successfully parsed JSON response")
            except json.JSONDecodeError as e:
                st.error(f"❌ JSON parsing error: {str(e)}")
                st.write("Raw response that caused the error:")
                st.code(content, language="text")
                st.write("Using default flow configuration...")
                flow_config = {
                    'next_steps': [],
                    'required_actions': [],
                    'team_coordination': {},
                    'success_criteria': []
                }
            
            # Create meeting configuration
            meeting_config = {
                'id': str(uuid.uuid4()),
                'goal': goal,
                'constraints': constraints,
                'space_config': space_config,
                'teams': roles,
                'flow_config': flow_config,
                'status': 'active',
                'created_at': datetime.now().isoformat()
            }
            
            st.write("✅ Meeting configuration created successfully")
            return meeting_config
            
        except Exception as e:
            st.error(f"❌ Failed to launch meeting: {str(e)}")
            return {}

    def update_meeting_flow(self, meeting_config: Dict, context: Dict = None) -> Dict:
        """Update meeting flow based on current context."""
        if context is None:
            context = {
                'stage': 'ongoing',
                'progress': '50%',
                'challenges': [],
                'time_elapsed': '30 minutes'
            }
        
        # Generate dynamic prompt for flow update
        prompt = f"""Based on the current meeting state, determine necessary flow adjustments.

        Current State:
        - Meeting Configuration: {json.dumps(meeting_config, indent=2)}
        - Context: {json.dumps(context, indent=2)}
        
        Consider:
        1. How has the meeting progressed?
        2. What adjustments are needed?
        3. How can we optimize the flow?
        4. What new challenges have emerged?
        
        Provide updated flow configuration that includes:
        1. Flow adjustments
        2. New steps
        3. Team coordination updates
        4. Success criteria updates
        """

        try:
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a meeting flow optimization expert that adapts to changing meeting dynamics."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            
            # Parse the response
            updated_flow = json.loads(response.choices[0].message.content)
            meeting_config['flow_config'] = updated_flow
            return meeting_config
            
        except Exception as e:
            st.error(f"Failed to update meeting flow: {str(e)}")
            return meeting_config

    def analyze_meeting_flow(self, meeting_config: Dict, context: Dict = None) -> Dict:
        """Analyze the effectiveness of current meeting flow."""
        if context is None:
            context = {
                'stage': 'ongoing',
                'progress': '50%',
                'challenges': [],
                'time_elapsed': '30 minutes'
            }
        
        # Generate dynamic prompt for flow analysis
        prompt = f"""Analyze the current meeting flow and its effectiveness.

        Current State:
        - Meeting Configuration: {json.dumps(meeting_config, indent=2)}
        - Context: {json.dumps(context, indent=2)}
        
        Consider:
        1. How effective is the current flow?
        2. What bottlenecks exist?
        3. What opportunities exist for improvement?
        4. How can we optimize the process?
        
        Provide analysis that includes:
        1. Flow effectiveness metrics
        2. Bottleneck analysis
        3. Improvement opportunities
        4. Optimization suggestions
        """

        try:
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a meeting flow analyst that provides dynamic insights."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            
            # Parse the response
            analysis_data = json.loads(response.choices[0].message.content)
            return analysis_data['analysis']
            
        except Exception as e:
            st.error(f"Failed to analyze meeting flow: {str(e)}")
            return {}

    def get_flow_history(self) -> List[Dict]:
        """Get the history of flow configurations."""
        return self.flow_history

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

    def _validate_meeting_summary(self, summary: Dict) -> bool:
        """Validate that a meeting summary has all required fields."""
        required_fields = {
            'key_discussion_points': list,
            'decisions_made': list,
            'action_items': list,
            'next_steps': list,
            'task_assignments': list
        }
        
        try:
            for field, field_type in required_fields.items():
                if field not in summary:
                    st.error(f"Missing required field in meeting summary: {field}")
                    return False
                if not isinstance(summary[field], field_type):
                    st.error(f"Invalid type for field {field}: expected {field_type}, got {type(summary[field])}")
                    return False
            return True
        except Exception as e:
            st.error(f"Error validating meeting summary: {str(e)}")
            return False

    def initialize_teams(self, roles: List[Dict]) -> bool:
        """Initialize teams based on roles."""
        try:
            for role in roles:
                team_name = role['team_name']
                self.teams[team_name] = {
                    'leader': role,
                    'members': [],  # Will be populated after C-level meeting
                    'status': 'pending',  # Will be 'active' after team creation
                    'requirements': role['initial_team_requirements']
                }
                self.team_tasks[team_name] = []
                self.team_artifacts[team_name] = []
                self.team_logs[team_name] = []
            return True
        except Exception as e:
            st.error(f"Failed to initialize teams: {str(e)}")
            return False

    def create_team(self, team_name: str, team_members: List[Dict]) -> bool:
        """Create a team with its members."""
        try:
            if team_name in self.teams:
                self.teams[team_name]['members'] = team_members
                self.teams[team_name]['status'] = 'active'
                self._add_team_log(team_name, f"Team created with {len(team_members)} members")
                return True
            return False
        except Exception as e:
            st.error(f"Failed to create team: {str(e)}")
            return False

    def add_task(self, team_name: str, task_name: str, description: str, 
                assigned_to: str = None, priority: str = 'medium') -> bool:
        """Add a task to a team."""
        try:
            if team_name in self.teams:
                task = {
                    'id': f"task_{len(self.team_tasks[team_name]) + 1}",
                    'name': task_name,
                    'description': description,
                    'assigned_to': assigned_to,
                    'priority': priority,
                    'status': 'pending',
                    'created_at': datetime.now().isoformat(),
                    'completed_at': None,
                    'dependencies': [],
                    'artifacts': []
                }
                self.team_tasks[team_name].append(task)
                self._add_team_log(team_name, f"New task created: {task_name}")
                return True
            return False
        except Exception as e:
            st.error(f"Failed to add task: {str(e)}")
            return False

    def complete_task(self, team_name: str, task_id: str, 
                     artifact: Dict = None) -> bool:
        """Mark a task as completed and store its artifact."""
        try:
            if team_name in self.teams:
                for task in self.team_tasks[team_name]:
                    if task['id'] == task_id:
                        task['status'] = 'completed'
                        task['completed_at'] = datetime.now().isoformat()
                        if artifact:
                            task['artifacts'].append(artifact)
                            self._add_team_artifact(team_name, artifact)
                        self._add_team_log(team_name, f"Task completed: {task['name']}")
                        return True
            return False
        except Exception as e:
            st.error(f"Failed to complete task: {str(e)}")
            return False

    def _add_team_artifact(self, team_name: str, artifact: Dict) -> None:
        """Add an artifact to a team's collection."""
        try:
            artifact['created_at'] = datetime.now().isoformat()
            self.team_artifacts[team_name].append(artifact)
        except Exception as e:
            st.error(f"Failed to add team artifact: {str(e)}")

    def _add_team_log(self, team_name: str, message: str) -> None:
        """Add a log entry for a team."""
        try:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'message': message
            }
            self.team_logs[team_name].append(log_entry)
        except Exception as e:
            st.error(f"Failed to add team log: {str(e)}")

    def get_team_status(self, team_name: str) -> Optional[Dict]:
        """Get current status of a team."""
        try:
            if team_name in self.teams:
                return {
                    'team': self.teams[team_name],
                    'tasks': self.team_tasks[team_name],
                    'artifacts': self.team_artifacts[team_name],
                    'logs': self.team_logs[team_name]
                }
            return None
        except Exception as e:
            st.error(f"Failed to get team status: {str(e)}")
            return None

    def run_team_meeting(self, team_name: str) -> Optional[Dict]:
        """Run a team meeting and generate meeting summary."""
        try:
            if team_name in self.teams:
                team = self.teams[team_name]
                # Log team structure for debugging
                st.markdown(f"#### [DEBUG] Team Structure for {team_name}")
                st.json(team)
                # Generate meeting prompt
                prompt = f"""
                Conduct a team meeting for {team_name}:
                Team Leader: {team['leader']['role_name']} ({team['leader']['agent_name']})
                Team Members: {', '.join(m['role'] for m in team['members'])}
                
                Current Tasks:
                {json.dumps(self.team_tasks[team_name], indent=2)}
                
                Generate a meeting summary with the following structure:
                {{
                    "key_discussion_points": ["string"],
                    "decisions_made": ["string"],
                    "action_items": ["string"],
                    "next_steps": ["string"],
                    "task_assignments": ["string"]
                }}
                
                Return ONLY the JSON object, with no additional text or explanation.
                """
                # Get meeting summary from GPT
                response = self.openai_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a team meeting facilitator. Always respond with valid JSON only."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7
                )
                # Clean and parse response
                cleaned_response = self._clean_json_response(response.choices[0].message.content)
                summary = json.loads(cleaned_response)
                # Validate summary structure
                if not self._validate_meeting_summary(summary):
                    st.error("Invalid meeting summary structure")
                    return None
                self._add_team_log(team_name, "Team meeting conducted")
                return summary
            else:
                st.error(f"Team '{team_name}' not found in self.teams. Available teams: {list(self.teams.keys())}")
                return None
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            st.error(f"Failed to run team meeting: {str(e)}\nTraceback:\n{tb}")
            # Log input data for debugging
            st.markdown(f"#### [DEBUG] Exception in run_team_meeting for {team_name}")
            st.json({
                'team_name': team_name,
                'teams': list(self.teams.keys()),
                'team_data': self.teams.get(team_name, {}),
                'team_tasks': self.team_tasks.get(team_name, [])
            })
            return None

    def run_c_level_meeting(self, c_level_roles: list) -> Optional[Dict]:
        """Run a C-Level meeting and generate meeting summary from a list of C-Level roles."""
        try:
            if not c_level_roles:
                st.error("No C-Level roles provided for C-Level meeting.")
                return None
            # Log C-Level roles for debugging
            st.markdown("#### [DEBUG] C-Level Roles for Meeting")
            st.json(c_level_roles)
            # Build executive and responsibility strings outside the f-string
            execs = ', '.join([f"{r['role_name']} ({r['agent_name']})" for r in c_level_roles])
            responsibilities = '\n'.join([f"- {r['role_name']}: {r['responsibilities']}" for r in c_level_roles])
            # Generate meeting prompt
            prompt = f"""
            Conduct a C-Level meeting for the following executives:
            {execs}

            Key Responsibilities:
            {responsibilities}

            Generate a meeting summary with the following structure:
            {{
                "key_discussion_points": ["string"],
                "decisions_made": ["string"],
                "action_items": ["string"],
                "next_steps": ["string"],
                "task_assignments": ["string"]
            }}

            Return ONLY the JSON object, with no additional text or explanation.
            """
            # Get meeting summary from GPT
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a C-Level meeting facilitator. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            # Clean and parse response
            cleaned_response = self._clean_json_response(response.choices[0].message.content)
            summary = json.loads(cleaned_response)
            # Validate summary structure
            if not self._validate_meeting_summary(summary):
                st.error("Invalid C-Level meeting summary structure")
                return None
            # Log meeting conducted
            st.success("C-Level meeting conducted successfully.")
            return summary
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            st.error(f"Failed to run C-Level meeting: {str(e)}\nTraceback:\n{tb}")
            st.markdown(f"#### [DEBUG] Exception in run_c_level_meeting")
            st.json({
                'c_level_roles': c_level_roles
            })
            return None

    def run_meeting_stage(self, 
                         meeting_config: Dict,
                         stage: str,
                         trust_matrix: Optional[np.ndarray] = None) -> List[Dict]:
        """
        Run a specific stage of the meeting process.
        
        Args:
            meeting_config (Dict): Meeting configuration
            stage (str): Stage name ('goal_decomposition', 'role_negotiation', 'consensus')
            trust_matrix (np.ndarray): Optional trust matrix for weighted decisions
            
        Returns:
            List[Dict]: List of agent interactions
        """
        stage_prompts = {
            'goal_decomposition': """
            Based on the main goal and constraints, suggest potential subtasks that need to be completed.
            Consider dependencies and logical order of tasks.
            """,
            'role_negotiation': """
            Review the proposed subtasks and suggest which agent(s) would be best suited for each task.
            Consider expertise, past performance, and current workload.
            """,
            'consensus': """
            Review the proposed task assignments and provide feedback or agreement.
            Consider team dynamics and task dependencies.
            """
        }
        
        interactions = []
        for agent in meeting_config['teams'][meeting_config['space_config']['name']]['agents']:
            # Prepare agent prompt with stage-specific context
            prompt = self._create_stage_prompt(
                agent,
                meeting_config,
                stage,
                stage_prompts[stage],
                trust_matrix
            )
            
            # Get agent's response
            response = self._get_agent_response(agent, prompt)
            
            # Record interaction
            interaction = {
                'from': agent['name'],
                'to': 'all',
                'content': response,
                'stage': stage,
                'timestamp': datetime.now().isoformat()
            }
            
            if trust_matrix is not None:
                interaction['trust_weight'] = self._calculate_trust_weight(
                    agent,
                    meeting_config['teams'][meeting_config['space_config']['name']]['agents'],
                    trust_matrix
                )
            
            interactions.append(interaction)
            self.meeting_log.append(interaction)
        
        return interactions

    def assign_tasks(self, 
                    meeting_config: Dict,
                    consensus_results: List[Dict]) -> Dict[str, Dict]:
        """
        Assign tasks to agents based on consensus results.
        
        Args:
            meeting_config (Dict): Meeting configuration
            consensus_results (List[Dict]): Results from consensus stage
            
        Returns:
            Dict[str, Dict]: Task assignments for each agent
        """
        assignments = {}
        
        for agent in meeting_config['teams'][meeting_config['space_config']['name']]['agents']:
            agent_tasks = []
            for result in consensus_results:
                if result.get('assigned_to') == agent['name']:
                    task = {
                        'task_id': f"task_{len(agent_tasks)}",
                        'description': result['task'],
                        'input_data': result.get('input_data', {}),
                        'output_format': result.get('output_format', 'text'),
                        'dependencies': result.get('dependencies', []),
                        'deadline': result.get('deadline')
                    }
                    agent_tasks.append(task)
            
            if agent_tasks:
                assignments[agent['name']] = {
                    'agent_id': agent['id'],
                    'tasks': agent_tasks
                }
                
                # Create task directories
                self._create_task_directories(agent['name'], agent_tasks)
        
        self.task_assignments = assignments
        return assignments

    def execute_tasks(self, 
                     assignments: Dict[str, Dict],
                     trust_matrix: Optional[np.ndarray] = None) -> Dict[str, List[Dict]]:
        """
        Execute assigned tasks and store results.
        
        Args:
            assignments (Dict[str, Dict]): Task assignments
            trust_matrix (np.ndarray): Optional trust matrix for collaboration
            
        Returns:
            Dict[str, List[Dict]]: Task execution results
        """
        results = {}
        
        for agent_name, agent_data in assignments.items():
            agent_results = []
            for task in agent_data['tasks']:
                # Execute task
                result = self._execute_single_task(
                    agent_name,
                    task,
                    trust_matrix
                )
                
                # Save result
                self._save_enhanced_task_result(agent_name, task, result)
                
                agent_results.append(result)
            
            results[agent_name] = agent_results
        
        return results

    def _create_stage_prompt(self,
                           agent: Dict,
                           meeting_config: Dict,
                           stage: str,
                           stage_prompt: str,
                           trust_matrix: Optional[np.ndarray] = None) -> str:
        """Create a prompt for a specific meeting stage."""
        prompt = f"""You are {agent['name']} in the {meeting_config['space_config']['name']}.

GOAL: {meeting_config['goal']}

CONSTRAINTS:
{chr(10).join(['- ' + c for c in meeting_config['constraints']])}

CURRENT STAGE: {stage}
{stage_prompt}

PREVIOUS INTERACTIONS:
{self._format_previous_interactions()}

Please provide your input for this stage.
"""
        return prompt

    def _get_agent_response(self, agent: Dict, prompt: str) -> str:
        """Get response from an agent using OpenAI API."""
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": agent['prompt']},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error getting agent response: {e}")
            return ""

    def _calculate_trust_weight(self,
                              agent: Dict,
                              all_agents: List[Dict],
                              trust_matrix: np.ndarray) -> float:
        """Calculate trust weight for an agent's input."""
        if trust_matrix is None:
            return 1.0
            
        agent_idx = next(i for i, a in enumerate(all_agents) if a['id'] == agent['id'])
        return np.mean(trust_matrix[:, agent_idx])

    def _create_task_directories(self, agent_name: str, tasks: List[Dict]) -> None:
        """Create directories for task execution and results."""
        base_dir = os.path.join('workspace', 'agents', agent_name, 'tasks')
        os.makedirs(base_dir, exist_ok=True)
        
        for task in tasks:
            task_dir = os.path.join(base_dir, task['task_id'])
            os.makedirs(task_dir, exist_ok=True)
            
            # Save task metadata
            with open(os.path.join(task_dir, 'task_meta.json'), 'w') as f:
                json.dump(task, f, indent=2)

    def _execute_single_task(self,
                           agent_name: str,
                           task: Dict,
                           trust_matrix: Optional[np.ndarray] = None) -> Dict:
        """Execute a single task and return the result."""
        # Create task-specific prompt
        prompt = f"""Execute the following task:

TASK: {task['description']}
INPUT DATA: {json.dumps(task['input_data'])}
OUTPUT FORMAT: {task['output_format']}

Please provide your solution.
"""
        
        # Get agent's response
        response = self._get_agent_response(
            {'name': agent_name, 'prompt': task.get('agent_prompt', '')},
            prompt
        )
        
        return {
            'task_id': task['task_id'],
            'status': 'completed',
            'result': response,
            'timestamp': datetime.now().isoformat()
        }

    def _save_enhanced_task_result(self, agent_name: str, task: Dict, result: Dict) -> None:
        """Enhanced task result saving with multiple output formats."""
        task_dir = os.path.join('workspace', 'agents', agent_name, 'tasks', task['task_id'])
        
        # Save main result as JSON
        with open(os.path.join(task_dir, 'result.json'), 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        # Save markdown report if available
        if 'report' in result:
            with open(os.path.join(task_dir, 'analysis_report.md'), 'w', encoding='utf-8') as f:
                f.write(result['report'])
        
        # Save code files if available
        if 'code' in result:
            with open(os.path.join(task_dir, 'analysis_code.py'), 'w', encoding='utf-8') as f:
                f.write(result['code'])
        
        # Save visualizations if available
        if 'visualizations' in result:
            viz_dir = os.path.join(task_dir, 'visualizations')
            os.makedirs(viz_dir, exist_ok=True)
            for viz_name, viz_data in result['visualizations'].items():
                if isinstance(viz_data, str) and viz_data.startswith('data:image'):
                    # Handle base64 encoded images
                    import base64
                    image_data = viz_data.split(',')[1]
                    image_bytes = base64.b64decode(image_data)
                    with open(os.path.join(viz_dir, f'{viz_name}.png'), 'wb') as f:
                        f.write(image_bytes)
                elif isinstance(viz_data, str):
                    # Handle SVG or other text-based visualizations
                    ext = '.svg' if '<svg' in viz_data else '.txt'
                    with open(os.path.join(viz_dir, f'{viz_name}{ext}'), 'w', encoding='utf-8') as f:
                        f.write(viz_data)
        
        # Save data files if available
        if 'data' in result:
            data_dir = os.path.join(task_dir, 'data')
            os.makedirs(data_dir, exist_ok=True)
            for data_name, data_content in result['data'].items():
                if isinstance(data_content, dict):
                    with open(os.path.join(data_dir, f'{data_name}.json'), 'w', encoding='utf-8') as f:
                        json.dump(data_content, f, indent=2, ensure_ascii=False)
                elif isinstance(data_content, str):
                    with open(os.path.join(data_dir, f'{data_name}.txt'), 'w', encoding='utf-8') as f:
                        f.write(data_content)
        
        # Save summary document
        summary = self._generate_task_summary(task, result)
        with open(os.path.join(task_dir, 'task_summary.md'), 'w', encoding='utf-8') as f:
            f.write(summary)
        
        # Update task metadata
        task['status'] = 'completed'
        task['completion_time'] = result['timestamp']
        task['artifacts_generated'] = list(result.keys())
        with open(os.path.join(task_dir, 'task_meta.json'), 'w', encoding='utf-8') as f:
            json.dump(task, f, indent=2, ensure_ascii=False)
    
    def _generate_task_summary(self, task: Dict, result: Dict) -> str:
        """Generate a summary document for the completed task."""
        summary = f"""# Task Summary: {task['task_id']}

## Task Information
- **Task ID**: {task['task_id']}
- **Description**: {task['description']}
- **Assigned To**: {task.get('assigned_to', 'Unknown')}
- **Status**: {result.get('status', 'Unknown')}
- **Completion Time**: {result.get('timestamp', 'Unknown')}

## Input Data
```json
{json.dumps(task.get('input_data', {}), indent=2)}
```

## Output Summary
- **Result Type**: {type(result.get('result', {})).__name__}
- **Generated Artifacts**: {', '.join(result.keys())}

## Key Findings
"""
        
        # Extract key findings from result
        if 'result' in result and isinstance(result['result'], dict):
            if 'key_findings' in result['result']:
                for finding in result['result']['key_findings']:
                    summary += f"- {finding}\n"
            elif 'summary' in result['result']:
                summary += f"{result['result']['summary']}\n"
        
        summary += f"""
## Files Generated
"""
        
        # List generated files
        if 'report' in result:
            summary += "- `analysis_report.md` - Detailed analysis report\n"
        if 'code' in result:
            summary += "- `analysis_code.py` - Analysis code and scripts\n"
        if 'visualizations' in result:
            summary += "- `visualizations/` - Generated charts and graphs\n"
        if 'data' in result:
            summary += "- `data/` - Processed data files\n"
        
        summary += f"""
## Metadata
- **Task Duration**: {self._calculate_task_duration(task, result)}
- **Output Quality**: {self._assess_output_quality(result)}
- **Trust Score**: {result.get('agent_metadata', {}).get('trust_score', 'N/A')}

---
*Generated by SituTrust Engine on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        return summary
    
    def _calculate_task_duration(self, task: Dict, result: Dict) -> str:
        """Calculate the duration of task execution."""
        try:
            start_time = task.get('start_time', result.get('timestamp'))
            end_time = result.get('timestamp')
            
            if start_time and end_time:
                start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
                duration = end_dt - start_dt
                return str(duration)
            else:
                return "Unknown"
        except:
            return "Unknown"
    
    def _assess_output_quality(self, result: Dict) -> str:
        """Assess the quality of the task output."""
        quality_score = 0
        
        # Check for comprehensive result
        if 'result' in result and isinstance(result['result'], dict):
            quality_score += 1
        
        # Check for additional artifacts
        if 'report' in result:
            quality_score += 1
        if 'code' in result:
            quality_score += 1
        if 'visualizations' in result:
            quality_score += 1
        if 'data' in result:
            quality_score += 1
        
        # Quality assessment based on score
        if quality_score >= 4:
            return "Excellent"
        elif quality_score >= 3:
            return "Good"
        elif quality_score >= 2:
            return "Fair"
        else:
            return "Basic"

    def _format_previous_interactions(self) -> str:
        """Format previous meeting interactions for context."""
        if not self.meeting_log:
            return "No previous interactions."
            
        formatted = []
        for interaction in self.meeting_log[-5:]:  # Show last 5 interactions
            formatted.append(
                f"{interaction['from']} ({interaction['stage']}): {interaction['content']}"
            )
        
        return "\n".join(formatted) 