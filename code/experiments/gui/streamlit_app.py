import streamlit as st
import json
import os
from datetime import datetime
import pandas as pd
import plotly.express as px
import networkx as nx
import sys
from pathlib import Path

# Add the project root to Python path
project_root = str(Path(__file__).parent.parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from code.experiments.utils.collaboration_flow import CollaborationFlow
from code.experiments.utils.spatial_manager import SpatialPrompting
from code.experiments.utils.trust_functions import TrustFunctions
from code.experiments.utils.role_generator import RoleGenerator
from code.experiments.utils.command_handler import CommandHandler
from code.experiments.config import OPENAI_API_KEY, AGENT_MODEL

from typing import Dict, List, Optional
import sqlite3
import time
import openai
import re

class CollaborationGUI:
    def __init__(self):
        # Create necessary directories
        self._create_required_directories()
        
        # Initialize components
        self.trust_functions = TrustFunctions()
        self.spatial_manager = SpatialPrompting()
        self.role_generator = RoleGenerator()
        self.collaboration_flow = CollaborationFlow()
        self.command_handler = CommandHandler()
        
        # Initialize database connections
        self._init_all_databases()
        
        # Initialize team structures
        self.teams = {}
        self.team_tasks = {}
        self.team_artifacts = {}
        self.team_logs = {}
        self.team_restructure_history = []
        
        # Initialize meeting state
        self.current_meeting = None

        self._init_session_state()

    def _create_required_directories(self):
        """Create all required directories for the application."""
        directories = [
            'workspace',
            'workspace/meetings',
            'workspace/artifacts',
            'workspace/logs',
            'workspace/databases'
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            st.sidebar.success(f"Created directory: {directory}")

    def _init_all_databases(self):
        """Initialize connections to all team databases."""
        try:
            # Create databases directory if it doesn't exist
            os.makedirs('workspace/databases', exist_ok=True)
            
            # Initialize default database
            self.default_db = sqlite3.connect('workspace/databases/default.db')
            self._create_team_tables(self.default_db)
            
            # Initialize team databases
            self.team_dbs = {}
            for team in ['cto_ai', 'cpo_ai', 'cmo_ai']:
                db_path = f'workspace/databases/{team}.db'
                self.team_dbs[team] = sqlite3.connect(db_path)
                self._create_team_tables(self.team_dbs[team])
            
            st.sidebar.success("Successfully initialized all databases")
            
        except Exception as e:
            st.error(f"Failed to initialize databases: {str(e)}")
            raise

    def _create_team_database(self, team_name: str) -> bool:
        """Create a new database for a team."""
        try:
            # Create data/teams directory if it doesn't exist
            os.makedirs("data/teams", exist_ok=True)
            
            # Create database file
            db_path = f"data/teams/{team_name}_ai.db"
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Create necessary tables
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS team_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    team_name TEXT NOT NULL,
                    data JSON,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    description TEXT,
                    status TEXT DEFAULT 'pending',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS artifacts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    type TEXT,
                    content TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            st.error(f"Failed to create database for team {team_name}: {str(e)}")
            return False

    def initialize_meeting(self, goal: str, constraints: List[str]):
        """Initialize a new meeting with dynamic team creation."""
        # Generate C-level roles and teams based on goals and constraints
        roles = self.role_generator.generate_c_level_roles(goal, constraints)
        
        # Create meeting configuration
        meeting_config = self.collaboration_flow.launch_meeting(goal, constraints, roles, {})
        self.current_meeting = meeting_config
        
        # Initialize teams dynamically
        for role in roles:
            team_name = role.get('team', 'default')
            if team_name not in self.teams:
                self.teams[team_name] = {
                    'name': team_name,
                    'agents': [],
                    'tasks': [],
                    'created_at': datetime.now().isoformat()
                }
                # Create team database
                self._create_team_database(team_name)
            
            self.teams[team_name]['agents'].append(role)
            
        return meeting_config

    def _create_team_tables(self, db):
        """Create necessary tables for a team's database."""
        try:
            cursor = db.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS team_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    team_name TEXT NOT NULL,
                    data TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                )
            ''')
            db.commit()
        except Exception as e:
            st.error(f"Failed to create tables for {db}: {str(e)}")

    def _get_team_db(self, team_name):
        """Get the appropriate database connection for a team."""
        db_mapping = {
            'cto_team': 'cto_ai',
            'cpo_team': 'cpo_ai',
            'cmo_team': 'cmo_ai'
        }
        return self.team_dbs.get(db_mapping.get(team_name, 'default'))

    def _save_team_data(self, team_name, data):
        """Save team-specific data to the appropriate database."""
        db = self._get_team_db(team_name)
        if db:
            try:
                cursor = db.cursor()
                cursor.execute('''
                    INSERT INTO team_data (team_name, data, timestamp)
                    VALUES (?, ?, ?)
                ''', (team_name, json.dumps(data), datetime.now().isoformat()))
                db.commit()
            except Exception as e:
                st.error(f"Failed to save team data: {str(e)}")

    def _load_team_data(self, team_name):
        """Load team-specific data from the appropriate database."""
        db = self._get_team_db(team_name)
        if db:
            try:
                cursor = db.cursor()
                cursor.execute('''
                    SELECT data FROM team_data 
                    WHERE team_name = ? 
                    ORDER BY timestamp DESC LIMIT 1
                ''', (team_name,))
                result = cursor.fetchone()
                return json.loads(result[0]) if result else None
            except Exception as e:
                st.error(f"Failed to load team data: {str(e)}")
        return None

    def _display_team_section(self, team_name):
        """Display team-specific section in the sidebar."""
        if team_name not in self.teams:
            return
            
        st.sidebar.subheader(f"{self.teams[team_name]['name']} Dashboard")
        
        # Load team data
        team_data = self._load_team_data(team_name)
        if team_data:
            st.sidebar.write("Active Tasks:", len(team_data.get('active_tasks', [])))
            st.sidebar.write("Completed Tasks:", len(team_data.get('completed_tasks', [])))
            
            # Display team members
            st.sidebar.write("Team Members:")
            for agent in team_data.get('agents', []):
                st.sidebar.write(f"- {agent['name']} ({agent['role']})")
            
            # Display trust scores
            trust_matrix = self.trust_functions.get_team_trust_matrix(team_name)
            if trust_matrix is not None:
                st.sidebar.write("Team Trust Score:", 
                               self.trust_functions.calculate_team_trust_score(team_name))

    def _create_new_task(self, team_name):
        """Create a new task for a specific team."""
        st.subheader(f"Create New Task for {self.teams[team_name]['name']}")
        
        task_name = st.text_input("Task Name", key=f"task_name_{team_name}")
        task_description = st.text_area("Task Description", key=f"task_desc_{team_name}")
        task_priority = st.selectbox("Priority", ["High", "Medium", "Low"], 
                                   key=f"task_priority_{team_name}")
        
        if st.button("Create Task", key=f"create_task_{team_name}"):
            if task_name and task_description:
                task_data = {
                    'name': task_name,
                    'description': task_description,
                    'priority': task_priority
                }
                self.collaboration_flow.add_task(team_name, task_data)
                st.success(f"Task created for {self.teams[team_name]['name']}")
            else:
                st.error("Please fill in all task fields")

    def _restructure_teams(self, new_goal: str = None, new_constraints: List[str] = None):
        """Dynamically restructure teams based on updated goals and constraints."""
        if not self.current_meeting:
            return False

        # Update meeting configuration
        if new_goal:
            self.current_meeting['goal'] = new_goal
        if new_constraints:
            self.current_meeting['constraints'] = new_constraints

        # Generate new roles based on updated goals and constraints
        new_roles = self.role_generator.generate_roles(
            self.current_meeting['goal'],
            self.current_meeting['constraints']
        )

        # Track team changes
        old_teams = self.teams.copy()
        new_teams = {}

        # Create new team structure
        for role in new_roles:
            team_name = role.get('team', 'default')
            if team_name not in new_teams:
                new_teams[team_name] = {
                    'name': team_name,
                    'agents': [],
                    'tasks': [],
                    'created_at': datetime.now().isoformat()
                }
                # Create team database if it doesn't exist
                if team_name not in self.team_dbs:
                    self._create_team_database(team_name)
            
            new_teams[team_name]['agents'].append(role)

        # Log team restructuring
        self.team_restructure_history.append({
            'timestamp': datetime.now().isoformat(),
            'old_teams': old_teams,
            'new_teams': new_teams,
            'reason': 'Goals/Constraints updated'
        })

        # Update teams
        self.teams = new_teams
        return True

    def _add_log(self, message: str):
        """Add a log message to the current meeting."""
        if self.current_meeting:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = f"[{timestamp}] {message}"
            self.current_meeting['logs'].append(log_entry)

    def _init_session_state(self):
        """Initialize session state variables."""
        if 'meeting_initialized' not in st.session_state:
            st.session_state.meeting_initialized = False
        if 'current_meeting' not in st.session_state:
            st.session_state.current_meeting = None
        if 'role_generation_history' not in st.session_state:
            st.session_state.role_generation_history = []
        if 'team_creation_status' not in st.session_state:
            st.session_state.team_creation_status = {}
        if 'error_messages' not in st.session_state:
            st.session_state.error_messages = []

    def _add_error(self, message: str):
        """Add an error message to the session state."""
        st.session_state.error_messages.append({
            'timestamp': datetime.now().isoformat(),
            'message': message
        })

    def _display_errors(self):
        """Display all error messages."""
        if st.session_state.error_messages:
            with st.expander("Error Log", expanded=True):
                for error in st.session_state.error_messages:
                    st.error(f"[{error['timestamp']}] {error['message']}")

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
            self._add_error(f"Failed to clean JSON response: {str(e)}")
            return response

    def _format_markdown(self, data: Dict) -> str:
        """Format data as markdown."""
        if isinstance(data, dict):
            return "\n".join([f"### {k}\n{self._format_markdown(v)}" for k, v in data.items()])
        elif isinstance(data, list):
            return "\n".join([f"- {self._format_markdown(item)}" for item in data])
        else:
            return str(data)

    def _get_c_level_roles(self, roles):
        """Filter and return only C-Level roles based on role_type or role_name."""
        c_level_roles = [r for r in roles if (r.get('role_type') == 'C-Level' or (r.get('role_name', '').startswith('C') and 'level' in r.get('role_name', '').lower()))]
        # Fallback: if role_type is missing, use role_name heuristic
        if not c_level_roles:
            c_level_roles = [r for r in roles if r.get('role_name', '').startswith('C')]
        return c_level_roles

    def run(self):
        """Run the Streamlit application."""
        st.title("🤖 SituTrust Engine: Multi AI Agent System in trust and space")
        
        # Display any error messages
        self._display_errors()
        
        # Meeting Initialization Section
        with st.expander("Initialize Meeting", expanded=not st.session_state.meeting_initialized):
            goal = st.text_area("Meeting Goal", height=100)
            constraints = st.text_area("Meeting Constraints", height=100)
            
            if st.button("Generate C-Level executive roles"):
                st.write("Generate Roles button clicked")
                if not goal or not constraints:
                    st.error("Please provide both goal and constraints")
                    self._add_error("Please provide both goal and constraints")
                    return
                with st.spinner("🔧 Generating C-Level executive roles for this project…"):
                    try:
                        roles = self.role_generator.generate_c_level_roles(goal, constraints)
                        st.write(" generate C-Level roles called", roles)
                        if roles:
                            st.session_state.current_meeting = {
                                'goal': goal,
                                'constraints': constraints,
                                'roles': roles,
                                'start_time': datetime.now().isoformat(),
                                'logs': []
                            }
                            st.session_state.meeting_initialized = True
                            if self.collaboration_flow.initialize_teams(roles):
                                st.success("Roles generated successfully!")
                                st.rerun()
                            else:
                                st.error("Failed to initialize teams")
                                self._add_error("Failed to initialize teams")
                        else:
                            st.error("Failed to generate roles")
                            self._add_error("Failed to generate roles")
                    except Exception as e:
                        import traceback
                        tb = traceback.format_exc()
                        st.error(f"Error during role generation: {str(e)}\nTraceback:\n{tb}")
                        self._add_error(f"Error during role generation: {str(e)}\nTraceback:\n{tb}")
        
        # Display Current Meeting Information
        if st.session_state.meeting_initialized:
            st.header("Current Meeting")
            st.markdown(f"""
            ### Meeting Goal
            {st.session_state.current_meeting['goal']}
            
            ### Meeting Constraints
            {st.session_state.current_meeting['constraints']}
            
            ### Meeting Details
            - Number of Roles: {len(st.session_state.current_meeting['roles'])}
            - Start Time: {st.session_state.current_meeting['start_time']}
            """)
            # Display roles structure for debugging
            st.markdown("#### 🚀 C-Level Executives Have Joined the Mission")
            st.json(st.session_state.current_meeting['roles'])
            
            # C-Level Meeting Section
            st.subheader("C-Level Meeting")
            if st.button("Start C-Level Meeting"):
                with st.spinner("🛬 C-Level executives are gathering from across the world, entering the conference space to kick off the project. ."):
                    try:
                        roles = st.session_state.current_meeting['roles']
                        c_level_roles = self._get_c_level_roles(roles)
                        if not c_level_roles:
                            self._add_error("No C-Level roles found. Please check role generation.")
                            return
                        # 1. Generate and display space prompt
                        goal = st.session_state.current_meeting['goal']
                        participants = [r['agent_name'] for r in c_level_roles]
                        tasks = [r['responsibilities'] for r in c_level_roles]
                        mood = "High trust, strategic collaboration, rapid iteration"
                        space_prompt = self.spatial_manager.generate_space_prompt(goal, participants, tasks, mood)
                        st.markdown("#### 🏛️ C-Level Meeting Space Prompt")
                        st.markdown(space_prompt)
                        # 2. Generate and display trust matrix
                        collaboration_history = []
                        for i, a in enumerate(c_level_roles):
                            for j, b in enumerate(c_level_roles):
                                if i != j:
                                    # Calculate trust score based on role attributes
                                    trust_score = 0.0
                                    
                                    # Communication compatibility (30%)
                                    a_comm = set(a['communication_channels'].lower().split(','))
                                    b_comm = set(b['communication_channels'].lower().split(','))
                                    comm_compat = len(a_comm.intersection(b_comm)) / len(a_comm.union(b_comm))
                                    trust_score += comm_compat * 0.3
                                    
                                    # Collaboration patterns (30%)
                                    a_collab = set(a['collaboration_patterns'].lower().split(','))
                                    b_collab = set(b['collaboration_patterns'].lower().split(','))
                                    collab_compat = len(a_collab.intersection(b_collab)) / len(a_collab.union(b_collab))
                                    trust_score += collab_compat * 0.3
                                    
                                    # Cross-functional impact (20%)
                                    a_impact = set(a['cross_functional_impact'].lower().split(','))
                                    b_impact = set(b['cross_functional_impact'].lower().split(','))
                                    impact_align = len(a_impact.intersection(b_impact)) / len(a_impact.union(b_impact))
                                    trust_score += impact_align * 0.2
                                    
                                    # Trust requirements (20%)
                                    a_trust = set(a['trust_requirements'].lower().split(','))
                                    b_trust = set(b['trust_requirements'].lower().split(','))
                                    trust_align = len(a_trust.intersection(b_trust)) / len(a_trust.union(b_trust))
                                    trust_score += trust_align * 0.2
                                    
                                    collaboration_history.append({
                                        'agent_a': a['agent_name'],
                                        'agent_b': b['agent_name'],
                                        'trust_score': round(trust_score, 2),
                                        'comm_compat': round(comm_compat, 2),
                                        'collab_compat': round(collab_compat, 2),
                                        'impact_align': round(impact_align, 2),
                                        'trust_align': round(trust_align, 2)
                                    })
                        
                        trust_data = self.trust_functions.generate_prompt_based_trust_matrix(c_level_roles, collaboration_history)
                        st.markdown("#### 🤝 C-Level Trust Matrix & Strategy")
                        
                        # Trust Matrix Formula Explanation
                        st.markdown("""
                        ##### 📊 Trust Score Calculation Formula
                        ```python
                        T_ij = σ(w₁·c_ij + w₂·p_ij + w₃·i_ij + w₄·t_ij)
                        
                        where:
                        - c_ij = Communication compatibility (30%)
                        - p_ij = Collaboration patterns (30%)
                        - i_ij = Cross-functional impact (20%)
                        - t_ij = Trust requirements (20%)
                        - σ = Sigmoid function for normalization
                        ```
                        """)
                        
                        # Trust Score Mapping
                        st.markdown("""
                        ##### 🎯 Trust Score to Behavior Mapping
                        | Trust Score Range | Behavior | UI Color |
                        |------------------|----------|----------|
                        | ≥ 0.85 | Accept & build on the idea | 🟢 Green |
                        | 0.65 - 0.84 | Ask for justification | 🟡 Yellow |
                        | < 0.65 | Rebut or defer | 🔴 Red |
                        """)
                        
                        # Display Trust Matrix with Color Coding
                        st.markdown("##### 🔄 Trust Matrix Visualization")
                        trust_matrix = trust_data.get('trust_matrix', {})
                        strategy = trust_data.get('strategy', {})
                        
                        # Create a DataFrame for better visualization
                        trust_df = pd.DataFrame(trust_matrix).fillna(0)
                        strategy_df = pd.DataFrame(strategy).fillna("")
                        
                        # Display trust matrix with color coding
                        st.dataframe(
                            trust_df.style.background_gradient(cmap='RdYlGn', vmin=0, vmax=1),
                            use_container_width=True
                        )
                        
                        # Display strategy matrix
                        st.markdown("##### 🎭 Behavioral Strategy")
                        st.dataframe(strategy_df, use_container_width=True)
                        
                        # Original JSON view (collapsed)
                        with st.expander("View Raw Trust Data"):
                            st.json(trust_data)
                        
                        # 3. 3-Phase C-Level Meeting Simulation
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
                        trust_matrix_str = json.dumps(trust_data.get('trust_matrix', {}), ensure_ascii=False)
                        strategy_str = json.dumps(trust_data.get('strategy', {}), ensure_ascii=False)
                        
                        for idx, (title, prompt) in enumerate(zip(phase_titles, phase_prompts)):
                            st.markdown(f"### 🧠 {title}")
                            # Compose phase context
                            phase_context = f"""
Meeting Objective: {goal}
Participants: {', '.join(participants)}
Environment: {space_prompt}
Trust Matrix: {trust_matrix_str}
Behavioral Strategy: {strategy_str}

{prompt}

Trust-based Response Guidelines:
Each C-Level participant must:
- Participate in *3 realistic, role-consistent dialogue turns*
- Explicitly demonstrate **at least one** of the following trust-based behaviors:
- ✅ Accept: Acknowledge and build upon another's point (if trust_score ≥ 0.85)
- ❓ Justify: Request clarification or evidence before accepting (0.65 ≤ trust_score < 0.85)
- ❌ Rebut: Disagree with reasoning or defer (if trust_score < 0.65)
- Use terminology and perspective aligned with their defined *responsibilities* and *domain*
- Refer to *trust dynamics* naturally in their speech (e.g., "Given our alignment on...", "I'm cautious because...")
role_emojis_map = {{
    "CEO": "🙋🏻‍♂️",
    "cto_gpt": "💻",
    "cfo_gpt": "😌",
    "COO": "🧐",
    "CMO": "😼",
    "CHRO": "🧑🏼‍🦲",
    "CIO": "🧔🏻‍♂️",
    "CSO": "👨🏾‍🦳",
    "CLO": "👩🏼‍🔧",
    "CDO": "👨🏻‍💻"
}}
📝 **Expected Format**:
- Use realistic conversation format:
```markdown
**emoji cto_gpt**: Here's my view based on our infrastructure plans...
**emoji cpo_gpt**: I agree, especially because our trust level supports this coordination.
```
"""
                            
                            # Call GPT for phase dialogue
                            response = self.trust_functions.openai_client.chat.completions.create(
                                model=AGENT_MODEL,
                                messages=[
                                    {"role": "system", "content": """
You are a C-Level AI Meeting Simulator trained in trust-aware collaboration. 
You simulate high-level strategic meetings between executives using markdown formatting. 
You must always:
- Reflect each participant's domain knowledge, priorities, and role-specific language
- Show realistic, human-like multi-turn dialogue
- Demonstrate decisions that align with trust-based behavioral rules
- End each meeting with a **clear markdown summary** that highlights alignment, conflicts, and next steps
"""},
                                    {"role": "user", "content": phase_context}
                                ],
                                temperature=0.7
                            )
                            phase_dialogue = response.choices[0].message.content.strip()
                            st.markdown(phase_dialogue)
                            phase_logs.append({
                                'phase': title,
                                'dialogue': phase_dialogue
                            })
                        # Save phase logs to meeting log
                        st.session_state.current_meeting['logs'].append({
                            'timestamp': datetime.now().isoformat(),
                            'type': 'c_level_meeting_phases',
                            'phases': phase_logs
                        })
                        # phase_logs를 session_state에 저장
                        st.session_state.phase_logs = phase_logs
                        st.success("✅ Conference among C-Level Executives  successfully completed!")
                        
                        # 바로 에이전트 생성 시작
                        print("[DEBUG] Starting agent generation directly after C-Level meeting")
                        st.info("Generating expert agents for each team...")

                        try:
                            st.write("[DEBUG] Starting to retrieve roles")
                            roles = st.session_state.current_meeting['roles']
                            c_level_roles = self._get_c_level_roles(roles)
                            st.write("[DEBUG] Completed retrieving roles")

                            st.write("[DEBUG] Starting to retrieve meeting decisions")
                            meeting_decisions_raw = st.session_state.get('phase_logs', [])
                        

                            summarized_meetings = self.role_generator.summarize_meeting_logs_with_gpt(
                                meeting_decisions_raw,
                                model=AGENT_MODEL
)
                            st.write("[DEBUG] Completed retrieving meeting decisions")

                            project_context = {
                                'goal': st.session_state.current_meeting['goal'],
                                'constraints': st.session_state.current_meeting['constraints']
                            }
                            agent_results = {}
                            agent_artifacts = {}

                            for c_role in c_level_roles:
                                print(f"c_role: {c_role}")
                                st.write(f"[DEBUG] Starting to generate expert agents for team {c_role['team_name']}")
                                expert_agents = self.role_generator.generate_team_expert_agents(
                                    c_role, summarized_meetings, project_context
                                )
                                st.write(f"[DEBUG] Completed generating expert agents for team {c_role['team_name']}")
                                for agent in expert_agents or []:
                                    # Safely extract all fields as strings
                                    agent_name = str(agent.get('agent_name', ''))
                                    role_name = str(agent.get('role_name', ''))
                                    expertise_level = str(agent.get('expertise_level', ''))
                                    team_name = str(agent.get('team_name', ''))
                                    role = str(agent.get('role', ''))
                                    responsibilities = str(agent.get('responsibilities', ''))
                                    required_skills = ', '.join(agent.get('required_skills', []))
                                    key_deliverables = ', '.join(agent.get('key_deliverables', []))
                                    collaboration_patterns = str(agent.get('collaboration_patterns', ''))
                                    trust_requirements = str(agent.get('trust_requirements', ''))
                                    decision_authority = str(agent.get('decision_authority', ''))
                                    communication_channels = str(agent.get('communication_channels', ''))
                                    success_criteria = str(agent.get('success_criteria', ''))
                                    specialization_areas = ', '.join(agent.get('specialization_areas', []))
                                    reasoning = str(agent.get('reasoning', ''))

                                    full_prompt = (
                                        f"You are **{agent_name}**, working as **{role_name}** (Expertise Level: {expertise_level}) on the **{team_name}**.\n"
                                        f"Your detailed role:\n{role}\n"
                                        f"### Responsibilities\n{responsibilities}\n"
                                        f"### Required Skills\n{required_skills}\n"
                                        f"### Key Deliverables\n{key_deliverables}\n"
                                        f"### Collaboration Pattern\n{collaboration_patterns}\n"
                                        f"### Trust Requirements\n{trust_requirements}\n"
                                        f"### Decision Authority\n{decision_authority}\n"
                                        f"### Communication Channels\n{communication_channels}\n"
                                        f"### Success Criteria\n{success_criteria}\n"
                                        f"### Specialization Areas\n{specialization_areas}\n"
                                        f"### Reasoning\n{reasoning}\n"
                                        "---🔧 **Task**: Based on the above, generate a realistic, high-quality sample output that demonstrates how you would execute your main deliverable(s) in this project context.\n\n"
                                        "Your response should reflect your expertise level, adhere to your communication and collaboration protocols, and model trust-aware coordination."
                                    )
                                    try:
                                        st.write(f"[DEBUG] Starting task for agent {agent_name}")
                                        response = self.trust_functions.openai_client.chat.completions.create(
                                            model=AGENT_MODEL,
                                            messages=[
                                                {
                                                    "role": "system",
                                                    "content": "You are an expert AI agent. Always produce outputs that reflect your defined role, expertise, and collaboration model."
                                                },
                                                {
                                                    "role": "user",
                                                    "content": full_prompt
                                                }
                                            ],
                                            temperature=0.7
                                        )
                                        output = response.choices[0].message.content.strip()
                                        st.write(f"[DEBUG] Completed task for agent {agent_name}")
                                    except Exception as e:
                                        output = f"[ERROR] {str(e)}"
                                        st.error(f"Error occurred while generating agent {agent_name}: {str(e)}")
                                    agent_dir = f"workspace/artifacts/{agent_name}"
                                    os.makedirs(agent_dir, exist_ok=True)
                                    output_path = os.path.join(agent_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_output.txt")
                                    with open(output_path, 'w', encoding='utf-8') as f:
                                        f.write(output)
                                    agent['artifact_path'] = output_path
                                    agent_artifacts[agent_name] = output_path
                                agent_results[c_role['team_name']] = expert_agents
                                st.write(f"[DEBUG] Completed generating expert agents for team {c_role['team_name']}")

                            st.session_state.agent_results = agent_results
                            st.session_state.agent_artifacts = agent_artifacts
                            st.success("Successfully generated and saved results for all expert agents!")

                        except Exception as e:
                            print(f"Error occurred during agent generation: {str(e)}")
                            st.error(f"Error occurred during agent generation: {str(e)}")
                            self._add_error(f"Agent generation error: {str(e)}")
                    except Exception as e:
                        print(f"Error occurred during C-Level meeting: {str(e)}")
                        import traceback
                        tb = traceback.format_exc()
                        self._add_error(f"Error during C-Level meeting: {str(e)}\nTraceback:\n{tb}")
                        print(f"Error occurred during C-Level meeting: {str(e)}\nTraceback:\n{tb}")
                    else:
                        print("agent_generation_in_progress is False, skipping agent generation")

            
            # Team Management Section
            st.subheader("Team Management")
            for team_name, team in self.collaboration_flow.teams.items():
                with st.expander(f"{team_name} Team"):
                    st.markdown(f"""
                    ### Team Leader
                    - Role: {team['leader']['role_name']}
                    - Agent: {team['leader']['agent_name']}
                    
                    ### Team Status
                    - Current Status: {team['status']}
                    """)
                    
                    if team['status'] == 'pending':
                        if st.button(f"Create {team_name} Team"):
                            with st.spinner(f"Creating {team_name} team..."):
                                try:
                                    # Generate team members based on requirements
                                    prompt = f"""
                                    Generate team members for {team_name} based on requirements:
                                    {json.dumps(team['requirements'], indent=2)}
                                    
                                    Return a JSON array of team members with:
                                    - role: Role name
                                    - agent_name: AI agent name
                                    - expertise: List of expertise areas
                                    - responsibilities: List of responsibilities
                                    
                                    Return ONLY the JSON array, with no additional text or explanation.
                                    """
                                    
                                    response = self.openai_client.chat.completions.create(
                                        model=AGENT_MODEL,
                                        messages=[
                                            {"role": "system", "content": "You are a team builder. Always respond with valid JSON only."},
                                            {"role": "user", "content": prompt}
                                        ],
                                        temperature=0.7
                                    )
                                    
                                    team_members = json.loads(response.choices[0].message.content)
                                    
                                    if self.collaboration_flow.create_team(team_name, team_members):
                                        st.session_state.team_creation_status[team_name] = 'created'
                                        st.success(f"{team_name} team created successfully!")
                                        st.rerun()
                                    else:
                                        self._add_error(f"Failed to create {team_name} team")
                                except Exception as e:
                                    self._add_error(f"Error creating {team_name} team: {str(e)}")
                    
                    elif team['status'] == 'active':
                        st.markdown("### Team Members")
                        for member in team['members']:
                            st.markdown(f"""
                            #### {member['role']} ({member['agent_name']})
                            - Expertise: {', '.join(member['expertise'])}
                            - Responsibilities: {', '.join(member['responsibilities'])}
                            """)
                        
                        # Team Meeting
                        if st.button(f"Start {team_name} Meeting"):
                            with st.spinner(f"Conducting {team_name} meeting..."):
                                try:
                                    meeting_summary = self.collaboration_flow.run_team_meeting(team_name)
                                    if meeting_summary:
                                        st.session_state.current_meeting['logs'].append({
                                            'timestamp': datetime.now().isoformat(),
                                            'type': 'team_meeting',
                                            'team': team_name,
                                            'summary': meeting_summary
                                        })
                                        st.success(f"{team_name} meeting completed!")
                                        st.markdown(f"### {team_name} Meeting Summary")
                                        st.markdown(self._format_markdown(meeting_summary))
                                    else:
                                        self._add_error(f"Failed to complete {team_name} meeting")
                                except Exception as e:
                                    self._add_error(f"Error during {team_name} meeting: {str(e)}")
                        
                        # Task Management
                        st.subheader("Task Management")
                        task_name = st.text_input("Task Name", key=f"task_name_{team_name}")
                        task_desc = st.text_area("Task Description", key=f"task_desc_{team_name}")
                        task_priority = st.selectbox("Priority", 
                                                   ["low", "medium", "high"],
                                                   key=f"task_priority_{team_name}")
                        
                        if st.button("Add Task", key=f"add_task_{team_name}"):
                            if not task_name or not task_desc:
                                self._add_error("Please provide both task name and description")
                                return
                            
                            try:
                                if self.collaboration_flow.add_task(team_name, task_name, 
                                                                 task_desc, priority=task_priority):
                                    st.success("Task added successfully!")
                                    st.rerun()
                                else:
                                    self._add_error(f"Failed to add task to {team_name}")
                            except Exception as e:
                                self._add_error(f"Error adding task: {str(e)}")
                        
                        # Display Tasks
                        st.markdown("### Current Tasks")
                        for task in self.collaboration_flow.team_tasks[team_name]:
                            with st.expander(f"{task['name']} ({task['status']})"):
                                st.markdown(f"""
                                #### Task Details
                                - Description: {task['description']}
                                - Priority: {task['priority']}
                                - Created: {task['created_at']}
                                """)
                                if task['status'] == 'completed':
                                    st.markdown(f"- Completed: {task['completed_at']}")
                                if task['artifacts']:
                                    st.markdown("#### Artifacts")
                                    for artifact in task['artifacts']:
                                        st.markdown(f"""
                                        - **{artifact['name']}**
                                          - Description: {artifact['description']}
                                        """)
            
            # Meeting Logs
            st.subheader("Meeting Logs")
            logs = st.session_state.current_meeting.get('logs', [])
            for log in logs:
                if log.get('type') == 'c_level_meeting_phases':
                    st.markdown("### C-Level 3step meeting log")
                    for phase in log.get('phases', []):
                        st.markdown(f"#### {phase['phase']}")
                        st.markdown(phase['dialogue'])
                elif log.get('type') == 'team_meeting':
                    st.markdown(f"### {log['team']} Team Meeting Summary")
                    st.markdown(self._format_markdown(log['summary']))
                else:
                    st.markdown(f"### Meeting Log ({log.get('type', 'unknown')})")
                    st.json(log)
        
        # Role Generation History
        if st.session_state.role_generation_history:
            st.sidebar.subheader("Role Generation History")
            for i, history in enumerate(st.session_state.role_generation_history):
                with st.sidebar.expander(f"Generation {i+1}"):
                    st.markdown("### Prompt")
                    st.code(history['prompt'])
                    st.markdown("### Response")
                    st.code(history['response'])

        # (선택) session_state 전체를 디버깅용으로 표시
        with st.expander("[DEBUG] session_state 전체 보기"):
            st.json(dict(st.session_state))
            

if __name__ == "__main__":
    gui = CollaborationGUI()
    gui.run() 