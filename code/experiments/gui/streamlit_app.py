import streamlit as st
import json
import os
from datetime import datetime
import pandas as pd
import plotly.express as px
import networkx as nx
from utils.collaboration_flow import CollaborationFlow
from utils.spatial_manager import SpatialPrompting
from utils.trust_functions import TrustFunctions
from utils.role_generator import RoleGenerator
from utils.command_handler import CommandHandler
from typing import Dict, List, Optional
import sqlite3
import time
import openai
from config import OPENAI_API_KEY, AGENT_MODEL
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
        roles = self.role_generator.generate_roles(goal, constraints)
        
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
        st.title("AI Collaboration System")
        
        # Display any error messages
        self._display_errors()
        
        # Meeting Initialization Section
        with st.expander("Initialize Meeting", expanded=not st.session_state.meeting_initialized):
            goal = st.text_area("Meeting Goal", height=100)
            constraints = st.text_area("Meeting Constraints", height=100)
            
            if st.button("Generate Roles"):
                if not goal or not constraints:
                    self._add_error("Please provide both goal and constraints")
                    return
                
                with st.spinner("Generating roles..."):
                    try:
                        roles = self.role_generator.generate_roles(goal, constraints)
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
                                self._add_error("Failed to initialize teams")
                        else:
                            self._add_error("Failed to generate roles")
                    except Exception as e:
                        self._add_error(f"Error during role generation: {str(e)}")
        
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
            st.markdown("#### [DEBUG] Roles Structure")
            st.json(st.session_state.current_meeting['roles'])
            
            # C-Level Meeting Section
            st.subheader("C-Level Meeting")
            if st.button("Start C-Level Meeting"):
                with st.spinner("Generating space prompt and trust matrix for C-Level meeting..."):
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
                                    collaboration_history.append({
                                        'agent_a': a['agent_name'],
                                        'agent_b': b['agent_name'],
                                        'history': 2,  # e.g., 2 past collaborations
                                        'success_rate': 0.8,
                                        'comm_compat': 0.9,
                                        'domain_align': 0.85
                                    })
                        trust_data = self.trust_functions.generate_prompt_based_trust_matrix(c_level_roles, collaboration_history)
                        st.markdown("#### 🤝 C-Level Trust Matrix & Strategy")
                        st.json(trust_data)
                        # Log C-Level roles for debugging
                        st.markdown("#### [DEBUG] C-Level Roles")
                        st.json(c_level_roles)
                        # 3. 3-Phase C-Level Meeting Simulation
                        phase_titles = [
                            "PHASE 1 – 문제 정의 및 구조 분석",
                            "PHASE 2 – 역할 분담 및 책임 설정",
                            "PHASE 3 – 전술적 전략 회의"
                        ]
                        phase_prompts = [
                            "각 C-Level 역할이 자신의 관점에서 문제를 정의하고, 구조를 분석하며, 서로 토론/반론/합의를 진행한다. 신뢰도 기반으로 수용/반론/근거요구 등 행동을 결정한다.",
                            "각 C-Level이 구체적인 역할 분담과 책임 설정을 논의한다. 각 팀/Agent의 역할과 책임, 협업 방식을 합의한다.",
                            "전술적 전략(실행 전략, 일정, 최종 합의 등)을 논의하고, 최종 실행 방안을 도출한다."
                        ]
                        phase_logs = []
                        for idx, (title, prompt) in enumerate(zip(phase_titles, phase_prompts)):
                            st.markdown(f"### 🧠 {title}")
                            # Compose phase context
                            phase_context = f"""
회의 목표: {goal}
참가자: {', '.join(participants)}
공간: {space_prompt}
신뢰도 행렬: {json.dumps(trust_data.get('trust_matrix', {}), ensure_ascii=False)}
행동전략: {json.dumps(trust_data.get('strategy', {}), ensure_ascii=False)}

{prompt}
각 역할별로 실제 대화체로 2~3턴씩 주고받으며, 신뢰도 기반 행동(수용/반론/근거요구 등)을 명확히 드러내고, 마지막엔 간략 요약을 붙여줘.
"""
                            # Call GPT for phase dialogue
                            response = self.trust_functions.openai_client.chat.completions.create(
                                model=AGENT_MODEL,
                                messages=[
                                    {"role": "system", "content": "You are a C-Level meeting simulator. Always respond in markdown, with dialogue and summary."},
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
                        st.success("C-Level 3단계 회의가 완료되었습니다!")
                        # 4. User feedback/OK selection
                        st.markdown("---")
                        st.markdown("#### 회의 결과에 대해 선택해 주세요:")
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("✅ OK 진행 (다음 단계)"):
                                st.session_state.c_level_meeting_ok = True
                                st.success("OK를 선택하셨습니다. 각 팀의 전문가 Agent를 생성하고 작업을 분배합니다.")
                                # --- Agent 생성 및 작업 분배/실행 ---
                                roles = st.session_state.current_meeting['roles']
                                c_level_roles = self._get_c_level_roles(roles)
                                # 1. 각 C-Level 팀별 전문가 Agent 생성 (예시: initial_team_requirements 기반)
                                agent_results = {}
                                for c_role in c_level_roles:
                                    team_name = c_role['team_name']
                                    requirements = c_role.get('initial_team_requirements', {})
                                    required_roles = requirements.get('required_roles', [])
                                    agents = []
                                    for r in required_roles:
                                        agent_name = f"{r.lower().replace(' ', '_')}_ai"
                                        agents.append({
                                            'agent_name': agent_name,
                                            'role': r,
                                            'team': team_name
                                        })
                                    # 2. 각 Agent별 작업(Task) 분배 및 실행 (예: GPT로 코드/텍스트/이미지 생성)
                                    agent_outputs = []
                                    for agent in agents:
                                        # 예시: 각 Agent가 자신의 역할에 맞는 결과물 생성
                                        task_prompt = f"You are {agent['agent_name']} in {team_name}. Your role: {agent['role']}. Generate a sample output (code, text, or image description) for your main responsibility."
                                        response = self.trust_functions.openai_client.chat.completions.create(
                                            model=AGENT_MODEL,
                                            messages=[
                                                {"role": "system", "content": "You are an expert agent. Always respond with a relevant output for your role."},
                                                {"role": "user", "content": task_prompt}
                                            ],
                                            temperature=0.7
                                        )
                                        output = response.choices[0].message.content.strip()
                                        agent_outputs.append({
                                            'agent': agent['agent_name'],
                                            'role': agent['role'],
                                            'output': output
                                        })
                                        # 3. 결과물 저장 (폴더/DB)
                                        agent_dir = f"workspace/artifacts/{agent['agent_name']}"
                                        os.makedirs(agent_dir, exist_ok=True)
                                        with open(os.path.join(agent_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_output.txt"), 'w', encoding='utf-8') as f:
                                            f.write(output)
                                    agent_results[team_name] = agent_outputs
                                # 4. UI에 결과물 표시
                                st.markdown("---")
                                st.markdown("### 팀/Agent별 작업 결과물")
                                for team, outputs in agent_results.items():
                                    st.markdown(f"#### {team}")
                                    for out in outputs:
                                        st.markdown(
                                            f"""- **{out['agent']}** ({out['role']}):\n\n```text\n{out['output']}\n```"""
                                        )
                        with col2:
                            if st.button("❌ 피드백 제출 (회의 반복)"):
                                st.session_state.c_level_meeting_ok = False
                                st.warning("피드백을 제출하셨습니다. 회의를 반복합니다.")
                    except Exception as e:
                        import traceback
                        tb = traceback.format_exc()
                        self._add_error(f"Error during C-Level meeting: {str(e)}\nTraceback:\n{tb}")
            
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
                    st.markdown("### C-Level 3단계 회의 로그")
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

if __name__ == "__main__":
    gui = CollaborationGUI()
    gui.run() 