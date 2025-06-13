from typing import Dict, List, Optional
import uuid
import sys
from pathlib import Path

# Add the project root to Python path
project_root = str(Path(__file__).parent.parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from code.experiments.config import OPENAI_API_KEY, AGENT_MODEL

import openai
import json
from datetime import datetime
import streamlit as st
import shutil
from typing import Dict, List, Optional, Union
import logging

class CommandHandler:
    def __init__(self, project_root: str = "SituTrust_Engine"):
        self.project_root = Path(project_root)
        self._setup_directories()
        self._setup_logging()
        
    def _setup_directories(self):
        """Initialize directory structure for persistence."""
        directories = [
            "meeting_logs",
            "agents",
            "reports",
            "checkpoints",
            "goals_and_constraints"
        ]
        
        for directory in directories:
            (self.project_root / directory).mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self):
        """Configure logging for the system."""
        log_dir = self.project_root / "logs"
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f"system_{datetime.now().strftime('%Y%m%d')}.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("CommandHandler")

    def handle_command(self, command: str, data: Optional[Dict] = None) -> Dict:
        """Process user commands and trigger appropriate actions."""
        command_map = {
            "OK 진행": self._handle_confirm,
            "피드백 제출": self._handle_feedback,
            "STOP 회의 중단": self._handle_stop,
            "사용자에게 결과 보고 후 OK 받기": self._handle_report,
            "Trust Score Matrix 보기": self._handle_trust_matrix,
            "XX_ai DB 확인": self._handle_agent_db,
            "회의 전체 로그 PDF로 보기": self._handle_meeting_log,
            "새로운 작업 요청": self._handle_new_task
        }
        
        if command not in command_map:
            raise ValueError(f"Unknown command: {command}")
        
        return command_map[command](data)

    def _handle_confirm(self, data: Optional[Dict]) -> Dict:
        """Handle confirmation of current phase."""
        self._create_checkpoint("Phase Confirmed", data)
        return {"status": "confirmed", "message": "Phase confirmed, proceeding to next stage"}

    def _handle_feedback(self, data: Optional[Dict]) -> Dict:
        """Handle user feedback and trigger remeeting."""
        if not data or "feedback" not in data:
            raise ValueError("Feedback content required")
        
        feedback_file = self.project_root / "meeting_logs" / f"feedback_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
        with open(feedback_file, "w") as f:
            json.dump(data, f, indent=2)
        
        return {"status": "feedback_received", "message": "Feedback recorded, initiating remeeting"}

    def _handle_stop(self, data: Optional[Dict]) -> Dict:
        """Handle meeting pause and serialization."""
        self._create_checkpoint("Meeting Paused", data)
        self._serialize_session(data)
        return {"status": "paused", "message": "Meeting paused and serialized"}

    def _handle_report(self, data: Optional[Dict]) -> Dict:
        """Generate and present final report."""
        if not data or "meeting_id" not in data:
            raise ValueError("Meeting ID required for report generation")
        
        report = self._generate_report(data["meeting_id"])
        return {"status": "report_generated", "report": report}

    def _handle_trust_matrix(self, data: Optional[Dict]) -> Dict:
        """Retrieve and format trust matrix."""
        if not data or "meeting_id" not in data:
            raise ValueError("Meeting ID required for trust matrix")
        
        trust_matrix = self._load_trust_matrix(data["meeting_id"])
        return {"status": "success", "trust_matrix": trust_matrix}

    def _handle_agent_db(self, data: Optional[Dict]) -> Dict:
        """Access specific agent's task database."""
        if not data or "agent_name" not in data:
            raise ValueError("Agent name required")
        
        agent_db = self._load_agent_db(data["agent_name"])
        return {"status": "success", "agent_db": agent_db}

    def _handle_meeting_log(self, data: Optional[Dict]) -> Dict:
        """Generate PDF of meeting logs."""
        if not data or "meeting_id" not in data:
            raise ValueError("Meeting ID required")
        
        log_file = self._generate_meeting_log_pdf(data["meeting_id"])
        return {"status": "success", "log_file": str(log_file)}

    def _handle_new_task(self, data: Optional[Dict]) -> Dict:
        """Process new task request."""
        if not data or "task_description" not in data:
            raise ValueError("Task description required")
        
        task_id = self._create_new_task(data)
        return {"status": "success", "task_id": task_id}

    def _create_checkpoint(self, phase: str, data: Optional[Dict] = None) -> None:
        """Create a checkpoint for the current state."""
        checkpoint = {
            "timestamp": datetime.now().isoformat(),
            "phase": phase,
            "agents_active": data.get("agents_active", []) if data else [],
            "summary": data.get("summary", "") if data else ""
        }
        
        checkpoint_file = self.project_root / "checkpoints" / f"checkpoint_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
        with open(checkpoint_file, "w") as f:
            json.dump(checkpoint, f, indent=2)

    def _serialize_session(self, data: Dict) -> None:
        """Serialize the current session state."""
        session_file = self.project_root / "meeting_logs" / f"session_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
        with open(session_file, "w") as f:
            json.dump(data, f, indent=2)

    def _generate_report(self, meeting_id: str) -> Dict:
        """Generate a comprehensive meeting report."""
        # Implementation for report generation
        return {"meeting_id": meeting_id, "status": "generated"}

    def _load_trust_matrix(self, meeting_id: str) -> Dict:
        """Load trust matrix for a specific meeting."""
        # Implementation for loading trust matrix
        return {"meeting_id": meeting_id, "matrix": {}}

    def _load_agent_db(self, agent_name: str) -> Dict:
        """Load agent's task database."""
        agent_dir = self.project_root / "agents" / agent_name
        if not agent_dir.exists():
            return {}
        
        task_history = []
        for task_file in (agent_dir / "task_history").glob("*.json"):
            with open(task_file) as f:
                task_history.append(json.load(f))
        
        return {"agent_name": agent_name, "task_history": task_history}

    def _generate_meeting_log_pdf(self, meeting_id: str) -> Path:
        """Generate PDF of meeting logs."""
        # Implementation for PDF generation
        return self.project_root / "reports" / f"meeting_log_{meeting_id}.pdf"

    def _create_new_task(self, data: Dict) -> str:
        """Create a new task and return its ID."""
        task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M')}"
        task_file = self.project_root / "goals_and_constraints" / f"{task_id}.json"
        
        with open(task_file, "w") as f:
            json.dump(data, f, indent=2)
        
        return task_id 