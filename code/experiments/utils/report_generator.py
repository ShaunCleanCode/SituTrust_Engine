import json
from pathlib import Path
from datetime import datetime
from fpdf import FPDF
import pandas as pd
from typing import Dict, List, Optional

class ReportGenerator:
    def __init__(self, project_root: str = "SituTrust_Engine"):
        self.project_root = Path(project_root)
        self.reports_dir = self.project_root / "reports"
        self.reports_dir.mkdir(exist_ok=True)

    def generate_meeting_log_pdf(self, meeting_id: str) -> Path:
        """Generate a PDF report from meeting logs."""
        # Load meeting data
        meeting_data = self._load_meeting_data(meeting_id)
        if not meeting_data:
            raise ValueError(f"No meeting data found for ID: {meeting_id}")

        # Create PDF
        pdf = FPDF()
        pdf.add_page()
        
        # Add title
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, f'Meeting Log: {meeting_id}', ln=True, align='C')
        pdf.ln(10)

        # Add meeting info
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Meeting Information', ln=True)
        pdf.set_font('Arial', '', 12)
        pdf.multi_cell(0, 10, f"Goal: {meeting_data.get('goal', 'N/A')}")
        pdf.multi_cell(0, 10, f"Date: {meeting_data.get('timestamp', 'N/A')}")
        pdf.ln(5)

        # Add participants
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Participants', ln=True)
        pdf.set_font('Arial', '', 12)
        for agent in meeting_data.get('agents', []):
            pdf.multi_cell(0, 10, f"• {agent['name']}: {agent['description']}")
        pdf.ln(5)

        # Add interactions
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Interactions', ln=True)
        pdf.set_font('Arial', '', 12)
        for interaction in meeting_data.get('interactions', []):
            pdf.multi_cell(0, 10, f"From: {interaction['from']}")
            pdf.multi_cell(0, 10, f"To: {interaction['to']}")
            pdf.multi_cell(0, 10, f"Content: {interaction['content']}")
            if 'trust_weight' in interaction:
                pdf.multi_cell(0, 10, f"Trust Weight: {interaction['trust_weight']}")
            pdf.ln(5)

        # Add task progress
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Task Progress', ln=True)
        pdf.set_font('Arial', '', 12)
        for agent_name, agent_data in meeting_data.get('task_assignments', {}).items():
            pdf.multi_cell(0, 10, f"Agent: {agent_name}")
            for task in agent_data.get('tasks', []):
                pdf.multi_cell(0, 10, f"• {task['description']}")
                pdf.multi_cell(0, 10, f"  Status: {task.get('status', 'N/A')}")
                if 'completion_time' in task:
                    pdf.multi_cell(0, 10, f"  Completed: {task['completion_time']}")
            pdf.ln(5)

        # Add trust matrix
        if 'trust_matrix' in meeting_data:
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, 'Trust Matrix', ln=True)
            pdf.set_font('Arial', '', 12)
            
            # Convert trust matrix to DataFrame for better formatting
            trust_df = pd.DataFrame(meeting_data['trust_matrix'])
            trust_table = trust_df.to_string()
            
            # Add trust matrix as text (simple format)
            pdf.multi_cell(0, 10, trust_table)
            pdf.ln(5)

        # Save PDF
        output_file = self.reports_dir / f"meeting_log_{meeting_id}.pdf"
        pdf.output(str(output_file))
        
        return output_file

    def generate_summary_report(self, meeting_id: str) -> Path:
        """Generate a summary report of the meeting."""
        # Load meeting data
        meeting_data = self._load_meeting_data(meeting_id)
        if not meeting_data:
            raise ValueError(f"No meeting data found for ID: {meeting_id}")

        # Create summary markdown
        summary = f"""# Meeting Summary: {meeting_id}

## Overview
- **Date**: {meeting_data.get('timestamp', 'N/A')}
- **Goal**: {meeting_data.get('goal', 'N/A')}

## Participants
{self._format_participants(meeting_data.get('agents', []))}

## Key Outcomes
{self._format_outcomes(meeting_data)}

## Task Completion
{self._format_task_completion(meeting_data.get('task_assignments', {}))}

## Trust Dynamics
{self._format_trust_dynamics(meeting_data.get('trust_matrix', {}))}
"""

        # Save summary
        output_file = self.reports_dir / f"summary_{meeting_id}.md"
        with open(output_file, 'w') as f:
            f.write(summary)
        
        return output_file

    def _load_meeting_data(self, meeting_id: str) -> Optional[Dict]:
        """Load meeting data from logs."""
        meeting_file = self.project_root / "meeting_logs" / f"meeting_{meeting_id}.json"
        if not meeting_file.exists():
            return None
        
        with open(meeting_file) as f:
            return json.load(f)

    def _format_participants(self, agents: List[Dict]) -> str:
        """Format participants section."""
        return "\n".join([
            f"- **{agent['name']}**: {agent['description']}"
            for agent in agents
        ])

    def _format_outcomes(self, meeting_data: Dict) -> str:
        """Format key outcomes section."""
        outcomes = []
        
        # Add consensus outcomes
        if 'consensus' in meeting_data:
            outcomes.append(f"- Consensus reached: {meeting_data['consensus']}")
        
        # Add major decisions
        if 'decisions' in meeting_data:
            for decision in meeting_data['decisions']:
                outcomes.append(f"- Decision: {decision}")
        
        return "\n".join(outcomes) if outcomes else "No specific outcomes recorded."

    def _format_task_completion(self, task_assignments: Dict) -> str:
        """Format task completion section."""
        completion = []
        
        for agent_name, agent_data in task_assignments.items():
            completed = sum(1 for task in agent_data.get('tasks', [])
                          if task.get('status') == 'completed')
            total = len(agent_data.get('tasks', []))
            
            completion.append(f"- **{agent_name}**: {completed}/{total} tasks completed")
        
        return "\n".join(completion) if completion else "No tasks recorded."

    def _format_trust_dynamics(self, trust_matrix: Dict) -> str:
        """Format trust dynamics section."""
        if not trust_matrix:
            return "No trust data available."
        
        # Calculate average trust scores
        trust_df = pd.DataFrame(trust_matrix)
        avg_trust = trust_df.mean().mean()
        max_trust = trust_df.max().max()
        
        return f"""- Average trust score: {avg_trust:.2f}
- Maximum trust score: {max_trust:.2f}
- Trust matrix available in detailed report""" 