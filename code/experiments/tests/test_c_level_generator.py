import unittest
import json
import os
from datetime import datetime
from pathlib import Path
import sys

# Add the project root to Python path
project_root = str(Path(__file__).parent.parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from code.experiments.utils.role_generator import RoleGenerator

class TestCLevelGenerator(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test."""
        self.role_generator = RoleGenerator()
        self.test_output_dir = Path(project_root) / "workspace" / "tests" / "c_level_generator"
        self.test_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for different types of test outputs
        self.responses_dir = self.test_output_dir / "responses"
        self.responses_dir.mkdir(exist_ok=True)
        
        self.analysis_dir = self.test_output_dir / "analysis"
        self.analysis_dir.mkdir(exist_ok=True)

    def _save_test_response(self, test_name: str, prompt: str, response: list, analysis: dict = None):
        """Save test response and analysis to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save response
        response_file = self.responses_dir / f"{test_name}_{timestamp}_response.json"
        with open(response_file, 'w', encoding='utf-8') as f:
            json.dump({
                'prompt': prompt,
                'response': response,
                'timestamp': timestamp
            }, f, indent=2, ensure_ascii=False)
        
        # Save analysis if provided
        if analysis:
            analysis_file = self.analysis_dir / f"{test_name}_{timestamp}_analysis.json"
            with open(analysis_file, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, indent=2, ensure_ascii=False)

    def _analyze_roles(self, roles: list) -> dict:
        """Analyze the generated roles and return metrics."""
        analysis = {
            'total_roles': len(roles),
            'role_types': {},
            'team_distribution': {},
            'responsibility_length': [],
            'skill_requirements': [],
            'trust_requirements': [],
            'cross_functional_impact': [],
            'risk_management': [],
            'innovation_focus': [],
            'stakeholder_management': []
        }
        
        for role in roles:
            # Count role types
            role_type = role.get('role_type', 'Unknown')
            analysis['role_types'][role_type] = analysis['role_types'].get(role_type, 0) + 1
            
            # Count team distribution
            team = role.get('team_name', 'Unknown')
            analysis['team_distribution'][team] = analysis['team_distribution'].get(team, 0) + 1
            
            # Analyze text lengths
            analysis['responsibility_length'].append(len(role.get('responsibilities', '')))
            analysis['skill_requirements'].append(len(role.get('required_skills', '')))
            analysis['trust_requirements'].append(len(role.get('trust_requirements', '')))
            analysis['cross_functional_impact'].append(len(role.get('cross_functional_impact', '')))
            analysis['risk_management'].append(len(role.get('risk_management', '')))
            analysis['innovation_focus'].append(len(role.get('innovation_focus', '')))
            analysis['stakeholder_management'].append(len(role.get('stakeholder_management', '')))
        
        # Calculate averages
        for key in ['responsibility_length', 'skill_requirements', 'trust_requirements',
                   'cross_functional_impact', 'risk_management', 'innovation_focus',
                   'stakeholder_management']:
            if analysis[key]:
                analysis[f'avg_{key}'] = sum(analysis[key]) / len(analysis[key])
            else:
                analysis[f'avg_{key}'] = 0
        
        return analysis

    def test_tech_startup_roles(self):
        """Test C-Level role generation for a tech startup scenario."""
        goal = """
        Develop and launch an AI-powered project management platform that helps teams 
        collaborate more effectively using natural language processing and machine learning.
        """
        constraints = [
            "Must be launched within 6 months",
            "Need to ensure data privacy and security compliance",
            "Must be scalable to handle enterprise-level usage",
            "Should integrate with popular project management tools"
        ]
        
        roles = self.role_generator.generate_c_level_roles(goal, constraints)
        self.assertIsNotNone(roles)
        self.assertTrue(len(roles) > 0)
        
        # Save and analyze response
        analysis = self._analyze_roles(roles)
        self._save_test_response(
            "tech_startup",
            f"Goal: {goal}\nConstraints: {constraints}",
            roles,
            analysis
        )

    def test_healthcare_platform_roles(self):
        """Test C-Level role generation for a healthcare platform scenario."""
        goal = """
        Create a secure healthcare platform that connects patients with specialists,
        manages medical records, and provides AI-powered health insights.
        """
        constraints = [
            "Must comply with HIPAA and other healthcare regulations",
            "Need to ensure patient data security",
            "Must be accessible to patients with disabilities",
            "Should integrate with existing healthcare systems"
        ]
        
        roles = self.role_generator.generate_c_level_roles(goal, constraints)
        self.assertIsNotNone(roles)
        self.assertTrue(len(roles) > 0)
        
        # Save and analyze response
        analysis = self._analyze_roles(roles)
        self._save_test_response(
            "healthcare_platform",
            f"Goal: {goal}\nConstraints: {constraints}",
            roles,
            analysis
        )

    def test_fintech_roles(self):
        """Test C-Level role generation for a fintech scenario."""
        goal = """
        Develop a blockchain-based payment system that enables instant cross-border
        transactions with minimal fees and maximum security.
        """
        constraints = [
            "Must comply with international financial regulations",
            "Need to ensure transaction security",
            "Must be scalable to handle high transaction volumes",
            "Should support multiple cryptocurrencies"
        ]
        
        roles = self.role_generator.generate_c_level_roles(goal, constraints)
        self.assertIsNotNone(roles)
        self.assertTrue(len(roles) > 0)
        
        # Save and analyze response
        analysis = self._analyze_roles(roles)
        self._save_test_response(
            "fintech",
            f"Goal: {goal}\nConstraints: {constraints}",
            roles,
            analysis
        )

    def test_role_validation(self):
        """Test the validation of generated roles."""
        goal = "Test role validation"
        constraints = ["Test constraint"]
        
        roles = self.role_generator.generate_c_level_roles(goal, constraints)
        self.assertIsNotNone(roles)
        
        for role in roles:
            # Test required fields
            self.assertIn('role_name', role)
            self.assertIn('role_type', role)
            self.assertIn('team_name', role)
            self.assertIn('agent_name', role)
            self.assertIn('responsibilities', role)
            self.assertIn('required_skills', role)
            self.assertIn('team_size', role)
            self.assertIn('key_metrics', role)
            self.assertIn('collaboration_patterns', role)
            self.assertIn('trust_requirements', role)
            self.assertIn('decision_authority', role)
            self.assertIn('communication_channels', role)
            self.assertIn('success_criteria', role)
            self.assertIn('initial_team_requirements', role)
            self.assertIn('cross_functional_impact', role)
            self.assertIn('risk_management', role)
            self.assertIn('innovation_focus', role)
            self.assertIn('stakeholder_management', role)
            self.assertIn('reasoning', role)
            
            # Test role type
            self.assertEqual(role['role_type'], 'C-Level')
            
            # Test team requirements structure
            team_req = role['initial_team_requirements']
            self.assertIn('required_roles', team_req)
            self.assertIn('required_skills', team_req)
            self.assertIn('team_structure', team_req)
            self.assertIn('collaboration_model', team_req)

if __name__ == '__main__':
    unittest.main() 