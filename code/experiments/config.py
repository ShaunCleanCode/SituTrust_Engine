import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

AGENT_MODEL = 'gpt-4'

# Room Configuration
DEFAULT_ROOM = 'solvay_strategy_room'

# Trust Model Configuration
TRUST_WEIGHTS = {
    'historical_collaboration': 0.3,
    'success_rate': 0.3,
    'communication_style': 0.2,
    'domain_alignment': 0.2
}

# Spatial Configuration
ROOM_TEMPLATES = {
    'solvay_strategy_room': {
        'description': 'A collaborative space for strategic planning and problem-solving',
        'capacity': 5,
        'features': ['whiteboard', 'projector', 'meeting_table']
    },
    'research_lab': {
        'description': 'A space for scientific research and experimentation',
        'capacity': 3,
        'features': ['lab_bench', 'computers', 'research_materials']
    }
}

# Logging Configuration
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s' 