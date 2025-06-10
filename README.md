# SituTrust Engine

## Overview
SituTrust Engine is a prompt-native AI conference system designed for orchestrated agent-to-agent (A2A) collaboration, grounded in trust and spatial context. It enables dynamic creation of C-level and expert AI agents, spatially situated meetings, trust-embedded decision making, and real-time task execution with persistent logs and artifacts.

## Features
- Dynamic C-level and expert agent generation based on goals and constraints
- Spatial prompting: creative, context-aware meeting space generation
- Trust matrix: prompt-based, experience-driven trust injection for all agents
- 3-phase C-level meeting simulation (problem definition, role assignment, tactical strategy)
- User feedback loop (OK/feedback, iterative improvement)
- Team/agent task assignment, execution, and artifact storage
- Streamlit-based interactive UI for real-time collaboration and monitoring

## Project Structure
```
SituTrust_Engine/
├── code/
│   └── experiments/
│       ├── gui/                # Streamlit app and GUI logic
│       ├── utils/              # Core modules: trust, spatial, role, flow, etc.
│       └── ...
├── paper/                      # Paper sections, figures, drafts
├── workspace/                  # Runtime data: meetings, logs, artifacts, databases
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
└── .gitignore                  # Git ignore rules
```

## Setup & Usage
1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/SituTrust_Engine.git
   cd SituTrust_Engine
   ```
2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. **Set your OpenAI API key**
   - Create a `.env` file in the project root:
     ```
     OPENAI_API_KEY=your-key-here
     ```
5. **Run the Streamlit app**
   ```bash
   PYTHONPATH=code/experiments streamlit run code/experiments/gui/streamlit_app.py
   ```
6. **Access the app**
   - Open your browser at [http://localhost:8501](http://localhost:8501)

## Contribution Guidelines
- Fork the repository and create feature branches (e.g., `feature/your-feature`)
- Write clear commit messages and document your code
- Open pull requests for review
- Report issues and suggest improvements via GitHub Issues

## License
MIT License (see LICENSE file)

### Description
SituTrust Engine is a prompt-native AI conference system grounded in trust and space. This repository contains the implementation and documentation of our research on AI-to-AI collaboration in spatially situated and trust-embedded conferences.

### Sections
- Abstract
- Introduction
- Related Works
- Method
- Experiments
- Discussion

### Getting Started
[To be added as the project progresses] 