# Method

## **3. Task / Problem and Method**

### **3.1 Problem Formulation: The Limitation of Monolithic Prompting in Collaborative AI**

Despite remarkable progress in large language models (LLMs), most multi-agent prompting systems still rely on **static role assignments**, **flat prompt architectures**, or **centralized orchestration**. These approaches fail to scale in collaborative, high-dimensional tasks such as software development, scientific planning, or open-ended ideation. More critically, they lack the organizational dynamics—such as trust-based delegation, relational memory, and spatial awareness—that are essential in human teams.

To address this gap, we define the following **core problem**:

> “How can we build a self-organizing multi-agent system (MAS) in which agents autonomously collaborate, reason, and delegate tasks based on spatial context and mutual trust—without relying on external orchestration or tool chains?”
> 

This research proposes **SituTrust**, a novel framework that reconceptualizes prompts as architectures for embedding space, trust, and structured collaboration into a purely language-based MAS.

## **3. Task / Problem and Method**

### **3.1 Problem Formulation: The Limitation of Monolithic Prompting in Collaborative AI**

Despite remarkable progress in large language models (LLMs), most multi-agent prompting systems still rely on **static role assignments**, **flat prompt architectures**, or **centralized orchestration**. These approaches fail to scale in collaborative, high-dimensional tasks such as software development, scientific planning, or open-ended ideation. More critically, they lack the organizational dynamics—such as trust-based delegation, relational memory, and spatial awareness—that are essential in human teams.

To address this gap, we define the following **core problem**:

> “How can we build a self-organizing multi-agent system (MAS) in which agents autonomously collaborate, reason, and delegate tasks based on spatial context and mutual trust—without relying on external orchestration or tool chains?”
> 

This research proposes **SituTrust**, a novel framework that reconceptualizes prompts as architectures for embedding space, trust, and structured collaboration into a purely language-based MAS.

| **Component** | **Description** |
| --- | --- |
| **Spatial Prompting** | Constructs shared workspaces (e.g., a virtual lab or boardroom) using natural language. Agents are situated in these environments, grounding their interactions spatially. |
| **Trust-Embedded Prompting** | Each agent maintains a trust vector toward others, influencing dialogue acceptance, task allocation, and rebuttal thresholds. |
| **A2A Conferencing** | Agents engage in structured, multi-turn dialogue resembling team discussions, where decisions emerge through negotiation, critique, and summarization. |

### **3.3 Trust Modeling in Agent Collaboration**

To regulate information flow and prevent uncritical acceptance among agents, SituTrust incorporates a **quantitative trust model**. Each agent A_i maintains a dynamic trust score T_{ij} \in [0, 1] toward agent A_j, calculated as:

T_{ij} = \sigma(w_1 \cdot h_{ij} + w_2 \cdot s_{ij} + w_3 \cdot c_{ij} + w_4 \cdot a_{ij})

Where:

| **Variable** | **Meaning** |
| --- | --- |
| h_{ij} | Historical collaboration count between A_i and A_j |
| s_{ij} | Success rate of past joint tasks |
| c_{ij} | Communication compatibility score (style, coherence) |
| a_{ij} | Domain alignment (e.g., shared expertise areas) |
| w_k | Predefined or learned weights |
| \sigma | Sigmoid function to bound trust score in [0,1] |

These trust scores are stored and updated in **relational memory vectors**, embedded directly into each prompt. The trust vector serves dual functions:

- As a **filter**: Low-trust agents’ arguments are weighted down or challenged.
- As a **delegator**: Tasks are distributed preferentially toward high-trust peers.

### **3.4 Spatial Architecture as Prompt Construct**

Unlike symbolic metaphors, SituTrust instantiates **literal spatial constructs** through prompt encoding. Agents are placed in spaces such as:

- Room(scene="AI Whiteboard Session", participants=[A1, A2, A3], goal="Design a caching system")
- Room(scene="Product Strategy Review", participants=[A4, A5], goal="Prioritize Q3 features")

These rooms are retained across turns, allowing agents to refer to previous whiteboard sketches, revisit decisions, and recall spatial relationships—mimicking how humans think in space.

### **3.5 Experimental Task Types**

We propose to evaluate SituTrust across a diverse range of **collaborative agent tasks**:

| **Task Type** | **Description** |
| --- | --- |
| **Code Review & Debugging** | Multiple agents critique and propose patches for a shared codebase. |
| **Creative Writing** | Co-authorship of stories with diverse narrative voices and goals. |
| **Game Design Ideation** | Distributed brainstorming for gameplay mechanics and balancing. |
| **Marketing Strategy** | Agents assume roles (designer, strategist, critic) to launch a product plan. |
| **Scientific Debate** | Agents challenge hypotheses and build consensus through multi-perspective reasoning. |

Each task is performed in two modes: (1) baseline prompting (single or flat agents), and (2) SituTrust prompting (spatial + trust + A2A). Performance will be measured using:

| **Evaluation Metric** | **Description** |
| --- | --- |
| **Diversity Score** | Lexical and conceptual variation among agent responses |
| **Consistency Score** | Logical and stylistic coherence within and across turns |
| **Trust Conformity Ratio** | Degree to which task delegation aligns with trust vectors |
| **Outcome Utility** | Task-specific human evaluation (e.g., creativity, accuracy, actionability) |
| **Responsiveness** | How well agents build on each other’s contributions |

### **3.6 Method Execution Flow**

1. **Initialization**:
    - Define agents, roles, prior history, and trust vectors.
    - Construct spatial scene prompt and task objective.
2. **Prompting Session**:
    - Agents receive initial scene prompt.
    - Respond iteratively based on context, prior turns, and trust-aware reasoning.
3. **Agent Behaviors**:
    - High-trust responses are accepted or extended.
    - Low-trust responses are filtered, critiqued, or ignored.
    - Summarization agents condense multi-turn logs into a final report.
4. **Human Feedback Loop**:
    - Optional human evaluation scores are used to adjust trust weights w_k.
    - Agents can retrain local heuristics or role behavior accordingly.

### **3.7 Core Hypothesis**

We hypothesize that:

> “By grounding agent interaction in spatial context and trust-weighted dynamics, SituTrust enables more autonomous, consistent, and effective multi-agent reasoning—without external orchestration.”
> 

This structure does not merely enhance prompting—it **transforms prompting into a new form of organizational cognition**, laying the foundation for scalable, autonomous AI teams.