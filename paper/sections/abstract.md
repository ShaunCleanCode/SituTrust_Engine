# Abstract

### 2. Abstract

---

**Prompt-Orchestrated A2A Collaboration via Spatially Situated and Trust-Embedded Conferences**

Despite their impressive generative capabilities, current large language models (LLMs) remain confined to a predominantly **1:1 human-to-model prompting paradigm**. This interaction mode requires constant human rephrasing and steering, positioning the LLM as a reactive oracle rather than an autonomous collaborator. Such a setup inherently limits the potential of LLMs for **sustained multi-agent collaboration, distributed reasoning, and long-term goal-oriented workflows**. While prior approaches—such as Chain-of-Thought prompting, tool-augmented planners, and role-based agent frameworks—attempt to address these issues, they often fall short due to **static role assignment, lack of spatial context, and minimal relational memory**.

In this work, we propose **SituTrust**, a novel prompt-native system that reconceptualizes prompts not merely as instructions, but as **constructive blueprints for cognitive environments**. SituTrust leverages **Spatial Prompting** to instantiate interactive spaces—such as conference rooms or project war rooms—through language alone, enabling LLMs to be contextually situated within them. Within these generated spaces, LLMs assume roles, build trust based on past interactions, and collaboratively execute complex reasoning tasks, **as if occupying a real shared environment**. This marks a shift toward **prompting-as-architecture**, where prompts do not just guide behavior—they **construct dynamic, intelligent organizations** within language.

SituTrust is built upon three foundational prompting principles:

1. **Conference Prompting** enables agent-to-agent (A2A) dialogues within a structured conference-like setup, facilitating adversarial reasoning, role-based negotiation, and iterative refinement of ideas.
2. **Spatial Prompting** expands the scope of role prompting by introducing a **prompt-constructed spatial environment**, where agents are instantiated not as abstract functions, but as contextually situated entities within a mental scene. This shift grounds collaborative reasoning in shared spatial contexts, akin to a virtual meeting room or task-specific workspace.
3. **Trust-Embedded Prompting** leverages structured relational vectors that encode historical collaboration, success rates, communication style, and domain expertise. These vectors are injected directly into prompts to influence task delegation, argument weighting, and emergent social dynamics among agents.

Unlike prior multi-agent frameworks that rely on external orchestration, SituTrust demonstrates how **prompts alone can instantiate coherent collaborative intelligence**, complete with spatial grounding and relational memory. Humans act only as supervisors or evaluators, while agents autonomously conduct discussions, generate outputs, and self-correct.

This abstract presents the conceptual design of SituTrust and its potential to shift the prompting paradigm toward **embedded organizational cognition**. While empirical evaluation is in progress, preliminary results suggest improved performance in ideation-heavy and multi-perspective reasoning tasks. Further experiments will quantify gains in diversity, consistency, and alignment quality across collaborative agent tasks.