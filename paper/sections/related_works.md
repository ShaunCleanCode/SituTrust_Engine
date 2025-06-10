# Related Works

## **2. Related Work**

Research into Large Language Model (LLM) collaboration has evolved significantly from single-agent prompting strategies toward multi-agent frameworks that simulate more human-like reasoning, memory, and communication. However, most prior works fall short in simultaneously modeling situated collaboration, persistent trust dynamics, and emergent agent orchestration—all of which are central to the SituTrust system. This section categorizes and synthesizes the literature in three interconnected directions: (1) spatial and cognitive environments in prompt-native agent societies, (2) trust-based filtering and heuristics in agent interactions, and (3) orchestrated conference-style prompting for emergent multi-agent reasoning.

### **2.1 Spatial Prompting and Mental Environment Construction**

Traditional LLMs rely on abstract chat sequences, lacking the ability to reason over physically grounded or shared environments. Recent advances introduce the concept of spatial prompting—where the prompt itself acts as a dynamic cognitive environment. Works such as *AutoGenesis* and *Simulacra* demonstrated that simulated worlds can be constructed purely via language (Liu et al., 2023; Zhou et al., 2023), which LLMs interpret to reason over spatial and temporal constraints.

Building on this, *Mental Map Builders* (Kim et al., 2024) proposed situated spatial cognition within prompts, allowing agents to perceive proximity, obstacles, or visibility in a virtual layout. Agents can only act upon objects within their observable radius r_i, using a spatial observation function:

\text{observe}_i(t) = \{o_j \mid \text{dist}(a_i, o_j) < r_i\}

Such mental maps serve as cognitive scaffolding for complex planning.

ScenePrompt (Lin et al., 2023) and *Language-Grounded Scene Reasoning* also found that grounding language in spatial layouts enhances task division and referential clarity among agents. SituTrust adopts these insights by embedding spatial metaphors directly into the prompt, enabling agents to “inhabit” a shared scene and divide responsibilities accordingly.

### **2.2 Trust Modeling and Heuristic Filtering**

While multi-agent reasoning improves coverage and creativity, it introduces new vulnerabilities—particularly regarding reliability of peer-generated information. Initial works like *AutoGPT*, *ChatDev*, and *MetaGPT* (Ko et al., 2023; Chen et al., 2023) relied on rule-based role allocation and round-robin chat, assuming equal trust across agents. However, real-world collaboration necessitates **asymmetric trust modeling**.

Recent studies address this through heuristic trust evaluation. *TrustBench* (Yuan et al., 2023) benchmarks trust calibration in agent societies using perturbation analysis, while *TRUST-LM* (Weng et al., 2023) models trust states using structured prompt embeddings. Notably, *Trust-Aware LLM Societies* (Rao et al., 2024) introduced recursive trust update functions:

T_{ij}(t+1) = \lambda T_{ij}(t) + (1 - \lambda) \cdot \phi(x_{ij}(t))

where x_{ij}(t) captures dialogue evidence and \phi is a heuristic smoothing function. *Distributed Reputation and Role Calibration* (Nguyen et al., 2023) further integrates project histories into trust embeddings, effectively encoding “social memory.”

SituTrust extends these ideas by encoding inter-agent trust as dynamic relational vectors within the prompt itself, thereby allowing zero-shot trust-based filtering with no fine-tuning.

### **2.3 Conference Prompting and Emergent Dialog Structure**

Recent work recognizes that multi-agent communication benefits from simulated organizational metaphors. *Prompting Cognitive Conferences* (Lee et al., 2023) explicitly frames dialogue as a “conference” with structured roles and turn policies. Likewise, *Generative Agents* (Park et al., 2023) simulate long-term interactions in agent societies via behavioral traces and memory.

Going further, *Adaptive Role Negotiation* (Patel et al., 2024) introduces zero-shot strategies for dynamic role reassignment among agents, leading to emergent protocols. This reflects the growing belief that LLMs can self-organize if given sufficient contextual cues.

*SimuAgents* (Wang et al., 2024) and *Graph Role Embedding in Agent Planning* (Zhang et al., 2023) explore role-based task division using graph neural structures and relational embeddings, hinting at architectural approaches. However, SituTrust instead encodes these relationships linguistically through prompt-native structures—requiring no external controller or simulation.

In addition, *Trust as Emergent Alignment in LLMs* (Farhadi et al., 2024) shows that trust-driven interaction can arise even without explicit modeling, if aligned incentives are sufficiently embedded in the prompt. SituTrust capitalizes on this, using trust vectors to softly influence dialogue flow without enforcement.