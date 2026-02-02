# NPC Data Layer: Technical Architecture

The NPC Data Layer forms the technical foundation of the npcsh system, providing a structured approach to organizing, managing, and executing AI agent interactions. This document provides a technical deep-dive into how the data layer is implemented and how it enables the system's capabilities.

## Core Data Structures

### NPC Class

The `NPC` class is the fundamental entity representing an AI agent within the system. NPCs are initialized from YAML files (with `.npc` extension) or directly with parameters. Key features include:
- Model/provider configuration for LLM interactions
- Access to jinxs (function-like capabilities)
- Database connection for persistence
- Integration with teams
- Tool usage configuration
- Jinja templating environment for dynamic content

Each NPC maintains its own shared context dictionary which serves as working memory during execution. The class provides methods for executing jinxs, generating LLM responses, handling agent passes between NPCs, and checking/executing commands.

### Jinx Class

The `Jinx` class provides Jinja template executions that NPCs can use as tool. Importantly, Jinxs are defined and operationalized through a prompt-based flow which allows them to be usable by models even if they don't have built-in tool calling capabilities. Thus, with jinxs, we can get more out of our small models. Jinxs support two execution engines:
1. `natural` - Uses LLM processing for text generation
2. `python` - Executes Python code with access to the NPC context and system modules

Each jinx contains a sequence of steps with preprocessing, execution, and postprocessing phases, all templated through Jinja2. This enables complex behaviors where Python code can prepare data, an LLM can analyze it, and additional code can post-process the results.

### Team Class

The `Team` class manages collections of NPCs and provides team-wide functionality:
- Hierarchical organization with sub-teams
- Shared context across NPCs
- Orchestrated execution through a forenpc (coordinator)
- Team-wide jinx availability
- Loading of team context from `.ctx` files

Teams implement an orchestration method that coordinates work across NPCs, tracking execution history and ensuring requests are fully processed.

### Pipeline Class

The `Pipeline` class represents a workflow of NPC interactions:
- Sequence of execution steps across different NPCs
- Jinja template references between steps
- Access to database sources through special template functions
- Support for batch processing vs. row-wise processing
- Mixture of agents (mixa) processing for consensus-building

## Execution Flow

1. **Initialization**: The system loads NPCs, jinxs, and teams from the filesystem
2. **Command Processing**: User input is parsed and routed to appropriate handlers
3. **Mode-Based Execution**: 
   - `agent` mode: Intelligent routing with pipeline processing
   - `chat` mode: Direct LLM interaction with shell command detection
   - `cmd` mode: LLM command execution without routing

4. **State Management**: Conversation history, knowledge graphs, and context are persisted between interactions

## Data Persistence

The system uses several persistence mechanisms:
- SQLite database for conversation history and execution tracking
- Vector database (ChromaDB) for embeddings and semantic search
- Filesystem for NPC, jinx, and team definitions
- Knowledge graphs for contextual memory across sessions

For a practical walkthrough of database setup, conversation tracking, memory, NPCSQL, and building persistent agent systems, see the [Building Applications guide](guides/building-applications.md).
