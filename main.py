# main.py

from config import DEBATE_TOPIC, AGENTS_CONFIG, NUMBER_OF_REBUTTAL_ROUNDS, ENABLE_RAG, KB_DIRECTORY
from debate_state import DebateState
from agents import DebateOrchestrator, AffirmativeAgent, NegativeAgent, JudgeAgent, Agent
from rag_pipeline import index_knowledge_base, get_retriever # Import RAG functions

# Mapping from config type string to Agent class
AGENT_TYPE_MAP = {
    'AffirmativeAgent': AffirmativeAgent,
    'NegativeAgent': NegativeAgent,
    'JudgeAgent': JudgeAgent,
    # Add other agent types here if you create them
}

def create_agent_instance(agent_config: dict, retriever=None) -> Agent:
    """Creates an agent instance based on configuration, passing the retriever."""
    agent_type_str = agent_config['type']
    agent_name = agent_config['name']
    agent_model = agent_config.get('model')

    agent_class = AGENT_TYPE_MAP.get(agent_type_str)
    if not agent_class:
        raise ValueError(f"Unknown agent type specified in config: {agent_type_str}")

    # Instantiate the agent, passing retriever if applicable
    if agent_model:
        # Debate agents need the retriever
        if agent_type_str in ['AffirmativeAgent', 'NegativeAgent']:
             return agent_class(name=agent_name, model=agent_model, retriever=retriever)
        # Other agents (like Judge) might not need it depending on your RAG design
        else:
             return agent_class(name=agent_name, model=agent_model)
    else:
        # Use default model, pass retriever if applicable
        if agent_type_str in ['AffirmativeAgent', 'NegativeAgent']:
             return agent_class(name=agent_name, retriever=retriever)
        else:
             return agent_class(name=agent_name)


def main():
    """Sets up and runs the multi-agent debate with optional RAG."""

    retriever = None
    if ENABLE_RAG:
        print("\n--- Setting up Knowledge Base (RAG) ---", flush=True)
        vector_store = index_knowledge_base(kb_directory=KB_DIRECTORY)
        if vector_store:
            retriever = get_retriever(vector_store)
            if not retriever:
                 print("Failed to get retriever from vector store. RAG will be disabled.", flush=True)
                 retriever = None
        else:
            print("Knowledge base indexing failed or no documents found. RAG will be disabled.", flush=True)
            retriever = None
        print("--- Knowledge Base Setup Complete ---", flush=True)


    debate_state = DebateState(topic=DEBATE_TOPIC)

    # Create agent instances from config, pass the retriever
    all_agents = []
    for agent_config in AGENTS_CONFIG:
        if agent_config.get('optional', False) and not True:
             continue
        try:
            # Pass the retriever to the agent creation function
            agent_instance = create_agent_instance(agent_config, retriever=retriever)
            all_agents.append(agent_instance)
            print(f"Created agent: {agent_instance.name} ({agent_instance.role_type}) using model {agent_instance.model}", flush=True)
        except ValueError as e:
            print(f"Skipping agent configuration due to error: {e}", flush=True)
        except Exception as e:
             print(f"An unexpected error occurred creating agent {agent_config.get('name', 'Unknown')}: {e}", flush=True)


    # Create the orchestrator
    try:
        # Orchestrator doesn't need the retriever itself, agents do
        orchestrator = DebateOrchestrator(
            name="The Moderator",
            debate_state=debate_state,
            agents=all_agents,
            model='dolphin-phi:latest' # Orchestrator model
        )
        print(f"Created orchestrator: {orchestrator.name}", flush=True)
    except ValueError as e:
         print(f"Failed to create orchestrator: {e}. Exiting.", flush=True)
         return


    # Run the debate
    try:
        orchestrator.run_debate(num_rebuttal_rounds=NUMBER_OF_REBUTTAL_ROUNDS)
    except Exception as e:
        print(f"\nAn error occurred during the debate execution: {e}", flush=True)

if __name__ == "__main__":
    main()