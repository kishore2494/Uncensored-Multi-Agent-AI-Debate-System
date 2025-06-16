from config import DEBATE_TOPIC, AGENTS_CONFIG, NUMBER_OF_REBUTTAL_ROUNDS
from debate_state import DebateState
from agents import DebateOrchestrator, AffirmativeAgent, NegativeAgent, JudgeAgent, Agent

# Mapping from config type string to Agent class
AGENT_TYPE_MAP = {
    'AffirmativeAgent': AffirmativeAgent,
    'NegativeAgent': NegativeAgent,
    'JudgeAgent': JudgeAgent,
    # Add other agent types here if you create them
}

def create_agent_instance(agent_config: dict) -> Agent:
    """Creates an agent instance based on configuration."""
    agent_type_str = agent_config['type']
    agent_name = agent_config['name']
    agent_model = agent_config.get('model') # Use .get for optional model key

    agent_class = AGENT_TYPE_MAP.get(agent_type_str)
    if not agent_class:
        raise ValueError(f"Unknown agent type specified in config: {agent_type_str}")

    # Instantiate the agent
    if agent_model:
        return agent_class(name=agent_name, model=agent_model)
    else:
        return agent_class(name=agent_name) # Use default model from config

def main():
    """Sets up and runs the multi-agent debate."""
    debate_state = DebateState(topic=DEBATE_TOPIC)

    # Create agent instances from config
    all_agents = []
    for agent_config in AGENTS_CONFIG:
        # Check if optional agent should be included
        if agent_config.get('optional', False) and not True: # Set to False to exclude optional agents
             continue # Skip this agent if optional and we choose not to include them
        try:
            agent_instance = create_agent_instance(agent_config)
            all_agents.append(agent_instance)
            print(f"Created agent: {agent_instance.name} ({agent_instance.role_type}) using model {agent_instance.model}")
        except ValueError as e:
            print(f"Skipping agent configuration due to error: {e}")
        except Exception as e:
             print(f"An unexpected error occurred creating agent {agent_config.get('name', 'Unknown')}: {e}")


    # Create the orchestrator
    # The orchestrator is special, it needs the list of other agents
    # We create it last after all other agents are instantiated
    try:
        orchestrator = DebateOrchestrator(
            name="The Moderator", # Orchestrator's name
            debate_state=debate_state,
            agents=all_agents, # Pass the list of participating agents
            model='dolphin-phi:latest' # Orchestrator can use a model too, though its LLM use is minimal here
        )
        print(f"Created orchestrator: {orchestrator.name}")
    except ValueError as e:
         print(f"Failed to create orchestrator: {e}. Exiting.")
         return # Cannot run debate without required agents


    # Run the debate
    try:
        orchestrator.run_debate(num_rebuttal_rounds=NUMBER_OF_REBUTTAL_ROUNDS)
    except Exception as e:
        print(f"\nAn error occurred during the debate execution: {e}")

if __name__ == "__main__":
    main()
