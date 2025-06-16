import ollama
import time
from config import (
    AGENT_SYSTEM_PROMPTS, STAGE_PROMPTS, DEFAULT_MODEL, SUMMARY_MODEL,
    DEBATE_TOPIC, PROMPT_EXAMPLES, SUMMARY_PROMPT_TEMPLATE,
    MAX_TOKENS_PER_STAGE, MAX_SUMMARY_TOKENS # Import new config items
)

class Agent:
    """Base class for all agents in the system."""
    def __init__(self, name: str, role_type: str, model: str = DEFAULT_MODEL):
        self.name = name
        self.role_type = role_type
        self.model = model
        self.system_prompt_template = AGENT_SYSTEM_PROMPTS.get(role_type)
        if self.system_prompt_template is None and role_type not in ['DebateOrchestrator', 'JudgeAgent', 'Summarizer']:
             self.system_prompt_template = AGENT_SYSTEM_PROMPTS.get('DebateAgent')

        self.system_prompt = self.system_prompt_template.format(topic=DEBATE_TOPIC) if self.system_prompt_template else ""
        if not self.system_prompt and role_type not in ['DebateOrchestrator', 'Summarizer']: # Exclude orchestrator/summarizer from this warning
             print(f"Warning: No system prompt template found for role type '{self.role_type}'", flush=True)


    # Modified to accept 'stage' and 'max_tokens'
    def generate_response(self, user_prompt: str, stage: str = None, max_tokens: int = -1, context: str = "") -> str:
        """Sends a prompt to the Ollama model and returns the response."""

        messages = []
        if self.system_prompt:
             messages.append({'role': 'system', 'content': self.system_prompt})

        # Add few-shot examples if available for this stage
        if stage and stage in PROMPT_EXAMPLES:
            messages.extend(PROMPT_EXAMPLES[stage])

        # Add context if available (future KB integration point)
        # if context:
        #     messages.append({'role': 'user', 'content': f"Context: {context}\n\n"})

        # Add the actual prompt for the current turn
        messages.append({'role': 'user', 'content': user_prompt})

        # --- Ollama Options ---
        options = {}
        if max_tokens > 0:
            options['num_predict'] = max_tokens # Set the max tokens limit

        # Debugging: Print the messages being sent (can be verbose)
        # import json
        # print(f"\n--- Messages for {self.name} ({self.role_type}), Max Tokens: {max_tokens} ---")
        # print(json.dumps(messages, indent=2))
        # print("-----------------------------------------------------\n")


        try:
            print(f"--- {self.name} ({self.role_type}) is thinking using model '{self.model}', Max Tokens: {max_tokens} ---", flush=True)
            response = ollama.chat(model=self.model, messages=messages, stream=False, options=options) # Pass options
            return response['message']['content'].strip()
        except ollama.ResponseError as e:
            print(f"Error from Ollama for {self.name}: {e}", flush=True)
            return f"ERROR: Agent failed to generate response due to Ollama error: {e}"
        except Exception as e:
             print(f"An unexpected error occurred for {self.name}: {e}", flush=True)
             return f"ERROR: Agent failed due to an unexpected error: {e}"


class DebateAgent(Agent):
    """Base class for debating agents (Affirmative/Negative)."""
    def __init__(self, name: str, role_type: str, stance: str, model: str = DEFAULT_MODEL):
        super().__init__(name, role_type, model)
        self.stance = stance

    # Modified act to accept debate_summary
    def act(self, debate_state: 'DebateState', stage: str, debate_summary: str = None) -> str:
        """Generates an argument based on the debate stage and provided summary."""
        prompt_template = STAGE_PROMPTS.get(stage)
        if not prompt_template:
            return f"ERROR: Unknown debate stage '{stage}'"

        # Format the stage prompt.
        if stage in ['rebuttal', 'closing_statement', 'judge_analysis'] and debate_summary is not None:
             user_prompt = prompt_template.format(topic=debate_state.topic, summary=debate_summary)
        else: # Opening statement
             user_prompt = prompt_template.format(topic=debate_state.topic)

        # Get the max tokens for this specific stage
        max_tokens = MAX_TOKENS_PER_STAGE.get(stage, -1) # Default to -1 (no limit) if stage not found

        # Pass the stage and max_tokens when calling generate_response
        argument = self.generate_response(user_prompt, stage=stage, max_tokens=max_tokens)
        return argument

class AffirmativeAgent(DebateAgent):
    """Agent arguing for the debate motion."""
    def __init__(self, name: str, model: str = DEFAULT_MODEL):
        super().__init__(name, 'AffirmativeAgent', 'Affirmative', model)

class NegativeAgent(DebateAgent):
    """Agent arguing against the debate motion."""
    def __init__(self, name: str, model: str = DEFAULT_MODEL):
        super().__init__(name, 'NegativeAgent', 'Negative', model)

class JudgeAgent(Agent):
    """Agent providing analysis at the end of the debate."""
    def __init__(self, name: str, model: str = DEFAULT_MODEL):
        super().__init__(name, 'JudgeAgent', model)

    # Modified act to accept debate_summary
    def act(self, debate_state: 'DebateState', debate_summary: str = None) -> str:
        """Analyzes the debate history and provides commentary based on summary."""
        stage = 'judge_analysis'
        prompt_template = STAGE_PROMPTS.get(stage)
        if not prompt_template:
            return "ERROR: Judge analysis prompt template not found."

        if debate_summary is None:
             return "ERROR: Judge needs a debate summary but none was provided."

        user_prompt = prompt_template.format(summary=debate_summary)

        # Get the max tokens for the judge stage
        max_tokens = MAX_TOKENS_PER_STAGE.get(stage, -1)

        # Pass the stage and max_tokens when calling generate_response
        analysis = self.generate_response(user_prompt, stage=stage, max_tokens=max_tokens)
        return analysis

class DebateOrchestrator(Agent):
    """Manages the flow of the debate."""
    def __init__(self, name: str, debate_state: 'DebateState', agents: list[Agent], model: str = DEFAULT_MODEL):
        super().__init__(name, 'DebateOrchestrator', model)
        self.debate_state = debate_state
        self.affirmative_agents = [a for a in agents if isinstance(a, AffirmativeAgent)]
        self.negative_agents = [a for a in agents if isinstance(a, NegativeAgent)]
        self.judge_agent = next((a for a in agents if isinstance(a, JudgeAgent)), None)

        if not self.affirmative_agents or not self.negative_agents:
            raise ValueError("Must have at least one Affirmative and one Negative agent configured.")

        self.turn_delay_seconds = 7 # Delay between agents speaking
        self.summary_model = SUMMARY_MODEL
        # System prompt for the summarizer functionality (used internally by orchestrator)
        self.summary_system_prompt = AGENT_SYSTEM_PROMPTS.get('Summarizer').format() # Get summarizer system prompt
        self.current_summary = None # To store the latest summary

    def _generate_summary(self) -> str:
        """Generates a summary of the current debate history using an LLM."""
        print("\n--- Orchestrator is summarizing debate history... ---", flush=True)
        history_text = self.debate_state.get_history_text()
        if not history_text.strip() or history_text == f"Debate Topic: {self.debate_state.topic}\n\n-- Debate History --\nNo arguments yet.\n\n-- End of History --\n":
             return "No debate history to summarize yet."

        user_prompt = SUMMARY_PROMPT_TEMPLATE.format(debate_history=history_text)

        messages = [
            {'role': 'system', 'content': self.summary_system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]

        # --- Ollama Options for Summary ---
        options = {}
        if MAX_SUMMARY_TOKENS > 0:
             options['num_predict'] = MAX_SUMMARY_TOKENS # Set the max tokens limit for summary

        try:
            # Use the summarization model and pass options
            response = ollama.chat(model=self.summary_model, messages=messages, stream=False, options=options)
            summary = response['message']['content'].strip()
            print("--- Summary Generated ---", flush=True)
            # print(summary) # Optional: print summary for debugging
            return summary
        except ollama.ResponseError as e:
            print(f"Error during summarization from Ollama: {e}", flush=True)
            return f"ERROR: Failed to generate summary due to Ollama error: {e}"
        except Exception as e:
             print(f"An unexpected error occurred during summarization: {e}", flush=True)
             return f"ERROR: Failed to generate summary due to unexpected error: {e}"


    def run_debate(self, num_rebuttal_rounds: int):
        """Runs the full debate sequence."""
        print("--- Starting Debate ---", flush=True)
        print(f"Topic: {self.debate_state.topic}\n", flush=True)

        # Stage 1: Opening Statements (No summary needed, get max tokens from config)
        print("\n--- Opening Statements ---", flush=True)
        self._run_stage('opening_statement', self.affirmative_agents, debate_summary=None)
        self._run_stage('opening_statement', self.negative_agents, debate_summary=None)

        # Stage 2: Rebuttal Rounds
        print(f"\n--- Rebuttal Rounds ({num_rebuttal_rounds} rounds) ---", flush=True)
        for i in range(num_rebuttal_rounds):
            print(f"\n--- Round {i+1} ---", flush=True)
            # Generate summary before the round starts
            self.current_summary = self._generate_summary()
            if "ERROR:" in self.current_summary:
                 print(f"Skipping remaining debate due to summarization error: {self.current_summary}", flush=True)
                 break

            # Affirmative rebuts Negative's last points (Pass the summary)
            # Max tokens for rebuttal is handled inside the agent's act method
            self._run_stage('rebuttal', self.affirmative_agents, debate_summary=self.current_summary)
            # Negative rebuts Affirmative's last points (Pass the summary)
            self._run_stage('rebuttal', self.negative_agents, debate_summary=self.current_summary)


        # Stage 3: Closing Statements
        print("\n--- Closing Statements ---", flush=True)
        # Generate summary before closing statements
        self.current_summary = self._generate_summary()
        if "ERROR:" not in self.current_summary:
            # Max tokens for closing is handled inside the agent's act method
            self._run_stage('closing_statement', self.affirmative_agents, debate_summary=self.current_summary)
            self._run_stage('closing_statement', self.negative_agents, debate_summary=self.current_summary)
        else:
             print(f"Skipping closing statements due to summarization error: {self.current_summary}", flush=True)


        # Stage 4: Judge Analysis (Optional)
        if self.judge_agent:
            print("\n--- Judge Analysis ---", flush=True)
            # Judge needs the final summary
            final_summary = self._generate_summary()
            if "ERROR:" not in final_summary:
                # Max tokens for judge analysis is handled inside the judge agent's act method
                analysis = self.judge_agent.act(self.debate_state, debate_summary=final_summary)
                print(f"[{self.judge_agent.name} - {self.judge_agent.role_type}]:\n{analysis}\n", flush=True)
            else:
                print(f"Skipping judge analysis due to summarization error: {final_summary}", flush=True)


        print("--- Debate Concluded ---", flush=True)
        print("\n--- Full Debate Transcript ---", flush=True)
        print(self.debate_state.get_history_text(), flush=True)


    def _run_stage(self, stage: str, agents: list[DebateAgent], debate_summary: str = None):
        """Helper to run a specific stage for a list of agents."""
        for agent in agents:
            print(f"\n[{agent.role_type} - {agent.name}] speaking...", flush=True)
            # Pass the debate_summary to the agent's act method
            # Max tokens is handled inside agent.act now
            argument_text = agent.act(self.debate_state, stage, debate_summary=debate_summary)
            if not argument_text.startswith("ERROR:"):
                 self.debate_state.add_argument(agent.name, agent.role_type, argument_text)
            else:
                 print(f"ERROR: {agent.name} failed to generate response. Not adding to history.", flush=True)

            print(f"[{agent.role_type} - {agent.name}]:\n{argument_text}\n", flush=True)
            time.sleep(self.turn_delay_seconds)

# ... (debate_state.py and main.py remain the same)
