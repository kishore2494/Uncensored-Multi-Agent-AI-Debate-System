# agents.py

# Import necessary libraries
import ollama
import time
import random
import queue
import concurrent.futures
from typing import Union # Import Union for type hinting

# Import necessary components for RAG context formatting and type hinting
from langchain.schema import Document
from langchain_core.retrievers import BaseRetriever # Type hint for retriever object

# Import configuration settings and debate state
from config import (
    AGENT_SYSTEM_PROMPTS, STAGE_PROMPTS, DEFAULT_MODEL, SUMMARY_MODEL,
    DEBATE_TOPIC, PROMPT_EXAMPLES, SUMMARY_PROMPT_TEMPLATE,
    MAX_TOKENS_PER_STAGE, MAX_SUMMARY_TOKENS,
    ENABLE_RAG, RETRIEVER_K # Ensure RETRIEVER_K is imported
)
from debate_state import DebateState # Import DebateState for type hinting


# --- Base Agent Class ---
class Agent:
    """Base class for all agents in the system, handling Ollama interaction."""
    # __init__ signature: 2 positional (self, name, role_type), then keyword-only (*)
    def __init__(self, name: str, role_type: str, *, model: str = DEFAULT_MODEL, retriever: Union[BaseRetriever, None] = None, agent_photo: Union[str, None] = None):
        self.name = name
        self.role_type = role_type
        self.model = model
        self.retriever = retriever # This can be None if not provided
        self.agent_photo = agent_photo # Path to the agent's photo

        # Get system prompt from config
        self.system_prompt_template = AGENT_SYSTEM_PROMPTS.get(role_type)
        if self.system_prompt_template is None:
             self.system_prompt = ""
             # Removed Streamlit warning print here
        else:
             try:
                # Format system prompt with topic if template uses {topic}
                self.system_prompt = self.system_prompt_template.format(topic=DEBATE_TOPIC)
             except KeyError:
                 # Use template directly if it doesn't use {topic}
                 self.system_prompt = self.system_prompt_template


    # Method to interact with the LLM (Ollama)
    # Takes user_prompt, stage (for examples/tokens), max_tokens, and retrieved_context
    def generate_response(self, user_prompt: str, stage: Union[str, None] = None, max_tokens: int = -1, retrieved_context: str = "") -> str:
        """Sends a prompt to the Ollama model and returns the raw response text."""

        messages = []
        if self.system_prompt:
             messages.append({'role': 'system', 'content': self.system_prompt})

        # Add few-shot examples if available for this stage
        if stage and stage in PROMPT_EXAMPLES:
            # Format examples to include the context placeholder if needed
            formatted_examples = []
            for example in PROMPT_EXAMPLES[stage]:
                 formatted_example = {'role': example['role']}
                 if example['role'] == 'user':
                     # Include the context placeholder in the example user prompt if RAG is conceptually used here
                     formatted_example['content'] = "Relevant information from knowledge base:\n\n[Context Placeholder]\n\n" + example['content']
                 else: # assistant role
                     # Assistant example response doesn't include the placeholder
                     formatted_example['content'] = example['content']
                 # CORRECTED: Append the single formatted_example to the list
                 formatted_examples.append(formatted_example)

            messages.extend(formatted_examples) # Add the list of formatted examples


        # Add the actual user prompt for the current turn
        full_user_prompt = user_prompt
        # Add retrieved context to the actual user prompt if RAG is enabled and context is provided
        if retrieved_context and ENABLE_RAG:
             full_user_prompt = f"Relevant information from knowledge base:\n\n{retrieved_context}\n\n" + user_prompt


        messages.append({'role': 'user', 'content': full_user_prompt})

        # Set Ollama options (like max_tokens)
        options = {}
        if max_tokens > 0:
            options['num_predict'] = max_tokens

        try:
            # Make the Ollama chat call
            response = ollama.chat(model=self.model, messages=messages, stream=False, options=options)
            return response['message']['content'].strip()
        except ollama.ResponseError as e:
            # Return an error message string on Ollama failure
            return f"ERROR: Agent failed to generate response due to Ollama error: {e}"
        except Exception as e:
             # Return an error message string on other exceptions
             return f"ERROR: Agent failed due to an unexpected error: {e}"

    # The base Agent class does NOT implement the 'act' method.
    # Subclasses that need to participate in a debate turn MUST implement their own 'act' method.
    # Keeping a placeholder here to avoid errors when checking if method exists
    def act(self, debate_state: DebateState, stage: str, *args, **kwargs):
        """Placeholder act method. Should be implemented by subclasses."""
        raise NotImplementedError(f"Agent type '{self.role_type}' must implement 'act' method.")


# --- Base Class for Debating Agents ---
# Inherits from Agent
class DebateAgent(Agent):
    """Base class for debating agents (Affirmative/Negative)."""
    # __init__ signature: 3 positional (self, name, role_type, stance), then keyword-only (*)
    def __init__(self, name: str, role_type: str, stance: str, *, model: str = DEFAULT_MODEL, retriever: Union[BaseRetriever, None] = None, agent_photo: Union[str, None] = None):
        # Call the parent Agent's __init__. Pass name and role_type positionally, then keywords.
        # Stance is specific to DebateAgent, not passed to Agent parent.
        super().__init__(name, role_type, model=model, retriever=retriever, agent_photo=agent_photo)
        self.stance = stance # Store the agent's stance ('Affirmative' or 'Negative')

    # THIS IS THE ACT METHOD FOR ALL DEBATING AGENTS (Affirmative and Negative inherit this)
    # It handles retrieving RAG context and formatting the prompt for debate stages.
    def act(self, debate_state: DebateState, stage: str, debate_summary: Union[str, None] = None) -> str:
        """Generates an argument based on the debate stage, summary, and retrieved context."""
        # Get the prompt template for the current stage from config
        prompt_template = STAGE_PROMPTS.get(stage)
        if not prompt_template:
            return f"ERROR: Unknown debate stage '{stage}'"

        retrieved_context = ""
        # Define which stages require RAG retrieval
        stages_using_rag = ['opening_statement', 'rebuttal', 'closing_statement']
        # Only retrieve if RAG is enabled, retriever is available for this agent, and the current stage uses RAG
        if ENABLE_RAG and self.retriever and stage in stages_using_rag:
            # Formulate a query for the retriever based on debate context and agent's task
            query = f"Provide information relevant to debating the topic: '{debate_state.topic}' from the {self.stance} perspective."
            if debate_summary and stage != 'opening_statement':
                 # If in rebuttal, query might also be based on recent points from summary
                 query += f" Specifically, provide information to rebut points made by the opposing side related to: {debate_summary[:200]}..." # Add part of summary to query
            elif stage == 'closing_statement' and debate_summary:
                 query += f" Specifically, provide information supporting key {self.stance} arguments summarized as: {debate_summary[:200]}..."

            # print(f"--- {self.name} ({self.role_type}) querying KB for stage '{stage}' with query: {query[:100]}... ---", flush=True) # Removed Streamlit print
            try:
                # Retrieve top K documents using the retriever
                # k is configured in rag_pipeline/config.py and passed when retriever is created
                relevant_docs = self.retriever.get_relevant_documents(query)
                if relevant_docs:
                     # Format retrieved documents into a string for the prompt
                     retrieved_context = "Relevant Information:\n" + "\n---\n".join([f"Source: {doc.metadata.get('source', 'N/A')}\nContent: {doc.page_content}" for doc in relevant_docs]) + "\n---\n"
                     # print(f"--- Retrieved {len(relevant_docs)} documents for {self.name} ---", flush=True) # Removed Streamlit print
                else:
                     # print(f"--- No relevant documents found for {self.name}'s query ---", flush=True) # Removed Streamlit print
                     pass # No context found

            except Exception as e:
                 # Return an internal error indicator if KB retrieval fails
                 # print(f"Error during KB retrieval for {self.name}: {e}", flush=True) # Removed Streamlit print
                 retrieved_context = f"ERROR_KB_RETRIEVAL: {e}"


        # --- Format the user prompt for the LLM ---
        prompt_args = {
             'topic': debate_state.topic,
             # Pass the retrieved_context string. If empty, the template will handle the placeholder.
             'retrieved_context': retrieved_context if retrieved_context else "[No relevant information found from knowledge base.]\n\n"
        }
        # Only add summary to prompt_args if the stage template uses the {summary} placeholder AND summary is provided
        # The prompt template for rebuttal and closing statements uses {summary}
        if stage in ['rebuttal', 'closing_statement'] and debate_summary is not None:
             prompt_args['summary'] = debate_summary
        # Note: JudgeAgent overrides act, its prompt only needs summary and doesn't use RAG context


        # Format the user prompt text using the appropriate template and the collected arguments
        try: # Add try-except for prompt formatting errors
             user_prompt_text = prompt_template.format(**prompt_args)
        except KeyError as e:
             return f"ERROR_PROMPT_FORMAT: Missing key in prompt args: {e}. Prompt template: {prompt_template}"
        except Exception as e:
             return f"ERROR_PROMPT_FORMAT: An unexpected error occurred during prompt formatting: {e}. Prompt template: {prompt_template}"


        # Get the max tokens limit for this specific stage from config
        max_tokens = MAX_TOKENS_PER_STAGE.get(stage, -1)

        # Call the generate_response method from the parent Agent class
        # Pass the formatted prompt, stage, max_tokens, and the retrieved_context string
        argument = self.generate_response(user_prompt_text, stage=stage, max_tokens=max_tokens, retrieved_context=retrieved_context)

        return argument


# --- Specific Debating Agent Classes ---
# Inherit from DebateAgent (and thus inherit the act method from DebateAgent)

class AffirmativeAgent(DebateAgent):
    """Agent arguing for the debate motion."""
    # __init__ signature: 1 positional (self, name), then keyword-only (*)
    def __init__(self, name: str, *, model: str = DEFAULT_MODEL, retriever: Union[BaseRetriever, None] = None, agent_photo: Union[str, None] = None):
        # Call DebateAgent's __init__. Pass name positionally, role_type and stance positionally, then keywords.
        # DebateAgent.__init__ expects (name, role_type, stance) positionally
        super().__init__(name, 'AffirmativeAgent', 'Affirmative', model=model, retriever=retriever, agent_photo=agent_photo)


class NegativeAgent(DebateAgent):
    """Agent arguing against the debate motion."""
    # __init__ signature: 1 positional (self, name), then keyword-only (*)
    def __init__(self, name: str, *, model: str = DEFAULT_MODEL, retriever: Union[BaseRetriever, None] = None, agent_photo: Union[str, None] = None):
        # Call DebateAgent's __init__. Pass name positionally, role_type and stance positionally, then keywords.
         super().__init__(name, 'NegativeAgent', 'Negative', model=model, retriever=retriever, agent_photo=agent_photo)


# --- Judge Agent Class ---
# Inherits from Agent (does NOT inherit from DebateAgent, has its own act method)
class JudgeAgent(Agent):
    """Agent providing analysis at the end of the debate."""
    # __init__ signature: 1 positional (self, name), then keyword-only (*)
    def __init__(self, name: str, *, model: str = DEFAULT_MODEL, retriever: Union[BaseRetriever, None] = None, agent_photo: Union[str, None] = None):
        # Call the parent Agent's __init__. Pass name and role_type positionally, then keywords.
        # Judge doesn't use RAG for its output generation task, so we pass retriever=None to Agent init
        super().__init__(name, 'JudgeAgent', model=model, retriever=None, agent_photo=agent_photo)

    # THIS IS THE ACT METHOD SPECIFICALLY FOR THE JUDGE AGENT
    # It overrides the base Agent.act (which raises NotImplementedError)
    # It handles getting summary and formatting prompt for judge analysis.
    def act(self, debate_state: DebateState, stage: str, debate_summary: Union[str, None] = None) -> str:
        """Analyzes the debate history and provides commentary based on summary."""
        # Judge's act method only needs the stage name 'judge_analysis' and the summary
        # Check if the stage is correct, although Orchestrator should call with 'judge_analysis'
        if stage != 'judge_analysis':
             # This should not happen if Orchestrator is correct
             return f"ERROR: Judge act called with incorrect stage: '{stage}'"

        # Get the prompt template for judge analysis
        prompt_template = STAGE_PROMPTS.get(stage)
        if not prompt_template:
            return f"ERROR: Judge analysis prompt template not found."

        # Judge prompt specifically uses the summary, requires summary to be provided
        if debate_summary is None:
             return "ERROR: Judge could not get summary."

        # Format the user prompt for the LLM using the template and summary
        try: # Add try-except for prompt formatting errors
             user_prompt = prompt_template.format(summary=debate_summary)
        except KeyError as e:
             return f"ERROR_PROMPT_FORMAT: Missing key in prompt args: {e}. Prompt template: {prompt_template}"
        except Exception as e:
             return f"ERROR_PROMPT_FORMAT: An unexpected error occurred during prompt formatting: {e}. Prompt template: {prompt_template}"


        # Get the max tokens limit for the judge stage
        max_tokens = MAX_TOKENS_PER_STAGE.get(stage, -1)

        # Call generate_response. Judge's task doesn't involve retrieving RAG context
        # for its output, so retrieved_context is an empty string.
        analysis = self.generate_response(user_prompt, stage=stage, max_tokens=max_tokens, retrieved_context="")

        return analysis


# --- Debate Orchestrator Class ---
# Inherits from Agent (Orchestrator is a type of agent in the system)
class DebateOrchestrator(Agent):
    """Manages the flow of the debate."""
    # __init__ signature: 3 positional (self, name, debate_state, agents), then keyword-only (*)
    def __init__(self, name: str, debate_state: DebateState, agents: list[Agent], *, model: str = 'llama3'):
         # Call parent Agent's __init__. Pass name positionally, role_type ('DebateOrchestrator') positionally, then keywords.
         # Orchestrator doesn't need retriever or agent_photo for its base Agent identity
         super().__init__(name, 'DebateOrchestrator', model=model, retriever=None, agent_photo=None)

         # Store positional arguments here
         self.debate_state = debate_state
         self.agents = agents # Store the full list of agent instances

         # Separate agents by role type
         self.affirmative_agents = [a for a in self.agents if isinstance(a, AffirmativeAgent)]
         self.negative_agents = [a for a in self.agents if isinstance(a, NegativeAgent)]
         self.judge_agent = next((a for a in self.agents if isinstance(a, JudgeAgent)), None)


         if not self.affirmative_agents or not self.negative_agents:
             # Removed Streamlit error here, raise Python ValueError
             raise ValueError("Must have at least one Affirmative and one Negative agent configured.")

         self.turn_delay_seconds = 1 # Delay between turns
         self.summary_model = SUMMARY_MODEL
         self.summary_system_prompt = AGENT_SYSTEM_PROMPTS.get('Summarizer').format() if AGENT_SYSTEM_PROMPTS.get('Summarizer') else ""

    # Method to generate a summary of debate history (used internally by orchestrator)
    def _generate_summary(self) -> str:
        """Generates a summary of the current debate history using an LLM."""
        # print messages to console
        print("\n--- Orchestrator is summarizing debate history... ---", flush=True)

        history_text = self.debate_state.get_history_text()
        if not history_text.strip() or "-- Debate History --\nNo arguments yet.\n\n-- End of History --" in history_text:
             return "No debate history to summarize yet."

        user_prompt = SUMMARY_PROMPT_TEMPLATE.format(debate_history=history_text)

        messages = []
        if self.summary_system_prompt:
            messages.append({'role': 'system', 'content': self.summary_system_prompt})
        messages.append({'role': 'user', 'content': user_prompt})

        options = {}
        if MAX_SUMMARY_TOKENS > 0:
            options['num_predict'] = MAX_SUMMARY_TOKENS

        try:
            # Use the generate_response method from the base Agent class for summary generation
            # The orchestrator is an Agent, so it can call its own generate_response
            summary = self.generate_response(user_prompt, stage='summary', max_tokens=MAX_SUMMARY_TOKENS, retrieved_context="")
            print("--- Summary Generated ---", flush=True)
            return summary
        except Exception as e:
            print(f"Error during summarization from Ollama: {e}", flush=True)
            return f"ERROR: Failed to generate summary due to error: {e}"


    # Main method to run the debate flow
    # This is a generator function that yields events back to the UI
    def run_debate(self, num_rebuttal_rounds: int):
        """Runs the full debate sequence, yielding output for the UI."""
        # Yield messages for the UI
        yield {"type": "status", "message": "Starting Debate...", "topic": self.debate_state.topic}
        yield {"type": "stage", "stage_name": "Opening Statements"}

        # Opening Statements Stage
        # Affirmative Team's Opening Statements
        for agent in self.affirmative_agents:
             # Yield status indicating who is speaking
             yield {"type": "status", "message": f"{agent.name} ({agent.role_type}) speaking..."}
             # Call the agent's act method
             argument_text = agent.act(self.debate_state, 'opening_statement', debate_summary=None) # Opening needs no summary
             # Add argument to debate history if not an error
             if not argument_text.startswith("ERROR:"):
                 self.debate_state.add_argument(agent.name, agent.role_type, argument_text)
             # Yield the argument for the UI
             yield {"type": "argument", "agent_name": agent.name, "agent_role": agent.role_type, "argument": argument_text, "agent_photo": agent.agent_photo}
             # Optional: Add a small pause between agents within a stage
             # time.sleep(self.turn_delay_seconds)

        # Negative Team's Opening Statements
        for agent in self.negative_agents:
             yield {"type": "status", "message": f"{agent.name} ({agent.role_type}) speaking..."}
             argument_text = agent.act(self.debate_state, 'opening_statement', debate_summary=None)
             if not argument_text.startswith("ERROR:"):
                 self.debate_state.add_argument(agent.name, agent.role_type, argument_text)
             yield {"type": "argument", "agent_name": agent.name, "agent_role": agent.role_type, "argument": argument_text, "agent_photo": agent.agent_photo}
             # Optional: Add a small pause
             # time.sleep(self.turn_delay_seconds)


        # Rebuttal Rounds Stage
        for i in range(num_rebuttal_rounds):
            yield {"type": "stage", "stage_name": f"--- Rebuttal Round {i+1} ---"}

            # Summarize debate history before each rebuttal round
            yield {"type": "status", "message": "Orchestrator summarizing debate..."}
            self.current_summary = self._generate_summary()
            yield {"type": "status", "message": "Summary Generated."}

            # Check if summarization failed
            if isinstance(self.current_summary, str) and self.current_summary.startswith("ERROR:"):
                 yield {"type": "status", "message": f"Skipping remaining debate due to summarization error: {self.current_summary}"}
                 break # Stop the debate loop if summarization fails

            # Affirmative Team's Rebuttals
            for agent in self.affirmative_agents:
                 yield {"type": "status", "message": f"{agent.name} ({agent.role_type}) speaking..."}
                 # Pass the current debate summary to the agent's act method
                 argument_text = agent.act(self.debate_state, 'rebuttal', debate_summary=self.current_summary)
                 if not argument_text.startswith("ERROR:"):
                     self.debate_state.add_argument(agent.name, agent.role_type, argument_text)
                 yield {"type": "argument", "agent_name": agent.name, "agent_role": agent.role_type, "argument": argument_text, "agent_photo": agent.agent_photo}
                 # Optional: Add a pause
                 # time.sleep(self.turn_delay_seconds)

            # Negative Team's Rebuttals
            for agent in self.negative_agents:
                 yield {"type": "status", "message": f"{agent.name} ({agent.role_type}) speaking..."}
                 argument_text = agent.act(self.debate_state, 'rebuttal', debate_summary=self.current_summary)
                 if not argument_text.startswith("ERROR:"):
                     self.debate_state.add_argument(agent.name, agent.role_type, argument_text)
                 yield {"type": "argument", "agent_name": agent.name, "agent_role": agent.role_type, "argument": argument_text, "agent_photo": agent.agent_photo}
                 # Optional: Add a pause
                 # time.sleep(self.turn_delay_seconds)


        # Closing Statements Stage
        yield {"type": "stage", "stage_name": "Closing Statements"}

        # Summarize debate history before closing statements
        yield {"type": "status", "message": "Orchestrator summarizing debate..."}
        self.current_summary = self._generate_summary()
        yield {"type": "status", "message": "Summary Generated."}

        # Check if summarization failed
        if isinstance(self.current_summary, str) and self.current_summary.startswith("ERROR:"):
             yield {"type": "status", "message": f"Skipping closing statements and judge due to summarization error: {self.current_summary}"}
        else:
            # Affirmative Team's Closing Statements
            for agent in self.affirmative_agents:
                 yield {"type": "status", "message": f"{agent.name} ({agent.role_type}) speaking..."}
                 # Pass the current debate summary
                 argument_text = agent.act(self.debate_state, 'closing_statement', debate_summary=self.current_summary)
                 if not argument_text.startswith("ERROR:"):
                     self.debate_state.add_argument(agent.name, agent.role_type, argument_text)
                 yield {"type": "argument", "agent_name": agent.name, "agent_role": agent.role_type, "argument": argument_text, "agent_photo": agent.agent_photo}
                 # Optional: Add a pause
                 # time.sleep(self.turn_delay_seconds)

            # Negative Team's Closing Statements
            for agent in self.negative_agents:
                 yield {"type": "status", "message": f"{agent.name} ({agent.role_type}) speaking..."}
                 # Pass the current debate summary
                 argument_text = agent.act(self.debate_state, 'closing_statement', debate_summary=self.current_summary)
                 if not argument_text.startswith("ERROR:"):
                     self.debate_state.add_argument(agent.name, agent.role_type, argument_text)
                 yield {"type": "argument", "agent_name": agent.name, "agent_role": agent.role_type, "argument": argument_text, "agent_photo": agent.agent_photo}
                 # Optional: Add a pause
                 # time.sleep(self.turn_delay_seconds)


            # Judge Analysis Stage (Optional)
            if self.judge_agent:
                yield {"type": "stage", "stage_name": "Judge Analysis"}
                # Summarize debate history for the judge's analysis
                yield {"type": "status", "message": "Orchestrator summarizing debate for Judge..."}
                final_summary = self._generate_summary() # Generate final summary
                yield {"type": "status", "message": "Summary Generated for Judge."}

                # Check if summarization failed for the judge
                if isinstance(final_summary, str) and final_summary.startswith("ERROR:"):
                     yield {"type": "status", "message": f"Skipping judge analysis due to summarization error: {final_summary}"}
                else:
                     # Call the Judge agent's act method, passing the final summary
                    analysis = self.judge_agent.act(self.debate_state, stage='judge_analysis', debate_summary=final_summary)
                    # Yield the judge's analysis argument
                    yield {"type": "argument", "agent_name": self.judge_agent.name, "agent_role": self.judge_agent.role_type, "argument": analysis, "agent_photo": self.judge_agent.agent_photo}


        # End of Debate
        yield {"type": "status", "message": "Debate Concluded."}