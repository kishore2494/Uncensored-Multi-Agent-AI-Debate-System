# --- Debate Configuration ---
DEBATE_TOPIC = "Should autonomous vehicles be implemented on a large scale within the next decade?"
NUMBER_OF_REBUTTAL_ROUNDS = 2 # Number of times each side gets to respond after opening statements

# --- Ollama Model Configuration ---
DEFAULT_MODEL = 'dolphin-phi:latest'
SUMMARY_MODEL = DEFAULT_MODEL 

# --- Agent Configuration ---
# Define each agent participating in the debate
AGENTS_CONFIG = [
    {'type': 'AffirmativeAgent', 'name': 'Affirmative Agent 1', 'model': DEFAULT_MODEL},
    {'type': 'AffirmativeAgent', 'name': 'Affirmative Agent 2', 'model': DEFAULT_MODEL},
    {'type': 'NegativeAgent', 'name': 'Negative Agent 1', 'model': DEFAULT_MODEL},
    {'type': 'NegativeAgent', 'name': 'Negative Agent 2', 'model': DEFAULT_MODEL},
    {'type': 'JudgeAgent', 'name': 'Debate Judge', 'model': DEFAULT_MODEL, 'optional': True}
]

# --- Agent Prompts (Base Instructions) ---
AGENT_SYSTEM_PROMPTS = {
    'DebateOrchestrator': (
        "You are a neutral debate moderator. Your role is to introduce the topic, "
        "call on speakers, maintain order, and conclude the debate. Do not offer your own opinions "
        "or arguments. Just manage the flow and report the arguments presented by the agents."
    ),
    # Add a system prompt for the summarizer functionality (used by Orchestrator)
    'Summarizer': (
        "You are a neutral summarization assistant. Your task is to read the provided debate history "
        "and produce a concise, impartial summary of the key arguments made by each side. "
        "Do not add external information or offer opinions. Focus on capturing the main points from both the Affirmative and Negative teams."
    ),
    'DebateAgent': ( # Base prompt for both Affirmative and Negative
        "You are an AI debater participating in a structured debate. "
        "Your goal is to present compelling arguments for your assigned stance on the topic, "
        "and respectfully rebut the points made by the opposing side. "
        "Be clear, logical, and focus on the arguments. Follow the format shown in the examples "
        "and base your responses on the provided debate summary. Keep your response within the requested token limit." # Added reference to token limit
    ),
    'AffirmativeAgent': (
        "You are an AI debater on the AFFIRMATIVE team. Your task is to argue STRONGLY in favor of the debate motion: '{topic}'. "
        "Present arguments supporting this position and defend it against the Negative team's points. "
        "Remember the base instructions for an AI debater."
    ),
    'NegativeAgent': (
         "You are an AI debater on the NEGATIVE team. Your task is to argue STRONGLY against the debate motion: '{topic}'. "
         "Present arguments opposing this position and defend it against the Affirmative team's points. "
         "Remember the base instructions for an AI debater."
    ),
     'JudgeAgent': (
        "You are an AI judge observing a debate on the topic: '{topic}'. "
        "Your sole role is to provide a brief, impartial summary of the key points made by each side *based only on the provided debate summary*. "
        "Do not add external information, offer opinions, or declare a winner. "
        "Summarize the arguments as shown in the example, using a list format. Keep your response within the requested token limit." # Added reference to token limit
    )
}

# --- Specific Prompts for Debate Stages ---
# Explicitly asking for a LIST of points and using a summary.
STAGE_PROMPTS = {
    'opening_statement': (
        "Deliver your opening statement for the topic: '{topic}'. "
        "Provide your main arguments as a numbered list of 3 to 4 concise points."
    ),
    # Modified prompts to refer to the '{summary}' placeholder
    'rebuttal': (
        "Here is a summary of the debate history so far:\n\n{summary}\n\n" # Use summary here
        "It is your turn to offer a rebuttal. Respond to the points made by the opposing team. "
        "Counter their claims and defend your own position based on the summary above. " # Reference summary
        "Provide your rebuttal points as a numbered list of 2 to 3 concise points."
    ),
    'closing_statement': (
         "Here is a summary of the debate history so far:\n\n{summary}\n\n" # Use summary here
        "Deliver your closing statement. Summarize your main arguments and explain why your stance on the topic is the most compelling, referencing points in the summary if helpful. " # Reference summary
        "Provide your summary points as a numbered list of 2 to 3 concise points."
    ),
    'judge_analysis': (
        "Here is a summary of the debate history:\n\n{summary}\n\n" # Use summary here
        "Provide a *brief*, impartial summary of the key arguments from the Affirmative team and the key arguments from the Negative team based *only* on the summary above. " # Reference summary
        "Format your response exactly as shown in the example, using headings and bullet points."
    )
}

# --- Prompt Template for Summarization ---
SUMMARY_PROMPT_TEMPLATE = (
    "Please provide a concise, neutral summary of the following debate history. "
    "Include the main arguments and counter-arguments presented by both the Affirmative and Negative teams:\n\n"
    "{debate_history}" # The summarizer needs the full history as input
    "\n\nProvide the summary in a few sentences or a short paragraph." # Added instruction for conciseness
)

# --- Max Tokens Configuration ---
# Define the maximum number of tokens each agent/stage can output.
# Adjust these values based on desired verbosity and model capability.
# These are rough estimates; you might need to experiment.
MAX_TOKENS_PER_STAGE = {
    'opening_statement': 200, # Enough for 3-4 concise points + intro/outro
    'rebuttal': 150,        # Enough for 2-3 concise points + intro/outro
    'closing_statement': 150,       # Enough for 2-3 concise points + intro/outro
    'judge_analysis': 200   # Enough for list of points for both sides
}

MAX_SUMMARY_TOKENS = 100 # Max tokens for the debate summary

# --- Few-Shot Examples ---
# ... (Keep PROMPT_EXAMPLES as is, they guide format, not strictly token count)
PROMPT_EXAMPLES = {
    'opening_statement': [
        {'role': 'user', 'content': "Deliver your opening statement for the topic: 'Should pineapple belong on pizza?'. Provide your main arguments as a numbered list of 3 to 4 concise points."},
        {'role': 'assistant', 'content': (
            "Here is my opening statement:\n"
            "1.  Pineapple adds a delicious sweet and tangy contrast to savory toppings.\n"
            "2.  Its juiciness helps prevent the pizza from being too dry.\n"
            "3.  It's a popular topping enjoyed by millions worldwide, indicating broad appeal.\n"
            "4.  Pairing fruit with savory dishes is common in many cuisines."
        )}
    ],
     'rebuttal': [
        {'role': 'user', 'content': (
            "Here is a summary of the debate history so far:\n\n"
            "Summary: Affirmative argued for safety, efficiency. Negative argued against based on risks, job losses. Most recently, Negative claimed AV tech isn't ready and job losses are certain.\n\n" # Example Summary Text
            "It is your turn to offer a rebuttal... Based on the summary, respond to the points made by the opposing side in their most recent arguments. Provide your rebuttal points as a numbered list of 2 to 3 concise points."
        )},
        {'role': 'assistant', 'content': (
            "Here is my rebuttal:\n"
            "1.  The claim that AV tech isn't ready ignores the rapid advancements and testing already underway by leading companies.\n"
            "2.  While job displacement is a concern, history shows technological shifts create new jobs, and focus should be on transition support, not halting progress."
        )}
    ],
     'closing_statement': [
        {'role': 'user', 'content': (
            "Here is a summary of the debate history so far:\n\n"
             "Summary: Affirmative argued safety, efficiency, accessibility benefits. Negative countered with safety risks, job losses, infrastructure costs. Rebuttals exchanged points on tech readiness, economic transition, and regulatory progress.\n\n" # Example Summary Text
            "Deliver your closing statement... Provide your summary points as a numbered list of 2 to 3 concise points."
        )},
        {'role': 'assistant', 'content': (
            "In closing, I reiterate my main points:\n"
            "1.  The potential safety and efficiency gains from AVs are transformative.\n"
            "2.  While challenges exist, they are surmountable with continued development and thoughtful policy, paving the way for significant societal benefits."
        )}
    ],
    'judge_analysis': [
        {'role': 'user', 'content': (
             "Here is a summary of the debate history:\n\n"
             "Summary: Affirmative highlighted safety from reducing human error, efficiency in traffic, and accessibility. Negative emphasized current safety risks, potential job losses, and infrastructure/regulatory hurdles. Rebuttals debated technological maturity and economic transition.\n\n" # Example Summary Text
            "Provide a brief, impartial summary of the key arguments from the Affirmative team and the key arguments from the Negative team based *only* on the summary above. Format your response exactly as shown in the example, using headings and bullet points."
        )},
        {'role': 'assistant', 'content': (
            "Affirmative Key Points:\n"
            "- AVs enhance safety by eliminating human error.\n"
            "- They improve efficiency and reduce congestion.\n"
            "- They offer increased accessibility.\n\n"
            "Negative Key Points:\n"
            "- AV technology is not yet sufficiently safe or reliable.\n"
            "- Large-scale implementation will cause significant job losses.\n"
            "- Infrastructure and regulatory challenges are major hurdles."
        )}
    ]
}
