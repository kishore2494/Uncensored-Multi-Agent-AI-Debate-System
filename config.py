# config.py

# --- Debate Configuration ---
DEBATE_TOPIC = "Should autonomous vehicles be implemented on a large scale within the next decade?"
NUMBER_OF_REBUTTAL_ROUNDS = 2 # Number of times each side gets to respond after opening statements

# --- Ollama Model Configuration ---
# Using dolphin-phi:latest for debate agents and summarization
DEFAULT_MODEL = 'dolphin-phi:latest'
SUMMARY_MODEL = DEFAULT_MODEL

# --- RAG Configuration ---
# Directory containing your PDF documents
KB_DIRECTORY = "./knowledge" # Create a folder named 'knowledge' in your project directory and put PDFs there
# Path where the ChromaDB vector store will be saved/loaded
VECTOR_STORE_PATH = "./chroma_db"
# Ollama model to use for creating embeddings (e.g., nomic-embed-text)
EMBEDDING_MODEL = 'nomic-embed-text'
# Chunk size and overlap for splitting documents
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
# Flag to indicate if RAG should be enabled
ENABLE_RAG = True
RETRIEVER_K = 3 # Number of relevant documents to retrieve for RAG

# --- Agent Configuration ---
# --- Agent Configuration ---
# AGENTS_CONFIG will be built dynamically by the UI, but we keep this
# placeholder structure or a base if needed.
# Let's define the POOL of possible names and photo paths instead.
SOUTH_INDIAN_NAMES = [
    "Ramesh", "Sita", "Krishna", "Lakshmi", "Raj", "Meena",
    "Arjun", "Priya", "Vikram", "Anjali", "Gopal", "Shanti",
    "Mohan", "Lalita", "Anand", "Radha", "Vivek", "Kavitha",
    "Sanjay", "Divya"
]

# Add placeholder image paths for your agents (make sure these files exist in ./images)
# You need at least as many photos as the maximum number of agents you might field (e.g., 5 pairs + 1 judge = 11)
AGENT_PHOTO_PATHS = [
    "D:\project\mulit agent\Images\5b658e1c-adec-4304-be53-fcad1c34d816.jpg",
    "D:\project\mulit agent\Images\259d5c05-4f36-475a-877e-5bbd277e5127.jpg",
    "D:\project\mulit agent\Images\85a26cbf-d583-47c8-b7da-67cf739b43ab.jpg",
    "D:\project\mulit agent\Images\bae93a3f-099c-46c8-ad4f-77777d6a195c.jpg",
    "D:\project\mulit agent\Images\cf245133-496d-4278-a1d6-ef88a0e4d6e0.jpg",
    "D:\project\mulit agent\Images\0cced509-f35c-4f0f-88ce-1cc474a8234c.jpg",
    "D:\project\mulit agent\Images\ec22cc1c-d385-4ef8-87d3-8fd64f86d20d.jpg",
    "D:\project\mulit agent\Images\ebcb1f39-f063-45d7-a617-6e6c463fe77d.jpg",
    "D:\project\mulit agent\Images\a8bafc9b-fc5b-4e96-9f0a-51ae736a84cf.jpg",
    "D:\project\mulit agent\Images\44166327-6f69-47e4-9072-74bab30ac5ea.jpg",
    "D:\project\mulit agent\Images\j.jpg", # A specific one for the judge perhaps
]

# --- Agent Prompts (Base Instructions) ---
AGENT_SYSTEM_PROMPTS = {
    'DebateOrchestrator': (
        "You are a neutral debate moderator. Your role is to introduce the topic, "
        "call on speakers, maintain order, and conclude the debate. Do not offer your own opinions "
        "or arguments. Just manage the flow and report the arguments presented by the agents."
    ),
    'Summarizer': (
        "You are a neutral summarization assistant. Your task is to read the provided debate history "
        "and produce a concise, impartial summary of the key arguments made by each side. "
        "Do not add external information or offer opinions. Focus on capturing the main points from both the Affirmative and Negative teams."
    ),
    'DebateAgent': ( # Base prompt for both Affirmative and Negative
        "You are an AI debater participating in a structured debate. "
        "Your goal is to present compelling arguments for your assigned stance on the topic, "
        "and respectfully rebut the points made by the opposing side. "
        "Base your response on the provided debate summary and *relevant information from the knowledge base* if available. " # Added RAG instruction
        "Be clear, logical, and focus on the arguments. Follow the format shown in the examples "
        "and keep your response within the requested token limit."
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
        "Summarize the arguments as shown in the example, using a list format. Keep your response within the requested token limit." # RAG context not needed for judge summary
    )
}

# --- Specific Prompts for Debate Stages ---
STAGE_PROMPTS = {
    'opening_statement': (
        "{retrieved_context}" # Placeholder for RAG context
        "Deliver your opening statement for the topic: '{topic}'. "
        "Provide your main arguments as a numbered list of 3 to 4 concise points, referencing the provided information if relevant." # Reference RAG context
    ),
    'rebuttal': (
        "{retrieved_context}" # Placeholder for RAG context
        "Here is a summary of the debate history so far:\n\n{summary}\n\n"
        "It is your turn to offer a rebuttal. Respond to the points made by the opposing team. "
        "Counter their claims and defend your own position based on the summary and *relevant provided information*. " # Reference RAG context
        "Provide your rebuttal points as a numbered list of 2 to 3 concise points."
    ),
    'closing_statement': (
         "{retrieved_context}" # Placeholder for RAG context
         "Here is a summary of the debate history so far:\n\n{summary}\n\n"
        "Deliver your closing statement. Summarize your main arguments and explain why your stance on the topic is the most compelling, referencing points in the summary and *relevant provided information* if helpful. " # Reference RAG context
        "Provide your summary points as a numbered list of 2 to 3 concise points."
    ),
    'judge_analysis': (
        "Here is a summary of the debate history:\n\n{summary}\n\n"
        "Provide a *brief*, impartial summary of the key arguments from the Affirmative team and the key arguments from the Negative team based *only* on the summary above. "
        "Format your response exactly as shown in the example, using headings and bullet points."
    )
}

# --- Prompt Template for Summarization ---
SUMMARY_PROMPT_TEMPLATE = (
    "Please provide a concise, neutral summary of the following debate history. "
    "Include the main arguments and counter-arguments presented by both the Affirmative and Negative teams:\n\n"
    "{debate_history}"
    "\n\nProvide the summary in a few sentences or a short paragraph."
)

# --- Max Tokens Configuration ---
MAX_TOKENS_PER_STAGE = {
    'opening_statement': 300, # Increased slightly to accommodate potential context
    'rebuttal': 250,        # Increased slightly to accommodate potential context
    'closing_statement': 250,       # Increased slightly to accommodate potential context
    'judge_analysis': 200   # Judge summary should still be concise
}

MAX_SUMMARY_TOKENS = 100

# --- Few-Shot Examples ---
# Update examples to show the *expected* format when context is present.
# We'll include a placeholder indicating where context *would* be.
# The LLM learns the format from these examples.
PROMPT_EXAMPLES = {
    'opening_statement': [
         {'role': 'user', 'content': (
            "Relevant information from knowledge base:\n\n[Context Placeholder]\n\n" # Indicate where context goes
            "Deliver your opening statement for the topic: 'Should pineapple belong on pizza?'. Provide your main arguments as a numbered list of 3 to 4 concise points."
         )},
        {'role': 'assistant', 'content': (
            "Here is my opening statement:\n"
            "1.  Pineapple adds a delicious sweet and tangy contrast to savory toppings.\n"
            "2.  Its juiciness helps prevent the pizza from being too dry.\n"
            "3.  It's a popular topping enjoyed by millions worldwide, indicating broad appeal.\n"
            "4.  Pairing fruit with savory dishes is common in many cuisines." # Example output doesn't need to explicitly use context if the point is general knowledge
        )}
    ],
     'rebuttal': [
        {'role': 'user', 'content': (
            "Relevant information from knowledge base:\n\n[Context Placeholder]\n\n" # Indicate where context goes
            "Here is a summary of the debate history so far:\n\n"
            "Summary: Affirmative argued for safety, efficiency. Negative argued against based on risks, job losses. Most recently, Negative claimed AV tech isn't ready and job losses are certain.\n\n"
            "It is your turn to offer a rebuttal... Based on the summary and *relevant provided information*, respond to the points made by the opposing side in their most recent arguments. Provide your rebuttal points as a numbered list of 2 to 3 concise points."
        )},
        {'role': 'assistant', 'content': (
            "Here is my rebuttal:\n"
            "1.  The claim that AV tech isn't ready ignores the rapid advancements and testing already underway by leading companies. [Reference info from context if possible]\n" # Added note for potential reference
            "2.  While job displacement is a concern, history shows technological shifts create new jobs, and focus should be on transition support, not halting progress."
        )}
    ],
     'closing_statement': [
        {'role': 'user', 'content': (
            "Relevant information from knowledge base:\n\n[Context Placeholder]\n\n" # Indicate where context goes
             "Here is a summary of the debate history so far:\n\n"
             "Summary: Affirmative argued safety, efficiency, accessibility benefits. Negative countered with safety risks, job losses, infrastructure costs. Rebuttals exchanged points on tech readiness, economic transition, and regulatory progress.\n\n"
            "Deliver your closing statement... Provide your summary points as a numbered list of 2 to 3 concise points, referencing the summary and *relevant provided information* if helpful." # Reference RAG context
        )},
        {'role': 'assistant', 'content': (
            "In closing, I reiterate my main points:\n"
            "1.  The potential safety and efficiency gains from AVs are transformative. [Reference info from context if possible]\n" # Added note for potential reference
            "2.  While challenges exist, they are surmountable with continued development and thoughtful policy, paving the way for significant societal benefits."
        )}
    ],
    'judge_analysis': [
        # Judge doesn't get RAG context in this design, so no placeholder needed here
        {'role': 'user', 'content': (
             "Here is a summary of the debate history:\n\n"
             "Summary: Affirmative highlighted safety from reducing human error, efficiency in traffic, and accessibility. Negative emphasized current safety risks, potential job losses, and infrastructure/regulatory hurdles. Rebuttals debated technological maturity and economic transition.\n\n"
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