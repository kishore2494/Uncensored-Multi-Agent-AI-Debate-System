#Autonomous AI Debating Society
<img width="1708" height="965" alt="Screenshot 2025-06-18 005800" src="https://github.com/user-attachments/assets/674a81dd-2b83-400f-8638-32b747937328" />

Okay, here's a comprehensive README.md file designed for GitHub. It includes a clear project overview, setup instructions, how to use it, the technologies used, and key features. This is formatted in Markdown, so you can copy and paste it directly.

Autonomous AI Debating Society
🌐 Overview

The Autonomous AI Debating Society is a cutting-edge multi-agent AI system designed to simulate structured debates on user-defined topics. Leveraging local Large Language Models (LLMs) and a Retrieval Augmented Generation (RAG) pipeline, this project demonstrates how autonomous agents can understand, argue, and synthesize information, all presented through an interactive Streamlit web interface.

This project serves as a comprehensive demonstration of advanced AI concepts, including multi-agent architecture, local LLM integration, dynamic knowledge retrieval, and responsive UI development.

✨ Features

Autonomous Multi-Agent Debate:

Debate Orchestrator: Manages debate flow, stages (Opening, Rebuttals, Closing), and turn-taking.

Affirmative Agents: Argue for the debate topic, generating compelling arguments and rebuttals.

Negative Agents: Argue against the debate topic, countering opposing points and presenting counter-arguments.

Judge Agent (Optional): Provides an impartial summary of the debate's key arguments.

Local LLM Integration (Ollama): Utilizes locally hosted open-source LLMs (e.g., dolphin-phi) as the "brains" for all agents, ensuring privacy and cost-efficiency.

Retrieval Augmented Generation (RAG):

Agents query a local vector database (ChromaDB) built from your own PDF documents (./knowledge directory).

Retrieved information is dynamically injected into LLM prompts, grounding arguments in factual context.

Uses Ollama-backed embedding models (e.g., nomic-embed-text) for vectorization.

Intelligent Context Management: Employs debate summarization (via a dedicated LLM call) to keep agent context windows focused and improve argumentative coherence across multiple turns.

Dynamic UI with Streamlit:

User-friendly interface to set debate topic, number of rounds, and agent count.

Toggle RAG system on/off.

Visual representation of agents with customizable photos and South Indian names.

Real-time status updates (e.g., "Thinking...", "Speaking...", "Summarizing...").

Interactive chat history displaying arguments as WhatsApp-like bubbles.

Support for light and dark modes (Streamlit's native theming).

Concurrency Handling: Implements threading and queue-based communication to ensure a responsive UI while handling long-running LLM and RAG operations in the background.

🚀 Technologies Used

Python (Core programming language)

Ollama (Local LLM and Embedding Model serving)

LangChain (Framework for RAG pipeline components: Document Loaders, Text Splitters, Embeddings, Retrievers)

ChromaDB (Local Vector Database)

Streamlit (Web Application Framework for UI)

concurrent.futures & queue (Python standard library for concurrency)

HTML/CSS (For custom UI styling and layout within Streamlit)

Prompt Engineering (Technique for guiding LLM behavior)

Modular Software Architecture

📦 Setup and Installation

Follow these steps to get the Autonomous AI Debating Society running on your local machine.

1. Prerequisites

Python 3.9+: Ensure Python is installed and added to your system's PATH.

Ollama:

Download and install Ollama from ollama.com.

Once installed, pull the required LLM and Embedding models:

Generated bash
ollama pull dolphin-phi:latest  # Or another preferred chat model like llama2:latest, mistral:latest
ollama pull nomic-embed-text    # Essential for RAG embeddings


Ensure the Ollama server is running in the background.

2. Project Setup

Clone the Repository (or create project files):

Generated bash
git clone https://github.com/your-username/autonomous-ai-debating-society.git
cd autonomous-ai-debating-society
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

(If you don't use Git, create the project directory and place all files there.)

Install Python Dependencies:
It's highly recommended to use a Python virtual environment to manage dependencies.

Generated bash
python -m venv .venv
.\.venv\Scripts\activate  # On Windows PowerShell
source ./.venv/bin/activate # On macOS/Linux/Git Bash

pip install -r requirements.txt
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

If you don't have a requirements.txt, create one or install manually:

Generated bash
pip install streamlit langchain langchain-community langchain-core pypdf chromadb ollama
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

Prepare Your Knowledge Base (PDFs):

Create a new directory named knowledge in the root of your project folder.

Place all the PDF documents you want your agents to learn from inside this knowledge directory.

Generated code
project_root/
├── app.py
├── config.py
├── debate_state.py
├── agents.py
├── rag_pipeline.py
├── knowledge/
│   ├── your_document_1.pdf
│   └── your_document_2.pdf
└── images/
    ├── agent_photo_1.png
    └── ...
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
IGNORE_WHEN_COPYING_END

Add Agent Photos:

Create a directory named images in the root of your project folder.

Place .png, .jpg, or other common image files for your agents here. Ensure the filenames you use in config.py under AGENT_PHOTO_PATHS accurately reflect these files. You need enough photos for the maximum number of agents you might configure (e.g., if you plan for 5 pairs + Judge, you need 11 unique photos). A photo named judge_photo.png (case-insensitive) will be prioritized for the Judge if it exists.

3. Running the Application

Activate your virtual environment (if you closed your terminal):

Generated bash
.\.venv\Scripts\activate  # On Windows PowerShell
source ./.venv/bin/activate # On macOS/Linux/Git Bash
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

Run the Streamlit app:

Generated bash
streamlit run app.py
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

Your web browser should automatically open to the Streamlit application (usually http://localhost:8501).

4. Using the Application

Configure Debate: Use the sidebar on the left to set:

Debate Topic: The central question for the agents.

Number of Rebuttal Rounds: How many turns each side gets to respond.

Number of Debating Pairs: Controls how many Affirmative and Negative agents participate.

Include Judge Agent: To enable or disable the Judge agent's analysis.

Knowledge Base Setup (RAG):

If Enable Knowledge Base (RAG) is checked and you've added PDFs to the knowledge folder, click the "Setup Knowledge Base" button.

Be patient! This step involves loading, chunking, and embedding your documents. It can take several minutes depending on the number/size of your PDFs and your machine's power. Watch your terminal for progress messages.

The "Start Debate" button will be disabled until RAG setup is complete (if enabled).

Start the Debate:

Once configured and RAG is ready (if enabled), click the "Start Debate" button.

Watch the agent visualization area for status updates ("Speaking...", "Listening...", "Summarizing...") and the chat panel for the debate unfolding in real-time.

Control and Reset:

"Stop Debate": Manually ends the current debate.

"Clear Debate History": Clears the chat log for a new debate.

⚙️ Configuration Notes

You can customize the debate further by editing config.py:

DEFAULT_MODEL / SUMMARY_MODEL: Change the Ollama models used by agents. Ensure you pull these models via ollama pull <model_name> beforehand.

KB_DIRECTORY / VECTOR_STORE_PATH / EMBEDDING_MODEL: Adjust paths or switch embedding models for RAG.

CHUNK_SIZE / CHUNK_OVERLAP: Fine-tune how documents are split for RAG.

RETRIEVER_K: Control how many top-k relevant documents are retrieved for agents.

AGENT_SYSTEM_PROMPTS / STAGE_PROMPTS / PROMPT_EXAMPLES: These are crucial for dictating agent behavior, argument style, and output format. Experiment with these to refine your agents' debating skills!

MAX_TOKENS_PER_STAGE / MAX_SUMMARY_TOKENS: Control the verbosity of LLM outputs.

🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page or open a pull request.

📝 License

This project is open source and available under the MIT License.

(Remember to replace https://github.com/your-username/your-repo-name with your actual GitHub repository URL and add a LICENSE file if you choose the MIT License)
