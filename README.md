Autonomous AI Debating Society
![alt text](https://img.shields.io/badge/Python-3.9+-blue.svg)

![alt text](https://img.shields.io/badge/Streamlit-App-ff4b4b.svg)

![alt text](https://img.shields.io/badge/Ollama-Local%20LLMs-0062FF.svg)

![alt text](https://img.shields.io/badge/LangChain-Framework-20B2AA.svg)

![alt text](https://img.shields.io/badge/ChromaDB-Vector%20DB-005C4D.svg)

![alt text](https://img.shields.io/badge/License-MIT-yellow.svg)
🚀 Overview
The Autonomous AI Debating Society is a sophisticated multi-agent AI system designed to simulate structured debates on user-defined topics. Leveraging local Large Language Models (LLMs) via Ollama, the system features distinct AI agents (Affirmative, Negative, Judge) that autonomously generate arguments, rebut opponents, and provide analysis. A core component of the system is its Retrieval Augmented Generation (RAG) pipeline, which grounds the agents' arguments in external knowledge extracted from local PDF documents. The entire experience is presented through an interactive and visually appealing web interface built with Streamlit.
This project serves as a comprehensive demonstration of integrating cutting-edge AI concepts like multi-agent systems, RAG, and prompt engineering with practical software development principles such as modular design and concurrency management.
✨ Features
Multi-Agent Architecture: Distinct AI agents with specialized roles (Affirmative Debaters, Negative Debaters, Optional Judge).
Local LLM Integration: Powered by Ollama, allowing debates to run entirely on your local machine using open-source models (e.g., dolphin-phi).
Retrieval Augmented Generation (RAG):
Agents dynamically query a local knowledge base (vector database) built from your own PDF documents.
Arguments are enriched with factual information retrieved from the documents, making debates more informed.
Uses LangChain for document loading, splitting, and embedding.
Employs ChromaDB as the local vector store for efficient similarity search.
Structured Debate Flow: Orchestrator agent manages debate stages (Opening Statements, Rebuttal Rounds, Closing Statements, Judge Analysis).
Context Management: Debate history is summarized by an LLM to maintain focus and reduce repetition in longer debates.
Advanced Prompt Engineering: Utilizes system prompts, few-shot examples, and token limits to control agent behavior, output format (e.g., concise bullet points), and argument quality.
Interactive Streamlit UI:
User-friendly interface for configuring debate topics, number of rounds, and number of agent pairs.
Toggle RAG system on/off.
Visualize agent photos, names, and real-time statuses ("Speaking...", "Listening...", "Summarizing...").
Chat-like display of debate history with avatars and aligned bubbles.
Dark/Light mode support (inherits from Streamlit's default theming).
Concurrency Handling: Implements threading to keep the UI responsive during long-running LLM and RAG operations.
🛠️ Technologies Used
Python 3.9+
Streamlit: Web application framework for the UI.
Ollama: For running local LLMs (e.g., dolphin-phi) and embedding models (e.g., nomic-embed-text).
LangChain: Framework for building LLM applications, used for RAG pipeline components (document loaders, text splitters, embeddings, retrievers).
ChromaDB: Lightweight, in-memory/disk-persisted vector database.
concurrent.futures & queue: Python standard library modules for multithreading and inter-thread communication.
os, random, collections.deque, base64: Python standard library modules for file system interaction, randomness, efficient queue-like data structures, and image encoding.
HTML/CSS: For custom styling within the Streamlit application.
🚀 Getting Started
Follow these steps to set up and run the Autonomous AI Debating Society on your local machine.
Prerequisites
Python 3.9+:
Ensure you have Python installed. You can download it from python.org.
Ollama:
Download and install Ollama from ollama.ai.
Once installed, open your terminal and pull the required models:
Generated bash
ollama pull dolphin-phi   # Main LLM for agents
ollama pull nomic-embed-text # Embedding model for RAG
Use code with caution.
Bash
Ensure your Ollama server is running before starting the Streamlit app. You can usually start it by just running ollama run dolphin-phi or similar, or it might run as a background service.
