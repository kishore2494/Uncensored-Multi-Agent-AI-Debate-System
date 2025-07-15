# app.py

import streamlit as st
import time
import os
import random
from collections import deque
import base64
import concurrent.futures
import queue


# Import backend components
from config import (
    DEBATE_TOPIC, NUMBER_OF_REBUTTAL_ROUNDS,
    DEFAULT_MODEL, SUMMARY_MODEL,
    ENABLE_RAG, KB_DIRECTORY, VECTOR_STORE_PATH, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, RETRIEVER_K,
    AGENT_PHOTO_PATHS, SOUTH_INDIAN_NAMES,
)
from debate_state import DebateState
from agents import Agent, DebateOrchestrator, AffirmativeAgent, NegativeAgent, JudgeAgent # Ensure Agent is imported
from rag_pipeline import index_knowledge_base, get_retriever


# --- Streamlit App Configuration ---
st.set_page_config(layout="wide", page_title="Autonomous AI Debating Society")


# --- Helper function to get base64 encoded image ---
def get_image_base64(image_path):
    try:
        if image_path and os.path.exists(image_path):
             with open(image_path, "rb") as img_file:
                 return base64.b64encode(img_file.read()).decode('utf-8')
        else:
            print(f"Image file not found or path is invalid: {image_path}", flush=True)
            return None
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}", flush=True)
        return None


# Inject custom CSS
st.markdown("""
<style>
    /* --- General Styles --- */
    /* Adjust padding of the main content area */
    .main .block-container {
        padding-top: 20px;
        padding-right: 10px; /* Reduced padding for side-by-side columns */
        padding-left: 10px;  /* Reduced padding */
        padding-bottom: 20px;
    }

    /* Hide default Streamlit header/title in the main area */
    .main .stApp > header {
        display: none;
    }

    /* --- Top Elements (Spanning or in Left Column) --- */
     /* Style for the main title element */
    .main-app-title {
        text-align: center;
        font-size: 3em; /* <-- Increased title font size */
        font-weight: bold;
        margin-bottom: 10px;
    }

    /* Style for the main status message below the title */
    .main-status-message {
        text-align: center;
        font-style: italic;
        color: gray;
        font-size: 0.9em;
        margin-bottom: 20px;
    }
     body[data-theme="dark"] .main-status-message { color: #aaa; }


    /* --- Layout: Sidebar | Main Content Area (Split into Agent Viz + Chat Panel) --- */
    /* Style for the left main column (Agent Viz + Top Elements) */
    .agent-viz-column {
        padding-right: 10px;
    }
     body[data-theme="dark"] .agent-viz-column { }

    /* --- Agent Visualization Area (Within the left main column) --- */
    .agent-viz-area {
         text-align: center;
         margin-top: 20px;
         margin-bottom: 20px;
         padding: 10px;
    }

    /* Individual agent container (within the visualization area) */
    .agent-container {
        text-align: center;
        margin: 10px;
        display: inline-block;
        vertical-align: top;
        width: 120px;
    }

    /* Agent Photo (Within agent-container) */
    .agent-container .agent-circle-img img {
        border-radius: 50%;
        width: 90px;
        height: 90px;
        object-fit: cover;
        margin-bottom: 5px;
        border: 3px solid #4CAF50;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    body[data-theme="dark"] .agent-container .agent-circle-img img {
         border: 3px solid #7cb342;
         box-shadow: 0 2px 5px rgba(255,255,255,0.1);
    }

    /* Agent Name (Below photo) */
    .agent-name-text {
        font-size: 0.9em;
        font-weight: bold;
        min-height: 1.2em;
        text-align: center;
    }

    /* Status Text (Below name) */
    .agent-status-text {
        font-size: 0.8em;
        font-style: italic;
        color: gray;
        min-height: 1.2em;
        text-align: center;
    }
    body[data-theme="dark"] .agent-status-text {
        color: #aaa;
    }


    /* --- Chat History Area (Right Main Column) --- */
     /* Style the entire right column to look like a panel */
    .chat-panel-column {
         border-left: 1px solid #ccc;
         padding-left: 20px;
         background-color: #f0f0f0; /* Light background for the panel */
         display: flex;
         flex-direction: column;
         border-radius: 5px;
         /* Remove default padding on the column itself if needed */
         /* padding: 0 !important; */
    }
     body[data-theme="dark"] .right-main-column {
         border-left: 1px solid #666;
         background-color: #333;
     }

     /* Chat header within the chat panel */
     .chat-panel-column h2 {
         text-align: center !important;
         margin-bottom: 10px;
     }


    /* Style the chat container within the chat panel */
    .chat-widget-container {
         border: 1px solid #ccc;
         border-radius: 10px;
         padding: 10px;
         background-color: #f9f9f9;
         margin-top: 0; /* Space handled by h2 margin-bottom */
         width: 100%;
         flex-grow: 1;
         min-height: 0;
         /* overflow-y handled by Streamlit's container(height=...) */
    }
     body[data-theme="dark"] .chat-widget-container {
         border: 1px solid #666;
         background-color: #0e1117;
     }


    /* Chat Message Layout (Inside Chat History Container) */
    .chat-message {
        display: flex; /* Use flexbox */
        margin-bottom: 15px;
        width: 100%;
        /* Removed margin-left/right: auto from here */
    }

    /* Chat Avatar */
    .chat-message .avatar {
        width: 45px;
        height: 45px;
        border-radius: 50%;
        object-fit: cover;
        margin-right: 10px; /* Space between avatar and text */
        border: 2px solid #bbb;
        flex-shrink: 0;
    }
     body[data-theme="dark"] .chat-message .avatar { border: 2px solid #777; }

     /* Message Content (Name + Bubble) */
     .chat-message .message-content {
         flex-grow: 1; /* Let content take up space */
         min-width: 0; /* Prevent overflow */
     }

     /* Agent Name in Chat */
    .chat-message .message-content strong {
        display: block;
        margin-bottom: 2px;
        font-size: 0.9em;
    }

    /* --- Chat Bubble Styling and Alignment --- */

    .agent-text-bubble {
        background-color: #e0e0e0;
        border-radius: 15px;
        padding: 10px 15px;
        margin-bottom: 5px;
        /* Control bubble width */
        max-width: calc(100% - 60px); /* Max width considering avatar + margin */
        display: inline-block; /* Make bubble take only necessary width */
        word-wrap: break-word;
        color: #000;
        line-height: 1.5;
        /* Removed default margins */
    }
    .agent-text-bubble.affirmative { background-color: #a5d6a7; }
     .agent-text-bubble.negative { background-color: #ef9a9a; }
     .agent-text-bubble.judge { background-color: #ce93d8; }


    /* Default (Negative/Judge): Avatar Left, Bubble Left */
    .chat-message:not(.affirmative) {
         flex-direction: row; /* Avatar then content */
         justify-content: flex-start; /* Align to the start (left) */
    }
     .chat-message:not(.affirmative) .message-content {
         text-align: left; /* Name and bubble content aligned left */
     }
      .chat-message:not(.affirmative) .message-content .agent-text-bubble {
          /* Bubble aligns left within its content div */
          margin-right: auto; /* Push bubble to the left (optional, depends on other styles) */
          margin-left: 0;
          text-align: left;
      }

    /* Affirmative: Avatar Right, Bubble Right */
    .chat-message.affirmative {
        flex-direction: row-reverse; /* Content then avatar */
        justify-content: flex-start; /* Align to the start (which is the right after reversing) */
    }
     .chat-message.affirmative .avatar {
         margin-left: 10px; /* Space between content and avatar */
         margin-right: 0;
     }
     .chat-message.affirmative .message-content {
          text-align: right; /* Name and bubble content aligned right */
     }
     .chat-message.affirmative .message-content .agent-text-bubble {
         /* Bubble aligns right within its content div */
         margin-left: auto; /* Push bubble to the right */
         margin-right: 0;
         text-align: left; /* Keep text aligned left inside the bubble */
      }


    /* --- Utility Styles --- */
    /* Status Messages (Used in chat history) */
    .status-message {
        font-size: 0.9em;
        font-style: italic;
        color: gray;
        text-align: center;
        margin: 5px 0;
    }
    body[data-theme="dark"] .status-message { color: #aaa; }

    /* Stage Separators (Used in chat history) */
    .stage-separator {
        font-weight: bold;
        color: #555;
        text-align: center;
        margin: 10px 0;
        border-bottom: 1px solid #ccc;
        line-height: 0.1em;
    }
     body[data-theme="dark"] .stage-separator { color: #aaa; border-bottom-color: #666;}

    .stage-separator span {
        background:#fff;
        padding:0 10px;
    }
     body[data-theme="dark"] .stage-separator span { background:#0e1117; }

     /* Style for the two white rectangles */
     .white-rectangle {
        background-color: white;
        height: 60px;
        margin: 15px 0;
        border: 1px solid #ddd;
        border-radius: 5px;
        width: 100%;
    }
     body[data-theme="dark"] .white-rectangle {
        background-color: #eee;
        border: 1px solid #555;
    }


    /* --- Dark Mode Adjustments --- */
    body[data-theme="dark"] .agent-text-bubble { background-color: #444; color: #eee; }
    body[data-theme="dark"] .agent-text-bubble.affirmative { background-color: #385d4a; }
    body[data-theme="dark"] .agent-text-bubble.negative { background-color: #68d43; }
    body[data-theme="dark"] .agent-text-bubble.judge { background-color: #5a436b; }

</style>
""", unsafe_allow_html=True)


# --- Session State Initialization ---
if 'debate_started' not in st.session_state:
    st.session_state.debate_started = False
if 'debate_finished' not in st.session_state:
    st.session_state.debate_finished = False
if 'orchestrator' not in st.session_state:
    st.session_state.orchestrator = None
if 'debate_history' not in st.session_state:
    st.session_state.debate_history = deque(maxlen=500)
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'agent_configs' not in st.session_state:
    st.session_state.agent_configs = []
if 'status_message' not in st.session_state:
    st.session_state.status_message = "Configure and start the debate."
if 'agent_statuses' not in st.session_state:
    st.session_state.agent_statuses = {}

# --- Session State for Threading ---
if 'debate_result_queue' not in st.session_state:
    st.session_state.debate_result_queue = queue.Queue()
if 'debate_step_processing' not in st.session_state:
    st.session_state.debate_step_processing = False


# --- Function to run one step of the debate generator in a thread ---
def run_debate_step(generator, result_queue):
    """Gets the next item from the generator and puts it in the queue."""
    try:
        item = next(generator, None)
        time.sleep(0.01)
        result_queue.put({"item": item, "error": None})
    except Exception as e:
        result_queue.put({"item": None, "error": e})


# --- Dynamic Agent Configuration ---
def create_dynamic_agent_configs(num_pairs, include_judge):
    total_debaters = num_pairs * 2
    total_agents = total_debaters + (1 if include_judge else 0)

    if total_agents == 0:
         st.warning("No agents configured. Adjust number of pairs or include judge.")
         return []

    if total_agents > len(SOUTH_INDIAN_NAMES):
        st.warning(f"Not enough names provided. Need {total_agents} but only have {len(SOUTH_INDIAN_NAMES)}. Reusing names.")
        selected_names = [SOUTH_INDIAN_NAMES[i % len(SOUTH_INDIAN_NAMES)] for i in range(total_agents)]
    else:
        selected_names = random.sample(SOUTH_INDIAN_NAMES, total_agents)
    random.shuffle(selected_names)

    available_photos = [f"images/{f}" for f in os.listdir("images") if os.path.isdir("images") and f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    if not available_photos:
        st.error("No image files found in the ./images directory!")
        return []

    if total_agents > len(available_photos):
        st.warning(f"Not enough valid image files found in ./images. Need {total_agents} but only found {len(available_photos)}. Reusing photos.")
        selected_photos = [available_photos[i % len(available_photos)] for i in range(total_agents)]
    else:
         selected_photos = random.sample(available_photos, total_agents)
    random.shuffle(selected_photos)


    agent_configs = []
    name_idx = 0
    photo_idx = 0

    # Create Debaters (Affirmative/Negative)
    aff_debaters_cfg = []
    neg_debaters_cfg = []
    for i in range(num_pairs):
        aff_name = selected_names[name_idx]
        aff_photo = selected_photos[photo_idx]
        aff_debaters_cfg.append({
            'type': 'AffirmativeAgent', 'name': aff_name,
            'model': DEFAULT_MODEL, 'agent_photo': aff_photo
        })
        name_idx += 1
        photo_idx += 1

        neg_name = selected_names[name_idx]
        neg_photo = selected_photos[photo_idx]
        neg_debaters_cfg.append({
            'type': 'NegativeAgent', 'name': neg_name,
            'model': DEFAULT_MODEL, 'agent_photo': neg_photo
        })
        name_idx += 1
        photo_idx += 1

    agent_configs.extend(aff_debaters_cfg)
    agent_configs.extend(neg_debaters_cfg)


    # Create Judge (if included)
    judge_cfg = None
    if include_judge:
         judge_photo_candidates = [p for p in available_photos if 'judge' in os.path.basename(p).lower()]
         judge_photo = judge_photo_candidates[0] if judge_photo_candidates else (random.choice(available_photos) if available_photos else None)

         if judge_photo:
             judge_cfg = {
                'type': 'JudgeAgent', 'name': "Judge", # Explicitly set Judge name to "Judge"
                'model': DEFAULT_MODEL, 'optional': True, 'agent_photo': judge_photo
             }
             agent_configs.append(judge_cfg)

         else:
             st.warning("Could not assign a photo to the Judge agent. Judge will not be included.")

    # Initialize agent statuses dictionary based on the created configs
    st.session_state.agent_statuses = {cfg['name']: "Waiting..." for cfg in agent_configs}


    return agent_configs


# --- RAG Setup Function ---
# ... (Keep setup_rag function as is)
def setup_rag(enable):
    st.session_state.status_message = "Setting up Knowledge Base (RAG)..."
    st.session_state.debate_history.append({"type": "status", "message": st.session_state.status_message})
    st.rerun()

    if enable:
        try:
            kb_dir_exists_and_not_empty = os.path.exists(KB_DIRECTORY) and os.path.isdir(KB_DIRECTORY) and len(os.listdir(KB_DIRECTORY)) > 0
            if not kb_dir_exists_and_not_empty:
                 st.session_state.status_message = f"RAG enabled, but '{KB_DIRECTORY}' is empty or missing. Cannot setup KB."
                 st.session_state.retriever = None
                 print(f"RAG Setup failed: KB directory empty or missing {KB_DIRECTORY}")
            else:
                vector_store = index_knowledge_base(
                    kb_directory=KB_DIRECTORY,
                    vector_store_path=VECTOR_STORE_PATH,
                    embedding_model=EMBEDDING_MODEL,
                    chunk_size=CHUNK_SIZE,
                    chunk_overlap=CHUNK_OVERLAP
                )
                if vector_store:
                    st.session_state.retriever = get_retriever(vector_store)
                    if not st.session_state.retriever:
                        st.session_state.status_message = "Failed to get retriever from vector store. RAG will be disabled."
                        st.session_state.retriever = None
                    else:
                        st.session_state.status_message = "Knowledge Base Setup Complete."
                else:
                    st.session_state.status_message = "Knowledge base indexing failed or no documents found. RAG will be disabled."
                    st.session_state.retriever = None
        except Exception as e:
            st.session_state.status_message = f"An error occurred during RAG setup: {e}. RAG will be disabled."
            st.session_state.retriever = None
            print(f"RAG Setup Error: {e}")
    else:
        st.session_state.status_message = "RAG disabled by user configuration."
        st.session_state.retriever = None

    st.session_state.debate_history.append({"type": "status", "message": st.session_state.status_message})
    st.rerun()


# --- Sidebar for controls ---
with st.sidebar:
    st.header("Debate Configuration")
    st.session_state.topic_input = st.text_area("Debate Topic", st.session_state.get('topic_input', DEBATE_TOPIC), height=100)
    st.session_state.rounds_input = st.slider("Number of Rebuttal Rounds", 0, 5, st.session_state.get('rounds_input', NUMBER_OF_REBUTTAL_ROUNDS))
    st.session_state.num_agent_pairs = st.slider("Number of Debating Pairs (Affirmative/Negative)", 1, 5, st.session_state.get('num_agent_pairs', 2))
    st.session_state.include_judge = st.checkbox("Include Judge Agent", st.session_state.get('include_judge', True))

    st.header("Advanced Settings")
    st.session_state.enable_rag_toggle = st.checkbox("Enable Knowledge Base (RAG)", st.session_state.get('enable_rag_toggle', ENABLE_RAG))
    st.write(f"KB Directory: `{KB_DIRECTORY}`")
    st.write(f"Vector Store: `{VECTOR_STORE_PATH}`")
    st.write(f"Embedding Model: `{EMBEDDING_MODEL}`")

    kb_dir_exists_and_not_empty = os.path.exists(KB_DIRECTORY) and os.path.isdir(KB_DIRECTORY) and len(os.listdir(KB_DIRECTORY)) > 0

    if st.button("Setup Knowledge Base", disabled = not kb_dir_exists_and_not_empty):
        setup_rag(st.session_state.enable_rag_toggle)

    if st.session_state.enable_rag_toggle:
         if kb_dir_exists_and_not_empty:
             if st.session_state.retriever is None:
                 st.warning("RAG enabled, but KB not loaded. Click 'Setup Knowledge Base'.")
             else:
                 st.success("RAG enabled and KB loaded.")
         else:
             st.warning(f"RAG enabled, but '{KB_DIRECTORY}' is empty or missing. Add PDF documents and click 'Setup Knowledge Base'.")
    else:
         st.info("RAG is disabled.")

    total_required_photos = (st.session_state.num_agent_pairs * 2) + (1 if st.session_state.include_judge else 0)
    available_photos_count = len([f for f in os.listdir("images") if os.path.isdir("images") and f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]) if os.path.exists("images") else 0


    can_create_agents = (st.session_state.num_agent_pairs * 2 + (1 if st.session_state.include_judge else 0)) > 0 and available_photos_count > 0

    start_button_disabled = st.session_state.debate_started or (st.session_state.enable_rag_toggle and kb_dir_exists_and_not_empty and st.session_state.retriever is None) or not can_create_agents or st.session_state.debate_step_processing


    if st.button("Start Debate", disabled=start_button_disabled):
        st.session_state.debate_started = True
        st.session_state.debate_finished = False
        st.session_state.debate_history = deque(maxlen=500) # Clear history
        st.session_state.status_message = "Initializing debate..."
        st.session_state.debate_step_processing = False # Ensure this is reset

        st.session_state.agent_configs = create_dynamic_agent_configs(
            st.session_state.num_agent_pairs,
            st.session_state.include_judge
        )
        if not st.session_state.agent_configs: # Check if config creation failed (due to names/photos)
             st.session_state.status_message = "Agent configuration failed. Cannot start debate."
             st.session_state.debate_started = False
             st.session_state.debate_history.append({"type": "status", "message": st.session_state.status_message})
             st.rerun()
        else:
            all_agents = []
            retriever_to_pass = st.session_state.retriever if st.session_state.enable_rag_toggle else None

            agent_map = {
                'AffirmativeAgent': AffirmativeAgent,
                'NegativeAgent': NegativeAgent,
                'JudgeAgent': JudgeAgent,
            }

            # Agent statuses are initialized in create_dynamic_agent_configs now
            # st.session_state.agent_statuses = {cfg['name']: "Waiting..." for cfg in st.session_state.agent_configs}


            try:
                for agent_cfg in st.session_state.agent_configs:
                    agent_type = agent_cfg['type']
                    AgentClass = agent_map[agent_type]
                    agent_name = agent_cfg['name']
                    agent_model = agent_cfg['model']
                    agent_photo = agent_cfg['agent_photo']

                    if agent_type in ['AffirmativeAgent', 'NegativeAgent']:
                         agent_instance = AgentClass(agent_name, model=agent_model, retriever=retriever_to_pass, agent_photo=agent_photo)
                    elif agent_type == 'JudgeAgent':
                         agent_instance = AgentClass(agent_name, model=agent_model, agent_photo=agent_photo)
                    else:
                         raise ValueError(f"Unknown agent type {agent_type}")

                    all_agents.append(agent_instance)

                debate_state = DebateState(topic=st.session_state.topic_input)
                st.session_state.orchestrator = DebateOrchestrator(
                    name="The Moderator",
                    debate_state=debate_state,
                    agents=all_agents,
                    model=DEFAULT_MODEL
                )
                st.session_state.debate_generator = st.session_state.orchestrator.run_debate(st.session_state.rounds_input)
                st.session_state.debate_step_processing = False # Ensure this is False initially
                st.rerun()

            except Exception as e:
                 st.session_state.status_message = f"Error initializing debate or agents: {e}"
                 st.session_state.debate_started = False
                 st.session_state.orchestrator = None
                 st.session_state.debate_history.append({"type": "status", "message": st.session_state.status_message + " -- Setup failed."})
                 print(f"Initialization Error: {e}")
                 st.rerun()


    if st.session_state.debate_started:
         if st.button("Stop Debate"):
             st.session_state.debate_started = False
             st.session_state.status_message = "Debate manually stopped."
             st.session_state.debate_history.append({"type": "status", "message": st.session_state.status_message})
             st.session_state.agent_statuses = {name: "Stopped" for name in st.session_state.agent_statuses}
             st.session_state.debate_step_processing = False
             st.rerun()


    if st.button("Clear Debate History"):
        st.session_state.debate_history = deque(maxlen=500)
        st.session_state.debate_finished = False
        st.session_state.debate_started = False
        st.session_state.status_message = "Debate history cleared."
        st.session_state.orchestrator = None
        st.session_state.agent_statuses = {}
        st.session_state.debate_step_processing = False
        st.rerun()


# --- Main Content Area (Contains everything to the right of the sidebar) ---
# This area will be split into two columns: Agent Visualization (Left) and Chat Panel (Right)
# Use st.columns to create these two main columns after the sidebar
# Adjust the ratios based on desired width
# Try [1, 1] for equal width initially, or [2, 1] if you want agents wider
agent_viz_col, chat_panel_col = st.columns([2, 1])


# --- Left Main Column: Agent Visualization Area + Top Elements ---
with agent_viz_col:
    # Place Title, Status, Topic, Rule here in the left column
    st.markdown("<div class='main-app-title'>Autonomous AI Debating Society</div>", unsafe_allow_html=True)
    st.markdown(f"<p class='main-status-message'>{st.session_state.status_message}</p>", unsafe_allow_html=True)
    st.header(st.session_state.get('topic_input', DEBATE_TOPIC))
    st.markdown("---")


    # Add the two horizontal white rectangles placeholder from the image
    #st.markdown("<div class='white-rectangle'></div>", unsafe_allow_html=True)
    #st.markdown("<div class='white-rectangle'></div>", unsafe_allow_html=True)


    # Create a container for the agent photos and statuses visualization below rectangles
    # This is the area with the 2x2 + 1 layout
    st.markdown("<div class='agent-viz-area'>", unsafe_allow_html=True)

    # Separate debaters from the judge for layout within this area
    aff_debaters_cfg = [a for a in st.session_state.agent_configs if a['type'] == 'AffirmativeAgent']
    neg_debaters_cfg = [a for a in st.session_state.agent_configs if a['type'] == 'NegativeAgent']
    judge_cfg = next((a for a in st.session_state.agent_configs if a['type'] == 'JudgeAgent'), None)


    # Display Debaters (Affirmative on left, Negative on right within this area's columns)
    num_aff_debaters = len(aff_debaters_cfg)
    num_neg_debaters = len(neg_debaters_cfg)

    if num_aff_debaters > 0 or num_neg_debaters > 0:
         debater_sides_cols = st.columns([1, 1])

         with debater_sides_cols[0]: # Affirmative Side Column (Left part of agent-viz-area)
              st.markdown("<div style='text-align: center; font-weight: bold;'>Affirmative Team</div>", unsafe_allow_html=True)
              for agent_cfg in aff_debaters_cfg:
                   status_text = st.session_state.agent_statuses.get(agent_cfg['name'], "Waiting...")
                   st.markdown(f"<div class='agent-status-text'>{status_text}</div>", unsafe_allow_html=True)
                   photo_src = agent_cfg.get('agent_photo', '')
                   base64_img_string = get_image_base64(photo_src)
                   if base64_img_string: img_tag = f'<img src="data:image/png;base64,{base64_img_string}">'
                   else: img_tag = '<div style="width: 90px; height: 90px; border: 1px solid red; border-radius: 50%; display: flex; justify-content: center; align-items: center; font-size: 0.7em;">Img Error</div>'
                   st.markdown(f"<div class='agent-container'>{img_tag}<div class='agent-name-text'>{agent_cfg['name']}</div></div>", unsafe_allow_html=True)


         with debater_sides_cols[1]: # Negative Side Column (Right part of agent-viz-area)
              st.markdown("<div style='text-align: center; font-weight: bold;'>Negative Team</div>", unsafe_allow_html=True)
              for agent_cfg in neg_debaters_cfg:
                   status_text = st.session_state.agent_statuses.get(agent_cfg['name'], "Waiting...")
                   st.markdown(f"<div class='agent-status-text'>{status_text}</div>", unsafe_allow_html=True)
                   photo_src = agent_cfg.get('agent_photo', '')
                   base64_img_string = get_image_base64(photo_src)
                   if base64_img_string: img_tag = f'<img src="data:image/png;base64,{base64_img_string}">'
                   else: img_tag = '<div style="width: 90px; height: 90px; border: 1px solid red; border-radius: 50%; display: flex; justify-content: center; align-items: center; font-size: 0.7em;">Img Error</div>'
                   st.markdown(f"<div class='agent-container'>{img_tag}<div class='agent-name-text'>{agent_cfg['name']}</div></div>", unsafe_allow_html=True)


    # Display Judge Centered Below Debaters within the Agent Display Area
    if judge_cfg:
         center_judge_cols = st.columns([1, 2, 1])
         with center_judge_cols[1]:
              st.markdown("<div style='text-align: center; font-weight: bold; margin-top: 20px;'>Judge</div>", unsafe_allow_html=True)
              status_text = st.session_state.agent_statuses.get(judge_cfg['name'], "Waiting...")
              st.markdown(f"<div class='agent-status-text' style='text-align: center;'>{status_text}</div>", unsafe_allow_html=True)

              photo_src = judge_cfg.get('agent_photo', '')
              base64_img_string = get_image_base64(photo_src)

              if base64_img_string:
                   img_tag = f'<img src="data:image/png;base64,{base64_img_string}">'
              else:
                   img_tag = '<div style="width: 90px; height: 90px; border: 1px solid red; border-radius: 50%; display: flex; justify-content: center; align-items: center; font-size: 0.7em;">Img Error</div>'

              st.markdown(f"<div class='agent-container'>{img_tag}<div class='agent-name-text'>{judge_cfg['name']}</div></div>", unsafe_allow_html=True)


    # Close the agent display area wrapper div
    st.markdown("</div>", unsafe_allow_html=True)


# --- Right Main Column: Chat History Area (Styled as a Panel) ---
with chat_panel_col: # This column will contain the chat header and container
    st.markdown("<div class='chat-panel-column'></div>", unsafe_allow_html=True)

    st.header("Debate Chat")
    chat_container = st.container(height=1000, border=True, key="chat_history_container")


    with chat_container:
         for item in reversed(st.session_state.debate_history):
            # Initialize bubble_class and avatar_img_tag for each item in the loop
            bubble_class = ""
            avatar_img_tag = '<div class="avatar" style="border: 1px solid red; border-radius: 50%; flex-shrink: 0; display: flex; justify-content: center; align-items: center; font-size: 0.6em;">Err</div>'


            if item["type"] == "status":
                st.markdown(f"<div class='status-message'>{item['message']}</div>", unsafe_allow_html=True)
            elif item["type"] == "stage":
                 st.markdown(f"<div class='stage-separator'><span>{item['stage_name']}</span></div>", unsafe_allow_html=True)
            elif item["type"] == "message":
                photo = item.get("photo", '')
                if 'affirmative' in item['role'].lower():
                    bubble_class = "affirmative"
                elif 'negative' in item['role'].lower():
                    bubble_class = "negative"
                elif 'judge' in item['role'].lower():
                     bubble_class = "judge"

                base64_avatar_string = get_image_base64(photo)

                if base64_avatar_string:
                     avatar_img_tag = f'<img src="data:image/png;base64,{base64_avatar_string}" class="avatar">'
                else:
                     pass


            if item["type"] == "message": # Only render message HTML for items of type 'message'
                 message_div_class = f"chat-message {bubble_class}"

                 st.markdown(f"""
                 <div class="{message_div_class}">
                     {avatar_img_tag}
                     <div class="message-content">
                         <strong>{item['name']}</strong>
                         <div class="agent-text-bubble {bubble_class}">
                             {item['text']}
                         </div>
                     </div>
                 </div>
                 """, unsafe_allow_html=True)


# --- Debate Loop ---
if st.session_state.debate_started and st.session_state.orchestrator and st.session_state.debate_generator and not st.session_state.debate_step_processing:
    st.session_state.debate_step_processing = True

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(run_debate_step, st.session_state.debate_generator, st.session_state.debate_result_queue)


# --- Processing Results from Thread ---
if not st.session_state.debate_result_queue.empty():
    try:
        result = st.session_state.debate_result_queue.get(timeout=0.01)

        if result["item"] is not None:
             st.session_state.debate_step_processing = False


        if result["error"]:
            error = result["error"]
            st.session_state.debate_started = False
            st.session_state.debate_step_processing = False
            st.session_state.status_message = f"An error occurred during the debate turn: {error}"
            st.session_state.debate_history.append({"type": "status", "message": st.session_state.status_message + " -- Debate ended."})
            print(f"Debate Thread Error: {error}", flush=True)
            st.session_state.agent_statuses = {name: "Error" for name in st.session_state.agent_statuses}
            st.rerun()

        else: # Successfully got an item from the generator
            yielded_item = result["item"]

            if yielded_item is not None:
                item_type = yielded_item.get("type")
                agent_name = yielded_item.get("agent_name")

                if item_type == "status":
                    st.session_state.status_message = yielded_item.get("message", "")
                    st.session_state.debate_history.append({"type": "status", "message": st.session_state.status_message}) # Store as status
                    # Update agent status display based on the message
                    if "speaking..." in yielded_item.get("message", ""):
                         speaking_name = yielded_item.get("message").split(" speaking...")[0]
                         if speaking_name in st.session_state.agent_statuses:
                             for name in st.session_state.agent_statuses:
                                if name == speaking_name:
                                    st.session_state.agent_statuses[name] = "Speaking..."
                                elif st.session_state.debate_started:
                                    st.session_state.agent_statuses[name] = "Listening..."
                                else:
                                     pass

                    elif "summarizing debate" in yielded_item.get("message", ""):
                         st.session_state.agent_statuses = {name: "Summarizing..." for name in st.session_state.agent_statuses}
                    elif "Summary Generated" in yielded_item.get("message", ""):
                         st.session_state.agent_statuses = {name: "Listening..." for name in st.session_state.agent_statuses}


                elif item_type == "stage":
                    st.session_state.status_message = f"Stage: {yielded_item.get('stage_name', 'Unknown')}"
                    st.session_state.debate_history.append({"type": "stage", "stage_name": yielded_item.get("stage_name", "Unknown")}) # Store as stage
                    st.session_state.agent_statuses = {name: "Listening..." for name in st.session_state.agent_statuses}


                elif item_type == "argument":
                    agent_name = yielded_item.get("agent_name")
                    agent_role = yielded_item.get("agent_role")
                    argument_text = yielded_item.get("argument", "")
                    agent_photo = yielded_item.get("agent_photo")

                    st.session_state.status_message = f"{agent_name} ({agent_role}) finished speaking."

                    st.session_state.debate_history.append({
                        "type": "message",
                        "name": agent_name,
                        'role': agent_role,
                        'text': argument_text,
                        'photo': agent_photo
                    })

                    if agent_name in st.session_state.agent_statuses:
                        st.session_state.agent_statuses[agent_name] = "Waiting..."


                elif item_type == "summary":
                     pass # Handled by status messages

                st.rerun()

            else:
                st.session_state.debate_started = False
                st.session_state.debate_finished = True
                st.session_state.debate_step_processing = False
                st.session_state.status_message = "Debate concluded."
                st.session_state.debate_history.append({"type": "status", "message": st.session_state.status_message})
                st.session_state.agent_statuses = {name: "Finished" for name in st.session_state.agent_statuses}
                st.rerun()

    except queue.Empty:
         pass
    except Exception as e:
        st.session_state.debate_started = False
        st.session_state.debate_step_processing = False
        st.session_state.status_message = f"An error occurred while processing debate item: {e}"
        st.session_state.debate_history.append({"type": "status", "message": st.session_state.status_message + " -- Debate ended."})
        print(f"Debate Item Processing Error: {e}", flush=True)
        st.session_state.agent_statuses = {name: "Error" for name in st.session_state.agent_statuses}
        st.rerun()


# --- End of UI Rendering ---
if st.session_state.debate_finished:
    st.balloons()

# --- Optional: Display Streamlit version for debugging ---
# st.sidebar.write(f"Streamlit Version: {st.__version__}")