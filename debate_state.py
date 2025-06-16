class DebateState:
    """Holds the state of the debate, including the topic and history."""
    def __init__(self, topic: str):
        self.topic = topic
        self.history = [] # List of dictionaries: [{'agent': name, 'role': role, 'argument': text}]

    def add_argument(self, agent_name: str, agent_role: str, argument: str):
        """Adds an argument to the debate history."""
        self.history.append({
            'agent': agent_name,
            'role': agent_role,
            'argument': argument
        })

    def get_history_text(self) -> str:
        """Returns the full debate history as formatted text."""
        history_text = f"Debate Topic: {self.topic}\n\n-- Debate History --\n"
        if not self.history:
            history_text += "No arguments yet.\n"
        else:
            for entry in self.history:
                history_text += f"[{entry['role']} - {entry['agent']}]:\n{entry['argument']}\n\n"
        history_text += "-- End of History --\n"
        return history_text

    def get_last_argument_text(self, from_role: str) -> str or None:
        """Returns the text of the last argument from a specific role."""
        for entry in reversed(self.history):
            if entry['role'] == from_role:
                return entry['argument']
        return None

    def get_full_history_for_prompt(self) -> list:
        """Returns history formatted for Ollama's chat message list."""
        messages = []
        # Add context from history, maybe last few turns or all depending on desired context length
        # For now, let's just pass the whole history as a user message in the prompt string
        # or include relevant previous messages directly if we manage history more granularly for the API
        # For simplicity with current STAGE_PROMPTS, we'll format it as a string in the prompt.
        return self.get_history_text() # Or a more structured format if needed
