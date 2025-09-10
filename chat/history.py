from langchain_core.messages import HumanMessage, AIMessage
from typing import List, Dict

class ChatHistory:
    def __init__(self):
        self.history: List[HumanMessage | AIMessage] = []

    def add_human_message(self, content: str):
        self.history.append(HumanMessage(content=content))

    def add_ai_message(self, content: str):
        self.history.append(AIMessage(content=content))

    def clear(self):
        self.history = []
        
    def get_messages(self) -> List[HumanMessage | AIMessage]:
        return self.history

    def get_formatted(self) -> List[Dict]:
        formatted = []
        for i, message in enumerate(self.history):
            formatted.append({
                "type": "human" if isinstance(message, HumanMessage) else "ai",
                "content": message.content,
                "index": i
            })
        return formatted

    def __len__(self):
        return len(self.history)