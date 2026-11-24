from typing import List
from langchain_core.messages import BaseMessage
from langchain_community.chat_message_histories import FileChatMessageHistory


class WindowedFileChatHistory(FileChatMessageHistory):
    """
    A file-based chat history that keeps a window of the last k interactions in memory
    for the prompt, while persisting the full history to disk.
    This implements the logic of ConversationBufferWindowMemory in a way
    compatible with RunnableWithMessageHistory.
    """

    def __init__(self, file_path: str, k: int = 5):
        super().__init__(file_path=file_path)
        self.k = k

    @property
    def messages(self) -> List[BaseMessage]:
        """Retrieve the last k interactions from the history."""
        all_messages = super().messages
        # Return the last k * 2 messages (user + AI per interaction)
        return all_messages[-(self.k * 2) :]