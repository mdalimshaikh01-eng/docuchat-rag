import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from vectordb import VectorDB
from memory import WindowedFileChatHistory
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
dotenv_path = os.path.join(project_root, ".env")
load_dotenv(dotenv_path)


def load_documents() -> List[Dict[str, Any]]:
    """
    Load .txt documents from the 'data' directory.

    Returns:
        A list of document dictionaries, where each dictionary has 'content' and 'metadata'.
    """
    results = []
    # Récupère le chemin du dossier data en remontant d'un niveau depuis src/
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(script_dir), "data")

    if not os.path.isdir(data_dir):
        print(f"Warning: Data directory '{data_dir}' not found.")
        return results

    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(data_dir, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    doc = {"content": content, "metadata": {"source": filename}}
                    results.append(doc)
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")

    return results


class RAGAssistant:
    """
    A simple RAG-based AI assistant using ChromaDB and multiple LLM providers.
    Supports OpenAI, Groq, and Google Gemini APIs.
    """

    def __init__(self):
        """Initialize the RAG assistant."""
        # Initialize LLM - check for available API keys in order of preference
        self.llm = self._initialize_llm()
        if not self.llm:
            raise ValueError(
                "No valid API key found. Please set one of: "
                "OPENAI_API_KEY, GROQ_API_KEY, or GOOGLE_API_KEY in your .env file"
            )

        # Initialize vector database
        self.vector_db = VectorDB()

        # Create RAG prompt template with strict constraints
        template = """Role:
            You are a helpful assistant that can answer the user's questions given some relevant documents and the conversation history.

            Style and Tone:
                - Use clear, concise language.
                - Use bullet points where appropriate.

            Constraints:
                - Only answer questions based on the provided documents and conversation history.
                - If the user's question is not related to the documents, then you SHOULD NOT answer the question. Say "The question is not answerable given the documents".
                - Never answer a question from your own knowledge.

            Instructions:
                Given the following conversation history and documents, answer the user's question.

            Chat History:
                {chat_history}

            Context:
                {context}

            Question:
                {question}
        """
        self.prompt_template = ChatPromptTemplate.from_template(template)

        # Define path for chat history and create a history object with a window of k=5
        chat_history_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chat_history.json")
        self.message_history = WindowedFileChatHistory(file_path=chat_history_path, k=5)

        # Create the chain with conversational memory
        base_chain = self.prompt_template | self.llm | StrOutputParser()
        self.chain_with_history = RunnableWithMessageHistory(
            base_chain,
            lambda session_id: self.message_history,  # Use the same history for all sessions
            input_messages_key="question",
            history_messages_key="chat_history",
        )

        print("RAG Assistant initialized successfully")

    def _initialize_llm(self):
        """
        Initialize the LLM by checking for available API keys.
        Tries OpenAI, Groq, and Google Gemini in that order.
        """
        # Check for OpenAI API key
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key and "your_openai_api_key_here" not in openai_key:
            model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            print(f"Using OpenAI model: {model_name}")
            return ChatOpenAI(
                api_key=openai_key, model=model_name, temperature=0.0
            )

        elif os.getenv("GROQ_API_KEY"):
            model_name = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
            print(f"Using Groq model: {model_name}")
            return ChatGroq(
                api_key=os.getenv("GROQ_API_KEY"), model=model_name, temperature=0.0
            )

        elif os.getenv("GOOGLE_API_KEY"):
            model_name = os.getenv("GOOGLE_MODEL", "gemini-2.0-flash")
            print(f"Using Google Gemini model: {model_name}")
            return ChatGoogleGenerativeAI(
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                model=model_name,
                temperature=0.0,
            )

        else:
            raise ValueError(
                "No valid API key found. Please set one of: OPENAI_API_KEY, GROQ_API_KEY, or GOOGLE_API_KEY in your .env file"
            )

    def add_documents(self, documents: List) -> None:
        """
        Add documents to the knowledge base.

        Args:
            documents: List of documents
        """
        self.vector_db.add_documents(documents)

    def invoke(self, input: str, n_results: int = 3) -> str:
        """
        Query the RAG assistant.

        Args:
            input: User's input
            n_results: Number of relevant chunks to retrieve

        Returns:
            Dictionary containing the answer and retrieved context
        """
        # 1. Retrieve relevant context chunks
        results = self.vector_db.search(input, n_results=n_results)
        documents = results.get("documents", [])
        
        # 2. Combine the retrieved document chunks into a single context string
        context = "\n\n".join(documents)

        # 3. Generate the response using the chain with history
        # Define a session ID. Since we have one history file, it's a fixed value.
        config = {"configurable": {"session_id": "main_session"}}

        # Invoke the chain with history, providing the necessary inputs
        response = self.chain_with_history.invoke(
            {"context": context, "question": input},
            config=config
        )

        return response


def main():
    """Main function to demonstrate the RAG assistant."""
    try:
        # --- Session Management: Clear chat history for a new session ---
        chat_history_path = os.path.join(project_root, "chat_history.json")
        if os.path.exists(chat_history_path):
            print("\nClearing chat history for new session...")
            with open(chat_history_path, "w") as f:
                f.write("[]")  # Initialize with empty JSON list
        # --- End Session Management ---

        # Initialize the RAG assistant
        print("Initializing RAG Assistant...")
        assistant = RAGAssistant()

        # Load sample documents
        print("\nLoading documents...")
        sample_docs = load_documents()
        print(f"Loaded {len(sample_docs)} sample documents")

        assistant.add_documents(sample_docs)

        # --- Verification Step ---
        collection_count = assistant.vector_db.collection.count()
        print(f"\n✅ Verification: {collection_count} chunks have been added to the vector database.")
        # --- End Verification ---

        done = False

        while not done:
            question = input("Enter a question or 'quit' to exit: ")
            if question.lower() == "quit":
                done = True
            else:
                result = assistant.invoke(question)
                print(result)

    except Exception as e:
        print(f"Error running RAG assistant: {e}")
        print("Make sure you have set up your .env file with at least one API key:")
        print("- OPENAI_API_KEY (OpenAI GPT models)")
        print("- GROQ_API_KEY (Groq Llama models)")
        print("- GOOGLE_API_KEY (Google Gemini models)")


if __name__ == "__main__":
    main()
