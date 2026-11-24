# DocuChat‑RAG 🧠📄

### A Local Retrieval‑Augmented Generation (RAG) AI Assistant

DocuChat‑RAG is a **local‑first conversational AI assistant** that
allows users to interact with their own documents using natural
language.

It implements a **Retrieval‑Augmented Generation (RAG)** architecture
that combines **semantic document search** with **powerful LLMs** to
generate grounded answers based strictly on the user's data.

The assistant can run with multiple LLM providers and keeps a
**lightweight conversational memory** while preserving **privacy and
efficiency**.

------------------------------------------------------------------------

# 🚀 Project Overview

DocuChat‑RAG enables users to:

• Load local documents (.txt files)\
• Convert documents into vector embeddings\
• Store them in a **local vector database (ChromaDB)**\
• Retrieve relevant document chunks using semantic similarity\
• Generate accurate answers using LLMs

Unlike generic chatbots, **DocuChat only answers using your documents**.

This significantly **reduces hallucinations** and ensures **trustworthy
responses**.

------------------------------------------------------------------------

## 🏗️ System Architecture

DocuChat‑RAG follows a modular Retrieval‑Augmented Generation
architecture composed of four main layers:

### 1️⃣ Document Processing Layer

-   Loads `.txt` documents from the `data/` directory
-   Splits documents into semantic chunks using
    `RecursiveCharacterTextSplitter`

### 2️⃣ Vector Retrieval Layer

-   Converts chunks into embeddings using `SentenceTransformer`
-   Stores embeddings in **ChromaDB**
-   Performs similarity search to retrieve the most relevant document
    chunks

### 3️⃣ Generation Layer

-   Injects retrieved context into a structured prompt
-   Uses an LLM (OpenAI, Groq, or Gemini)
-   Generates answers grounded strictly in the retrieved documents

### 4️⃣ Conversation Memory

-   Uses `FileChatMessageHistory`
-   Maintains a window of the **last 5 interactions**
-   Persists conversation history to a JSON file

The system ensures low hallucination through strict prompt constraints
and context‑grounded responses.

------------------------------------------------------------------------

## 🔁 RAG Pipeline

``` mermaid
flowchart TD

User[User Question]

User --> EmbedQuery[Query Embedding]

EmbedQuery --> VectorSearch[Similarity Search - ChromaDB]

VectorSearch --> RetrieveDocs[Retrieve Relevant Chunks]

RetrieveDocs --> Prompt[Prompt Template]

Prompt --> LLM[LLM Generation]

LLM --> Answer[Final Response]

ChatHistory[Conversation Memory] --> Prompt
```

------------------------------------------------------------------------

# 🧠 Key Features

## 📂 Document Ingestion

The system automatically reads text files from the **data/** directory
and prepares them for semantic search.

Steps: 1. Load documents 2. Split text into chunks 3. Generate
embeddings 4. Store vectors in ChromaDB

------------------------------------------------------------------------

## 🔎 Semantic Retrieval

When a user asks a question:

1.  The query is converted into an embedding
2.  The vector database searches for the **most relevant chunks**
3.  The top results are sent to the language model as context

This ensures answers are **grounded in the source documents**.

------------------------------------------------------------------------

## 💬 Conversational Memory (Optimized)

DocuChat uses:

• **ConversationBufferWindowMemory**\
• **FileChatMessageHistory**

This allows the assistant to:

• Remember the **last few conversation turns**\
• Maintain conversational coherence\
• Keep memory usage extremely low

Only a **small window of recent messages** is injected into the prompt,
while the full history can optionally be saved to JSON.

To keep the system lightweight:

✔ Only the last **5 exchanges** are used as context\
✔ The history file is **reset on each new session**

------------------------------------------------------------------------

## 🤖 Multi‑LLM Support

DocuChat automatically selects the available model depending on which
API key is provided.

Supported providers:

• **OpenAI** (GPT‑4o)\
• **Groq** (Llama‑3)\
• **Google Gemini**

This architecture allows developers to **switch providers without
changing code**.

------------------------------------------------------------------------

# 🏗️ Project Architecture

The project follows a modular architecture to ensure clarity,
maintainability, and extensibility.

    DocuChat-RAG/
    │
    ├── data/                    # Knowledge base documents
    │   └── *.txt
    │
    ├── src/
    │   ├── app.py               # Main application entry point
    │   ├── vectordb.py          # Vector database logic
    │   └── memory.py            # Conversational memory management
    │
    ├── chroma_db/               # Persistent vector database
    │
    ├── chat_history.json        # Persistent conversation history
    │
    ├── requirements.txt         # Core project dependencies (minimal packages required to run project)
    ├── requirements-frozen.txt  # Fully pinned dependencies generated with pip freeze for reproducibility
    ├── .env.example             # Environment variables template
    ├── LICENSE                  # MIT License
    └── README.md                # Project documentation

------------------------------------------------------------------------

# 🧩 Core Components

## app.py

Main application entry point.

Responsibilities: - Load documents - Initialize vector database - Manage
the RAG pipeline - Handle conversation memory - Run the interactive chat
loop

## vectordb.py

Wrapper around ChromaDB.

Responsibilities: - Text chunking - Embedding generation - Document
ingestion - Similarity search

## memory.py

Implements conversation memory.

Features: - File‑based history persistence - Windowed memory (last 5
interactions) - Optimized context injection into prompts

------------------------------------------------------------------------

# ⚙️ Technical Components

## Vector Database

DocuChat uses **ChromaDB**.

Advantages:

• Local storage\
• Fast semantic search\
• No external infrastructure required

------------------------------------------------------------------------

## Embedding Model

Embeddings are generated using **Sentence Transformers**.

Default model:

    sentence-transformers/all-MiniLM-L6-v2

Characteristics:

• 384‑dimension embeddings\
• Fast CPU inference\
• Excellent semantic similarity performance

------------------------------------------------------------------------

## Text Chunking

Documents are split using **RecursiveCharacterTextSplitter**.

Typical parameters:

    chunk_size = 500
    chunk_overlap = 50

This ensures chunks preserve enough context while remaining searchable.

------------------------------------------------------------------------

# 🛠️ Installation

## 1️⃣ Clone the Repository

``` bash
git clone https://github.com/yourusername/docuchat-rag.git
cd docuchat-rag
```

------------------------------------------------------------------------

## 2️⃣ Create Virtual Environment

``` bash
python -m venv venv
```

Activate it:

Windows

    venv\Scripts\activate

Linux / Mac

    source venv/bin/activate

------------------------------------------------------------------------

## 3️⃣ Install Dependencies

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

# 🔑 Environment Configuration

Create a `.env` file:

    OPENAI_API_KEY=your_key
    GROQ_API_KEY=your_key
    GOOGLE_API_KEY=your_key

Only **one key is required**.

The system automatically detects the available provider.

------------------------------------------------------------------------

# ▶️ Running the Application

Start the assistant:

``` bash
python src/app.py
```

Example interaction:

    You: What is RAG?

    Assistant:
    Retrieval‑Augmented Generation (RAG) is a technique that combines
    document retrieval with language models to produce grounded answers.

------------------------------------------------------------------------

# 📊 RAG Pipeline Explained

1️⃣ User asks a question\
2️⃣ The query is converted into an embedding\
3️⃣ ChromaDB performs a similarity search\
4️⃣ The most relevant document chunks are retrieved\
5️⃣ Retrieved context is injected into a prompt template\
6️⃣ The LLM generates a grounded response\
7️⃣ The conversation memory updates with the new interaction

------------------------------------------------------------------------

# 💡 Design Principles

DocuChat‑RAG was built following modern AI engineering principles:

-   **Modular architecture**
-   **LLM‑agnostic design**
-   **Retrieval‑grounded generation**
-   **Conversation context awareness**
-   **Local vector database for privacy**

------------------------------------------------------------------------

# 🧪 Example Use Cases

DocuChat can be adapted for many applications:

• Personal knowledge assistants\
• Company internal documentation search\
• Research paper assistants\
• Local AI assistants for private data\
• Developer documentation bots

------------------------------------------------------------------------


# 🔐 Privacy

DocuChat is designed for **local AI usage**.

Benefits:

✔ Documents stay on your machine\
✔ No external database required\
✔ Vector search runs locally

Only the **LLM API call** leaves the machine when using cloud models.

------------------------------------------------------------------------

# 🧭 Future Improvements

Planned upgrades:

• Local LLM support (Ollama)\
• Web interface (Streamlit / React)\
• Document support for PDF & Markdown\
• Advanced reranking for retrieval\
• Hybrid search (keyword + vector)

------------------------------------------------------------------------

# 👨‍💻 Author

**MOKEMI TCHAKOUTE BAUTREL Duval**\
AI Engineer & Developer

------------------------------------------------------------------------

# 📜 License

MIT License

Please see the **LICENSE** file for more information.
#   d o c u c h a t - r a g  
 #   d o c u c h a t - r a g  
 