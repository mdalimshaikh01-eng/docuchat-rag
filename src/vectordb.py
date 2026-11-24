import os
import chromadb
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


class VectorDB:
    """
    A simple vector database wrapper using ChromaDB with HuggingFace embeddings.
    """

    def __init__(self, collection_name: str = None, embedding_model: str = None):
        """
        Initialize the vector database.

        Args:
            collection_name: Name of the ChromaDB collection
            embedding_model: HuggingFace model name for embeddings
        """
        self.collection_name = collection_name or os.getenv(
            "CHROMA_COLLECTION_NAME", "rag_documents"
        )
        self.embedding_model_name = embedding_model or os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )

        # Initialize ChromaDB client, ensuring the path is relative to the project root
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        db_path = os.path.join(project_root, "chroma_db")
        self.client = chromadb.PersistentClient(path=db_path)

        # Load embedding model
        print(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "RAG document collection"},
        )

        print(f"Vector database initialized with collection: {self.collection_name}")

    def chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """
        Simple text chunking by splitting on spaces and grouping into chunks.

        Args:
            text: Input text to chunk
            chunk_size: Approximate number of characters per chunk

        Returns:
            List of text chunks
        """
        # Using LangChain's RecursiveCharacterTextSplitter (Option 2)
        # It tries to split on different characters in order to keep chunks semantically coherent.
        # Common separators include ["\n\n", "\n", " ", ""]
        # We'll use a chunk_overlap of 10% of the chunk_size for better context preservation.
        chunk_overlap = int(chunk_size * 0.1)
        if chunk_overlap < 0: # Ensure overlap is not negative for very small chunk_size
            chunk_overlap = 0

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        chunks = text_splitter.split_text(text)

        return chunks

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add documents to the vector database.

        Args:
            documents: List of documents
        """
        print(f"Processing {len(documents)} documents...")
        
        ids = []
        doc_texts = []
        metadatas = []

        for doc_idx, doc in enumerate(documents):
            if "content" not in doc:
                continue
            
            chunks = self.chunk_text(doc["content"])
            base_metadata = doc.get("metadata", {})

            for chunk_idx, chunk in enumerate(chunks):
                ids.append(f"doc_{doc_idx}_chunk_{chunk_idx}")
                doc_texts.append(chunk)
                metadatas.append(base_metadata)

        if ids:
            print(f"Generating embeddings for {len(ids)} chunks...")
            embeddings = self.embedding_model.encode(doc_texts)
            if hasattr(embeddings, "tolist"):
                embeddings = embeddings.tolist()

            self.collection.upsert(
                ids=ids,
                documents=doc_texts,
                embeddings=embeddings,
                metadatas=metadatas
            )
            print("Documents added to vector database")
        else:
            print("No documents to add.")

    def search(self, query: str, n_results: int = 5, threshold: float = 1) -> Dict[str, Any]:
        """
        Search for similar documents in the vector database.

        Args:
            query: Search query
            n_results: Number of results to return
            threshold: Similarity threshold (lower distance is better). Results above this distance are filtered out.

        Returns:
            Dictionary containing search results with keys: 'documents', 'metadatas', 'distances', 'ids'
        """
        # Create an embedding for the search query
        query_embedding = self.embedding_model.encode([query])
        if hasattr(query_embedding, "tolist"):
            query_embedding = query_embedding.tolist()

        # Perform the query on the collection
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
        )

        # ChromaDB returns a dict with lists of lists (one for each query). We only have one query.
        # We extract the results for our single query.
        if not results or not results.get("ids") or not results["ids"][0]:
            return {"documents": [], "metadatas": [], "distances": [], "ids": []}

        # Filter results based on threshold
        # We zip the lists to handle them together, then filter
        filtered_results = {
            "documents": [],
            "metadatas": [],
            "distances": [],
            "ids": []
        }

        for i, distance in enumerate(results["distances"][0]):
            if distance < threshold:
                filtered_results["ids"].append(results["ids"][0][i])
                filtered_results["documents"].append(results["documents"][0][i])
                filtered_results["distances"].append(distance)
                filtered_results["metadatas"].append(results["metadatas"][0][i])

        return {
            "documents": filtered_results["documents"],
            "metadatas": filtered_results["metadatas"],
            "distances": filtered_results["distances"],
            "ids": filtered_results["ids"],
        }


# --- Bloc de test pour la fonction chunk_text ---
# if __name__ == "__main__":
#     print("--- Lancement du test pour VectorDB.search() ---")
#     # Utilise un nom de collection différent pour ne pas polluer la base principale
#     vdb = VectorDB(collection_name="test_collection")
#     # Vide la collection de test, mais seulement s'il y a des éléments à supprimer
#     ids_to_delete = vdb.collection.get()["ids"]
#     if ids_to_delete:
#         vdb.collection.delete(ids=ids_to_delete)

#     # 1. Ajout de documents de test
#     print("\n1. Ajout de documents de test...")
#     docs_to_add = [
#         {"content": "L'intelligence artificielle (IA) est une branche de l'informatique.", "metadata": {"source": "ai.txt"}},
#         {"content": "L'exploration de Mars est une priorité pour la NASA.", "metadata": {"source": "space.txt"}},
#     ]
#     vdb.add_documents(docs_to_add)
#     print(f"Total d'éléments dans la base de test : {vdb.collection.count()}")

#     # 2. Test avec une question pertinente
#     print("\n2. Test avec une question pertinente : 'Qu'est-ce que l'IA ?'")
#     relevant_results = vdb.search("Qu'est-ce que l'IA ?")
#     print(f"   -> Trouvé {len(relevant_results['documents'])} résultat(s).")
#     if relevant_results['documents']:
#         print(f"   -> Document le plus proche : '{relevant_results['documents'][0]}' (Distance: {relevant_results['distances'][0]:.4f})")

#     # 3. Test avec une question non pertinente
#     print("\n3. Test avec une question non pertinente : 'Quelle est la recette de la tarte aux pommes ?'")
#     irrelevant_results = vdb.search("Quelle est la recette de la tarte aux pommes ?")
#     print(f"   -> Trouvé {len(irrelevant_results['documents'])} résultat(s).")
#     if not irrelevant_results['documents']:
#         print("   -> ✅ Succès : Aucun document retourné, le filtre de pertinence (threshold) a fonctionné.")

#     print("\n--- Test terminé ---")
