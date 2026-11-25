"""
=====================================================
Module: rag.py
Purpose: Retrieval-Augmented Generation (RAG) system.
         Loads the company knowledge base (TXT/PDF),
         creates semantic embeddings with Sentence
         Transformers, stores them in FAISS, and
         retrieves relevant chunks for each query.

Flow:
  1. Load documents → split into chunks
  2. Embed chunks → store in FAISS index
  3. For each query → find top-K relevant chunks
  4. Return chunks as context for the LLM
=====================================================
"""

import os
import logging
from typing import List, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

# Knowledge base directory
DATA_DIR = "data"
KNOWLEDGE_BASE_FILE = os.path.join(DATA_DIR, "sample_company_knowledge.txt")

# RAG Configuration
CHUNK_SIZE = 500          # Characters per chunk
CHUNK_OVERLAP = 50        # Overlap between chunks to preserve context
TOP_K_RESULTS = 3         # Number of chunks to retrieve per query
MIN_SIMILARITY = 0.3      # Minimum cosine similarity score


class SimpleRAG:
    """
    A lightweight RAG system built with:
    - SentenceTransformers for embeddings (free, local)
    - FAISS for vector similarity search (free, local)
    - Custom text chunking (no LangChain dependency required)

    Can optionally use LangChain document loaders for PDFs.
    """

    def __init__(self):
        self.chunks: List[str] = []          # Text chunks
        self.chunk_sources: List[str] = []   # Source file for each chunk
        self.embeddings = None               # numpy array of embeddings
        self.index = None                    # FAISS index
        self.model = None                    # SentenceTransformer model
        self._initialized = False

    def _load_embedding_model(self) -> bool:
        """
        Load the SentenceTransformer embedding model.
        'all-MiniLM-L6-v2' is fast, free and ~80MB.
        Perfect for semantic similarity tasks.
        """
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading embedding model (all-MiniLM-L6-v2)...")
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("✅ Embedding model loaded")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load embedding model: {e}")
            return False

    def _chunk_text(self, text: str, source: str = "knowledge_base") -> List[Tuple[str, str]]:
        """
        Split text into overlapping chunks of CHUNK_SIZE characters.
        Returns list of (chunk_text, source) tuples.
        """
        chunks = []
        # Split by paragraphs first for more natural boundaries
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        current_chunk = ""
        for para in paragraphs:
            # If adding this paragraph exceeds chunk size, save current and start new
            if len(current_chunk) + len(para) > CHUNK_SIZE and current_chunk:
                chunks.append((current_chunk.strip(), source))
                # Keep overlap
                current_chunk = current_chunk[-CHUNK_OVERLAP:] + " " + para
            else:
                current_chunk = current_chunk + " " + para if current_chunk else para

        # Don't forget the last chunk
        if current_chunk.strip():
            chunks.append((current_chunk.strip(), source))

        return chunks

    def load_document(self, file_path: str) -> bool:
        """
        Load a document file (TXT or PDF) into the RAG system.
        Returns True on success.
        """
        file_path = str(file_path)

        if not os.path.exists(file_path):
            logger.warning(f"Document not found: {file_path}")
            return False

        file_ext = Path(file_path).suffix.lower()
        source_name = Path(file_path).name
        text = ""

        try:
            if file_ext == ".txt":
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()

            elif file_ext == ".pdf":
                # Try LangChain PDF loader first (handles complex PDFs)
                try:
                    from langchain_community.document_loaders import PyPDFLoader
                    loader = PyPDFLoader(file_path)
                    pages = loader.load()
                    text = "\n\n".join([page.page_content for page in pages])
                    logger.info(f"Loaded PDF with LangChain: {len(pages)} pages")
                except ImportError:
                    # Fallback to pypdf directly
                    import pypdf
                    with open(file_path, "rb") as f:
                        reader = pypdf.PdfReader(f)
                        text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
                    logger.info(f"Loaded PDF with pypdf: {len(reader.pages)} pages")

            else:
                logger.warning(f"Unsupported file type: {file_ext}")
                return False

            if not text.strip():
                logger.warning(f"No text extracted from: {file_path}")
                return False

            # Chunk the document
            new_chunks = self._chunk_text(text, source=source_name)
            self.chunks.extend([c[0] for c in new_chunks])
            self.chunk_sources.extend([c[1] for c in new_chunks])

            logger.info(f"✅ Loaded '{source_name}': {len(new_chunks)} chunks created")
            return True

        except Exception as e:
            logger.error(f"Error loading document '{file_path}': {e}")
            return False

    def build_index(self) -> bool:
        """
        Create FAISS vector index from all loaded document chunks.
        Must be called after loading documents.
        """
        if not self.chunks:
            logger.warning("No documents loaded. Cannot build index.")
            return False

        if not self._load_embedding_model():
            return False

        try:
            import faiss
            import numpy as np

            logger.info(f"Building FAISS index for {len(self.chunks)} chunks...")

            # Generate embeddings for all chunks
            self.embeddings = self.model.encode(
                self.chunks,
                batch_size=32,
                show_progress_bar=False,
                convert_to_numpy=True
            )

            # Create FAISS index (Inner Product for cosine similarity)
            # Normalize embeddings first for cosine similarity
            faiss.normalize_L2(self.embeddings)
            dimension = self.embeddings.shape[1]

            self.index = faiss.IndexFlatIP(dimension)  # IP = Inner Product
            self.index.add(self.embeddings)

            self._initialized = True
            logger.info(f"✅ FAISS index built with {self.index.ntotal} vectors")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to build FAISS index: {e}")
            return False

    def initialize(self, additional_files: List[str] = None) -> bool:
        """
        Full initialization: load default knowledge base + any extra files,
        then build the FAISS index.
        """
        # Load default knowledge base
        self.load_document(KNOWLEDGE_BASE_FILE)

        # Load any additional uploaded files
        if additional_files:
            for file_path in additional_files:
                self.load_document(file_path)

        # Build the index
        return self.build_index()

    def retrieve(self, query: str, top_k: int = TOP_K_RESULTS) -> List[str]:
        """
        Find the most relevant text chunks for a given query.

        Args:
            query: The user's question or message
            top_k: Number of chunks to retrieve

        Returns:
            List of relevant text chunks (empty list if RAG not initialized)
        """
        if not self._initialized or self.index is None:
            logger.warning("RAG not initialized. Returning empty context.")
            return []

        try:
            import faiss
            import numpy as np

            # Embed the query
            query_embedding = self.model.encode(
                [query],
                convert_to_numpy=True
            )
            faiss.normalize_L2(query_embedding)

            # Search the FAISS index
            distances, indices = self.index.search(query_embedding, min(top_k, len(self.chunks)))

            # Filter by minimum similarity and collect results
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx >= 0 and dist >= MIN_SIMILARITY:
                    results.append(self.chunks[idx])

            logger.debug(f"RAG retrieved {len(results)} chunks for query: '{query[:50]}...'")
            return results

        except Exception as e:
            logger.error(f"RAG retrieval failed: {e}")
            return []

    def get_context_string(self, query: str) -> str:
        """
        Retrieve relevant chunks and format them as a context string
        to inject into the LLM prompt.
        """
        chunks = self.retrieve(query)
        if not chunks:
            return ""

        context = "RELEVANT COMPANY INFORMATION:\n" + "─" * 40 + "\n"
        for i, chunk in enumerate(chunks, 1):
            context += f"\n[Source {i}]:\n{chunk}\n"
        context += "─" * 40

        return context

    def add_uploaded_file(self, file_content: bytes, filename: str) -> bool:
        """
        Add a new document uploaded by the user and rebuild the index.

        Args:
            file_content: Raw bytes of the uploaded file
            filename: Original filename (used to determine type)

        Returns:
            True on success
        """
        # Save to uploads directory
        uploads_dir = os.path.join(DATA_DIR, "uploads")
        os.makedirs(uploads_dir, exist_ok=True)

        file_path = os.path.join(uploads_dir, filename)
        with open(file_path, "wb") as f:
            f.write(file_content)

        # Load the new file
        if self.load_document(file_path):
            # Rebuild index with the new document included
            return self.build_index()

        return False


# --- Module-level singleton instance ---
# Used across all pages of the Streamlit app
_rag_instance: Optional[SimpleRAG] = None


def get_rag_instance() -> SimpleRAG:
    """Get or create the singleton RAG instance."""
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = SimpleRAG()
    return _rag_instance


def initialize_rag(additional_files: List[str] = None) -> bool:
    """Initialize the global RAG instance."""
    rag = get_rag_instance()
    if not rag._initialized:
        return rag.initialize(additional_files=additional_files)
    return True
