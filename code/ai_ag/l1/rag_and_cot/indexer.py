"""
Indexer - Create and manage the vector database
"""

import os
import logging
import chromadb
from chromadb.config import Settings
from typing import List, Dict
from document_loader import load_all_documents, process_documents
from embeddings import generate_embeddings_batch
import config

# Get logger for this module
logger = logging.getLogger(__name__)


class VectorDatabase:
    """
    Manages the vector database for storing and retrieving documents.
    """
    
    def __init__(self, db_path: str = None, collection_name: str = None):
        """
        Initialize the vector database.
        
        Args:
            db_path: Path to store the database
            collection_name: Name of the collection
        """
        self.db_path = db_path or config.VECTOR_DB_DIR
        self.collection_name = collection_name or config.COLLECTION_NAME
        
        logger.debug(f"Initializing VectorDatabase at: {self.db_path}")
        logger.debug(f"Collection name: {self.collection_name}")
        
        # Create directory if it doesn't exist
        os.makedirs(self.db_path, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=self.db_path)
        logger.info("ChromaDB client initialized")
        
        # Get or create collection
        self.collection = None
    
    def create_collection(self):
        """
        Create a new collection (deletes existing if present).
        """
        logger.info(f"Creating collection: {self.collection_name}")
        # Delete existing collection if it exists
        try:
            self.client.delete_collection(name=self.collection_name)
            print(f"  Deleted existing collection: {self.collection_name}")
            logger.info(f"Deleted existing collection: {self.collection_name}")
        except:
            logger.debug("No existing collection to delete")
            pass
        
        # Create new collection
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"description": "Personal knowledge base"}
        )
        print(f"  Created new collection: {self.collection_name}")
        logger.info(f"Created new collection: {self.collection_name}")
    
    def get_collection(self):
        """
        Get existing collection.
        """
        logger.debug(f"Attempting to get collection: {self.collection_name}")
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
            logger.info(f"Successfully retrieved collection: {self.collection_name}")
            return True
        except Exception as e:
            logger.debug(f"Collection not found: {e}")
            return False
    
    def add_documents(self, chunks: List[Dict[str, str]]):
        """
        Add documents to the vector database.
        
        Args:
            logger.error("Collection not initialized")
            raise ValueError("Collection not initialized. Call create_collection() first.")
        """
        
        logger.info(f"Adding {len(chunks)} chunks to database")
        print("\n  Generating embeddings...")
        
        # Extract texts
        texts = [chunk["content"] for chunk in chunks]
        
        # Generate embeddings
        embeddings = generate_embeddings_batch(texts, config.EMBEDDING_MODEL)
        
        # Prepare metadata
        metadatas = [
            {
                "source": chunk["source"],
                "type": chunk["type"],
                "chunk_id": str(chunk["chunk_id"])
            }
            for chunk in chunks
        ]
        
        # Generate IDs
        ids = [f"{chunk['source']}_{chunk['chunk_id']}" for chunk in chunks]
        
        logger.debug(f"Generated {len(ids)} unique IDs for chunks")
        print(f"\n  Adding {len(chunks)} chunks to database...")
        
        # Add to collection
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"  ✓ Successfully added {len(chunks)} chunks to vector database")
        logger.info(f"  ✓ Successfully added {len(chunks)} chunks to vector database")
        
        print(f"  ✓ Successfully added {len(chunks)} chunks to vector database")
    
    def query(self, query_text: str, n_results: int = 5) -> Dict:
        """
        Query the vector database.
        
        Args:
            query_text: Query string
            n_results: Number of results to return
        """
        logger.debug(f"Querying database with: '{query_text[:50]}...' (top {n_results})")
        if not self.collection:
            if not self.get_collection():
                logger.error("Collection not found for query")
                raise ValueError("Collection not found. Please index documents first.")
        
        # Generate embedding for query
        from embeddings import generate_embedding
        query_embedding = generate_embedding(query_text, config.EMBEDDING_MODEL)
        
        # Query the collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        logger.debug(f"Query returned {len(results.get('documents', [[]])[0])} results")
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        return results


def build_vector_database():
    logger.info("=== Starting vector database build process ===")
    print("\n" + "=" * 60)
    print("Building Vector Database")
    print("=" * 60)
    
    # Step 1: Load documents
    print("\n[1/4] Loading documents...")
    logger.info("Step 1: Loading documents")
    documents = load_all_documents(
        config.LINKEDIN_FILE,
        config.GITHUB_FILE,
        config.RESUME_FILE
    )
    
    if not documents:
        logger.error("No documents found to index")
        print("\n❌ No documents found! Please add your data files to the data/ directory.")
        return False
    
    # Step 2: Process documents into chunks
    print("\n[2/4] Processing documents into chunks...")
    logger.info("Step 2: Processing documents into chunks")
    chunks = process_documents(documents, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    print(f"  ✓ Total chunks created: {len(chunks)}")
    
    # Step 3: Initialize vector database
    print("\n[3/4] Initializing vector database...")
    logger.info("Step 3: Initializing vector database")
    db = VectorDatabase()
    db.create_collection()
    
    # Step 4: Add documents to database
    print("\n[4/4] Indexing documents...")
    logger.info("Step 4: Indexing documents")
    db.add_documents(chunks)
    
    print("\n" + "=" * 60)
    print("✓ Vector Database Built Successfully!")
    print("=" * 60)
    logger.info("=== Vector database build completed successfully ===")
    print("\n" + "=" * 60)
    print("✓ Vector Database Built Successfully!")
    print("=" * 60)
    
    return True


def check_database_exists():
    logger.debug(f"Checking if database exists at: {config.VECTOR_DB_DIR}")
    if not os.path.exists(config.VECTOR_DB_DIR):
        logger.debug("Database directory does not exist")
        return False
    
    try:
        db = VectorDatabase()
        exists = db.get_collection()
        logger.info(f"Database exists check result: {exists}")
        return exists
    except Exception as e:
        logger.error(f"Error checking database existence: {e}")
    if not os.path.exists(config.VECTOR_DB_DIR):
        return False
    
    try:
        db = VectorDatabase()
        return db.get_collection()
    except:
        return False
