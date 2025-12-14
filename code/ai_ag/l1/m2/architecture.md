# Module 2 Architecture: Personal Alter-Ego RAG Chatbot

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DATA COLLECTION PHASE                        │
└─────────────────────────────────────────────────────────────────────┘

    ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
    │   LinkedIn   │      │    GitHub    │      │  PDF Resume  │
    │   Profile    │      │   Profile    │      │              │
    │   (JSON)     │      │   (JSON)     │      │   (PDF)      │
    └──────┬───────┘      └──────┬───────┘      └──────┬───────┘
           │                     │                     │
           └─────────────────────┼─────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      DOCUMENT PROCESSING PHASE                       │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │  Document Loader        │
                    │  - Parse JSON files     │
                    │  - Extract PDF text     │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │  Text Chunker          │
                    │  - Split into chunks   │
                    │  - 500-1000 tokens     │
                    │  - Preserve context    │
                    └────────────┬───────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        INDEXING PHASE                                │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │  Embedding Model        │
                    │  (nomic-embed-text)     │
                    │  - Generate vectors     │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │  Vector Database        │
                    │  (ChromaDB)             │
                    │  - Store embeddings     │
                    │  - Enable similarity    │
                    └────────────┬────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      QUERY/CHAT PHASE (RAG)                          │
└─────────────────────────────────────────────────────────────────────┘

         ┌──────────────────────────────────────────┐
         │       User asks a question               │
         └─────────────────┬────────────────────────┘
                           │
                           ▼
         ┌──────────────────────────────────────────┐
         │    Convert question to embedding         │
         │    (nomic-embed-text)                    │
         └─────────────────┬────────────────────────┘
                           │
                           ▼
         ┌──────────────────────────────────────────┐
         │  Search Vector DB for similar chunks     │
         │  (Top 3-5 most relevant)                 │
         └─────────────────┬────────────────────────┘
                           │
                           ▼
         ┌──────────────────────────────────────────┐
         │  Retrieve relevant context chunks        │
         └─────────────────┬────────────────────────┘
                           │
                           ▼
         ┌──────────────────────────────────────────┐
         │  Build prompt with:                      │
         │  1. Retrieved context                    │
         │  2. User question                        │
         │  3. System instructions                  │
         └─────────────────┬────────────────────────┘
                           │
                           ▼
         ┌──────────────────────────────────────────┐
         │  Send to LLM (llama3.2 or mistral)       │
         └─────────────────┬────────────────────────┘
                           │
                           ▼
         ┌──────────────────────────────────────────┐
         │  LLM generates personalized response     │
         └─────────────────┬────────────────────────┘
                           │
                           ▼
         ┌──────────────────────────────────────────┐
         │     Return answer to user                │
         └──────────────────────────────────────────┘
```

## Component Explanation

### 1. **Data Collection**
- Export your LinkedIn profile as JSON
- Fetch GitHub profile data via API
- Provide PDF resume

### 2. **Document Processing**
- Load and parse all documents
- Split text into manageable chunks
- Maintain context between chunks

### 3. **Indexing (Creating Vector Database)**
- Convert text chunks to embeddings (numerical vectors)
- Store embeddings in ChromaDB
- Enable fast similarity search

### 4. **Query Processing (RAG)**
- **R**etrieval: Find relevant information from vector DB
- **A**ugmented: Add context to the user's question
- **G**eneration: LLM generates accurate response based on your data

## Why RAG?

Without RAG, the LLM would only have general knowledge. With RAG:
- ✅ Answers are based on YOUR specific data
- ✅ Reduces hallucinations
- ✅ Keeps data local and private
- ✅ Can update knowledge without retraining

## Key Technologies

- **Ollama**: Local LLM runtime
- **ChromaDB**: Lightweight vector database
- **nomic-embed-text**: Embedding model for creating vectors
- **llama3.2 / mistral**: Language models for generating responses
