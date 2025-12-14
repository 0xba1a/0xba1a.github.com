# Quick Start Guide - Module 2

## Setup

1. **Install dependencies:**
   ```bash
   cd code/ai_ag/l1/m2
   pip install -r requirements.txt
   ```

2. **Install and run Ollama:**
   ```bash
   # Install Ollama (if not already installed)
   curl -fsSL https://ollama.com/install.sh | sh
   
   # Pull required models
   ollama pull nomic-embed-text
   ollama pull llama3.3
   ```

3. **Prepare your data:**

   ### Option A: Use the GitHub profile downloader
   ```bash
   python get_github_profile.py YOUR_GITHUB_USERNAME
   ```
   This creates `data/github_profile.json`

   ### Option B: Manual GitHub export
   ```bash
   curl https://api.github.com/users/YOUR_USERNAME > data/github_profile.json
   ```

   ### LinkedIn Profile
   **Option 1: Export from LinkedIn (Recommended)**
   1. Go to LinkedIn Settings & Privacy → Data Privacy
   2. Request "Download larger data archive"
   3. Wait for email with download link
   4. Extract ZIP file
   5. Copy `Profile.csv` to `data/Profile.csv`

   **Option 2: Create manually (for testing)**
   Create a CSV file `data/Profile.csv` with basic info:
   ```csv
   First Name,Last Name,Headline,Summary
   John,Doe,Software Engineer,Experienced developer...
   ```

   ### Resume
   Place your resume as `data/resume.pdf`

## Run the Chatbot

```bash
python alter_ego.py
```

The first run will:
1. ✅ Check for existing vector database
2. 🔨 Build the database (if needed) - shows progress
3. 🤖 Start the interactive chat

## Example Session

```
============================================================
          WELCOME TO ALTER-EGO CHATBOT
============================================================

📊 Vector database not found. Building it now...
This is a one-time process and may take a few minutes.

============================================================
Building Vector Database
============================================================

[1/4] Loading documents...
  Loading LinkedIn profile...
    ✓ Loaded (2345 characters)
  Loading GitHub profile...
    ✓ Loaded (8921 characters)
  Loading resume PDF...
    ✓ Loaded (3456 characters)

[2/4] Processing documents into chunks...

  Chunking documents...
    linkedin: 4 chunks created
    github: 12 chunks created
    resume: 5 chunks created
  ✓ Total chunks created: 21

[3/4] Initializing vector database...
  Created new collection: personal_knowledge

[4/4] Indexing documents...

  Generating embeddings...
    Processing chunk 10/21...
    Processing chunk 20/21...

  Adding 21 chunks to database...
  ✓ Successfully added 21 chunks to vector database

============================================================
✓ Vector Database Built Successfully!
============================================================

🤖 Starting chatbot...

✓ Chatbot initialized and ready!

============================================================
Alter-Ego Chatbot
============================================================

Ask me anything! (Type 'quit' or 'exit' to stop)

You: What programming languages do you know?

Alter-Ego: I'm proficient in Python, JavaScript, and Go. I have extensive 
experience with Python for backend development and data science projects, 
JavaScript for web development, and Go for building scalable microservices...

You: Tell me about your work experience

Alter-Ego: I've worked as a Software Engineer at Company X for the past 3 years, 
where I focused on building cloud-native applications...

You: quit

Alter-Ego: Goodbye! Have a great day!
```

## File Structure

```
m2/
├── alter_ego.py              # Main application (RUN THIS)
├── get_github_profile.py     # GitHub profile downloader
├── config.py                 # Configuration settings
├── document_loader.py        # Document parsing
├── embeddings.py             # Embedding generation
├── indexer.py                # Vector database builder
├── chatbot.py                # RAG chatbot logic
├── requirements.txt          # Python dependencies
├── data/                     # Your personal data
│   ├── Profile.csv           # LinkedIn Profile.csv
│   ├── github_profile.json
│   └── resume.pdf
└── vector_db/                # Created automatically
    └── chroma.sqlite3
```

## Troubleshooting

### "Collection not found" error
```bash
# Delete the vector_db directory and rebuild
rm -rf vector_db/
python alter_ego.py
```

### Slow responses
- First query is slower (model loading)
- Use a smaller model in config.py (already using llama3.2)
- Reduce TOP_K_RESULTS in config.py

### Out of memory
- Close other applications
- Use an even smaller model if available

## Customization

Edit [config.py](config.py) to customize:
- `CHAT_MODEL`: Change the LLM model
- `EMBEDDING_MODEL`: Change the embedding model
- `CHUNK_SIZE`: Adjust chunk size
- `TOP_K_RESULTS`: Number of relevant chunks to retrieve

## Next Steps

- Add more documents to `data/` and rebuild
- Experiment with different models
- Modify the chatbot personality in [chatbot.py](chatbot.py)
