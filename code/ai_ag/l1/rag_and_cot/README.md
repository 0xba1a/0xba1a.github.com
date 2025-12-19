# Module 2: Personal Alter-Ego RAG Chatbot

Build an AI-powered chatbot that answers questions about you using your LinkedIn profile, GitHub profile, and resume. This chatbot uses **RAG (Retrieval Augmented Generation)** to provide accurate, personalized responses based on your actual data.

## 📋 Table of Contents

- [Prerequisites](#prerequisites)
- [Architecture](#architecture)
- [Data Preparation](#data-preparation)
- [Installation](#installation)
- [How It Works](#how-it-works)
- [Recommended Models](#recommended-models)

---

## Prerequisites

### Software Requirements

1. **Python 3.8+**
2. **Ollama** (for local LLM)
   ```bash
   # Install Ollama from https://ollama.ai
   curl -fsSL https://ollama.com/install.sh | sh
   ```

3. **Required Python packages**:
   ```bash
   pip install chromadb langchain langchain-community pypdf requests
   ```

### Ollama Models

Pull the required models:

```bash
# Embedding model (converts text to vectors)
ollama pull nomic-embed-text

# Language model (generates responses)
ollama pull llama3.2

# Alternative language models (optional)
ollama pull mistral
ollama pull qwen2.5
```

---

## Architecture

See [architecture.md](architecture.md) for detailed system design and flow diagrams.

**Quick Summary:**
1. **Collect** your LinkedIn, GitHub, and Resume data
2. **Process** documents into smaller chunks
3. **Index** chunks using embeddings in a vector database
4. **Query** by finding relevant chunks and generating answers

---

## Data Preparation

### 1. Export LinkedIn Profile

#### Request Your Data Archive (Recommended)
1. Go to **Settings & Privacy** → **Data Privacy**
2. Click **"Get a copy of your data"**
3. Select **"Download larger data archive"**
4. Wait for email (takes 10 minutes to 24 hours)
5. Extract the ZIP file
6. Copy `Profile.csv` to the `data/` directory

**The system will automatically parse the `Profile.csv` file.**

Other useful CSV files from LinkedIn export (optional):
- `Positions.csv` - Work experience
- `Education.csv` - Education history
- `Skills.csv` - Skills list

**Note:** The code is designed to work with LinkedIn's `Profile.csv` format. Just place the file in `data/Profile.csv` and you're ready to go!

---

### 2. Export GitHub Profile

Use the GitHub API to fetch your profile data:

#### Using Command Line:

```bash
# Replace YOUR_USERNAME with your GitHub username
curl https://api.github.com/users/YOUR_USERNAME > github_profile.json

# Get your repositories
curl https://api.github.com/users/YOUR_USERNAME/repos?per_page=100 > github_repos.json
```

#### Using Python:

```python
import requests
import json

username = "YOUR_USERNAME"

# Get profile
profile = requests.get(f"https://api.github.com/users/{username}").json()

# Get repositories
repos = requests.get(f"https://api.github.com/users/{username}/repos?per_page=100").json()

# Create simplified summary
github_data = {
    "name": profile.get("name"),
    "bio": profile.get("bio"),
    "location": profile.get("location"),
    "company": profile.get("company"),
    "blog": profile.get("blog"),
    "public_repos": profile.get("public_repos"),
    "followers": profile.get("followers"),
    "repositories": [
        {
            "name": repo["name"],
            "description": repo["description"],
            "language": repo["language"],
            "stars": repo["stargazers_count"],
            "url": repo["html_url"]
        }
        for repo in repos
    ]
}

with open("github_profile.json", "w") as f:
    json.dump(github_data, f, indent=2)
```

**What to include:**
- ✅ Profile information (name, bio, location)
- ✅ Repository names and descriptions
- ✅ Primary programming languages
- ❌ Skip: Individual commits, pull request details (too much noise)

---

### 3. PDF Resume

Simply have your resume in PDF format. The system will extract text automatically.

**Recommended filename:** `resume.pdf`

---

## How It Works

### Step 1: Document Loading
```python
# Load all your documents
- linkedin_profile.json → Parsed and converted to text
- github_profile.json → Parsed and converted to text  
- resume.pdf → Text extracted using PDF parser
```

### Step 2: Text Chunking
```python
# Break documents into smaller chunks
# Why? Embeddings work better with focused content
- Chunk size: 500-1000 characters
- Overlap: 100-200 characters (preserves context)
```

### Step 3: Create Embeddings
```python
# Convert each chunk to a vector (list of numbers)
# These vectors capture semantic meaning

chunk = "Worked as Software Engineer at Company X"
embedding = [0.123, -0.456, 0.789, ...] # 768 dimensions
```

### Step 4: Store in Vector Database
```python
# ChromaDB stores embeddings and enables similarity search
db = ChromaDB()
db.add(chunks, embeddings, metadata)
```

### Step 5: Query (RAG in Action)
```python
# User asks: "What projects has the person worked on?"

1. Convert question → embedding vector
2. Search vector DB for similar chunks
3. Retrieve top 3-5 most relevant chunks
4. Build prompt:
   """
   Context: [Retrieved chunks about projects]
   Question: What projects has the person worked on?
   Answer as if you are the person.
   """
5. Send to LLM → Get personalized answer
```

---

## Recommended Models

### For Embeddings (Text → Vectors)

**Primary Choice: `nomic-embed-text`**
- Size: ~274MB
- Quality: Excellent for English text
- Speed: Fast
- Use case: Perfect for personal documents

```bash
ollama pull nomic-embed-text
```

### For Text Generation (Answering Questions)

#### Option 1: **`llama3.2` (Recommended for beginners)**
```bash
ollama pull llama3.2
```
- Size: ~2GB (3B parameters)
- Quality: Good balance of speed and accuracy
- RAM: ~4GB needed
- Best for: Quick responses, lower-end hardware

#### Option 2: **`mistral`**
```bash
ollama pull mistral
```
- Size: ~4.1GB (7B parameters)
- Quality: Better understanding, more detailed answers
- RAM: ~8GB needed
- Best for: More nuanced responses

#### Option 3: **`qwen2.5`**
```bash
ollama pull qwen2.5:7b
```
- Size: ~4.7GB (7B parameters)
- Quality: Excellent for technical content
- RAM: ~8GB needed
- Best for: Technical profiles with code

### Model Size vs Quality Trade-off

```
Model Size (Parameters)  │  Quality  │  Speed  │  RAM Needed
─────────────────────────┼───────────┼─────────┼─────────────
llama3.2 (3B)           │    ⭐⭐⭐   │  ⚡⚡⚡   │    4GB
mistral (7B)            │   ⭐⭐⭐⭐  │   ⚡⚡    │    8GB
llama3.1 (8B)           │   ⭐⭐⭐⭐  │   ⚡⚡    │    8GB
qwen2.5 (7B)            │   ⭐⭐⭐⭐  │   ⚡⚡    │    8GB
```

**Recommendation:** Start with `llama3.2` for development, upgrade to `mistral` or `qwen2.5` for production.

---

## Project Structure

```
m2/
├── README.md              # This file
├── architecture.md        # Architecture diagrams
├── data/                  # Your personal data
│   ├── Profile.csv        # LinkedIn Profile.csv
│   ├── github_profile.json
│   └── resume.pdf
├── vector_db/             # ChromaDB storage (created automatically)
├── document_loader.py     # Load and parse documents
├── indexer.py             # Create embeddings and store in DB
├── chatbot.py             # Main chatbot interface
└── config.py              # Configuration settings
```

---

## Learning Objectives

By completing this module, you'll understand:

- ✅ **RAG Architecture**: How to combine retrieval and generation
- ✅ **Vector Databases**: Why and how to use embeddings
- ✅ **Document Processing**: Chunking strategies for better retrieval
- ✅ **Prompt Engineering**: Building effective context-aware prompts
- ✅ **Local LLMs**: Running AI models privately on your machine

---

## Next Steps

1. Follow the data preparation steps above
2. Place your data files in the `data/` directory:
   - `Profile.csv` (from LinkedIn data export)
   - `github_profile.json` (use get_github_profile.py)
   - `resume.pdf`
3. Run `python alter_ego.py` - it will automatically index and start chatting!

---

## Privacy Note

🔒 **All data stays local on your machine**
- No cloud APIs required
- Your personal information never leaves your computer
- Ollama runs entirely locally
- ChromaDB is a local file-based database
- LinkedIn CSV files are parsed locally without any external calls

---

## Troubleshooting

### "Out of memory" errors
- Use a smaller model (llama3.2 instead of mistral)
- Reduce chunk size
- Close other applications

### Slow responses
- First query is always slower (model loading)
- Use smaller model for faster inference
- Reduce number of retrieved chunks

### Irrelevant answers
- Increase chunk overlap for better context
- Retrieve more chunks (top 5 instead of 3)
- Improve your source documents with more details

---

## Additional Resources

- [Ollama Documentation](https://github.com/ollama/ollama)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [LangChain Documentation](https://python.langchain.com/)
- [RAG Explanation (Visual)](https://www.youtube.com/watch?v=T-D1OfcDW1M)
