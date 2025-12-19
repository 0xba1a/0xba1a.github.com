# AI Agent Course - Level 1

## Module 1: Joke Explainer

A simple Python program that demonstrates basic AI agent concepts by fetching programming jokes and using a local LLM to explain them.

### Prerequisites

1. **Python 3.7+** installed
2. **Ollama** running locally with a model (e.g., llama3.2)
3. **Required Python packages**:
   ```bash
   pip install requests
   ```

### Project Structure

```
l1/
├── lib/              # Shared library modules
│   ├── __init__.py
│   └── llm.py        # LLM interaction library
└── m1/               # Module 1 - Joke Explainer
    ├── __init__.py
    ├── joke_api.py   # Joke API interaction
    ├── main.py       # Main application
    └── README.md
```

### How to Run

1. Make sure Ollama is running locally:
   ```bash
   ollama serve
   ```

2. Navigate to the module directory:
   ```bash
   cd code/ai_ag/l1/m1
   ```

3. Run the program:
   ```bash
   python main.py
   ```

### What It Does

1. **Fetches a joke**: Calls the official jokes API to get a random programming joke
2. **Requests explanation**: Sends the joke to a locally running Ollama LLM
3. **Displays results**: Shows both the joke and the LLM's explanation

### Learning Objectives

- Making HTTP API calls
- Interacting with local LLMs
- Structuring code with libraries and modules
- Simple prompt engineering
