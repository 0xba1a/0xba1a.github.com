import sys
import os
import json
import subprocess
import re
import logging

# Add the lib directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lib import ask_llm, setup_logging

# Get logger for this module
logger = logging.getLogger(__name__)

BUGGY_FILE_PATH = os.path.join(os.path.dirname(__file__), 'data/buggy_code.py')

SYSTEM_PROMPT = """You are an autonomous bug fixing agent.
You have access to a bash terminal.
You can read files, write files, and run commands.

Your task is to fix the bug in the python file located at: {file_path}

You must respond in STRICT JSON format with the following keys:
- "task_done": boolean (true if the bug is fixed and verified, false otherwise)
- "command": string (the bash command to execute, empty if task_done is true)
- "thoughts": string (your reasoning and plan)

Do not include any text outside the JSON object.
Do not use markdown formatting (like ```json ... ```). Just the raw JSON string.

Example response:
{{
    "task_done": false,
    "command": "cat {file_path}",
    "thoughts": "I need to read the file to understand the code and the bug."
}}
"""

def run_command(command):
    """Executes a bash command and returns the output."""
    logger.info(f"Executing command: {command}")
    print(f"\n[Executing]: {command}")
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            cwd=os.path.dirname(__file__) # Run in the agent's directory
        )
        output = result.stdout + result.stderr
        logger.debug(f"Command output: {output}")
        return output
    except Exception as e:
        logger.error(f"Command execution failed: {e}")
        return str(e)

def clean_json_response(response):
    """Attempts to extract and parse JSON from the LLM response."""
    # Remove markdown code blocks if present
    response = re.sub(r'^```json\s*', '', response, flags=re.MULTILINE)
    response = re.sub(r'^```\s*', '', response, flags=re.MULTILINE)
    response = re.sub(r'\s*```$', '', response, flags=re.MULTILINE)
    return response.strip()

def main():
    # Setup logging
    setup_logging("bug_fixing_agent")
    logger.info("=== Bug Fixing Agent Started ===")
    print("=== Bug Fixing Agent Started ===")
    
    # Initial prompt
    history = SYSTEM_PROMPT.format(file_path="data/buggy_code.py")
    
    step = 1
    while True:
        logger.info(f"--- Step {step} ---")
        print(f"\n--- Step {step} ---")
        
        # Ask LLM
        logger.debug("Sending request to LLM")
        response_str = ask_llm(history)
        logger.debug(f"LLM Response: {response_str}")
        
        # Parse JSON
        try:
            cleaned_response = clean_json_response(response_str)
            response_data = json.loads(cleaned_response)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON response from LLM. Response: {response_str}")
            print(f"[Error]: Failed to parse JSON response from LLM.\nResponse:\n{response_str}")
            # Feed the error back to the LLM
            history += f"\n\nInvalid JSON response. Please respond with valid JSON only.\nResponse was: {response_str}"
            continue
            
        thoughts = response_data.get("thoughts", "")
        command = response_data.get("command", "")
        task_done = response_data.get("task_done", False)
        
        logger.info(f"Thoughts: {thoughts}")
        print(f"[Thoughts]: {thoughts}")
        
        if task_done:
            logger.info("Task completed successfully")
            print("\n=== Task Completed ===")
            break
        
        # Execute command
        if command:
            output = run_command(command)
            print(f"[Output]:\n{output}")
            
            # Update history
            history += f"\n\nResponse:\n{cleaned_response}\n\nCommand Output:\n{output}"
        else:
            logger.warning("No command provided but task_done is false")
            print("[Warning]: No command provided but task_done is false.")
            history += f"\n\nResponse:\n{cleaned_response}\n\nError: No command provided."

        step += 1

if __name__ == "__main__":
    main()
