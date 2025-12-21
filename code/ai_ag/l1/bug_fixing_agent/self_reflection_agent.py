import sys
import os
import json
import subprocess
import re
import logging

# Add the lib directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lib import ask_llm, ask_llm_azure_openai, setup_logging

# Get logger for this module
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an autonomous AI agent.
You have access to a bash terminal.
You can read files, write files, and run commands.
You are working in the 'code_base' directory of the project.

Your task is described below. You should understand it and fix the issue:
{problem_statement}

You must respond in STRICT JSON format with the following keys:
- "task_done": boolean (true if the task is completed, false otherwise)
- "command": string (the bash command to execute, empty if task_done is true)
- "thoughts": string (your reasoning and plan)

Do not include any text outside the JSON object.
Do not use markdown formatting (like ```json ... ```). Just the raw JSON string.

Example response:
{{
    "task_done": false,
    "command": "ls -F",
    "thoughts": "I need to explore the directory to understand the project structure."
}}
"""

def run_command(command, cwd):
    """Executes a bash command and returns the output."""
    logger.info(f"Executing command: {command}")
    print(f"\n[Executing]: {command}")
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            cwd=cwd
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
    if len(sys.argv) < 2:
        print("Usage: python self_reflection_agent.py <working_directory>")
        sys.exit(1)

    working_dir = os.path.abspath(sys.argv[1])
    if not os.path.exists(working_dir):
        print(f"Error: Directory {working_dir} does not exist.")
        sys.exit(1)

    problem_file = os.path.join(working_dir, "problem.txt")
    if not os.path.exists(problem_file):
        print(f"Error: problem.txt not found in {working_dir}")
        sys.exit(1)

    with open(problem_file, "r") as f:
        problem_statement = f.read()

    code_base_dir = os.path.join(working_dir, "code_base")
    if not os.path.exists(code_base_dir):
        print(f"Error: code_base directory not found in {working_dir}")
        sys.exit(1)

    # Setup logging
    setup_logging("bug_fixing_agent")
    logger.info("=== Bug Fixing Agent Started ===")
    print("=== Bug Fixing Agent Started ===")
    print(f"Working Directory: {working_dir}")
    print(f"Code Base Directory: {code_base_dir}")
    
    # Initial prompt
    history = SYSTEM_PROMPT.format(problem_statement=problem_statement)
    
    step = 1
    in_reflection_mode = False
    final_patch = None

    while True:
        logger.info(f"--- Step {step} ---")
        print(f"\n--- Step {step} ---")
        
        # Ask LLM
        logger.debug("Sending request to LLM")
        if in_reflection_mode:
            response_str = ask_llm(history, model="qwen3-coder")
        else:
            response_str = ask_llm_azure_openai(history)
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
            if not in_reflection_mode:
                logger.info("Entering self-reflection mode")
                print("\n=== Entering Self-Reflection Mode ===")
                
                # Get git diff
                final_patch = run_command("git diff", code_base_dir)
                
                # Reset code base to original state
                run_command("git checkout .", code_base_dir)
                run_command("git clean -fd", code_base_dir)
                
                reflection_prompt = f"""
You have indicated that the task is done.
I have reset the code base to the original state.
Here is the patch of your changes:
```diff
{final_patch}
```

Please review your changes carefully against the problem statement.
1. Did you fix the issue?
2. Did you introduce any new bugs?
3. Did you remove any necessary code?
4. Verify the buggy state if needed.

If you are confident in your solution, respond with "task_done": true.
If you find issues, respond with "task_done": false and provide the command to fix them (you will need to re-apply changes or start over as the code is reset).
"""
                history += f"\n\nResponse:\n{cleaned_response}\n\nSystem: {reflection_prompt}"
                in_reflection_mode = True
                continue
            else:
                logger.info("Task verified and completed")
                print("\n=== Task Verified and Completed ===")
                print(f"\nFinal Patch:\n{final_patch}")
                print(f"\nJustification:\n{thoughts}")
                break
        
        # If task is not done, we are definitely not in reflection mode anymore
        if in_reflection_mode:
            logger.info("Self-reflection failed. Returning to work.")
            print("\n=== Self-Reflection Failed. Returning to Work. ===")
            in_reflection_mode = False

        # Execute command
        if command:
            output = run_command(command, code_base_dir)
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
