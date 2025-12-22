import sys
import os
import json
import subprocess
import re
import logging
import time

# Add the lib directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lib import ask_llm, setup_logging

# Get logger for this module
logger = logging.getLogger(__name__)

class LLMInteraction:
    def __init__(self):
        self.dangerous_commands = ["rm -rf", "tree"]

    def call_llm(self, prompt):
        logger.debug("Sending request to LLM")
        return ask_llm(prompt)

    def validate_response(self, response_str):
        """Attempts to extract and parse JSON from the LLM response."""
        # Remove markdown code blocks if present
        response = re.sub(r'^```json\s*', '', response_str, flags=re.MULTILINE)
        response = re.sub(r'^```\s*', '', response, flags=re.MULTILINE)
        response = re.sub(r'\s*```$', '', response, flags=re.MULTILINE)
        response = response.strip()

        try:
            data = json.loads(response)
            return data
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON response from LLM. Response: {response_str}")
            return None

    def is_safe_command(self, command):
        for dangerous in self.dangerous_commands:
            if dangerous in command:
                logger.warning(f"Dangerous command detected: {command}")
                return False
        return True

class PromptManager:
    def __init__(self, system_prompt):
        self.system_prompt = system_prompt
        self.history = []

    def add_user_message(self, message):
        self.history.append(f"User: {message}")

    def add_assistant_message(self, message):
        self.history.append(f"Assistant: {message}")

    def get_prompt(self):
        # Combine system prompt and history
        full_prompt = self.system_prompt + "\n\n" + "\n".join(self.history)
        return full_prompt

class AgentLoop:
    def __init__(self, llm_interaction, prompt_manager, max_iterations, max_timeout):
        self.llm_interaction = llm_interaction
        self.prompt_manager = prompt_manager
        self.max_iterations = max_iterations
        self.max_timeout = max_timeout
        self.start_time = None

    def run_command(self, command):
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

    def run(self):
        self.start_time = time.time()
        step = 1

        while step <= self.max_iterations:
            current_time = time.time()
            if current_time - self.start_time > self.max_timeout:
                print("Max timeout reached. Exiting.")
                break

            logger.info(f"--- Step {step} ---")
            print(f"\n--- Step {step} ---")

            prompt = self.prompt_manager.get_prompt()
            response_str = self.llm_interaction.call_llm(prompt)
            logger.debug(f"LLM Response: {response_str}")

            response_data = self.llm_interaction.validate_response(response_str)

            if not response_data:
                print(f"[Error]: Failed to parse JSON response from LLM.\nResponse:\n{response_str}")
                self.prompt_manager.add_assistant_message(response_str)
                self.prompt_manager.add_user_message(f"Invalid JSON response. Please respond with valid JSON only.\nResponse was: {response_str}")
                continue

            self.prompt_manager.add_assistant_message(json.dumps(response_data))

            thoughts = response_data.get("thoughts", "")
            command = response_data.get("command", "")
            task_done = response_data.get("task_done", False)

            print(f"Thoughts: {thoughts}")

            if task_done:
                print("Task marked as done.")
                break

            if command:
                if self.llm_interaction.is_safe_command(command):
                    output = self.run_command(command)
                    self.prompt_manager.add_user_message(f"Command Output:\n{output}")
                else:
                    print(f"[Security Alert]: Dangerous command blocked: {command}")
                    self.prompt_manager.add_user_message(f"Security Alert: The command '{command}' is not allowed.")
            else:
                self.prompt_manager.add_user_message("No command provided. If you are done, set task_done to true.")

            step += 1

        if step > self.max_iterations:
            print("Max iterations reached.")

SYSTEM_PROMPT = """You are an autonomous bug fixing agent.
You have access to a bash terminal.
You can read files, write files, and run commands.

Your task is to fix the bug in the python file that will be given in the user message.

You must respond in STRICT JSON format with the following keys:
- "task_done": boolean (true if the bug is fixed and verified, false otherwise)
- "command": string (the bash command to execute, empty if task_done is true)
- "thoughts": string (your reasoning and plan)

Do not include any text outside the JSON object.
Do not use markdown formatting (like ```json ... ```). Just the raw JSON string.

Example response:
{{
    "task_done": false,
    "command": "cat <file_name>,
    "thoughts": "I need to read the file to understand the code and the bug."
}}
"""

def main():
    setup_logging("bug_fixing_agent_guardrails")

    llm = LLMInteraction()

    file_path = "data/buggy_code.py"
    system_prompt = SYSTEM_PROMPT.format()
    prompt_manager = PromptManager(system_prompt)

    # Initial user message to start the conversation
    prompt_manager.add_user_message(f"Please fix the bug in {file_path}")

    agent = AgentLoop(llm, prompt_manager, max_iterations=10, max_timeout=60)
    agent.run()

if __name__ == "__main__":
    main()
