from pathlib import Path
import sys
import os
import json
import subprocess
import re
import logging
import time

# Add the lib directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lib import ask_llm, setup_logging, ask_llm_azure_openai

# Get logger for this module
logger = logging.getLogger(__name__)

class LLMInteraction:
    def __init__(self):
        self.dangerous_commands = ["rm -rf", "tree"]

    def call_llm(self, prompt):
        logger.debug("Sending request to LLM")
        # return ask_llm(prompt)
        return ask_llm_azure_openai(prompt)

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

class Sandbox:
    def __init__(self, workspace_dir):
        self.workspace_dir = os.path.abspath(workspace_dir)
        self.container_name = f"sandbox_{int(time.time())}"
        self.ssh_port = self._get_free_port()
        self.key_file = os.path.abspath(os.path.join(os.path.dirname(__file__), f"id_rsa_{self.container_name}"))

    def __enter__(self):
        self._generate_ssh_key()
        self._start_container()
        self._setup_ssh()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    def _get_free_port(self):
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            return s.getsockname()[1]

    def _generate_ssh_key(self):
        if os.path.exists(self.key_file):
            os.remove(self.key_file)
        if os.path.exists(self.key_file + ".pub"):
            os.remove(self.key_file + ".pub")
        subprocess.run(f"ssh-keygen -t rsa -b 2048 -f {self.key_file} -N ''", shell=True, check=True, capture_output=True)

    def _start_container(self):
        logger.info(f"Starting sandbox container {self.container_name}...")
        cmd = [
            "sudo", "docker", "run", "-d",
            "-p", f"{self.ssh_port}:22",
            "--name", self.container_name,
            "-v", f"{self.workspace_dir}:/workspace",
            "-w", "/workspace",
            "ubuntu:latest",
            "sleep", "infinity"
        ]
        subprocess.run(cmd, check=True, capture_output=True)

    def _setup_ssh(self):
        logger.info("Setting up SSH in sandbox...")
        # Install ssh and setup keys
        commands = [
            "apt-get update && apt-get install -y openssh-server",
            "mkdir -p /root/.ssh",
            "chmod 700 /root/.ssh",
            f"echo '{open(self.key_file + '.pub').read().strip()}' > /root/.ssh/authorized_keys",
            "chmod 600 /root/.ssh/authorized_keys",
            "service ssh start"
        ]
        for cmd in commands:
            subprocess.run(["sudo", "docker", "exec", self.container_name, "sh", "-c", cmd], check=True, capture_output=True)
        
        # Wait for SSH to be ready
        time.sleep(2)

    def run_command(self, command):
        logger.debug(f"Executing command in sandbox: {command}")
        ssh_cmd = [
            "ssh", "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "LogLevel=ERROR",
            "-i", self.key_file,
            "-p", str(self.ssh_port),
            "root@localhost",
            command
        ]
        result = subprocess.run(ssh_cmd, capture_output=True, text=True)
        logger.debug(f"Command output:\n{result.stdout}\n\nError:\n{result.stderr}")
        return result.stdout + result.stderr

    def cleanup(self):
        logger.info(f"Cleaning up sandbox {self.container_name}")
        subprocess.run(["sudo", "docker", "rm", "-f", self.container_name], capture_output=True)
        if os.path.exists(self.key_file):
            logger.debug(f"Removing SSH key files {self.key_file}")
            os.remove(self.key_file)
        if os.path.exists(self.key_file + ".pub"):
            logger.debug(f"Removing SSH public key file {self.key_file}.pub")
            os.remove(self.key_file + ".pub")

class ToolCall:
    def __init__(self, prompt_manager, sandbox):
        self.prompt_manager = prompt_manager
        self.sandbox = sandbox
        self.tool_name = "anthropic-text-editor"
        self.tool_description = """
You have access to a tool called 'anthropic-text-editor'.
This tool allows you to view, create, replace strings, insert, and delete lines in files.
It communicates through JSON over stdin/stdout.

Usage:
echo '{"input": <JSON_INPUT>}' | anthropic-text-editor

<JSON_INPUT> format:
{
  "command": "view|create|str_replace|insert|delete",
  "path": "/absolute/path/to/file",
  "view_range": [start_line, end_line], // Optional for view
  "old_str": "text to replace", // Required for str_replace
  "new_str": "replacement text", // Optional for str_replace, required for insert
  "insert_line": line_number, // Required for insert
  "delete_range": [start_line, end_line], // Required for delete
  "file_text": "content" // Required for create
}

IMPORTANT: Use this tool for making any code changes.
"""
        self._install_tool()
        # Test the tool
        test_cmd = 'echo \'{"input": {"command": "view", "path": "/workspace/data/buggy_code.py", "view_range": [1, 5]}}\' | anthropic-text-editor'
        output = self.sandbox.run_command(test_cmd)
        if "error" in output.lower():
            logger.error("Failed to verify anthropic-text-editor installation.")
            print("Failed to verify anthropic-text-editor installation. Exiting.")
            sys.exit(-1)
        self._update_prompt()

    def _install_tool(self):
        logger.info(f"Installing {self.tool_name}...")
        print(f"Installing {self.tool_name} (this may take a while)...")
        
        # Install dependencies
        self.sandbox.run_command("apt-get update && apt-get install -y curl build-essential")
        
        # Install Rust
        self.sandbox.run_command("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y")
        
        # Install anthropic-text-editor
        cmd = "export PATH=$HOME/.cargo/bin:$PATH && cargo install --locked anthropic-text-editor"
        self.sandbox.run_command(cmd)

        # Create a symlink to make it accessible globally
        self.sandbox.run_command("ln -s /root/.cargo/bin/anthropic-text-editor /usr/local/bin/anthropic-text-editor")
        
    def _update_prompt(self):
        self.prompt_manager.system_prompt += "\n\n" + self.tool_description

    def cleanup(self):
        logger.info(f"Removing {self.tool_name}...")
        cmd = "export PATH=$HOME/.cargo/bin:$PATH && cargo uninstall anthropic-text-editor"
        self.sandbox.run_command(cmd)

class AgentLoop:
    def __init__(self, llm_interaction, prompt_manager, sandbox, max_iterations, max_timeout):
        self.llm_interaction = llm_interaction
        self.prompt_manager = prompt_manager
        self.sandbox = sandbox
        self.max_iterations = max_iterations
        self.max_timeout = max_timeout
        self.start_time = None

    def run_command(self, command):
        """Executes a bash command inside the sandbox and returns the output."""
        logger.info(f"Executing command in sandbox: {command}")
        print(f"\n[Executing]: {command}")
        try:
            output = self.sandbox.run_command(command)
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
                # Print the final code state
                final_code = self.sandbox.run_command("cat /workspace/data/buggy_code.py")
                print(f"\n--- Final Code ---\n{final_code}\n")
                break

            if command:
                if self.llm_interaction.is_safe_command(command):
                    output = self.run_command(command)
                    try:
                        output_json = json.loads(output)
                        if isinstance(output_json, dict) and "content" in output_json:
                            print(f"Command Output:\n{output_json['content']}")
                        else:
                            print(f"Command Output:\n{output}")
                    except json.JSONDecodeError:
                        print(f"Command Output:\n{output}")
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
    setup_logging(Path(__file__).name)

    llm = LLMInteraction()

    file_path = "data/buggy_code.py"
    system_prompt = SYSTEM_PROMPT.format()
    prompt_manager = PromptManager(system_prompt)

    # Initial user message to start the conversation
    prompt_manager.add_user_message(f"Please fix the bug in /workspace/{file_path}")

    workspace_dir = os.path.dirname(os.path.abspath(__file__))
    
    with Sandbox(workspace_dir) as sandbox:
        tool_call = ToolCall(prompt_manager, sandbox)
        try:
            agent = AgentLoop(llm, prompt_manager, sandbox, max_iterations=10, max_timeout=60)
            agent.run()
        finally:
            tool_call.cleanup()

if __name__ == "__main__":
    main()
