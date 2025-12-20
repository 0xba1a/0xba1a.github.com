import sys
import os
import subprocess

def setup_environment(target_dir):
    # Ensure target directory exists
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    repo_url = "https://github.com/django/django"
    base_commit = "27c09043da52ca1f02605bf28600bfd5ace95ae4"
    problem_statement = """DatabaseCache._cull implementation could fail if no key was found to perform a deletion in the table. This prevented the new cache key/value from being correctly added.
The error manifested as 'NoneType' object is not subscriptable because cursor.fetchone() returned None when no rows were found, and the code attempted to access index [0] on it."""

    repo_dir = os.path.join(target_dir, "code_base")
    
    # Clone repository
    if not os.path.exists(repo_dir):
        print(f"Cloning {repo_url} into {repo_dir}...")
        subprocess.run(["git", "clone", repo_url, repo_dir], check=True)
    else:
        print(f"Repository already exists at {repo_dir}")

    # Checkout base commit
    print(f"Checking out base commit {base_commit}...")
    subprocess.run(["git", "checkout", base_commit], cwd=repo_dir, check=True)

    # Write problem statement
    problem_file = os.path.join(target_dir, "problem.txt")
    print(f"Writing problem statement to {problem_file}...")
    with open(problem_file, "w") as f:
        f.write(problem_statement)

    print("Setup complete.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python setup_swe_bench.py <target_directory>")
        sys.exit(1)
    
    target_directory = sys.argv[1]
    setup_environment(target_directory)
