#!/usr/bin/env python3
"""Generate the DVC DAG and update the README.md file."""

import re
import subprocess


def get_dvc_dag_mermaid() -> str | None:
    """Get DVC DAG in Mermaid format."""
    try:
        result = subprocess.run(["dvc", "dag", "--mermaid"], capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running dvc dag: {e}")
        return None


def update_readme_dag(mermaid_dag: str) -> bool:
    """Update README.md with the Mermaid DAG."""
    try:
        with open("README.md", "r") as f:
            content = f.read()

        # Define the markers for the DAG section
        start_marker = "<!-- DVC DAG START -->"
        end_marker = "<!-- DVC DAG END -->"

        # Create the new DAG section
        dag_section = f"{start_marker}\n```mermaid\n{mermaid_dag}```\n{end_marker}"

        # If markers exist, replace content between them
        if start_marker in content and end_marker in content:
            pattern = f"{start_marker}.*?{end_marker}"
            new_content = re.sub(pattern, dag_section, content, flags=re.DOTALL)
        else:
            # If markers don't exist, append to the end
            new_content = f"{content}\n\n## DVC Pipeline\n{dag_section}\n"

        with open("README.md", "w") as f:
            f.write(new_content)

        return True
    except Exception as e:
        print(f"Error updating README.md: {e}")
        return False


def main() -> None:
    """Main function to generate the DVC DAG and update the README.md file."""
    mermaid_dag = get_dvc_dag_mermaid()
    if mermaid_dag:
        success = update_readme_dag(mermaid_dag)
        if success:
            print("Successfully updated README.md with DVC DAG")
        else:
            print("Failed to update README.md")
            exit(1)
    else:
        print("Failed to generate DVC DAG")
        exit(1)


if __name__ == "__main__":
    main()
