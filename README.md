# Project name

Project description

## Project initialisation steps:

1. Initialize virtual environment.
2. Install uv with conda
    ```
    conda install -y poetry
    ```
    or with pip
    ```
    pip install poetry
    ```

3. Install dependencies with poetry
    ```
    poetry install
    ```

## DVC Pipeline
<!-- DVC DAG START -->
```mermaid
flowchart TD
	node1["data/raw_data/customers.csv.dvc"]
	node2["data/raw_data/test_data.csv.dvc"]
	node3["dvc/dvc.yaml:evaluate"]
	node4["dvc/dvc.yaml:featurize"]
	node5["dvc/dvc.yaml:inference"]
	node6["dvc/dvc.yaml:split"]
	node7["dvc/dvc.yaml:train_model"]
	node1-->node6
	node2-->node5
	node4-->node5
	node4-->node7
	node6-->node3
	node6-->node4
	node6-->node5
	node6-->node7
	node7-->node3
	node7-->node5
```
<!-- DVC DAG END -->
