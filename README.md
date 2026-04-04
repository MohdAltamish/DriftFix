---
title: Schema Migration Environment
emoji: 🗄️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 7860
base_path: /web
tags:
  - openenv
  - database
  - schema-migration
---

# 🗄️ Schema Migration Environment

An RL environment where an AI agent receives a broken SQLite database schema with failing SQL queries, and must issue SQL DDL/DML statements step-by-step to fix the schema until all target queries pass. Simulates real-world database migration workflows.

## Why Schema Migration?

Database schema migrations are one of the most error-prone tasks in software engineering. A single wrong `ALTER TABLE` can destroy data, break queries, or cascade failures across services. This environment lets you train RL agents to:

- **Diagnose** broken schemas from failing query error messages
- **Plan** migration strategies (add columns, normalize tables, rename schemas)
- **Execute** SQL DDL/DML safely without data loss
- **Verify** that all target queries pass after migration

## Quick Start

```python
import asyncio
from schema_migration_env import SchemaMigrationEnv, SchemaMigrationAction

async def main():
    # Connect to a running server
    env = SchemaMigrationEnv.from_url("http://localhost:7860")

    # Reset with a specific task
    result = await env.reset(task_id="add_missing_column")
    print(result.observation.schema_dump)
    print(result.observation.query_results)

    # Take a step
    action = SchemaMigrationAction(
        sql="ALTER TABLE employees ADD COLUMN salary INTEGER DEFAULT 60000;",
        action_type="execute"
    )
    step = await env.step(action)
    print(f"Reward: {step.reward}, Done: {step.done}")

    await env.close()

async def main_docker():
    # Or start a container automatically
    env = await SchemaMigrationEnv.from_docker_image("mohdaltamish/driftfix-env", port=7860)
    ...
    await env.close()

asyncio.run(main())
```

## Action Space

| Field | Type | Description |
|-------|------|-------------|
| `sql` | `str` | The SQL statement the agent wants to execute |
| `action_type` | `str` | One of: `"execute"` (run SQL), `"submit"` (finish episode), `"reset"` (restart) |

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `schema_dump` | `str` | Current CREATE TABLE statements of all tables |
| `query_results` | `list[QueryResult]` | Each target query's pass/fail + error message |
| `last_sql_error` | `str | None` | Error from last executed SQL (if any) |
| `last_sql_output` | `str | None` | Output/result from last executed SQL |
| `step_count` | `int` | Current step number |
| `done` | `bool` | Whether the episode has ended |
| `task_id` | `str` | Current task identifier |
| `task_description` | `str` | Human-readable task description |
| `hint` | `str | None` | Optional hint (easy task only) |

## State Space

| Field | Type | Description |
|-------|------|-------------|
| `task_id` | `str` | Current task identifier |
| `db_schema` | `str` | Current database schema |
| `target_queries` | `list[str]` | Queries that must all pass |
| `queries_passed` | `int` | Number of currently passing queries |
| `queries_total` | `int` | Total number of target queries |
| `step_count` | `int` | Current step number |
| `max_steps` | `int` | Maximum allowed steps |
| `done` | `bool` | Whether the episode has ended |
| `reward_so_far` | `float` | Cumulative reward |
| `episode_id` | `str` | UUID for this episode |

## Tasks

### 1. `add_missing_column` (Easy)
**Max Steps:** 10 | **Target Queries:** 2

The `employees` table is missing a `salary` column. The agent must add it and populate varied values.

### 2. `normalize_table` (Medium)
**Max Steps:** 20 | **Target Queries:** 3

A denormalized `orders` table must be split into `customers`, `products`, and normalized `orders` tables.

### 3. `breaking_version_migration` (Hard)
**Max Steps:** 30 | **Target Queries:** 5

A legacy v1 schema (`user_accounts`, `transactions`) must be migrated to v2 (`users`, `transactions` with renamed columns) without any data loss.

## Reward Function

```
Per-step reward:
  progress_delta = (queries_passing_now - queries_passing_before) / total_queries

  if progress_delta > 0:  reward = 0.4 × progress_delta   (progress bonus)
  if progress_delta < 0:  reward = 0.2 × progress_delta   (regression penalty)
  if progress_delta == 0: reward = -0.01                   (step cost)

  SQL error:              reward -= 0.05
  Destructive action:     reward -= 0.30
  All queries pass:       reward += 0.30 + efficiency_bonus

  efficiency_bonus = max(0, (max_steps - step_count) / max_steps) × 0.2

  Episode score = sum(step_rewards), clamped to [0.0, 1.0]
```

## Setup

### Install Dependencies

```bash
pip install -e .
```

### Run Locally

```bash
# Start the server
uvicorn server.app:app --host 0.0.0.0 --port 7860

# Visit the web UI
open http://localhost:7860/web
```

### Docker Build & Run

```bash
# Build
docker build -t driftfix-env .

# Run
docker run -p 7860:7860 driftfix-env

# Test health
curl http://localhost:7860/health
```

### Run Inference

```bash
export HF_TOKEN="your-hf-token"
export IMAGE_NAME="driftfix-env"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"

python inference.py
```

## Baseline Scores

| Task | Random Agent | GPT-4o | Qwen2.5-72B |
|------|-------------|--------|-------------|
| `add_missing_column` | 0.05 | 0.92 | 0.85 |
| `normalize_table` | 0.02 | 0.78 | 0.71 |
| `breaking_version_migration` | 0.01 | 0.65 | 0.58 |

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `HF_TOKEN` | Yes | — | Hugging Face API token |
| `API_BASE_URL` | No | `https://router.huggingface.co/v1` | LLM API endpoint |
| `MODEL_NAME` | No | `Qwen/Qwen2.5-72B-Instruct` | Model to use |
| `IMAGE_NAME` | Yes* | — | Docker image name (*required for docker mode) |
| `TASK_NAME` | No | `add_missing_column` | Default task to run |

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/reset` | Reset environment (body: `{}` or `{"task_id": "..."}`) |
| `POST` | `/step` | Execute SQL action |
| `GET` | `/state` | Get current state |
| `GET` | `/health` | Health check |
| `GET` | `/web` | Web UI |
| `WebSocket` | `/ws` | Persistent session |

## License

MIT
