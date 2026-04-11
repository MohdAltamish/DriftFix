---
title: "DriftFix: Schema Migration Environment"
emoji: 🗄️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - database
  - schema-migration
  - sql
  - reinforcement-learning
---

# 🗄️ DriftFix: Schema Migration Environment

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-blueviolet)](https://github.com/openenv-core)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**DriftFix** is a high-fidelity Reinforcement Learning (RL) environment designed to train and evaluate AI agents on real-world database schema migration tasks. Built for the **Meta PyTorch OpenEnv Hackathon**, it challenges agents to diagnose broken schemas from failing query results and issue precise SQL DDL/DML statements to restore system health.

---

## 🚀 Why DriftFix?

Database migrations are inherently high-stakes. A single flawed `ALTER TABLE` or `DROP` can lead to data loss or service outages. DriftFix provides a sandboxed, deterministic environment where agents can:

- **Diagnose** structural issues from raw SQL error messages.
- **Plan** complex multi-step migrations (e.g., normalization).
- **Execute** DDL/DML safely using an in-memory SQLite backend.
- **Verify** success through a suite of passing target queries.

---

## 🛠️ System Architecture

```mermaid
graph TD
    A[Agent] -->|SQL Action| S[FastAPI Server]
    S -->|Execute| DB[(SQLite In-Memory)]
    DB -->|Results| E[Grader Engine]
    E -->|Reward/Observation| S
    S -->|State Update| UI[Web UI / /web]
    S -->|Response| A
```

---

## 🏁 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/mohdaltamish/DriftFix.git
cd DriftFix

# Install dependencies
pip install -e .
```

### Local Testing & Evaluation

Testing the evaluation sandbox locally requires building the Docker image and utilizing the Hugging Face Serverless API:

```bash
# 1. Build the OpenEnv compatible Docker image
docker build -t driftfix-env .

# 2. Set your Hugging Face Access Token
export HF_TOKEN="your_hf_api_key_here"
export IMAGE_NAME="driftfix-env"

# 3. Simulate the evaluation pipeline
python3 inference.py
```

### Judge Simulation (No Docker)

To simulate the exact judge environment without Docker:

```bash
unset IMAGE_NAME
export HF_TOKEN="your_hf_token"
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"

python3 inference.py
```

### Basic Usage

```python
import asyncio
from schema_migration_env import SchemaMigrationEnv, SchemaMigrationAction

async def main():
    # Start environment via Docker
    env = await SchemaMigrationEnv.from_docker_image("driftfix-env")

    # Reset to a specific task
    result = await env.reset(task_id="add_missing_column")
    print(f"Task: {result.observation.task_description}")

    # Take an action
    action = SchemaMigrationAction(
        sql="ALTER TABLE employees ADD COLUMN salary INTEGER DEFAULT 60000;",
        action_type="execute"
    )
    step = await env.step(action)
    print(f"Reward: {step.reward} | Done: {step.done}")

    await env.close()

asyncio.run(main())
```

---

## 📊 Inference Output Format

The inference script emits **exactly three line types** to stdout:

```
[START] task=<task_name> env=<benchmark> model=<model_name>
[STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
[END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
```

**Rules:**
- `reward` and `rewards` formatted to 2 decimal places
- `done` and `success` are lowercase booleans: `true` or `false`
- `error` is the raw error string, or `null` if none
- All `[DEBUG]` output goes to stderr (not stdout)
- `[END]` is always emitted, even on exception

**Example output:**
```
[START] task=add_missing_column env=schema_migration_env model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=ALTER TABLE employees ADD COLUMN salary REAL; reward=0.88 done=true error=null
[END] success=true steps=1 rewards=0.88
```

---

## 📊 Evaluation Benchmark

DriftFix features three core tasks of increasing complexity:

| Task ID | Difficulty | Target Queries | Focus Area |
| :--- | :--- | :---: | :--- |
| `add_missing_column` | **Easy** | 2 | Schema discovery & basic DDL |
| `normalize_table` | **Medium** | 3 | Data restructuring & FK management |
| `breaking_version_migration` | **Hard** | 5 | Data preservation during breaking changes |

### Baseline Scores (Success Rate)

| Model | Easy | Medium | Hard |
| :--- | :---: | :---: | :--- |
| **GPT-4o** | 92% | 78% | 65% |
| **Qwen2.5-72B-Instruct** | 100% (Score: 0.880) | 100% (Score: 0.850) | 100% (Score: 0.767) |
| **Random Agent** | 5% | 2% | 1% |

---

## 💎 Features

### 🖥️ Modern Web UI

DriftFix includes a built-in **Glassmorphism Web Dashboard** accessible at `/web`. It provides real-time visualization of:
- Current Database Schema (DDL dump)
- Live Query Pass/Fail status
- Agent Progress & Cumulative Rewards

### ⚡ Persistent Sessions
Leverage **WebSockets (`/ws`)** for high-frequency agent interaction, reducing HTTP overhead and enabling real-time observability.

### 🛡️ Safety Engine
The environment includes a `Destructive Action` detector that penalizes agents for `DROP TABLE` or `DELETE` operations that result in unrecoverable data loss in tables containing critical data.

---

## 📈 Reward Function

The reward system is designed to encourage both correctness and efficiency:

$$R = \text{Progress Bonus} + \text{Completion Bonus} + \text{Efficiency Bonus} - \text{Penalties}$$

- **Progress Bonus (+0.4 per $\Delta$query)**: Awarded for each new target query that starts passing.
- **Completion Bonus (+0.3)**: Awarded when all target queries pass.
- **Efficiency Bonus (+0.0 to +0.2)**: Proportional to remaining steps upon completion.
- **SQL Error Penalty (-0.05)**: Applied for invalid SQL syntax.
- **Destructive Action (-0.30)**: High penalty for unrecoverable data loss.
- **Step Cost (-0.01)**: To discourage loops and favor shorter paths.

---

## ⚙️ Configuration

| Variable | Default | Description |
| :--- | :--- | :--- |
| `HF_TOKEN` | **Required** (no default) | Hugging Face API token for inference |
| `API_BASE_URL` | `https://router.huggingface.co/v1` | LLM Gateway URL |
| `MODEL_NAME` | `Qwen/Qwen2.5-72B-Instruct` | LLM used by `inference.py` |
| `IMAGE_NAME` | *(optional)* | Docker image name (local mode only) |
| `ENV_BASE_URL` | `https://mohdaltamish-driftfix-env.hf.space` | HF Space URL when IMAGE_NAME is not set |

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  Developed with ❤️ for the Meta OpenEnv Hackathon
</p>