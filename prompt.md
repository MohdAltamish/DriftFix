You are building a complete, production-ready OpenEnv RL environment called "schema_migration_env" for the Meta PyTorch OpenEnv Hackathon. Build the ENTIRE project in one shot, ready to push to Hugging Face Spaces and pass all validation checks.

═══════════════════════════════════════════════
PROJECT OVERVIEW
═══════════════════════════════════════════════

Name: schema_migration_env
Domain: Database Schema Migration Agent
Description: An RL environment where an AI agent receives a broken SQLite database schema with failing SQL queries, and must issue SQL DDL/DML statements step-by-step to fix the schema until all target queries pass. Simulates real-world database migration workflows.

═══════════════════════════════════════════════
DIRECTORY STRUCTURE — BUILD EXACTLY THIS
═══════════════════════════════════════════════

schema_migration_env/
├── inference.py                         ← root level, mandatory
├── openenv.yaml                         ← root level, mandatory
├── README.md
├── pyproject.toml
├── .dockerignore
├── schema_migration_env/
│   ├── __init__.py                      ← exports SchemaMigrationEnv, SchemaMigrationAction
│   ├── client.py                        ← async EnvClient subclass + .sync() wrapper
│   ├── models.py                        ← Pydantic Action, Observation, State models
│   └── server/
│       ├── __init__.py
│       ├── app.py                       ← FastAPI app, /step /reset /state /health endpoints + WebSocket /ws
│       ├── environment.py               ← core environment logic
│       ├── tasks.py                     ← 3 tasks definitions + graders
│       └── Dockerfile                   ← placed at schema_migration_env/server/Dockerfile

═══════════════════════════════════════════════
OPENENV SPEC — FOLLOW EXACTLY
═══════════════════════════════════════════════

1. ALL models must be Pydantic BaseModel with full type annotations
2. step(action) → StepResult(observation, reward, done, info)
3. reset() → ResetResult(observation)
4. state() → current SchemaMigrationState
5. openenv.yaml must include: name, version, description, tags: [openenv], app_port: 8000, base_path: /web
6. Run openenv validate must pass
7. /reset endpoint accepts POST with empty body {}
8. /step endpoint accepts POST with action JSON
9. /state endpoint accepts GET
10. /health endpoint accepts GET, returns {"status": "ok"}

═══════════════════════════════════════════════
MODELS — schema_migration_env/models.py
═══════════════════════════════════════════════

SchemaMigrationAction:
  - sql: str                             ← the SQL statement the agent wants to execute
  - action_type: str = "execute"         ← "execute" | "submit" | "reset"

SchemaMigrationObservation:
  - schema_dump: str                     ← current CREATE TABLE statements of all tables
  - query_results: list[QueryResult]     ← each target query's pass/fail + error message
  - last_sql_error: str | None           ← error from last executed SQL if any
  - last_sql_output: str | None          ← output/result from last executed SQL
  - step_count: int
  - done: bool
  - task_id: str
  - task_description: str
  - hint: str | None                     ← optional hint for easy task only

QueryResult:
  - query_id: str
  - query: str
  - passed: bool
  - error: str | None
  - expected_row_count: int | None
  - actual_row_count: int | None

SchemaMigrationState:
  - task_id: str
  - db_schema: str
  - target_queries: list[str]
  - queries_passed: int
  - queries_total: int
  - step_count: int
  - max_steps: int
  - done: bool
  - reward_so_far: float
  - episode_id: str

═══════════════════════════════════════════════
3 TASKS — schema_migration_env/server/tasks.py
═══════════════════════════════════════════════

TASK 1 — "add_missing_column" (EASY)
  Seed schema:
    CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT, department TEXT);
    INSERT 5 rows of sample data
  Problem: A SELECT query references employees.salary which doesn't exist
  Target queries (must ALL pass for done=True):
    Q1: SELECT id, name, salary FROM employees WHERE salary > 50000; → must return rows without error
    Q2: SELECT department, AVG(salary) FROM employees GROUP BY department; → must work
  Agent must: ALTER TABLE employees ADD COLUMN salary INTEGER DEFAULT 60000;
              UPDATE employees SET salary = ... (varied values)
  Grader: Q1 passes=0.4, Q2 passes=0.4, salary values are varied (not all same)=0.2
  Hint provided: "The employees table is missing a column referenced in the queries"
  Max steps: 10

TASK 2 — "normalize_table" (MEDIUM)
  Seed schema:
    CREATE TABLE orders (
      id INTEGER PRIMARY KEY,
      customer_name TEXT,
      customer_email TEXT,
      customer_phone TEXT,
      product_name TEXT,
      product_price REAL,
      quantity INTEGER
    );
    INSERT 8 rows with repeated customer data (same customer, multiple orders)
  Problem: Schema is denormalized. Target queries require normalized tables.
  Target queries:
    Q1: SELECT c.name, COUNT(o.id) as order_count FROM customers c JOIN orders o ON c.id = o.customer_id GROUP BY c.id; → must work
    Q2: SELECT p.name, SUM(o.quantity) FROM products p JOIN orders o ON p.id = o.product_id GROUP BY p.id; → must work
    Q3: SELECT c.email FROM customers c WHERE c.id = 1; → must return a row
  Agent must:
    - CREATE TABLE customers (id, name, email, phone)
    - CREATE TABLE products (id, name, price)  
    - ALTER TABLE orders ADD COLUMN customer_id, product_id
    - Migrate data, add FKs
    - Drop old columns from orders
  Grader: each query passing = 0.3, all 3 = 0.1 bonus, total max 1.0
  Max steps: 20

TASK 3 — "breaking_version_migration" (HARD)
  Seed schema (legacy v1):
    CREATE TABLE user_accounts (
      uid INTEGER PRIMARY KEY,
      full_name TEXT,
      usr_email TEXT,
      usr_phone TEXT,
      acct_balance REAL,
      created_ts TEXT
    );
    CREATE TABLE transactions (
      txn_id INTEGER PRIMARY KEY,
      user_uid INTEGER,
      txn_amount REAL,
      txn_date TEXT,
      txn_type TEXT,
      FOREIGN KEY(user_uid) REFERENCES user_accounts(uid)
    );
    INSERT 10 user rows, 20 transaction rows
  Problem: New application code expects v2 schema with renamed columns
  Target queries (v2 schema expected):
    Q1: SELECT id, name, email FROM users LIMIT 1; → table renamed, columns renamed
    Q2: SELECT u.name, SUM(t.amount) FROM users u JOIN transactions t ON u.id = t.user_id GROUP BY u.id; → FK column renamed too
    Q3: SELECT COUNT(*) FROM users; → must equal original row count (data preservation check)
    Q4: SELECT COUNT(*) FROM transactions; → must equal original row count
    Q5: SELECT id, amount, created_at FROM transactions LIMIT 1; → columns renamed
  Agent must:
    - CREATE TABLE users with new column names
    - INSERT INTO users SELECT uid as id, full_name as name, ...
    - ALTER TABLE transactions ADD COLUMN user_id, amount, created_at
    - UPDATE transactions SET new cols from old cols
    - DROP old columns or recreate table
    - Preserve ALL row counts exactly
  Grader:
    Q1=0.15, Q2=0.20, Q3=0.25 (data preservation), Q4=0.25, Q5=0.15
    Rollback penalty: if any table has 0 rows that previously had rows = -0.3 applied to final score
    Final score clamped to [0.0, 1.0]
  Max steps: 30

═══════════════════════════════════════════════
REWARD FUNCTION — CRITICAL
═══════════════════════════════════════════════

Per step reward formula:
  queries_passing_now = count of target queries currently passing
  queries_passing_before = count before this step
  progress_delta = (queries_passing_now - queries_passing_before) / total_queries
  
  if progress_delta > 0:
      step_reward = 0.4 * progress_delta    ← progress bonus
  elif progress_delta < 0:
      step_reward = 0.2 * progress_delta    ← regression penalty
  else:
      step_reward = -0.01                   ← small step cost to discourage loops

  if last_sql had an error:
      step_reward -= 0.05                   ← SQL error penalty

  if done and all queries pass:
      step_reward += 0.3                    ← completion bonus
      efficiency_bonus = max(0, (max_steps - step_count) / max_steps) * 0.2
      step_reward += efficiency_bonus       ← reward finishing fast

  if destructive action (DROP TABLE on wrong table, deletes all rows):
      step_reward -= 0.3                    ← safety penalty

  Final episode score = sum(all step rewards) normalized to [0.0, 1.0]

═══════════════════════════════════════════════
SERVER — schema_migration_env/server/app.py
═══════════════════════════════════════════════

Use FastAPI. Implement:
  POST /reset     → accepts {} or TaskRequest{task_id: str}, returns ResetResult
  POST /step      → accepts SchemaMigrationAction, returns StepResult  
  GET  /state     → returns SchemaMigrationState
  GET  /health    → returns {"status": "ok"}
  GET  /web       → returns simple HTML UI showing current schema + query status
  WebSocket /ws   → persistent session support

Support concurrent sessions: use session_id in headers or query params.
max_concurrent_envs = 32

Each session gets its own SQLite in-memory database: sqlite3.connect(":memory:")
Sessions stored in dict keyed by session_id.
session_id generated as uuid4 on reset if not provided.
Return session_id in all responses inside info dict.

═══════════════════════════════════════════════
ENVIRONMENT CORE — server/environment.py
═══════════════════════════════════════════════

Class SchemaMigrationEnvironment:
  - Uses sqlite3 in-memory DB per episode
  - reset(task_id) → loads task schema + data, returns initial observation
  - step(action) → executes SQL safely, evaluates graders, returns StepResult
  - state() → returns full state
  - _execute_sql_safe(sql) → wraps in try/except, catches all sqlite3 errors, never crashes server
  - _evaluate_queries() → runs each target query, checks results, returns list[QueryResult]
  - _compute_reward() → implements reward formula above
  - _get_schema_dump() → SELECT sql FROM sqlite_master WHERE type='table'
  - _check_destructive(sql) → detect DROP TABLE on tables that have data, DELETE without WHERE
  - Episode isolation: each reset() creates fresh sqlite3.connect(":memory:") connection

SAFETY RULES for _execute_sql_safe:
  - Never allow: DROP DATABASE, ATTACH DATABASE, shell commands
  - Allow all DDL: CREATE, ALTER, DROP TABLE (but penalize if destructive)
  - Allow all DML: INSERT, UPDATE, DELETE, SELECT
  - Wrap every execution in try/except sqlite3.Error
  - Return error message in observation, never raise to FastAPI

═══════════════════════════════════════════════
DOCKERFILE — server/Dockerfile
═══════════════════════════════════════════════

FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "schema_migration_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]

requirements.txt must include:
  fastapi>=0.104.0
  uvicorn>=0.24.0
  pydantic>=2.0.0
  openenv-core
  websockets
  python-multipart

NO heavy ML dependencies. sqlite3 is stdlib, no install needed.
Total image must build in under 5 minutes.
Must work on linux/amd64.

═══════════════════════════════════════════════
openenv.yaml — ROOT LEVEL
═══════════════════════════════════════════════

name: schema-migration-env
version: "0.1.0"
description: "RL environment for training agents to perform database schema migrations. Agent executes SQL DDL/DML statements step-by-step to fix broken schemas until target queries pass."
author: "your-hf-username"
tags:
  - openenv
  - database
  - schema-migration
  - sql
  - real-world
sdk: docker
app_port: 8000
base_path: /web
tasks:
  - id: add_missing_column
    difficulty: easy
    description: "Add a missing column to fix failing SELECT queries"
  - id: normalize_table
    difficulty: medium
    description: "Normalize a denormalized table into proper relational schema"
  - id: breaking_version_migration
    difficulty: hard
    description: "Migrate legacy schema to new version without data loss"

═══════════════════════════════════════════════
HF SPACE HEADER — top of README.md
═══════════════════════════════════════════════

---
title: Schema Migration Environment
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
---

═══════════════════════════════════════════════
inference.py — ROOT LEVEL — MANDATORY FORMAT
═══════════════════════════════════════════════

CRITICAL: The stdout log format is auto-parsed by judges. ANY deviation = wrong score.

Implement exactly this structure:

  import asyncio, os, textwrap
  from typing import List, Optional
  from openai import OpenAI
  from schema_migration_env import SchemaMigrationEnv, SchemaMigrationAction

  API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
  API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
  MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
  IMAGE_NAME = os.getenv("IMAGE_NAME")
  TASK_NAME = os.getenv("TASK_NAME") or "add_missing_column"
  BENCHMARK = "schema_migration_env"
  MAX_STEPS = 15
  SUCCESS_SCORE_THRESHOLD = 0.5

Log functions — COPY EXACTLY, do not rename fields:
  def log_start(task, env, model):
      print(f"[START] task={task} env={env} model={model}", flush=True)

  def log_step(step, action, reward, done, error):
      error_val = error if error else "null"
      print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}", flush=True)

  def log_end(success, steps, score, rewards):
      rewards_str = ",".join(f"{r:.2f}" for r in rewards)
      print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

System prompt for the agent:
  You are a database migration expert. You receive a SQLite database schema with failing SQL queries.
  Your goal is to issue SQL DDL/DML statements one at a time to fix the schema so all target queries pass.
  Rules:
  - Output ONLY a single valid SQL statement per turn. No explanation, no markdown, no backticks.
  - Read the schema_dump and query_results carefully before acting.
  - If a query fails with an error, fix the root cause.
  - Prefer ALTER TABLE over DROP+RECREATE when possible.
  - Always preserve existing data.
  - When all queries pass, output: SUBMIT

Run all 3 tasks sequentially in main():
  tasks = ["add_missing_column", "normalize_table", "breaking_version_migration"]
  For each task, run a full episode, emit [START] [STEP]* [END]
  
  episode score = sum(rewards) normalized, clamped to [0.0, 1.0]
  success = score >= SUCCESS_SCORE_THRESHOLD

  [END] must always emit, even on exception (use try/finally)

env init: env = await SchemaMigrationEnv.from_docker_image(IMAGE_NAME)

═══════════════════════════════════════════════
CLIENT — schema_migration_env/client.py
═══════════════════════════════════════════════

Follow the exact pattern from TB2 and ReasoningGym examples:
  - Class SchemaMigrationEnv(EnvClient or simple HTTP client)
  - async methods: reset(task_id=None), step(action), state(), close()
  - .sync() wrapper returns synchronous version
  - @classmethod async from_docker_image(cls, image_name) → starts container, waits for health, returns client
  - @classmethod from_url(cls, base_url) → connects to running server
  - Use httpx for async HTTP calls
  - Timeout: 30 seconds per request
  - Wait for /health to return 200 before returning from from_docker_image (poll every 1s, max 60s)

═══════════════════════════════════════════════
__init__.py exports
═══════════════════════════════════════════════

from schema_migration_env.client import SchemaMigrationEnv
from schema_migration_env.models import SchemaMigrationAction, SchemaMigrationObservation, SchemaMigrationState
__all__ = ["SchemaMigrationEnv", "SchemaMigrationAction", "SchemaMigrationObservation", "SchemaMigrationState"]

═══════════════════════════════════════════════
pyproject.toml
═══════════════════════════════════════════════

[project]
name = "schema-migration-env"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
  "fastapi>=0.104.0",
  "uvicorn>=0.24.0", 
  "pydantic>=2.0.0",
  "httpx>=0.25.0",
  "openenv-core",
  "websockets",
  "python-multipart",
  "openai>=1.0.0",
]

═══════════════════════════════════════════════
README.md CONTENT
═══════════════════════════════════════════════

Write a detailed README with:
1. HF Space YAML header (shown above)
2. Project description — what it is, why it matters
3. Quick start code snippet using SchemaMigrationEnv
4. Action space table (fields, types, description)
5. Observation space table
6. State space table
7. Task descriptions with difficulty and expected agent behavior
8. Reward function explanation with formula
9. Setup instructions: pip install, docker build, local run
10. Baseline scores table (fill with realistic placeholder scores)
11. Environment variables table (API_BASE_URL, MODEL_NAME, HF_TOKEN, IMAGE_NAME, TASK_NAME)

═══════════════════════════════════════════════
CRITICAL RULES — DO NOT VIOLATE
═══════════════════════════════════════════════

1. sqlite3 only — no PostgreSQL, no MySQL, no external DB dependencies
2. Every endpoint must handle exceptions and return proper HTTP error responses, never crash
3. In-memory DB per session — no files written to disk during episodes
4. inference.py MUST be in the root directory, not inside any subdirectory
5. openenv.yaml MUST be in the root directory
6. Dockerfile MUST be at schema_migration_env/server/Dockerfile OR root level
7. /reset must accept POST with empty body {} without errors
8. All rewards must be float, clamped to [-1.0, 1.0] per step, episode score to [0.0, 1.0]
9. log_end score field formatted to .3f, rewards formatted to .2f — exactly as shown
10. No print statements except the mandatory [START] [STEP] [END] log lines and [DEBUG] lines
11. All async functions must be properly awaited — no blocking calls in async context
12. from_docker_image must use subprocess or docker SDK to start container, not shell=True
13. Session cleanup: close SQLite connection on session end or timeout (5 min idle timeout)
14. The /web endpoint must return valid HTML — even a minimal page showing current state
15. max_concurrent_envs=32 in app creation
16. DO NOT use localhost hardcoded — use 0.0.0.0 for server bind

═══════════════════════════════════════════════
VALIDATION CHECKLIST — verify before finishing
═══════════════════════════════════════════════

After writing all files, verify:
[ ] inference.py exists at root level
[ ] openenv.yaml exists at root level with correct tags
[ ] Dockerfile exists and has no syntax errors
[ ] /reset accepts POST {} and returns 200
[ ] /health returns {"status": "ok"}
[ ] All 3 tasks defined with graders returning 0.0–1.0
[ ] Reward function never returns NaN or values outside [-1, 1]
[ ] log_start, log_step, log_end format exactly matches mandatory format
[ ] [END] is inside finally block
[ ] No import errors — all imports resolve
[ ] SchemaMigrationEnv exported from __init__.py
[ ] pyproject.toml has all dependencies
[ ] README has HF Space YAML header

═══════════════════════════════════════════════
BUILD ORDER
═══════════════════════════════════════════════

Build files in this exact order:
1. models.py
2. server/tasks.py
3. server/environment.py
4. server/app.py
5. client.py
6. __init__.py
7. inference.py
8. openenv.yaml
9. Dockerfile
10. pyproject.toml
11. README.md
12. .dockerignore

After all files are written, output a summary of:
- File tree
- Any assumptions made
- How to run locally: docker build + docker run command
- How to run inference: python inference.py command with env vars