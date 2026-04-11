"""
inference.py — Schema Migration RL Environment
Root-level inference script for the Meta PyTorch OpenEnv Hackathon.

STDOUT FORMAT (exact — do not modify):
  [START] task=<task> env=<env> model=<model>
  [STEP]  step=<n> action=<sql> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<bool> steps=<n> rewards=<r1,r2,...,rn>
"""

import asyncio
import os
import sys
import textwrap
from typing import List, Optional

from openai import OpenAI
from schema_migration_env import SchemaMigrationEnv, SchemaMigrationAction

# ─── Required env vars (per judge spec) ───────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")  # must have default
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")         # must have default
HF_TOKEN     = os.getenv("HF_TOKEN")                                           # mandatory, no default

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

# ─── Optional env vars ─────────────────────────────────────────────────────────
IMAGE_NAME   = os.getenv("IMAGE_NAME")   # local Docker mode — optional
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "https://mohdaltamish-driftfix-env.hf.space")

# ─── Constants ─────────────────────────────────────────────────────────────────
BENCHMARK               = "schema_migration_env"
MAX_STEPS               = 15
SUCCESS_SCORE_THRESHOLD = 0.5
ALL_TASKS               = ["add_missing_column", "normalize_table", "breaking_version_migration"]

# Initialize OpenAI client (per judge spec)
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
)

# ─── Logging (EXACT judge format — do not modify field names or order) ─────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    # Must be single line — strip all newlines from action
    action_clean = action.replace("\n", " ").replace("\r", "").strip()
    print(
        f"[STEP] step={step} action={action_clean} reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)


# ─── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = textwrap.dedent("""\
    You are a database migration expert working with a SQLite database.
    Your goal: issue SQL DDL/DML statements one at a time to fix the schema so all target queries pass.

    Rules:
    - Output ONLY a single valid SQL statement per turn. No explanation, no markdown, no backticks.
    - Read schema_dump and query_results carefully before each action.
    - Fix the root cause of each failing query.
    - SQLite ALTER TABLE cannot add PRIMARY KEY, UNIQUE, or STORED columns.
      For unsupported changes: CREATE new table, INSERT SELECT from old, DROP old, RENAME new.
    - Always preserve existing row data.
    - When ALL target queries pass, output exactly: SUBMIT
""")


# ─── Observation → prompt ──────────────────────────────────────────────────────
def obs_to_prompt(obs) -> str:
    lines = [
        f"Task: {obs.task_id} — {obs.task_description}",
        f"Step: {obs.step_count}",
    ]
    if obs.hint:
        lines.append(f"Hint: {obs.hint}")

    lines.append("\n=== Current Schema ===")
    lines.append(obs.schema_dump or "(empty)")

    lines.append("\n=== Target Query Status ===")
    for qr in obs.query_results:
        status = "PASS" if qr.passed else "FAIL"
        lines.append(f"[{status}] {qr.query}")
        if qr.error:
            lines.append(f"  Error: {qr.error}")
        if qr.actual_row_count is not None:
            lines.append(f"  Rows: {qr.actual_row_count}")

    if obs.last_sql_error:
        lines.append(f"\n=== Last SQL Error ===\n{obs.last_sql_error}")

    lines.append("\nOutput your next SQL statement (or SUBMIT if all queries pass):")
    return "\n".join(lines)


# ─── LLM call ──────────────────────────────────────────────────────────────────
def get_action(obs, messages: list) -> str:
    messages.append({"role": "user", "content": obs_to_prompt(obs)})

    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=256,
            temperature=0.2,
        )
        raw = (resp.choices[0].message.content or "").strip()
        messages.append({"role": "assistant", "content": raw})
    except Exception as e:
        print(f"[DEBUG] LLM call failed: {e}", file=sys.stderr, flush=True)
        raw = "SELECT 1;"
        messages.append({"role": "assistant", "content": raw})

    # Strip markdown fences
    sql = raw
    for fence in ("```sql", "```SQL", "```"):
        if sql.startswith(fence):
            sql = sql[len(fence):].lstrip()
    if sql.endswith("```"):
        sql = sql[:-3].rstrip()
    sql = sql.strip()

    # Enforce single statement only
    if sql.upper() != "SUBMIT":
        parts = [s.strip() for s in sql.split(";") if s.strip()]
        sql = (parts[0] + ";") if parts else "SELECT 1;"

    return sql


# ─── Episode runner ────────────────────────────────────────────────────────────
async def run_episode(env: SchemaMigrationEnv, task_id: str) -> None:
    rewards: List[float] = []
    steps_taken = 0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        reset_result = await env.reset(task_id=task_id)
        obs = reset_result.observation
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        for step_num in range(1, MAX_STEPS + 1):
            if obs.done:
                break

            sql = get_action(obs, messages)
            action_type = "submit" if sql.strip().upper() == "SUBMIT" else "execute"
            action = SchemaMigrationAction(sql=sql, action_type=action_type)

            try:
                step_result = await env.step(action)
            except Exception as e:
                print(f"[DEBUG] env.step() failed: {e}", file=sys.stderr, flush=True)
                break

            obs        = step_result.observation
            reward     = float(step_result.reward) if step_result.reward is not None else 0.0
            done       = step_result.done
            error      = obs.last_sql_error

            rewards.append(reward)
            steps_taken = step_num

            log_step(step=step_num, action=sql, reward=reward, done=done, error=error)

            if done:
                break

        score   = sum(rewards)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Episode error ({task_id}): {e}", file=sys.stderr, flush=True)

    finally:
        # [END] always emitted — even on exception — per judge rules
        if not rewards:
            rewards = [0.0]
        log_end(success=success, steps=steps_taken, rewards=rewards)


# ─── Main ──────────────────────────────────────────────────────────────────────
async def main() -> None:
    env = None

    # Docker mode (local/dev) → HF Space mode (judge/CI)
    if IMAGE_NAME:
        try:
            print(f"[DEBUG] Launching Docker: {IMAGE_NAME}", file=sys.stderr, flush=True)
            env = await SchemaMigrationEnv.from_docker_image(IMAGE_NAME)
        except Exception as e:
            print(f"[DEBUG] Docker failed ({e}), falling back to HF Space", file=sys.stderr, flush=True)
            env = None

    if env is None:
        try:
            print(f"[DEBUG] Connecting to: {ENV_BASE_URL}", file=sys.stderr, flush=True)
            env = SchemaMigrationEnv.from_url(ENV_BASE_URL)
        except Exception as e:
            print(f"[DEBUG] Connection failed: {e}", file=sys.stderr, flush=True)
            for task_id in ALL_TASKS:
                log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
                log_end(success=False, steps=0, rewards=[0.0])
            return

    try:
        for task_id in ALL_TASKS:
            try:
                await run_episode(env, task_id)
            except Exception as e:
                print(f"[DEBUG] Task failed ({task_id}): {e}", file=sys.stderr, flush=True)
                log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
                log_end(success=False, steps=0, rewards=[0.0])
    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"[DEBUG] Fatal: {e}", file=sys.stderr, flush=True)
        for task_id in ALL_TASKS:
            log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
            log_end(success=False, steps=0, rewards=[0.0])