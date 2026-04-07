"""
Inference Script — Schema Migration Environment
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    IMAGE_NAME     The name of the local image to use for the environment (optional).

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import asyncio
import os
import textwrap
from typing import List, Optional

from openai import OpenAI

from schema_migration_env import SchemaMigrationEnv, SchemaMigrationAction

# ── Environment variables ────────────────────────────────────────────────────
IMAGE_NAME    = os.getenv("IMAGE_NAME")          # optional — only used for local Docker mode
API_KEY       = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL  = os.getenv("API_BASE_URL")  or "https://router.huggingface.co/v1"
MODEL_NAME    = os.getenv("MODEL_NAME")    or "Qwen/Qwen2.5-72B-Instruct"
HF_SPACE_URL  = os.getenv("HF_SPACE_URL") or "https://mohdaltamish-driftfix-env.hf.space"

BENCHMARK              = "schema_migration_env"
MAX_STEPS              = 15
SUCCESS_SCORE_THRESHOLD = 0.5

ALL_TASKS = ["add_missing_column", "normalize_table", "breaking_version_migration"]

# ── System prompt ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = textwrap.dedent("""\
    You are a database migration expert. You receive a SQLite database schema with failing SQL queries.
    Your goal is to issue SQL DDL/DML statements one at a time to fix the schema so all target queries pass.
    Rules:
    - CRITICAL: Output exactly ONE single SQL statement per turn. No combined statements.
    - No explanation, no markdown, no backticks.
    - Read the schema_dump and query_results carefully before acting.
    - If a query fails with an error, fix the root cause.
    - SQLite ALTER TABLE is limited. Do NOT add UNIQUE, PRIMARY KEY, or STORED constraints via ALTER TABLE ADD COLUMN.
    - For unsupported structural changes: create new table, copy data, drop old, rename new.
    - Always preserve existing data.
    - When all queries pass, output: SUBMIT
""")

# ── Logging helpers (mandatory format — do not modify) ───────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


# ── Prompt builder ───────────────────────────────────────────────────────────
def build_user_prompt(observation) -> str:
    qr_lines = []
    for qr in observation.query_results:
        status = "PASS" if qr.passed else "FAIL"
        line = f"  {qr.query_id}: {status} — {qr.query}"
        if qr.error:
            line += f"\n    Error: {qr.error}"
        qr_lines.append(line)
    queries_str = "\n".join(qr_lines)
    hint_str = f"\nHint: {observation.hint}" if observation.hint else ""

    return textwrap.dedent(f"""\
        Task: {observation.task_id} — {observation.task_description}
        Step: {observation.step_count}
        {hint_str}

        Current Schema:
        {observation.schema_dump}

        Target Query Results:
        {queries_str}

        Last SQL Error: {observation.last_sql_error or 'None'}
        Last SQL Output: {observation.last_sql_output or 'None'}

        Issue your next SQL statement (or SUBMIT if all queries pass):
    """)


# ── LLM call ─────────────────────────────────────────────────────────────────
def get_model_action(client: OpenAI, observation, history: List[str]) -> str:
    user_prompt = build_user_prompt(observation)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in history[-6:]:
        messages.append({"role": "assistant", "content": h})
    messages.append({"role": "user", "content": user_prompt})

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.2,
            max_tokens=500,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()

        # Strip markdown backticks if present
        if text.startswith("```"):
            lines = [l for l in text.split("\n") if not l.strip().startswith("```")]
            text = "\n".join(lines).strip()

        # Enforce single statement
        if text.strip().upper() != "SUBMIT":
            statements = [s for s in text.split(";") if s.strip()]
            if statements:
                text = statements[0].strip() + ";"

        return text if text else "SELECT 1;"
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return "SELECT 1;"


# ── Episode runner ───────────────────────────────────────────────────────────
async def run_episode(env: SchemaMigrationEnv, client: OpenAI, task_id: str) -> None:
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    history: List[str] = []

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(task_id=task_id)
        observation = result.observation

        for step in range(1, MAX_STEPS + 1):
            if observation.done:
                break

            sql = get_model_action(client, observation, history)
            history.append(sql)

            action_type = "submit" if sql.strip().upper() == "SUBMIT" else "execute"
            action = SchemaMigrationAction(sql=sql, action_type=action_type)

            step_result = await env.step(action)
            observation = step_result.observation
            reward = step_result.reward or 0.0
            done = step_result.done
            error = observation.last_sql_error

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=sql, reward=reward, done=done, error=error)

            if done:
                break

        score = max(min(sum(rewards), 1.0), 0.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Episode error for task {task_id}: {e}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ── Main ─────────────────────────────────────────────────────────────────────
async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Connect to environment:
    # - If IMAGE_NAME is set → start local Docker container
    # - Otherwise → connect directly to HF Space (judge environment)
    env = None
    try:
        if IMAGE_NAME:
            print(f"[DEBUG] Starting Docker container: {IMAGE_NAME}", flush=True)
            env = await SchemaMigrationEnv.from_docker_image(IMAGE_NAME, port=8007)
        else:
            print(f"[DEBUG] Connecting to HF Space: {HF_SPACE_URL}", flush=True)
            env = SchemaMigrationEnv.from_url(HF_SPACE_URL)

    except Exception as e:
        print(f"[DEBUG] Failed to connect to environment: {e}", flush=True)
        # Emit mandatory [START]/[END] for all tasks so validator sees output
        for task_id in ALL_TASKS:
            log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
            log_end(success=False, steps=0, score=0.0, rewards=[])
        return

    try:
        for task_id in ALL_TASKS:
            await run_episode(env, client, task_id)
    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except BaseException as e:
        print(f"[DEBUG] Fatal error: {e}", flush=True)
        # Last resort — emit END for any tasks that didn't complete
        for task_id in ALL_TASKS:
            log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
            log_end(success=False, steps=0, score=0.0, rewards=[])