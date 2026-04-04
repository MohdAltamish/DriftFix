"""
Inference Script — Schema Migration Environment
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    IMAGE_NAME     The name of the local image to use for the environment.

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

IMAGE_NAME = os.getenv("IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
TASK_NAME = os.getenv("TASK_NAME") or "add_missing_column"
BENCHMARK = "schema_migration_env"
MAX_STEPS = 15
SUCCESS_SCORE_THRESHOLD = 0.5

SYSTEM_PROMPT = textwrap.dedent("""\
    You are a database migration expert. You receive a SQLite database schema with failing SQL queries.
    Your goal is to issue SQL DDL/DML statements one at a time to fix the schema so all target queries pass.
    Rules:
    - Output ONLY a single valid SQL statement per turn. No explanation, no markdown, no backticks.
    - Read the schema_dump and query_results carefully before acting.
    - If a query fails with an error, fix the root cause.
    - Prefer ALTER TABLE over DROP+RECREATE when possible.
    - Always preserve existing data.
    - When all queries pass, output: SUBMIT
""")

ALL_TASKS = ["add_missing_column", "normalize_table", "breaking_version_migration"]


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


def build_user_prompt(observation) -> str:
    """Build the user prompt from an observation."""
    # Format query results
    qr_lines = []
    for qr in observation.query_results:
        status = "✅ PASS" if qr.passed else "❌ FAIL"
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


def get_model_action(client: OpenAI, observation, history: List[str]) -> str:
    """Get the next SQL action from the LLM."""
    user_prompt = build_user_prompt(observation)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Add recent history for context
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

        # Clean up: remove markdown backticks if present
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first and last lines (backtick markers)
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines).strip()

        return text if text else "SELECT 1;"
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return "SELECT 1;"


async def run_episode(
    env: SchemaMigrationEnv,
    client: OpenAI,
    task_id: str,
) -> None:
    """Run a single episode for one task."""
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

            # Get action from LLM
            sql = get_model_action(client, observation, history)
            history.append(sql)

            # Determine action type
            action_type = "submit" if sql.strip().upper() == "SUBMIT" else "execute"
            action = SchemaMigrationAction(sql=sql, action_type=action_type)

            step_result = await env.step(action)
            observation = step_result.observation
            reward = step_result.reward
            done = step_result.done
            error = observation.last_sql_error

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=sql, reward=reward, done=done, error=error)

            if done:
                break

        # Compute episode score
        total = sum(rewards)
        score = max(min(total, 1.0), 0.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    env = await SchemaMigrationEnv.from_docker_image(IMAGE_NAME)

    try:
        for task_id in ALL_TASKS:
            await run_episode(env, client, task_id)
    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error (container cleanup): {e}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())