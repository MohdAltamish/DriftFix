"""
test_local.py — Full local pipeline test (no LLM required).

Runs all 3 tasks with hardcoded optimal SQL solutions to verify:
  - Environment reset/step/reward
  - Judge-compliant log format
  - Correct reward values (0.00 or 1.00)

Usage:
    .venv/bin/python test_local.py
"""

import asyncio
import os
import sys
from typing import List, Optional

from schema_migration_env import SchemaMigrationEnv, SchemaMigrationAction

# ── Constants ────────────────────────────────────────────────────────────────
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")
MODEL_NAME   = "local-oracle"          # Fake model name for local testing
BENCHMARK    = "schema_migration_env"
SUCCESS_SCORE_THRESHOLD = 0.5

# ── Optimal SQL sequences for each task ─────────────────────────────────────

TASK_SOLUTIONS = {
    "add_missing_column": [
        "ALTER TABLE employees ADD COLUMN salary REAL;",
        "UPDATE employees SET salary = 60000 WHERE id = 1;",
        "UPDATE employees SET salary = 55000 WHERE id = 2;",
        "UPDATE employees SET salary = 72000 WHERE id = 3;",
        "UPDATE employees SET salary = 48000 WHERE id = 4;",
        "UPDATE employees SET salary = 65000 WHERE id = 5;",
        "SUBMIT",
    ],
    "normalize_table": [
        # Step 1: Create customers table
        "CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, email TEXT, phone TEXT);",
        # Step 2: Populate customers
        "INSERT INTO customers (name, email, phone) SELECT DISTINCT customer_name, customer_email, customer_phone FROM orders;",
        # Step 3: Create products table
        "CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT, price REAL);",
        # Step 4: Populate products
        "INSERT INTO products (name, price) SELECT DISTINCT product_name, product_price FROM orders;",
        # Step 5: Create new normalized orders table
        "CREATE TABLE orders_new (id INTEGER PRIMARY KEY, customer_id INTEGER, product_id INTEGER, quantity INTEGER, FOREIGN KEY(customer_id) REFERENCES customers(id), FOREIGN KEY(product_id) REFERENCES products(id));",
        # Step 6: Populate new orders
        "INSERT INTO orders_new (id, customer_id, product_id, quantity) SELECT o.id, c.id, p.id, o.quantity FROM orders o JOIN customers c ON o.customer_email = c.email JOIN products p ON o.product_name = p.name;",
        # Step 7: Drop old orders and rename new
        "DROP TABLE orders;",
        "ALTER TABLE orders_new RENAME TO orders;",
        "SUBMIT",
    ],
    "breaking_version_migration": [
        # Step 1: Create renamed users table
        "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, email TEXT, phone TEXT, balance REAL, created_at TEXT);",
        # Step 2: Migrate data from user_accounts → users
        "INSERT INTO users (id, name, email, phone, balance, created_at) SELECT uid, full_name, usr_email, usr_phone, acct_balance, created_ts FROM user_accounts;",
        # Step 3: Rename transactions to backup
        "ALTER TABLE transactions RENAME TO transactions_old;",
        # Step 4: Create new transactions table with renamed columns
        "CREATE TABLE transactions (id INTEGER PRIMARY KEY, user_id INTEGER, amount REAL, created_at TEXT, type TEXT, FOREIGN KEY(user_id) REFERENCES users(id));",
        # Step 5: Migrate transactions data
        "INSERT INTO transactions (id, user_id, amount, created_at, type) SELECT txn_id, user_uid, txn_amount, txn_date, txn_type FROM transactions_old;",
        # Step 6: Clean up
        "DROP TABLE transactions_old;",
        "DROP TABLE user_accounts;",
        "SUBMIT",
    ],
}

# ── Judge-format logging ──────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    action_clean = action.replace("\n", " ").replace("\r", "").strip()
    print(
        f"[STEP] step={step} action={action_clean} reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)


# ── Episode runner ────────────────────────────────────────────────────────────

async def run_episode(env: SchemaMigrationEnv, task_id: str) -> None:
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    sql_sequence = TASK_SOLUTIONS.get(task_id, ["SUBMIT"])

    try:
        reset_result = await env.reset(task_id=task_id)
        obs = reset_result.observation

        for step_num, sql in enumerate(sql_sequence, start=1):
            if obs.done:
                break

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
                score = step_result.info.get("score", sum(rewards))
                break

        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Episode error ({task_id}): {e}", file=sys.stderr, flush=True)

    finally:
        if not rewards:
            rewards = [0.0]
        log_end(success=success, steps=steps_taken, rewards=rewards)


# ── Main ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    print(f"[DEBUG] Connecting to: {ENV_BASE_URL}", file=sys.stderr, flush=True)

    try:
        env = SchemaMigrationEnv.from_url(ENV_BASE_URL)
    except Exception as e:
        print(f"[DEBUG] Connection failed: {e}", file=sys.stderr, flush=True)
        for task_id in TASK_SOLUTIONS:
            log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
            log_end(success=False, steps=0, rewards=[0.0])
        return

    try:
        for task_id in TASK_SOLUTIONS:
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

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "="*60, flush=True)
    print("✅  Local test complete — all tasks ran end-to-end.", flush=True)
    print("="*60, flush=True)


if __name__ == "__main__":
    asyncio.run(main())
