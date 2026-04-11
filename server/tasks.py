"""Task definitions and graders for the Schema Migration Environment.

Each grader returns a float strictly in (0.001, 0.999) via _clamp().
Exact 0.0 or 1.0 are rejected by the Phase 2 validator.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from server.models import QueryResult


def _clamp(score: float) -> float:
    """Return score strictly in (0.001, 0.999) — validator rejects exact 0.0 or 1.0."""
    return max(0.001, min(0.999, float(score)))


@dataclass
class TaskDefinition:
    task_id: str
    difficulty: str
    description: str
    seed_sql: List[str]
    target_queries: List[dict]
    grader: Callable[[List[QueryResult], sqlite3.Connection], float]
    hint: Optional[str] = None
    max_steps: int = 10


# ────────────────────────────────────────────────────────────────────
# TASK 1 — add_missing_column (EASY)
# ────────────────────────────────────────────────────────────────────

_TASK1_SEED = [
    "CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT, department TEXT);",
    "INSERT INTO employees (id, name, department) VALUES (1, 'Alice', 'Engineering');",
    "INSERT INTO employees (id, name, department) VALUES (2, 'Bob', 'Marketing');",
    "INSERT INTO employees (id, name, department) VALUES (3, 'Carol', 'Engineering');",
    "INSERT INTO employees (id, name, department) VALUES (4, 'Dave', 'Sales');",
    "INSERT INTO employees (id, name, department) VALUES (5, 'Eve', 'Marketing');",
]

_TASK1_QUERIES = [
    {"query_id": "q1", "query": "SELECT id, name, salary FROM employees WHERE salary > 50000;"},
    {"query_id": "q2", "query": "SELECT department, AVG(salary) FROM employees GROUP BY department;"},
]


def _grader_add_missing_column(
    results: List[QueryResult], conn: sqlite3.Connection
) -> float:
    if results is None:
        return _clamp(0.0)
    score = 0.0
    for r in results:
        if r.query_id == "q1" and r.passed:
            score += 0.4
        elif r.query_id == "q2" and r.passed:
            score += 0.4
    # Bonus: salary values are varied
    try:
        cursor = conn.execute("SELECT salary FROM employees WHERE salary IS NOT NULL")
        salaries = [row[0] for row in cursor.fetchall()]
        if salaries and len(set(salaries)) > 1:
            score += 0.2
    except Exception:
        pass
    return _clamp(score)


# ────────────────────────────────────────────────────────────────────
# TASK 2 — normalize_table (MEDIUM)
# ────────────────────────────────────────────────────────────────────

_TASK2_SEED = [
    """CREATE TABLE orders (
        id INTEGER PRIMARY KEY,
        customer_name TEXT,
        customer_email TEXT,
        customer_phone TEXT,
        product_name TEXT,
        product_price REAL,
        quantity INTEGER
    );""",
    "INSERT INTO orders VALUES (1, 'John Doe', 'john@example.com', '555-0101', 'Widget A', 9.99, 3);",
    "INSERT INTO orders VALUES (2, 'John Doe', 'john@example.com', '555-0101', 'Widget B', 19.99, 1);",
    "INSERT INTO orders VALUES (3, 'Jane Smith', 'jane@example.com', '555-0202', 'Widget A', 9.99, 5);",
    "INSERT INTO orders VALUES (4, 'Jane Smith', 'jane@example.com', '555-0202', 'Widget C', 29.99, 2);",
    "INSERT INTO orders VALUES (5, 'Bob Wilson', 'bob@example.com', '555-0303', 'Widget B', 19.99, 4);",
    "INSERT INTO orders VALUES (6, 'John Doe', 'john@example.com', '555-0101', 'Widget C', 29.99, 1);",
    "INSERT INTO orders VALUES (7, 'Jane Smith', 'jane@example.com', '555-0202', 'Widget A', 9.99, 2);",
    "INSERT INTO orders VALUES (8, 'Bob Wilson', 'bob@example.com', '555-0303', 'Widget A', 9.99, 6);",
]

_TASK2_QUERIES = [
    {"query_id": "q1", "query": "SELECT c.name, COUNT(o.id) as order_count FROM customers c JOIN orders o ON c.id = o.customer_id GROUP BY c.id;"},
    {"query_id": "q2", "query": "SELECT p.name, SUM(o.quantity) FROM products p JOIN orders o ON p.id = o.product_id GROUP BY p.id;"},
    {"query_id": "q3", "query": "SELECT c.email FROM customers c WHERE c.id = 1;"},
]


def _grader_normalize_table(
    results: List[QueryResult], conn: sqlite3.Connection
) -> float:
    if results is None:
        return _clamp(0.0)
    score = 0.0
    passed_count = 0
    for r in results:
        if r.passed:
            score += 0.3
            passed_count += 1
    if passed_count == 3:
        score += 0.1
    return _clamp(score)


# ────────────────────────────────────────────────────────────────────
# TASK 3 — breaking_version_migration (HARD)
# ────────────────────────────────────────────────────────────────────

_TASK3_SEED = [
    """CREATE TABLE user_accounts (
        uid INTEGER PRIMARY KEY,
        full_name TEXT,
        usr_email TEXT,
        usr_phone TEXT,
        acct_balance REAL,
        created_ts TEXT
    );""",
    """CREATE TABLE transactions (
        txn_id INTEGER PRIMARY KEY,
        user_uid INTEGER,
        txn_amount REAL,
        txn_date TEXT,
        txn_type TEXT,
        FOREIGN KEY(user_uid) REFERENCES user_accounts(uid)
    );""",
    "INSERT INTO user_accounts VALUES (1, 'Alice Johnson', 'alice@example.com', '555-0001', 1500.00, '2023-01-15');",
    "INSERT INTO user_accounts VALUES (2, 'Bob Smith', 'bob@example.com', '555-0002', 2300.50, '2023-02-20');",
    "INSERT INTO user_accounts VALUES (3, 'Carol White', 'carol@example.com', '555-0003', 890.75, '2023-03-10');",
    "INSERT INTO user_accounts VALUES (4, 'Dave Brown', 'dave@example.com', '555-0004', 4200.00, '2023-04-05');",
    "INSERT INTO user_accounts VALUES (5, 'Eve Davis', 'eve@example.com', '555-0005', 3100.25, '2023-05-12');",
    "INSERT INTO user_accounts VALUES (6, 'Frank Miller', 'frank@example.com', '555-0006', 750.00, '2023-06-18');",
    "INSERT INTO user_accounts VALUES (7, 'Grace Lee', 'grace@example.com', '555-0007', 5600.80, '2023-07-22');",
    "INSERT INTO user_accounts VALUES (8, 'Hank Moore', 'hank@example.com', '555-0008', 920.30, '2023-08-30');",
    "INSERT INTO user_accounts VALUES (9, 'Ivy Clark', 'ivy@example.com', '555-0009', 1800.00, '2023-09-14');",
    "INSERT INTO user_accounts VALUES (10, 'Jack Taylor', 'jack@example.com', '555-0010', 3400.60, '2023-10-01');",
    "INSERT INTO transactions VALUES (1,  1,  100.00, '2023-06-01', 'credit');",
    "INSERT INTO transactions VALUES (2,  1,  -50.00, '2023-06-02', 'debit');",
    "INSERT INTO transactions VALUES (3,  2,  200.00, '2023-06-03', 'credit');",
    "INSERT INTO transactions VALUES (4,  2,  -75.50, '2023-06-04', 'debit');",
    "INSERT INTO transactions VALUES (5,  3,  300.00, '2023-06-05', 'credit');",
    "INSERT INTO transactions VALUES (6,  3, -120.00, '2023-06-06', 'debit');",
    "INSERT INTO transactions VALUES (7,  4,  450.00, '2023-06-07', 'credit');",
    "INSERT INTO transactions VALUES (8,  4, -200.00, '2023-06-08', 'debit');",
    "INSERT INTO transactions VALUES (9,  5,  150.00, '2023-06-09', 'credit');",
    "INSERT INTO transactions VALUES (10, 5,  -80.00, '2023-06-10', 'debit');",
    "INSERT INTO transactions VALUES (11, 6,  500.00, '2023-06-11', 'credit');",
    "INSERT INTO transactions VALUES (12, 7, -250.00, '2023-06-12', 'debit');",
    "INSERT INTO transactions VALUES (13, 7,  600.00, '2023-06-13', 'credit');",
    "INSERT INTO transactions VALUES (14, 8, -100.00, '2023-06-14', 'debit');",
    "INSERT INTO transactions VALUES (15, 8,  350.00, '2023-06-15', 'credit');",
    "INSERT INTO transactions VALUES (16, 9, -175.00, '2023-06-16', 'debit');",
    "INSERT INTO transactions VALUES (17, 9,  400.00, '2023-06-17', 'credit');",
    "INSERT INTO transactions VALUES (18, 10,-300.00, '2023-06-18', 'debit');",
    "INSERT INTO transactions VALUES (19, 10, 225.00, '2023-06-19', 'credit');",
    "INSERT INTO transactions VALUES (20, 1,  175.00, '2023-06-20', 'credit');",
]

_TASK3_QUERIES = [
    {"query_id": "q1", "query": "SELECT id, name, email FROM users LIMIT 1;"},
    {"query_id": "q2", "query": "SELECT u.name, SUM(t.amount) FROM users u JOIN transactions t ON u.id = t.user_id GROUP BY u.id;"},
    {"query_id": "q3", "query": "SELECT COUNT(*) FROM users;",        "expected_row_count": 1},
    {"query_id": "q4", "query": "SELECT COUNT(*) FROM transactions;", "expected_row_count": 1},
    {"query_id": "q5", "query": "SELECT id, amount, created_at FROM transactions LIMIT 1;"},
]


def _grader_breaking_version_migration(
    results: List[QueryResult], conn: sqlite3.Connection
) -> float:
    if results is None:
        return _clamp(0.0)
    weights = {"q1": 0.15, "q2": 0.20, "q3": 0.25, "q4": 0.25, "q5": 0.15}
    score = 0.0

    for r in results:
        if r.passed:
            score += weights.get(r.query_id, 0.0)

    # Data preservation checks
    for r in results:
        if r.query_id == "q3" and r.passed:
            try:
                count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
                if count != 10:
                    score -= weights["q3"]
            except Exception:
                pass
        elif r.query_id == "q4" and r.passed:
            try:
                count = conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
                if count != 20:
                    score -= weights["q4"]
            except Exception:
                pass

    # Rollback penalty — any critical table emptied
    for table in ["users", "user_accounts", "transactions"]:
        try:
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            if count == 0:
                score -= 0.3
                break
        except Exception:
            continue

    return _clamp(score)


# ────────────────────────────────────────────────────────────────────
# TASK REGISTRY
# ────────────────────────────────────────────────────────────────────

TASKS: Dict[str, TaskDefinition] = {
    "add_missing_column": TaskDefinition(
        task_id="add_missing_column",
        difficulty="easy",
        description="Add a missing column to fix failing SELECT queries",
        seed_sql=_TASK1_SEED,
        target_queries=_TASK1_QUERIES,
        grader=_grader_add_missing_column,
        hint="The employees table is missing a column referenced in the queries",
        max_steps=10,
    ),
    "normalize_table": TaskDefinition(
        task_id="normalize_table",
        difficulty="medium",
        description="Normalize a denormalized table into proper relational schema",
        seed_sql=_TASK2_SEED,
        target_queries=_TASK2_QUERIES,
        grader=_grader_normalize_table,
        hint=None,
        max_steps=20,
    ),
    "breaking_version_migration": TaskDefinition(
        task_id="breaking_version_migration",
        difficulty="hard",
        description="Migrate legacy schema to new version without data loss",
        seed_sql=_TASK3_SEED,
        target_queries=_TASK3_QUERIES,
        grader=_grader_breaking_version_migration,
        hint=None,
        max_steps=30,
    ),
}


def get_task(task_id: str) -> TaskDefinition:
    if task_id not in TASKS:
        raise KeyError(f"Unknown task '{task_id}'. Available: {list(TASKS.keys())}")
    return TASKS[task_id]


def list_task_ids() -> List[str]:
    return list(TASKS.keys())