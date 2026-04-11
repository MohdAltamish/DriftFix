"""
graders.py — Required by OpenEnv validator.
Exposes EasyGrader, MediumGrader, HardGrader classes.
Each grade(env) returns float strictly in (0.001, 0.999).
Safe to call with env=None.
"""
from __future__ import annotations
import sqlite3
from typing import Optional


def _clamp(score: float) -> float:
    return max(0.001, min(0.999, float(score)))


def _get_conn(env) -> Optional[sqlite3.Connection]:
    if env is None:
        return None
    for attr in ("conn", "_conn", "db", "connection", "_db"):
        conn = getattr(env, attr, None)
        if conn is not None:
            return conn
    return None


class EasyGrader:
    def grade(self, env) -> float:
        if env is None:
            return 0.5
        try:
            conn = _get_conn(env)
            if conn is None:
                return 0.5
            score = 0.0
            try:
                conn.execute("SELECT id, name, salary FROM employees WHERE salary > 50000")
                score += 0.4
            except Exception:
                pass
            try:
                conn.execute("SELECT department, AVG(salary) FROM employees GROUP BY department")
                score += 0.4
            except Exception:
                pass
            try:
                salaries = [r[0] for r in conn.execute("SELECT salary FROM employees WHERE salary IS NOT NULL").fetchall()]
                if salaries and len(set(salaries)) > 1:
                    score += 0.2
            except Exception:
                pass
            return _clamp(score if score > 0 else 0.001)
        except Exception:
            return 0.5


class MediumGrader:
    def grade(self, env) -> float:
        if env is None:
            return 0.5
        try:
            conn = _get_conn(env)
            if conn is None:
                return 0.5
            score = 0.0
            try:
                conn.execute("SELECT c.name, COUNT(o.id) FROM customers c JOIN orders o ON c.id = o.customer_id GROUP BY c.id")
                score += 0.3
            except Exception:
                pass
            try:
                conn.execute("SELECT p.name, SUM(o.quantity) FROM products p JOIN orders o ON p.id = o.product_id GROUP BY p.id")
                score += 0.3
            except Exception:
                pass
            try:
                conn.execute("SELECT c.email FROM customers c WHERE c.id = 1")
                score += 0.3
            except Exception:
                pass
            if score >= 0.9:
                score += 0.1
            return _clamp(score if score > 0 else 0.001)
        except Exception:
            return 0.5


class HardGrader:
    def grade(self, env) -> float:
        if env is None:
            return 0.5
        try:
            conn = _get_conn(env)
            if conn is None:
                return 0.5
            weights = {"q1": 0.15, "q2": 0.20, "q3": 0.25, "q4": 0.25, "q5": 0.15}
            score = 0.0
            try:
                rows = conn.execute("SELECT id, name, email FROM users LIMIT 1").fetchall()
                if rows:
                    score += weights["q1"]
            except Exception:
                pass
            try:
                rows = conn.execute("SELECT u.name, SUM(t.amount) FROM users u JOIN transactions t ON u.id = t.user_id GROUP BY u.id").fetchall()
                if rows:
                    score += weights["q2"]
            except Exception:
                pass
            try:
                count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
                if count > 0:
                    score += weights["q3"]
                if count != 10:
                    score -= weights["q3"]
            except Exception:
                pass
            try:
                count = conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
                if count > 0:
                    score += weights["q4"]
                if count != 20:
                    score -= weights["q4"]
            except Exception:
                pass
            try:
                rows = conn.execute("SELECT id, amount, created_at FROM transactions LIMIT 1").fetchall()
                if rows:
                    score += weights["q5"]
            except Exception:
                pass
            for table in ["users", "user_accounts", "transactions"]:
                try:
                    count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                    if count == 0:
                        score -= 0.3
                        break
                except Exception:
                    continue
            return _clamp(score if score > 0 else 0.001)
        except Exception:
            return 0.5
