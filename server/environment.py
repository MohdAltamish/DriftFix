"""Core environment logic for the Schema Migration Environment.

Each episode uses an isolated in-memory SQLite database.

REWARD DESIGN:
  - Per-step rewards are progress-based (delta in queries passing) — range (-0.3, 0.5)
  - Episode final score = task grader called ONCE at episode end — strictly (0.001, 0.999)
  - The grader is NOT called on every step (was the bug causing 0.0/1.0 scores)
"""

from __future__ import annotations

import re
import sqlite3
import uuid
from typing import List, Optional, Tuple

from server.models import (
    QueryResult,
    ResetResult,
    SchemaMigrationAction,
    SchemaMigrationObservation,
    SchemaMigrationState,
    StepResult,
)
from server.tasks import TaskDefinition, get_task, list_task_ids


def _clamp(score: float) -> float:
    """Clamp to strictly (0.001, 0.999) — validator rejects exact 0.0 or 1.0."""
    return max(0.001, min(0.999, float(score)))


class SchemaMigrationEnvironment:
    """RL environment for database schema migration."""

    def __init__(self) -> None:
        self.conn: Optional[sqlite3.Connection] = None
        self.task: Optional[TaskDefinition] = None
        self.episode_id: str = ""
        self.step_count: int = 0
        self.done: bool = False
        self.reward_so_far: float = 0.0
        self.rewards: List[float] = []
        self.last_sql_error: Optional[str] = None
        self.last_sql_output: Optional[str] = None
        self._queries_passing_before: int = 0
        self._initial_table_row_counts: dict = {}

    def reset(self, task_id: Optional[str] = None) -> ResetResult:
        """Reset the environment with a new or specified task."""
        if self.conn:
            try:
                self.conn.close()
            except Exception:
                pass

        if task_id is None:
            task_id = list_task_ids()[0]
        self.task = get_task(task_id)

        self.conn = sqlite3.connect(":memory:")
        self.conn.execute("PRAGMA foreign_keys = ON;")

        for sql in self.task.seed_sql:
            self.conn.execute(sql)
        self.conn.commit()

        self._record_initial_row_counts()

        self.episode_id = str(uuid.uuid4())
        self.step_count = 0
        self.done = False
        self.reward_so_far = 0.0
        self.rewards = []
        self.last_sql_error = None
        self.last_sql_output = None

        query_results = self._evaluate_queries()
        self._queries_passing_before = sum(1 for r in query_results if r.passed)

        observation = SchemaMigrationObservation(
            schema_dump=self._get_schema_dump(),
            query_results=query_results,
            last_sql_error=None,
            last_sql_output=None,
            step_count=0,
            done=False,
            task_id=self.task.task_id,
            task_description=self.task.description,
            hint=self.task.hint,
        )

        return ResetResult(
            observation=observation,
            info={"session_id": self.episode_id, "task_id": self.task.task_id},
        )

    def step(self, action: SchemaMigrationAction) -> StepResult:
        """Execute one step in the environment."""
        if self.task is None or self.conn is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        if self.done:
            query_results = self._evaluate_queries()
            obs = SchemaMigrationObservation(
                schema_dump=self._get_schema_dump(),
                query_results=query_results,
                last_sql_error="Episode already done.",
                last_sql_output=None,
                step_count=self.step_count,
                done=True,
                task_id=self.task.task_id,
                task_description=self.task.description,
                hint=self.task.hint,
            )
            return StepResult(
                observation=obs,
                reward=0.01,  # small nonzero — never exact 0.0
                done=True,
                info={"session_id": self.episode_id},
            )

        self.step_count += 1

        # ── SUBMIT action ───────────────────────────────────────────
        if action.action_type == "submit" or action.sql.strip().upper() == "SUBMIT":
            query_results = self._evaluate_queries()
            self.done = True

            # Call grader ONCE at episode end
            final_score = self._get_final_score(query_results)
            reward = final_score  # already clamped to (0.001, 0.999)

            self.rewards.append(reward)
            self.reward_so_far += reward

            all_pass = sum(1 for r in query_results if r.passed) == len(self.task.target_queries)
            obs = SchemaMigrationObservation(
                schema_dump=self._get_schema_dump(),
                query_results=query_results,
                last_sql_error=None,
                last_sql_output="Submitted. All queries pass." if all_pass else "Submitted. Not all queries pass.",
                step_count=self.step_count,
                done=True,
                task_id=self.task.task_id,
                task_description=self.task.description,
                hint=self.task.hint,
            )
            return StepResult(
                observation=obs,
                reward=reward,
                done=True,
                info={"session_id": self.episode_id, "score": final_score},
            )

        # ── RESET action ────────────────────────────────────────────
        if action.action_type == "reset":
            result = self.reset(self.task.task_id)
            return StepResult(
                observation=result.observation,
                reward=0.01,
                done=False,
                info=result.info,
            )

        # ── EXECUTE SQL action ──────────────────────────────────────
        is_destructive = self._check_destructive(action.sql)
        sql_error, sql_output = self._execute_sql_safe(action.sql)

        self.last_sql_error = sql_error
        self.last_sql_output = sql_output

        query_results = self._evaluate_queries()
        queries_passing_now = sum(1 for r in query_results if r.passed)
        total_queries = len(self.task.target_queries)
        all_pass = queries_passing_now == total_queries
        at_max_steps = self.step_count >= self.task.max_steps

        if all_pass or at_max_steps:
            self.done = True

        # ── Progress-based step reward (NOT the grader) ─────────────
        # This produces natural partial values, never exactly 0.0 or 1.0
        reward = self._compute_step_reward(
            queries_passing_now=queries_passing_now,
            total_queries=total_queries,
            had_sql_error=sql_error is not None,
            is_destructive=is_destructive,
            all_pass=all_pass,
            at_max_steps=at_max_steps,
            query_results=query_results,
        )

        self._queries_passing_before = queries_passing_now
        self.rewards.append(reward)
        self.reward_so_far += reward

        obs = SchemaMigrationObservation(
            schema_dump=self._get_schema_dump(),
            query_results=query_results,
            last_sql_error=sql_error,
            last_sql_output=sql_output,
            step_count=self.step_count,
            done=self.done,
            task_id=self.task.task_id,
            task_description=self.task.description,
            hint=self.task.hint,
        )

        info = {"session_id": self.episode_id}
        if self.done:
            final_score = self._get_final_score(query_results)
            info["score"] = final_score

        return StepResult(
            observation=obs,
            reward=reward,
            done=self.done,
            info=info,
        )

    def _compute_step_reward(
        self,
        queries_passing_now: int,
        total_queries: int,
        had_sql_error: bool,
        is_destructive: bool,
        all_pass: bool,
        at_max_steps: bool,
        query_results: List[QueryResult],
    ) -> float:
        """
        Progress-based step reward — never calls the grader.
        Produces values naturally in (-0.5, 0.9), never exact 0.0 or 1.0.
        """
        reward = 0.0
        delta = queries_passing_now - self._queries_passing_before

        if total_queries > 0:
            if delta > 0:
                # Progress bonus: +0.4 per newly passing query (normalized)
                reward += 0.4 * (delta / total_queries)
            elif delta < 0:
                # Regression penalty
                reward += 0.2 * (delta / total_queries)  # negative
            else:
                # No change — small step cost to discourage loops
                reward -= 0.01

        # SQL error penalty
        if had_sql_error:
            reward -= 0.05

        # Destructive action penalty
        if is_destructive:
            reward -= 0.3

        # Completion bonus
        if all_pass:
            reward += 0.3
            # Efficiency bonus for finishing early
            if self.task:
                steps_remaining = self.task.max_steps - self.step_count
                efficiency = steps_remaining / self.task.max_steps
                reward += 0.2 * efficiency

        # Clamp step reward to (-0.5, 0.95) — never exact 0.0 or 1.0
        reward = max(-0.49, min(0.94, reward))

        # Nudge away from exact 0.0
        if reward == 0.0:
            reward = 0.01

        return round(reward, 4)

    def _get_final_score(self, query_results: List[QueryResult]) -> float:
        """
        Call the task grader ONCE to get the final episode score.
        Always returns a value strictly in (0.001, 0.999).
        """
        if self.task is None or self.conn is None:
            return 0.001
        try:
            score = float(self.task.grader(query_results, self.conn))
        except Exception:
            score = 0.001
        return _clamp(score)

    def state(self) -> SchemaMigrationState:
        if self.task is None:
            return SchemaMigrationState(
                task_id="", db_schema="", target_queries=[],
                queries_passed=0, queries_total=0, step_count=0,
                max_steps=0, done=True, reward_so_far=0.0, episode_id="",
            )
        query_results = self._evaluate_queries()
        queries_passed = sum(1 for r in query_results if r.passed)
        return SchemaMigrationState(
            task_id=self.task.task_id,
            db_schema=self._get_schema_dump(),
            target_queries=[q["query"] for q in self.task.target_queries],
            queries_passed=queries_passed,
            queries_total=len(self.task.target_queries),
            step_count=self.step_count,
            max_steps=self.task.max_steps,
            done=self.done,
            reward_so_far=self.reward_so_far,
            episode_id=self.episode_id,
        )

    def close(self) -> None:
        if self.conn:
            try:
                self.conn.close()
            except Exception:
                pass
            self.conn = None

    # ── Private helpers ─────────────────────────────────────────────

    def _execute_sql_safe(self, sql: str) -> Tuple[Optional[str], Optional[str]]:
        if self.conn is None:
            return ("No database connection.", None)
        sql_upper = sql.strip().upper()
        blocked = ["DROP DATABASE", "ATTACH DATABASE", "DETACH DATABASE",
                   "LOAD_EXTENSION", ".SHELL", ".SYSTEM"]
        for b in blocked:
            if b in sql_upper:
                return (f"Blocked operation: {b}", None)
        try:
            cursor = self.conn.execute(sql)
            self.conn.commit()
            if sql_upper.startswith("SELECT"):
                rows = cursor.fetchall()
                if rows:
                    col_names = [desc[0] for desc in cursor.description] if cursor.description else []
                    header = " | ".join(col_names)
                    row_strs = [" | ".join(str(v) for v in row) for row in rows[:20]]
                    output = f"{header}\n" + "\n".join(row_strs)
                    if len(rows) > 20:
                        output += f"\n... ({len(rows)} total rows)"
                    return (None, output)
                else:
                    return (None, "Query returned 0 rows.")
            else:
                return (None, f"OK. Rows affected: {cursor.rowcount}")
        except sqlite3.Error as e:
            return (str(e), None)
        except Exception as e:
            return (f"Unexpected error: {str(e)}", None)

    def _evaluate_queries(self) -> List[QueryResult]:
        if self.task is None or self.conn is None:
            return []
        results = []
        for qdef in self.task.target_queries:
            query_id = qdef["query_id"]
            query = qdef["query"]
            expected_row_count = qdef.get("expected_row_count")
            try:
                cursor = self.conn.execute(query)
                rows = cursor.fetchall()
                actual_count = len(rows)
                passed = True
                if expected_row_count is not None and actual_count < 1:
                    passed = False
                results.append(QueryResult(
                    query_id=query_id, query=query, passed=passed,
                    error=None, expected_row_count=expected_row_count,
                    actual_row_count=actual_count,
                ))
            except sqlite3.Error as e:
                results.append(QueryResult(
                    query_id=query_id, query=query, passed=False,
                    error=str(e), expected_row_count=expected_row_count,
                    actual_row_count=None,
                ))
        return results

    def _get_schema_dump(self) -> str:
        if self.conn is None:
            return ""
        try:
            cursor = self.conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL ORDER BY name"
            )
            return "\n\n".join(row[0] for row in cursor.fetchall())
        except Exception:
            return ""

    def _check_destructive(self, sql: str) -> bool:
        sql_upper = sql.strip().upper()
        drop_match = re.search(r"DROP\s+TABLE\s+(?:IF\s+EXISTS\s+)?(\w+)", sql_upper)
        if drop_match and self.conn:
            table_name = drop_match.group(1).lower()
            try:
                count = self.conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
                if count > 0:
                    return True
            except Exception:
                pass
        if "DELETE" in sql_upper and "WHERE" not in sql_upper:
            return True
        return False

    def _record_initial_row_counts(self) -> None:
        self._initial_table_row_counts = {}
        if self.conn is None:
            return
        try:
            cursor = self.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            for (name,) in cursor.fetchall():
                try:
                    c = self.conn.execute(f"SELECT COUNT(*) FROM {name}")
                    self._initial_table_row_counts[name] = c.fetchone()[0]
                except Exception:
                    pass
        except Exception:
            pass