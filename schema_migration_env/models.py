"""Pydantic models for the Schema Migration Environment."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class SchemaMigrationAction(BaseModel):
    """Action issued by the agent — a single SQL statement."""

    sql: str = Field(..., description="The SQL statement the agent wants to execute")
    action_type: str = Field(
        default="execute",
        description="Type of action: 'execute' | 'submit' | 'reset'",
    )


class QueryResult(BaseModel):
    """Result of evaluating a single target query."""

    query_id: str
    query: str
    passed: bool
    error: Optional[str] = None
    expected_row_count: Optional[int] = None
    actual_row_count: Optional[int] = None


class SchemaMigrationObservation(BaseModel):
    """Observation returned after each step."""

    schema_dump: str = Field(
        ..., description="Current CREATE TABLE statements of all tables"
    )
    query_results: List[QueryResult] = Field(
        ..., description="Each target query's pass/fail + error message"
    )
    last_sql_error: Optional[str] = Field(
        None, description="Error from last executed SQL if any"
    )
    last_sql_output: Optional[str] = Field(
        None, description="Output/result from last executed SQL"
    )
    step_count: int
    done: bool
    task_id: str
    task_description: str
    hint: Optional[str] = Field(None, description="Optional hint for easy task only")


class SchemaMigrationState(BaseModel):
    """Full environment state."""

    task_id: str
    db_schema: str
    target_queries: List[str]
    queries_passed: int
    queries_total: int
    step_count: int
    max_steps: int
    done: bool
    reward_so_far: float
    episode_id: str


class StepResult(BaseModel):
    """Result returned from env.step()."""

    observation: SchemaMigrationObservation
    reward: float
    done: bool
    info: dict = Field(default_factory=dict)


class ResetResult(BaseModel):
    """Result returned from env.reset()."""

    observation: SchemaMigrationObservation
    info: dict = Field(default_factory=dict)


class TaskRequest(BaseModel):
    """Optional request body for /reset."""

    task_id: Optional[str] = None
    session_id: Optional[str] = None
