"""FastAPI server for the Schema Migration Environment.

Endpoints:
  POST /reset     → reset environment
  POST /step      → execute one step
  GET  /state     → get current state
  GET  /health    → health check
  GET  /web       → web UI
  WebSocket /ws   → persistent session
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response

from server.models import (
    ResetResult,
    SchemaMigrationAction,
    SchemaMigrationState,
    StepResult,
    TaskRequest,
)
from server.environment import SchemaMigrationEnvironment
from server.tasks import TASKS, list_task_ids

# ── Session management ──────────────────────────────────────────────

MAX_CONCURRENT_ENVS = 32
SESSION_IDLE_TIMEOUT = 300  # 5 minutes


class SessionEntry:
    """Wraps an environment instance with metadata."""

    def __init__(self, env: SchemaMigrationEnvironment):
        self.env = env
        self.last_accessed = time.time()

    def touch(self) -> None:
        self.last_accessed = time.time()


_sessions: Dict[str, SessionEntry] = {}
_cleanup_task: Optional[asyncio.Task] = None


async def _cleanup_idle_sessions() -> None:
    """Background task to remove idle sessions."""
    while True:
        await asyncio.sleep(30)
        now = time.time()
        expired = [
            sid
            for sid, entry in _sessions.items()
            if now - entry.last_accessed > SESSION_IDLE_TIMEOUT
        ]
        for sid in expired:
            entry = _sessions.pop(sid, None)
            if entry:
                entry.env.close()


def _get_session(session_id: Optional[str]) -> tuple[str, SessionEntry]:
    """Get or create a session."""
    if session_id and session_id in _sessions:
        entry = _sessions[session_id]
        entry.touch()
        return session_id, entry

    if len(_sessions) >= MAX_CONCURRENT_ENVS:
        # Remove oldest session
        oldest_id = min(_sessions, key=lambda k: _sessions[k].last_accessed)
        old_entry = _sessions.pop(oldest_id)
        old_entry.env.close()

    new_id = session_id or str(uuid.uuid4())
    env = SchemaMigrationEnvironment()
    entry = SessionEntry(env)
    _sessions[new_id] = entry
    return new_id, entry


# ── FastAPI app ──────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _cleanup_task
    _cleanup_task = asyncio.create_task(_cleanup_idle_sessions())
    yield
    if _cleanup_task:
        _cleanup_task.cancel()
    for entry in _sessions.values():
        entry.env.close()
    _sessions.clear()


app = FastAPI(
    title="Schema Migration Environment",
    description="RL environment for database schema migration",
    version="0.1.0",
    lifespan=lifespan,
)


# ── Endpoints ────────────────────────────────────────────────────────

@app.get("/")
async def root():
    """Redirect to web UI."""
    return RedirectResponse(url="/web")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Handle favicon request to avoid 404."""
    return Response(status_code=204)



@app.get("/health")
async def health():
    """Health check."""
    return {"status": "healthy"}


@app.get("/metadata")
async def metadata():
    """Return environment metadata for OpenEnv validator."""
    return {
        "name": "schema-migration-env",
        "description": "RL environment for training agents to perform database schema migrations. Agent executes SQL DDL/DML statements step-by-step to fix broken schemas until target queries pass.",
        "version": "0.1.0",
        "author": "mohdaltamish",
        "tags": ["openenv", "database", "schema-migration", "sql", "real-world"],
    }


@app.get("/schema")
async def schema():
    """Return action/observation/state JSON schemas for OpenEnv validator."""
    from server.models import SchemaMigrationAction, SchemaMigrationObservation, SchemaMigrationState
    return {
        "action": SchemaMigrationAction.model_json_schema(),
        "observation": SchemaMigrationObservation.model_json_schema(),
        "state": SchemaMigrationState.model_json_schema(),
    }


@app.post("/reset")
async def reset_env(
    request: Request,
    session_id: Optional[str] = Query(None),
):
    """Reset the environment. Accepts {} or TaskRequest."""
    try:
        body = await request.json()
    except Exception:
        body = {}

    task_id = body.get("task_id")
    req_session_id = body.get("session_id") or session_id

    sid, entry = _get_session(req_session_id)

    try:
        result = entry.env.reset(task_id=task_id)
        result.info["session_id"] = sid
        return result.model_dump()
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")


@app.post("/step")
async def step_env(
    action: SchemaMigrationAction,
    session_id: Optional[str] = Query(None),
):
    """Execute one step."""
    if not session_id or session_id not in _sessions:
        # Try to find any active session or error
        if not _sessions:
            raise HTTPException(
                status_code=400,
                detail="No active session. Call /reset first.",
            )
        if session_id and session_id not in _sessions:
            raise HTTPException(
                status_code=404,
                detail=f"Session '{session_id}' not found. Call /reset first.",
            )
        # Use most recent session
        session_id = max(_sessions, key=lambda k: _sessions[k].last_accessed)

    entry = _sessions[session_id]
    entry.touch()

    try:
        result = entry.env.step(action)
        result.info["session_id"] = session_id
        return result.model_dump()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Step failed: {str(e)}")


@app.get("/state")
async def get_state(session_id: Optional[str] = Query(None)):
    """Get current environment state."""
    if not session_id or session_id not in _sessions:
        if not _sessions:
            return SchemaMigrationState(
                task_id="",
                db_schema="",
                target_queries=[],
                queries_passed=0,
                queries_total=0,
                step_count=0,
                max_steps=0,
                done=True,
                reward_so_far=0.0,
                episode_id="",
            ).model_dump()
        session_id = max(_sessions, key=lambda k: _sessions[k].last_accessed)

    entry = _sessions[session_id]
    entry.touch()
    return entry.env.state().model_dump()


@app.get("/web", response_class=HTMLResponse)
async def web_ui(session_id: Optional[str] = Query(None)):
    """Simple HTML UI showing current schema + query status."""
    state_data = None
    if session_id and session_id in _sessions:
        entry = _sessions[session_id]
        state_data = entry.env.state()
    elif _sessions:
        sid = max(_sessions, key=lambda k: _sessions[k].last_accessed)
        state_data = _sessions[sid].env.state()

    tasks_html = ""
    for tid, tdef in TASKS.items():
        tasks_html += f'<li><strong>{tid}</strong> ({tdef.difficulty}) — {tdef.description}</li>'

    if state_data:
        schema_display = state_data.db_schema or "<em>No schema loaded</em>"
        queries_display = "<br>".join(state_data.target_queries) if state_data.target_queries else "<em>None</em>"
        status_display = f"""
        <div class="status-row"><span>Task:</span> <strong>{state_data.task_id or 'None'}</strong></div>
        <div class="status-row"><span>Episode:</span> <code>{state_data.episode_id or 'N/A'}</code></div>
        <div class="status-row"><span>Step:</span> {state_data.step_count} / {state_data.max_steps}</div>
        <div class="status-row"><span>Queries Passing:</span> {state_data.queries_passed} / {state_data.queries_total}</div>
        <div class="status-row"><span>Reward So Far:</span> {state_data.reward_so_far:.3f}</div>
        <div class="status-row"><span>Done:</span> {'✅ Yes' if state_data.done else '⏳ No'}</div>
        """
    else:
        schema_display = "<em>No session active. POST to /reset to start.</em>"
        queries_display = "<em>N/A</em>"
        status_display = "<p>No active session.</p>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Schema Migration Environment</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
            color: #e2e8f0;
            min-height: 100vh;
            padding: 2rem;
        }}
        .container {{ max-width: 960px; margin: 0 auto; }}
        h1 {{
            font-size: 2rem;
            background: linear-gradient(90deg, #38bdf8, #818cf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }}
        .subtitle {{ color: #94a3b8; margin-bottom: 2rem; }}
        .card {{
            background: rgba(30, 41, 59, 0.8);
            border: 1px solid rgba(148, 163, 184, 0.1);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            backdrop-filter: blur(10px);
        }}
        .card h2 {{
            color: #38bdf8;
            font-size: 1.1rem;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        pre {{
            background: #0f172a;
            padding: 1rem;
            border-radius: 8px;
            overflow-x: auto;
            font-size: 0.85rem;
            color: #a5f3fc;
            line-height: 1.5;
        }}
        .status-row {{
            display: flex;
            justify-content: space-between;
            padding: 0.5rem 0;
            border-bottom: 1px solid rgba(148, 163, 184, 0.1);
        }}
        .status-row:last-child {{ border-bottom: none; }}
        code {{
            background: rgba(56, 189, 248, 0.1);
            padding: 0.15em 0.4em;
            border-radius: 4px;
            font-size: 0.85em;
        }}
        ul {{ padding-left: 1.5rem; }}
        li {{ margin-bottom: 0.5rem; }}
        .badge {{
            display: inline-block;
            padding: 0.2em 0.6em;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 600;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🗄️ Schema Migration Environment</h1>
        <p class="subtitle">RL environment for training agents to perform database schema migrations</p>

        <div class="card">
            <h2>📊 Current Status</h2>
            {status_display}
        </div>

        <div class="card">
            <h2>🗃️ Database Schema</h2>
            <pre>{schema_display}</pre>
        </div>

        <div class="card">
            <h2>🎯 Target Queries</h2>
            <pre>{queries_display}</pre>
        </div>

        <div class="card">
            <h2>📋 Available Tasks</h2>
            <ul>{tasks_html}</ul>
        </div>

        <div class="card">
            <h2>🔗 API Endpoints</h2>
            <ul>
                <li><code>POST /reset</code> — Reset environment (body: {{"task_id": "add_missing_column"}})</li>
                <li><code>POST /step</code> — Execute SQL (body: {{"sql": "ALTER TABLE ...", "action_type": "execute"}})</li>
                <li><code>GET /state</code> — Get current state</li>
                <li><code>GET /health</code> — Health check</li>
                <li><code>WebSocket /ws</code> — Persistent session</li>
            </ul>
        </div>
    </div>
</body>
</html>"""
    return HTMLResponse(content=html)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for persistent sessions."""
    await websocket.accept()

    session_id = str(uuid.uuid4())
    env = SchemaMigrationEnvironment()
    entry = SessionEntry(env)
    _sessions[session_id] = entry

    try:
        await websocket.send_json({"type": "connected", "session_id": session_id})

        while True:
            data = await websocket.receive_text()
            entry.touch()

            try:
                msg = json.loads(data)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "detail": "Invalid JSON"})
                continue

            msg_type = msg.get("type", "")

            if msg_type == "reset":
                task_id = msg.get("task_id")
                try:
                    result = env.reset(task_id=task_id)
                    result.info["session_id"] = session_id
                    await websocket.send_json(
                        {"type": "reset_result", "data": result.model_dump()}
                    )
                except Exception as e:
                    await websocket.send_json(
                        {"type": "error", "detail": str(e)}
                    )

            elif msg_type == "step":
                sql = msg.get("sql", "")
                action_type = msg.get("action_type", "execute")
                action = SchemaMigrationAction(sql=sql, action_type=action_type)
                try:
                    result = env.step(action)
                    result.info["session_id"] = session_id
                    await websocket.send_json(
                        {"type": "step_result", "data": result.model_dump()}
                    )
                except Exception as e:
                    await websocket.send_json(
                        {"type": "error", "detail": str(e)}
                    )

            elif msg_type == "state":
                state = env.state()
                await websocket.send_json(
                    {"type": "state_result", "data": state.model_dump()}
                )

            else:
                await websocket.send_json(
                    {"type": "error", "detail": f"Unknown message type: {msg_type}"}
                )

    except WebSocketDisconnect:
        pass
    finally:
        _sessions.pop(session_id, None)
        env.close()


def main():
    """Main entry point for running the server."""
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
