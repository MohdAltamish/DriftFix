"""Async HTTP client for the Schema Migration Environment.

Provides SchemaMigrationEnv with:
  - from_docker_image(image_name) → starts container, returns client
  - from_url(base_url) → connects to running server
  - .sync() → returns synchronous wrapper
"""

from __future__ import annotations

import asyncio
import subprocess
import time
from typing import Optional

import httpx

from schema_migration_env.models import (
    ResetResult,
    SchemaMigrationAction,
    SchemaMigrationObservation,
    SchemaMigrationState,
    StepResult,
)


class SchemaMigrationEnv:
    """Async client for the Schema Migration Environment server."""

    def __init__(self, base_url: str, session_id: Optional[str] = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.session_id = session_id
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=30.0)
        self._container_id: Optional[str] = None

    @classmethod
    async def from_docker_image(
        cls, image_name: Optional[str], port: int = 8000
    ) -> "SchemaMigrationEnv":
        """Start a Docker container and return client connected to it."""
        if not image_name:
            raise ValueError("IMAGE_NAME environment variable is required")

        # Start container
        process = await asyncio.create_subprocess_exec(
            "docker",
            "run",
            "-d",
            "--rm",
            "-p",
            f"{port}:7860",
            image_name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            raise RuntimeError(f"Failed to start container: {stderr.decode().strip()}")

        container_id = stdout.decode().strip()
        base_url = f"http://localhost:{port}"

        env = cls(base_url=base_url)
        env._container_id = container_id

        # Wait for health endpoint
        start_time = time.time()
        while time.time() - start_time < 60:
            try:
                resp = await env._client.get("/health")
                if resp.status_code == 200:
                    return env
            except Exception:
                pass
            await asyncio.sleep(1)

        raise TimeoutError("Server did not become healthy within 60 seconds")

    @classmethod
    def from_url(cls, base_url: str) -> "SchemaMigrationEnv":
        """Connect to a running server."""
        return cls(base_url=base_url)

    async def reset(self, task_id: Optional[str] = None) -> ResetResult:
        """Reset the environment."""
        body = {}
        if task_id:
            body["task_id"] = task_id
        if self.session_id:
            body["session_id"] = self.session_id

        params = {}
        if self.session_id:
            params["session_id"] = self.session_id

        resp = await self._client.post("/reset", json=body, params=params)
        resp.raise_for_status()
        data = resp.json()

        # Extract session_id from response
        info = data.get("info", {})
        if "session_id" in info:
            self.session_id = info["session_id"]

        return ResetResult(**data)

    async def step(self, action: SchemaMigrationAction) -> StepResult:
        """Execute one step."""
        params = {}
        if self.session_id:
            params["session_id"] = self.session_id

        resp = await self._client.post(
            "/step",
            json=action.model_dump(),
            params=params,
        )
        resp.raise_for_status()
        data = resp.json()

        return StepResult(**data)

    async def state(self) -> SchemaMigrationState:
        """Get current state."""
        params = {}
        if self.session_id:
            params["session_id"] = self.session_id

        resp = await self._client.get("/state", params=params)
        resp.raise_for_status()
        data = resp.json()

        return SchemaMigrationState(**data)

    async def close(self) -> None:
        """Close the client and stop the container if we started one."""
        try:
            await self._client.aclose()
        except Exception:
            pass

        if self._container_id:
            try:
                subprocess.run(
                    ["docker", "stop", self._container_id],
                    capture_output=True,
                    timeout=15,
                )
            except Exception:
                pass
            self._container_id = None

    def sync(self) -> "SyncSchemaMigrationEnv":
        """Return a synchronous wrapper."""
        return SyncSchemaMigrationEnv(self)


class SyncSchemaMigrationEnv:
    """Synchronous wrapper around SchemaMigrationEnv."""

    def __init__(self, async_env: SchemaMigrationEnv) -> None:
        self._async_env = async_env
        try:
            self._loop = asyncio.get_event_loop()
            if self._loop.is_closed():
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
        except RuntimeError:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

    def reset(self, task_id: Optional[str] = None) -> ResetResult:
        return self._loop.run_until_complete(self._async_env.reset(task_id))

    def step(self, action: SchemaMigrationAction) -> StepResult:
        return self._loop.run_until_complete(self._async_env.step(action))

    def state(self) -> SchemaMigrationState:
        return self._loop.run_until_complete(self._async_env.state())

    def close(self) -> None:
        self._loop.run_until_complete(self._async_env.close())
