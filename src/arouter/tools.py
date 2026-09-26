"""The agent's tools, over an in-memory service-config workspace that is exactly gradable."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

# Tools the prefetcher is allowed to speculate on. Anything that mutates state or has
# an external side effect is excluded: a discarded write is not free, it is a bug.
READ_ONLY_TOOLS = frozenset({"list_files", "read_file", "grep"})


class ToolError(Exception):
    """A tool was called in a way the workspace cannot honour."""


@dataclass
class Workspace:
    """A flat `key: value` config tree, cloned per task so runs cannot contaminate each other."""

    files: dict[str, str] = field(default_factory=dict)

    def clone(self) -> Workspace:
        return Workspace(files=dict(self.files))

    def read(self, path: str) -> str:
        if path not in self.files:
            raise ToolError(f"no such file: {path}")
        return self.files[path]

    def get_int(self, path: str, key: str) -> int | None:
        for line in self.read(path).splitlines():
            if line.strip().startswith(f"{key}:"):
                raw = line.split(":", 1)[1].strip()
                try:
                    return int(raw)
                except ValueError:
                    return None
        return None

    def get_raw(self, path: str, key: str) -> str | None:
        for line in self.read(path).splitlines():
            if line.strip().startswith(f"{key}:"):
                return line.split(":", 1)[1].strip()
        return None

    def set_value(self, path: str, key: str, value: str) -> str:
        text = self.read(path)
        lines = text.splitlines()
        for i, line in enumerate(lines):
            if line.strip().startswith(f"{key}:"):
                indent = line[: len(line) - len(line.lstrip())]
                lines[i] = f"{indent}{key}: {value}"
                self.files[path] = "\n".join(lines) + "\n"
                return self.files[path]
        raise ToolError(f"{path} has no key {key!r}")

    def digest(self) -> dict[str, str]:
        return dict(sorted(self.files.items()))


@dataclass
class ToolResult:
    ok: bool
    payload: dict[str, Any]
    latency_ms: float
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "payload": self.payload, "latency_ms": round(self.latency_ms, 2), "error": self.error}


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "list_files",
        "description": "List every file path in the workspace.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "read_file",
        "description": "Read one file's full text.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Workspace-relative path."}},
            "required": ["path"],
        },
    },
    {
        "name": "grep",
        "description": "Find every 'path:line' in the workspace matching a regular expression.",
        "parameters": {
            "type": "object",
            "properties": {"pattern": {"type": "string"}},
            "required": ["pattern"],
        },
    },
    {
        "name": "set_value",
        "description": "Set one top-level 'key: value' line in one file. The only way to change state.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "key": {"type": "string"},
                "value": {"type": "string", "description": "The new value, as a string."},
            },
            "required": ["path", "key", "value"],
        },
    },
    {
        "name": "finish",
        "description": "End the run. Provide the summary the task asked for, or '{}' if none was requested.",
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "description": "A JSON object as a string, or '{}'."},
                "confidence": {
                    "type": "number",
                    "description": "Your own probability, 0 to 1, that the task is now correctly done.",
                },
            },
            "required": ["summary"],
        },
    },
]

TOOL_NAMES = frozenset(schema["name"] for schema in TOOL_SCHEMAS)


def execute(ws: Workspace, tool: str, args: dict[str, Any], latency_ms: float = 0.0) -> ToolResult:
    """Run one tool against the workspace, charging the configured tool latency."""
    started = time.perf_counter()
    if latency_ms:
        time.sleep(latency_ms / 1000.0)
    try:
        payload = _dispatch(ws, tool, args)
        elapsed = (time.perf_counter() - started) * 1000.0
        return ToolResult(ok=True, payload=payload, latency_ms=elapsed)
    except ToolError as exc:
        elapsed = (time.perf_counter() - started) * 1000.0
        return ToolResult(ok=False, payload={}, latency_ms=elapsed, error=str(exc))


def _dispatch(ws: Workspace, tool: str, args: dict[str, Any]) -> dict[str, Any]:
    if tool == "list_files":
        return {"paths": sorted(ws.files)}
    if tool == "read_file":
        path = str(args.get("path", ""))
        return {"path": path, "content": ws.read(path)}
    if tool == "grep":
        pattern = str(args.get("pattern", ""))
        try:
            rx = re.compile(pattern)
        except re.error as exc:
            raise ToolError(f"bad pattern: {exc}") from exc
        matches = [
            f"{path}:{n}: {line}"
            for path, text in sorted(ws.files.items())
            for n, line in enumerate(text.splitlines(), 1)
            if rx.search(line)
        ]
        return {"matches": matches[:40]}
    if tool == "set_value":
        path, key = str(args.get("path", "")), str(args.get("key", ""))
        if "value" not in args:
            raise ToolError("set_value requires a value")
        content = ws.set_value(path, key, str(args["value"]))
        return {"path": path, "key": key, "value": str(args["value"]), "content": content}
    if tool == "finish":
        return {"summary": str(args.get("summary", "{}"))}
    raise ToolError(f"unknown tool: {tool}")


def render_workspace_listing(ws: Workspace) -> str:
    return "\n".join(f"- {path}" for path in sorted(ws.files))
