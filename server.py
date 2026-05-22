#!/usr/bin/env python3
"""Local web dashboard for active Claude Code sessions.

Reads ~/.claude/sessions/*.json + per-session JSONL transcripts, surfaces
who's busy, what they're working on, and recent tool activity. Stdlib only.

    python3 server.py [--port 8765] [--host 127.0.0.1]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

CLAUDE_HOME = Path(os.environ.get("CLAUDE_HOME", Path.home() / ".claude"))
SESSIONS_DIR = CLAUDE_HOME / "sessions"
PROJECTS_DIR = CLAUDE_HOME / "projects"
# Above this size, fall back to tailing the file to keep the dashboard responsive.
TRANSCRIPT_FULL_READ_MAX = 16 * 1024 * 1024
TRANSCRIPT_TAIL_BYTES = 2 * 1024 * 1024  # used only when files exceed the cap


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def encode_cwd(cwd: str) -> str:
    # CC encodes the project dir name by replacing '/' and '.' with '-'.
    return re.sub(r"[/.]", "-", cwd)


def read_session_files() -> list[dict[str, Any]]:
    out = []
    if not SESSIONS_DIR.exists():
        return out
    for f in SESSIONS_DIR.glob("*.json"):
        try:
            out.append(json.loads(f.read_text()))
        except (OSError, json.JSONDecodeError):
            continue
    return out


def read_transcript_lines(path: Path) -> list[str]:
    """Return all decoded lines. For very large files, fall back to tailing."""
    try:
        size = path.stat().st_size
    except OSError:
        return []
    with path.open("rb") as f:
        if size > TRANSCRIPT_FULL_READ_MAX:
            f.seek(size - TRANSCRIPT_TAIL_BYTES)
            f.readline()  # drop partial leading line
        data = f.read()
    try:
        text = data.decode("utf-8", errors="replace")
    except Exception:
        return []
    return [ln for ln in text.splitlines() if ln.strip()]


def summarize_tool_input(name: str, inp: dict[str, Any]) -> str:
    if not isinstance(inp, dict):
        return ""
    if name == "Bash":
        cmd = inp.get("command", "")
        return cmd if len(cmd) < 120 else cmd[:117] + "..."
    if name in ("Read", "Write", "Edit", "NotebookEdit"):
        return str(inp.get("file_path", ""))
    if name == "Agent":
        desc = inp.get("description") or ""
        st = inp.get("subagent_type") or ""
        return f"[{st}] {desc}" if st else desc
    if name in ("Grep", "Glob"):
        return str(inp.get("pattern") or inp.get("path") or "")
    if name == "WebFetch":
        return str(inp.get("url", ""))
    if name == "Skill":
        return str(inp.get("skill", ""))
    # Generic fallback: short JSON-ish preview
    s = json.dumps(inp, default=str)
    return s if len(s) < 120 else s[:117] + "..."


def parse_transcript(jsonl_path: Path) -> dict[str, Any]:
    """Extract title, last prompt, recent tool calls, sub-agent status, etc."""
    title: str | None = None
    last_prompt: str | None = None
    git_branch: str | None = None
    permission_mode: str | None = None
    last_user_ts: float | None = None
    last_assistant_ts: float | None = None
    recent_tool_calls: list[dict[str, Any]] = []
    # sub-agent (sidechain) tracking
    sidechain_spawns: dict[str, dict[str, Any]] = {}  # tool_use_id -> spawn info
    sidechain_completed: set[str] = set()
    last_sidechain_activity: float | None = None
    sidechain_active_now = False
    last_sidechain_text: str | None = None
    user_prompts: list[dict[str, Any]] = []  # real human prompts, chronological
    # awaiting-user detection: any unanswered AskUserQuestion or ExitPlanMode tool_use
    awaiting_tools: dict[str, dict[str, Any]] = {}  # tool_use_id -> {name, ts}
    completed_tool_ids: set[str] = set()

    lines = read_transcript_lines(jsonl_path)
    for line in lines:
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        et = ev.get("type")
        if et == "ai-title":
            title = ev.get("aiTitle") or title
        elif et == "last-prompt":
            last_prompt = ev.get("lastPrompt") or last_prompt
        elif et == "permission-mode":
            permission_mode = ev.get("permissionMode") or permission_mode
        elif et == "user":
            git_branch = ev.get("gitBranch") or git_branch
            ts = parse_ts(ev.get("timestamp"))
            if ts:
                last_user_ts = ts if last_user_ts is None else max(last_user_ts, ts)
            msg = ev.get("message") or {}
            content = msg.get("content")
            # tool_result content arrives as user messages — handle separately from real prompts
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "tool_result":
                        tid = item.get("tool_use_id")
                        if tid:
                            completed_tool_ids.add(tid)
                            if tid in sidechain_spawns:
                                sidechain_completed.add(tid)
            # Real human prompts have promptId and are not sidechain
            if ev.get("promptId") and not ev.get("isSidechain"):
                text_parts: list[str] = []
                if isinstance(content, str):
                    text_parts.append(content)
                elif isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            text_parts.append(item.get("text") or "")
                text = "\n".join(p for p in text_parts if p and p.strip()).strip()
                if text:
                    user_prompts.append({"text": text, "ts": ev.get("timestamp")})
        elif et == "assistant":
            git_branch = ev.get("gitBranch") or git_branch
            ts = parse_ts(ev.get("timestamp"))
            if ts:
                last_assistant_ts = ts if last_assistant_ts is None else max(last_assistant_ts, ts)
            msg = ev.get("message") or {}
            content = msg.get("content")
            is_sidechain = bool(ev.get("isSidechain"))
            if is_sidechain and ts:
                last_sidechain_activity = ts if last_sidechain_activity is None else max(last_sidechain_activity, ts)
            if isinstance(content, list):
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    if item.get("type") == "tool_use":
                        name = item.get("name", "")
                        inp = item.get("input") or {}
                        recent_tool_calls.append({
                            "name": name,
                            "summary": summarize_tool_input(name, inp),
                            "ts": ev.get("timestamp"),
                            "sidechain": is_sidechain,
                        })
                        tid = item.get("id")
                        if name == "Agent" and not is_sidechain and tid:
                            sidechain_spawns[tid] = {
                                "description": (inp.get("description") if isinstance(inp, dict) else None) or "",
                                "subagent_type": (inp.get("subagent_type") if isinstance(inp, dict) else None) or "",
                                "ts": ev.get("timestamp"),
                            }
                        if name in ("AskUserQuestion", "ExitPlanMode") and not is_sidechain and tid:
                            awaiting_tools[tid] = {"name": name, "ts": ev.get("timestamp")}
                    elif item.get("type") == "text" and is_sidechain:
                        txt = (item.get("text") or "").strip()
                        if txt:
                            last_sidechain_text = txt[:200]

    # active subagents = spawned but not yet completed
    active_subagents = [
        {"tool_use_id": tid, **info}
        for tid, info in sidechain_spawns.items()
        if tid not in sidechain_completed
    ]
    # awaiting user = an AskUserQuestion / ExitPlanMode is pending without tool_result
    pending_awaiting = [info for tid, info in awaiting_tools.items() if tid not in completed_tool_ids]
    awaiting_user = len(pending_awaiting) > 0
    awaiting_tool_name = pending_awaiting[-1]["name"] if pending_awaiting else None

    # consider a sidechain "active now" if the latest assistant activity is sidechain
    if last_sidechain_activity and last_assistant_ts and last_sidechain_activity >= last_assistant_ts - 0.001:
        sidechain_active_now = True

    return {
        "title": title,
        "last_prompt": last_prompt,
        "permission_mode": permission_mode,
        "git_branch": git_branch,
        "last_user_ts": last_user_ts,
        "last_assistant_ts": last_assistant_ts,
        "recent_tool_calls": recent_tool_calls[-12:],
        "active_subagents": active_subagents,
        "sidechain_active_now": sidechain_active_now,
        "last_sidechain_text": last_sidechain_text,
        "recent_user_prompts": user_prompts[-4:],  # chronological, oldest first
        "last_prompt": (user_prompts[-1]["text"] if user_prompts else last_prompt),
        "awaiting_user": awaiting_user,
        "awaiting_tool_name": awaiting_tool_name,
    }


# --------------------------------------------------------------------------- #
# Summary generation via `claude -p` (background, hash-cached per session).
# --------------------------------------------------------------------------- #

_summary_cache: dict[str, dict[str, Any]] = {}
_summary_lock = threading.Lock()
_summary_threads: dict[str, threading.Thread] = {}
_summary_subprocess_pids: set[int] = set()
_summary_subprocess_pids_lock = threading.Lock()
SUMMARY_MAX_PROMPT_CHARS = 1200
SUMMARY_TIMEOUT_SEC = 90


def _prompts_hash(prompts: list[dict[str, Any]], title: str | None) -> str:
    h = hashlib.sha256()
    h.update((title or "").encode("utf-8"))
    h.update(b"\x00")
    for p in prompts:
        h.update((p.get("text") or "").encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def _build_summary_prompt(prompts: list[dict[str, Any]], title: str | None) -> str:
    lines = [
        "You will see a sequence of user prompts from a Claude Code session.",
        "Respond with ONE short sentence (max ~12 words) describing the concrete task the user is having Claude work on.",
        "Focus on what is being built or fixed, not the meta of how the session is going.",
        "Do not start with 'The user' or 'This session'. Just state the task as a noun phrase or short imperative.",
        "Output ONLY the summary, no preamble.",
        "",
    ]
    if title:
        lines.append(f"Session auto-title: {title}")
        lines.append("")
    lines.append("Recent prompts (oldest first):")
    for i, p in enumerate(prompts, 1):
        text = (p.get("text") or "").strip()
        if len(text) > SUMMARY_MAX_PROMPT_CHARS:
            text = text[:SUMMARY_MAX_PROMPT_CHARS] + "…"
        lines.append(f"[{i}] {text}")
    lines.append("")
    lines.append("Summary:")
    return "\n".join(lines)


def _run_claude_summary(prompt: str) -> tuple[str | None, str | None]:
    """Returns (summary, error_message). Either is None on success/failure."""
    if not shutil.which("claude"):
        return None, "claude CLI not found in PATH"
    try:
        proc = subprocess.Popen(
            [
                "claude",
                "--no-session-persistence",
                "--tools", "",
                "--model", "haiku",
                "-p", prompt,
            ],
            cwd="/tmp",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except Exception as e:
        return None, f"subprocess error: {e}"
    with _summary_subprocess_pids_lock:
        _summary_subprocess_pids.add(proc.pid)
    try:
        try:
            stdout, stderr = proc.communicate(timeout=SUMMARY_TIMEOUT_SEC)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            return None, f"timeout after {SUMMARY_TIMEOUT_SEC}s"
    finally:
        with _summary_subprocess_pids_lock:
            _summary_subprocess_pids.discard(proc.pid)
    result = subprocess.CompletedProcess(args=proc.args, returncode=proc.returncode, stdout=stdout, stderr=stderr)
    if result.returncode != 0:
        err = (result.stderr or "").strip().splitlines()
        msg = err[-1] if err else f"exit {result.returncode}"
        return None, msg[:200]
    out = (result.stdout or "").strip()
    if not out:
        return None, "empty output"
    # take first non-empty line
    first = next((ln.strip() for ln in out.splitlines() if ln.strip()), "")
    # strip trailing period, quotes
    first = first.strip('"\'').rstrip(".")
    if len(first) > 240:
        first = first[:240] + "…"
    return first, None


def _generate_summary_worker(session_id: str, prompts: list[dict[str, Any]], title: str | None, content_hash: str) -> None:
    prompt = _build_summary_prompt(prompts, title)
    started = time.time()
    summary, err = _run_claude_summary(prompt)
    with _summary_lock:
        # only write if our hash is still current (avoid overwriting a newer attempt)
        cur = _summary_cache.get(session_id)
        if cur and cur.get("hash") != content_hash:
            return
        _summary_cache[session_id] = {
            "hash": content_hash,
            "summary": summary,
            "state": "ready" if summary else "error",
            "error": err,
            "ts": time.time(),
            "elapsed": time.time() - started,
        }


def get_or_kick_summary(session_id: str, prompts: list[dict[str, Any]], title: str | None) -> dict[str, Any]:
    if not prompts:
        return {"summary": None, "state": "empty"}
    content_hash = _prompts_hash(prompts, title)
    with _summary_lock:
        cached = _summary_cache.get(session_id)
        if cached and cached.get("hash") == content_hash:
            # ready / generating / error — return as-is until prompts change
            return dict(cached)
        # mark generating; another thread may be running for an older hash — that's fine, it will be discarded on write
        _summary_cache[session_id] = {"hash": content_hash, "summary": None, "state": "generating", "error": None, "ts": time.time()}
        prior = _summary_threads.get(session_id)
        if prior and prior.is_alive():
            # an older thread is still running for a different hash; let the new one race it
            pass
        t = threading.Thread(
            target=_generate_summary_worker,
            args=(session_id, list(prompts), title, content_hash),
            daemon=True,
        )
        _summary_threads[session_id] = t
        t.start()
        return dict(_summary_cache[session_id])


# --------------------------------------------------------------------------- #
# Terminal automation: focus a session's tab, open a new session, kill a PID.
# macOS-specific (Terminal.app / iTerm2 via osascript).
# --------------------------------------------------------------------------- #


def _ps_field(pid: int, fmt: str) -> str | None:
    try:
        r = subprocess.run(
            ["ps", "-p", str(pid), "-o", fmt + "="],
            capture_output=True, text=True, timeout=2,
        )
        if r.returncode != 0:
            return None
        out = r.stdout.strip()
        return out or None
    except Exception:
        return None


def get_tty_for_pid(pid: int) -> str | None:
    tty = _ps_field(pid, "tty")
    if not tty or tty in ("??", "?"):
        return None
    return tty if tty.startswith("/") else f"/dev/{tty}"


def _pid_parent_chain(pid: int, max_depth: int = 25) -> list[tuple[int, str]]:
    """Walk parent process tree returning [(pid, command), ...] up to launchd."""
    chain: list[tuple[int, str]] = []
    current = pid
    while current and current > 1 and len(chain) < max_depth:
        try:
            r = subprocess.run(
                ["ps", "-p", str(current), "-o", "ppid=,command="],
                capture_output=True, text=True, timeout=2,
            )
            if r.returncode != 0:
                break
            line = r.stdout.strip()
            if not line:
                break
            parts = line.split(None, 1)
            if len(parts) < 2:
                break
            ppid_str, command = parts
            chain.append((current, command))
            current = int(ppid_str)
        except Exception:
            break
    return chain


def detect_terminal_app(pid: int) -> str:
    """Return 'iTerm', 'Terminal', or 'Terminal' as fallback."""
    for _, cmd in _pid_parent_chain(pid):
        cl = cmd.lower()
        if "iterm" in cl:
            return "iTerm"
        if "terminal.app" in cl or cl.endswith("/terminal"):
            return "Terminal"
    return "Terminal"


def _osascript(script: str) -> tuple[str, str, int]:
    try:
        r = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True, text=True, timeout=15,
        )
        return r.stdout.strip(), r.stderr.strip(), r.returncode
    except Exception as e:
        return "", str(e), -1


def _app_running(name: str) -> bool:
    out, _, _ = _osascript(f'application "{name}" is running')
    return out == "true"


_FOCUS_TERMINAL = '''
tell application "Terminal"
    activate
    repeat with w in windows
        repeat with t in tabs of w
            try
                if (tty of t) is "TTY" then
                    set selected of t to true
                    set index of w to 1
                    return "ok"
                end if
            end try
        end repeat
    end repeat
    return "not_found"
end tell
'''

_FOCUS_ITERM = '''
tell application "iTerm"
    activate
    repeat with w in windows
        repeat with t in tabs of w
            repeat with s in sessions of t
                try
                    if (tty of s) is "TTY" then
                        tell t to select
                        return "ok"
                    end if
                end try
            end repeat
        end repeat
    end repeat
    return "not_found"
end tell
'''


def focus_terminal_for_pid(pid: int) -> tuple[bool, str]:
    tty = get_tty_for_pid(pid)
    if not tty:
        return False, "could not determine tty for that PID"
    preferred = detect_terminal_app(pid)
    order = (
        [("iTerm", _FOCUS_ITERM), ("Terminal", _FOCUS_TERMINAL)]
        if preferred == "iTerm"
        else [("Terminal", _FOCUS_TERMINAL), ("iTerm", _FOCUS_ITERM)]
    )
    for app_name, tmpl in order:
        if app_name == "iTerm" and not _app_running("iTerm"):
            continue
        if app_name == "Terminal" and not _app_running("Terminal"):
            continue
        out, err, _ = _osascript(tmpl.replace("TTY", tty))
        if out == "ok":
            return True, f"focused {app_name}"
        if err:
            # try the next one
            continue
    return False, f"tab with tty {tty} not found in any running terminal app"


_NEW_TERMINAL = '''
tell application "Terminal"
    activate
    do script "CMD"
end tell
'''

_NEW_ITERM = '''
tell application "iTerm"
    activate
    if (count of windows) = 0 then
        create window with default profile
    else
        tell current window to create tab with default profile
    end if
    tell current session of current window
        write text "CMD"
    end tell
end tell
'''


def open_new_session(cwd: str) -> tuple[bool, str]:
    cwd_expanded = os.path.expanduser(cwd or "~")
    if not os.path.isdir(cwd_expanded):
        return False, f"not a directory: {cwd_expanded}"
    cmd = f"cd {shlex.quote(cwd_expanded)} && claude"
    cmd_for_applescript = cmd.replace("\\", "\\\\").replace('"', '\\"')
    if _app_running("iTerm"):
        out, err, rc = _osascript(_NEW_ITERM.replace("CMD", cmd_for_applescript))
        if rc == 0:
            return True, f"opened iTerm tab in {cwd_expanded}"
    out, err, rc = _osascript(_NEW_TERMINAL.replace("CMD", cmd_for_applescript))
    if rc == 0:
        return True, f"opened Terminal window in {cwd_expanded}"
    return False, err or "AppleScript failed"


def stop_session(pid: int, tracked_pids: set[int]) -> tuple[bool, str]:
    if pid not in tracked_pids:
        return False, "PID is not a tracked Claude Code session"
    try:
        os.kill(pid, signal.SIGTERM)
        return True, f"sent SIGTERM to {pid}"
    except ProcessLookupError:
        return False, "process not found"
    except PermissionError:
        return False, "permission denied"
    except Exception as e:
        return False, str(e)


def parse_ts(ts: Any) -> float | None:
    if not ts:
        return None
    if isinstance(ts, (int, float)):
        return float(ts) / 1000.0 if ts > 1e12 else float(ts)
    if isinstance(ts, str):
        try:
            # ISO 8601 like "2026-05-22T09:56:17.902Z"
            from datetime import datetime
            t = ts.replace("Z", "+00:00")
            return datetime.fromisoformat(t).timestamp()
        except Exception:
            return None
    return None


def collect_state() -> dict[str, Any]:
    sessions_meta = read_session_files()
    sessions = []
    with _summary_subprocess_pids_lock:
        worker_pids = set(_summary_subprocess_pids)
    for meta in sessions_meta:
        pid = meta.get("pid")
        if not pid or not pid_alive(int(pid)):
            continue
        if int(pid) in worker_pids:
            continue
        sid = meta.get("sessionId")
        cwd = meta.get("cwd") or ""
        project_dir = PROJECTS_DIR / encode_cwd(cwd)
        jsonl = project_dir / f"{sid}.jsonl" if sid else None
        details: dict[str, Any] = {}
        if jsonl and jsonl.exists():
            try:
                details = parse_transcript(jsonl)
            except Exception as e:
                details = {"_parse_error": str(e)}
        # convert started/updated to seconds
        started_at = meta.get("startedAt")
        if isinstance(started_at, (int, float)) and started_at > 1e12:
            started_at = started_at / 1000.0
        updated_at = meta.get("updatedAt")
        if isinstance(updated_at, (int, float)) and updated_at > 1e12:
            updated_at = updated_at / 1000.0
        summary_info = get_or_kick_summary(
            sid or str(pid),
            details.get("recent_user_prompts") or [],
            details.get("title"),
        )
        status_raw = meta.get("status", "?")
        # CC writes status="waiting" while paused for user input (AskUserQuestion / permission prompts).
        # Trust it as a primary signal; OR it with the JSONL-based detection.
        awaiting_user = bool(details.get("awaiting_user")) or status_raw == "waiting"
        sessions.append({
            "pid": pid,
            "sessionId": sid,
            "cwd": cwd,
            "project": Path(cwd).name if cwd else "",
            "status": status_raw,
            "kind": meta.get("kind", ""),
            "entrypoint": meta.get("entrypoint", ""),
            "version": meta.get("version", ""),
            "startedAt": started_at,
            "updatedAt": updated_at,
            **details,
            "awaiting_user": awaiting_user,
            "summary": summary_info.get("summary"),
            "summary_state": summary_info.get("state"),
            "summary_error": summary_info.get("error"),
        })
    # sort: awaiting-you first, then busy, then most recently updated
    sessions.sort(key=lambda s: (
        not s.get("awaiting_user"),
        s.get("status") != "busy",
        -(s.get("updatedAt") or 0),
    ))
    return {
        "now": time.time(),
        "sessions": sessions,
        "claude_home": str(CLAUDE_HOME),
    }


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Claude Agent Watch</title>
<meta name="viewport" content="width=device-width,initial-scale=1" />
<style>
  :root {
    --bg: #0e1116;
    --panel: #161b22;
    --panel-2: #1f2630;
    --border: #2a313c;
    --text: #e6edf3;
    --muted: #8b949e;
    --busy: #d29922;
    --idle: #3fb950;
    --awaiting: #4d8eff;
    --accent: #58a6ff;
    --danger: #f85149;
    --mono: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; background: var(--bg); color: var(--text); font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }
  header { padding: 14px 22px; border-bottom: 1px solid var(--border); display: flex; align-items: center; gap: 16px; flex-wrap: wrap; }
  header h1 { font-size: 16px; margin: 0; font-weight: 600; }
  header .meta { color: var(--muted); font-size: 12px; font-family: var(--mono); }
  header .spacer { flex: 1; }
  /* buttons */
  .btn { background: var(--panel-2); color: var(--text); border: 1px solid var(--border); border-radius: 6px; padding: 5px 11px; font-size: 12px; cursor: pointer; font-family: inherit; }
  .btn:hover { background: #2a313c; border-color: #3a414c; }
  .btn:disabled { opacity: 0.5; cursor: not-allowed; }
  .btn.primary { background: var(--accent); color: #0e1116; border-color: var(--accent); font-weight: 600; }
  .btn.primary:hover { background: #79b8ff; border-color: #79b8ff; }
  .btn.danger { color: var(--danger); }
  .btn.danger:hover { background: rgba(248, 81, 73, 0.12); border-color: var(--danger); }
  /* new-session form */
  .new-form { display: none; align-items: center; gap: 8px; }
  .new-form.open { display: flex; }
  .new-form input { background: var(--panel-2); color: var(--text); border: 1px solid var(--border); border-radius: 6px; padding: 5px 9px; font-size: 12px; font-family: var(--mono); width: 320px; }
  .new-form input:focus { outline: none; border-color: var(--accent); }
  /* per-card actions */
  .actions { display: flex; gap: 8px; margin-top: 12px; padding-top: 10px; border-top: 1px solid var(--border); }
  .toast { position: fixed; bottom: 20px; right: 20px; background: var(--panel); border: 1px solid var(--border); border-radius: 6px; padding: 10px 14px; font-size: 12px; color: var(--text); box-shadow: 0 4px 16px rgba(0,0,0,0.4); max-width: 360px; }
  .toast.error { border-color: var(--danger); color: var(--danger); }
  .toast.success { border-color: var(--idle); }
  main { padding: 18px 22px; }
  .empty { color: var(--muted); padding: 40px; text-align: center; border: 1px dashed var(--border); border-radius: 8px; }
  .grid { display: flex; flex-direction: column; gap: 14px; }
  .card { background: var(--panel); border: 2px solid var(--border); border-radius: 8px; padding: 13px 15px; width: 100%; }
  .card.busy { border-color: var(--busy); }
  .card.awaiting { border-color: var(--awaiting); box-shadow: 0 0 0 2px rgba(77, 142, 255, 0.35); }
  .card-head { display: flex; align-items: center; gap: 8px; margin-bottom: 6px; }
  .pill { font-size: 11px; font-family: var(--mono); padding: 2px 8px; border-radius: 999px; background: var(--panel-2); color: var(--muted); }
  .pill.busy { background: rgba(210, 153, 34, 0.15); color: var(--busy); }
  .pill.idle { background: rgba(63, 185, 80, 0.12); color: var(--idle); }
  .pill.awaiting { background: rgba(77, 142, 255, 0.18); color: var(--awaiting); }
  .pill.sidechain { background: rgba(88, 166, 255, 0.12); color: var(--accent); }
  .title { font-weight: 600; font-size: 14px; flex: 1; }
  .row { display: flex; flex-wrap: wrap; gap: 6px 14px; color: var(--muted); font-size: 12px; font-family: var(--mono); margin: 4px 0 10px; }
  .row span { white-space: nowrap; }
  .prompt { background: var(--panel-2); border-radius: 6px; padding: 12px 14px 14px; font-size: 13px; color: var(--text); margin-bottom: 10px; line-height: 1.5; }
  .summary { background: rgba(88, 166, 255, 0.08); border-left: 3px solid var(--accent); border-radius: 4px; padding: 10px 14px 12px; font-size: 13.5px; color: var(--text); margin-bottom: 10px; line-height: 1.5; }
  .summary.generating { color: var(--muted); font-style: italic; border-left-color: var(--border); background: var(--panel-2); }
  .summary.error { color: var(--danger); font-style: italic; border-left-color: var(--danger); background: rgba(248, 81, 73, 0.05); }
  .summary .err-detail { color: var(--muted); font-style: normal; font-size: 11px; font-family: var(--mono); margin-top: 4px; }
  /* markdown content inside prompt blocks */
  .md { word-wrap: break-word; }
  .md > *:first-child { margin-top: 0; }
  .md > *:last-child { margin-bottom: 0; }
  .md p { margin: 0 0 0.5em; }
  .md ul, .md ol { margin: 0.3em 0 0.5em; padding-left: 1.4em; }
  .md li { margin: 0.15em 0; }
  .md code { background: rgba(255, 255, 255, 0.08); padding: 1px 5px; border-radius: 3px; font-family: var(--mono); font-size: 12px; }
  .md pre { background: rgba(0, 0, 0, 0.35); padding: 8px 10px; border-radius: 4px; overflow-x: auto; margin: 0.4em 0; }
  .md pre code { background: transparent; padding: 0; font-size: 12px; }
  .md a { color: var(--accent); text-decoration: none; }
  .md a:hover { text-decoration: underline; }
  .md strong { color: #fff; font-weight: 600; }
  .md blockquote { border-left: 3px solid var(--border); margin: 0.4em 0; padding: 0 0 0 10px; color: var(--muted); }
  .md h1, .md h2, .md h3, .md h4 { font-size: 14px; margin: 0.6em 0 0.3em; font-weight: 600; }
  .label { font-size: 10px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); margin-bottom: 4px; }
  details.activity { margin-top: 4px; border: 1px solid var(--border); border-radius: 6px; background: var(--panel-2); }
  details.activity > summary { list-style: none; cursor: pointer; padding: 8px 10px; font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); display: flex; align-items: center; gap: 8px; user-select: none; }
  details.activity > summary::-webkit-details-marker { display: none; }
  details.activity > summary::before { content: "▸"; display: inline-block; transition: transform 0.15s ease; color: var(--muted); font-size: 10px; }
  details.activity[open] > summary::before { transform: rotate(90deg); }
  details.activity > summary .count { margin-left: auto; color: var(--muted); text-transform: none; letter-spacing: 0; font-family: var(--mono); font-size: 11px; }
  details.activity .tools { padding: 0 10px 8px; }
  .tools { list-style: none; padding: 0; margin: 0 0 6px; }
  .tools li { font-family: var(--mono); font-size: 12px; padding: 3px 0; border-top: 1px solid var(--border); display: flex; gap: 8px; }
  .tools li:first-child { border-top: 0; }
  .tools .name { color: var(--accent); flex-shrink: 0; min-width: 78px; }
  .tools .name.sc { color: #d2a8ff; }
  .tools .summary { color: var(--text); opacity: 0.85; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .agents { margin: 6px 0 0; }
  .agents .ag { font-size: 12px; padding: 4px 8px; background: rgba(88, 166, 255, 0.08); border: 1px solid rgba(88, 166, 255, 0.3); border-radius: 6px; margin-top: 4px; }
  .agents .ag .at { color: var(--accent); font-family: var(--mono); }
  .err { color: var(--danger); font-family: var(--mono); font-size: 12px; }
  a { color: var(--accent); }
  .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: var(--idle); margin-right: 6px; }
  .dot.busy { background: var(--busy); animation: pulse 1.4s ease-in-out infinite; }
  .dot.awaiting { background: var(--awaiting); animation: pulse 1.0s ease-in-out infinite; box-shadow: 0 0 0 2px rgba(77, 142, 255, 0.30); }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.4; } }
</style>
</head>
<body>
<script src="https://cdn.jsdelivr.net/npm/marked@12.0.2/marked.min.js" defer></script>
<header>
  <h1>Claude Agent Watch</h1>
  <span class="meta" id="meta">loading…</span>
  <span class="spacer"></span>
  <button class="btn primary" id="new-toggle">+ New session</button>
  <form class="new-form" id="new-form" autocomplete="off">
    <input type="text" id="new-cwd" placeholder="~" spellcheck="false" />
    <button type="submit" class="btn primary" id="new-submit">Open</button>
    <button type="button" class="btn" id="new-cancel">Cancel</button>
  </form>
</header>
<main>
  <div id="root" class="grid"></div>
</main>
<script>
function fmtAgo(tsSec, now) {
  if (!tsSec) return "—";
  const s = Math.max(0, Math.round(now - tsSec));
  if (s < 60) return s + "s ago";
  if (s < 3600) return Math.round(s/60) + "m ago";
  if (s < 86400) return Math.round(s/3600) + "h ago";
  return Math.round(s/86400) + "d ago";
}
function el(tag, attrs, children) {
  const e = document.createElement(tag);
  if (attrs) for (const [k,v] of Object.entries(attrs)) {
    if (k === "class") e.className = v;
    else if (k === "text") e.textContent = v;
    else e.setAttribute(k, v);
  }
  if (children) for (const c of children) { if (c) e.appendChild(c); }
  return e;
}
const openActivity = new Set(); // sessionIds whose activity block is expanded
async function postJSON(path, body) {
  const r = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  let data = null;
  try { data = await r.json(); } catch (_) {}
  return { ok: r.ok && data && data.ok, message: (data && data.message) || (r.ok ? "ok" : "request failed"), status: r.status };
}
function toast(message, kind) {
  const t = document.createElement("div");
  t.className = "toast " + (kind || "");
  t.textContent = message;
  document.body.appendChild(t);
  setTimeout(() => t.remove(), 4500);
}
function escapeHtml(s) {
  return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
function renderMarkdown(text) {
  if (window.marked && window.marked.parse) {
    try {
      return window.marked.parse(text, { breaks: true, gfm: true });
    } catch (e) { /* fall through */ }
  }
  return "<p>" + escapeHtml(text).replace(/\n/g, "<br>") + "</p>";
}
function mdEl(text, extraClass) {
  const d = document.createElement("div");
  d.className = "md" + (extraClass ? " " + extraClass : "");
  d.innerHTML = renderMarkdown(text);
  return d;
}
function render(state) {
  const root = document.getElementById("root");
  root.innerHTML = "";
  document.getElementById("meta").textContent =
    state.sessions.length + " active session(s) · refreshed " + new Date(state.now*1000).toLocaleTimeString();
  if (state.sessions.length === 0) {
    root.appendChild(el("div", { class: "empty", text: "No active Claude Code sessions detected." }));
    return;
  }
  for (const s of state.sessions) {
    const isBusy = s.status === "busy";
    const isAwaiting = !!s.awaiting_user;
    const card = el("div", { class: "card" + (isAwaiting ? " awaiting" : isBusy ? " busy" : "") });
    // head
    const head = el("div", { class: "card-head" });
    const dotCls = isAwaiting ? " awaiting" : (isBusy ? " busy" : "");
    const dot = el("span", { class: "dot" + dotCls });
    head.appendChild(dot);
    head.appendChild(el("div", { class: "title", text: s.title || s.project || "(untitled)" }));
    if (isAwaiting) {
      const lbl = s.awaiting_tool_name === "ExitPlanMode" ? "awaiting plan approval" : "awaiting your input";
      head.appendChild(el("span", { class: "pill awaiting", text: lbl }));
    } else {
      head.appendChild(el("span", { class: "pill " + (isBusy ? "busy" : "idle"), text: s.status }));
    }
    if (s.sidechain_active_now) head.appendChild(el("span", { class: "pill sidechain", text: "sub-agent" }));
    card.appendChild(head);
    // meta row
    const row = el("div", { class: "row" }, [
      el("span", { text: "pid " + s.pid }),
      el("span", { text: s.project ? "📁 " + s.project : "" }),
      el("span", { text: s.git_branch ? "⎇ " + s.git_branch : "" }),
      el("span", { text: "updated " + fmtAgo(s.updatedAt, state.now) }),
      el("span", { text: "v" + s.version }),
    ]);
    card.appendChild(row);
    // working on — synthesized one-line summary from `claude -p`
    const promptsAll = s.recent_user_prompts || [];
    const lastPromptText = promptsAll.length > 0
      ? promptsAll[promptsAll.length - 1].text
      : s.last_prompt;
    if (s.summary_state && s.summary_state !== "empty") {
      card.appendChild(el("div", { class: "label", text: "Working on" }));
      let cls = "summary";
      let text = "";
      if (s.summary_state === "ready" && s.summary) {
        text = s.summary;
      } else if (s.summary_state === "generating") {
        cls += " generating";
        text = "Summarizing recent work…";
      } else if (s.summary_state === "error") {
        cls += " error";
        text = "Could not generate summary.";
      }
      const sumEl = el("div", { class: cls });
      sumEl.appendChild(mdEl(text));
      if (s.summary_state === "error" && s.summary_error) {
        sumEl.appendChild(el("div", { class: "err-detail", text: s.summary_error }));
      }
      card.appendChild(sumEl);
    }
    // last prompt (most recent, full)
    if (lastPromptText) {
      card.appendChild(el("div", { class: "label", text: "Last prompt" }));
      const promptEl = el("div", { class: "prompt" });
      promptEl.appendChild(mdEl(lastPromptText));
      card.appendChild(promptEl);
    }
    // active sub-agents
    if (s.active_subagents && s.active_subagents.length > 0) {
      card.appendChild(el("div", { class: "label", text: "Active sub-agent(s)" }));
      const wrap = el("div", { class: "agents" });
      for (const a of s.active_subagents) {
        const line = el("div", { class: "ag" });
        line.appendChild(el("span", { class: "at", text: (a.subagent_type || "agent") + ": " }));
        line.appendChild(document.createTextNode(a.description || "(no description)"));
        wrap.appendChild(line);
      }
      card.appendChild(wrap);
    }
    // recent tool calls (expandable)
    if (s.recent_tool_calls && s.recent_tool_calls.length > 0) {
      const details = el("details", { class: "activity" });
      if (openActivity.has(s.sessionId)) details.setAttribute("open", "");
      details.addEventListener("toggle", () => {
        if (details.open) openActivity.add(s.sessionId);
        else openActivity.delete(s.sessionId);
      });
      const summary = el("summary");
      summary.appendChild(document.createTextNode("Recent activity"));
      summary.appendChild(el("span", { class: "count", text: s.recent_tool_calls.length + " call(s)" }));
      details.appendChild(summary);
      const ul = el("ul", { class: "tools" });
      for (const t of s.recent_tool_calls.slice().reverse()) {
        const li = el("li");
        li.appendChild(el("span", { class: "name" + (t.sidechain ? " sc" : ""), text: t.name }));
        li.appendChild(el("span", { class: "summary", text: t.summary || "" }));
        ul.appendChild(li);
      }
      details.appendChild(ul);
      card.appendChild(details);
    }
    if (s._parse_error) {
      card.appendChild(el("div", { class: "err", text: "transcript parse error: " + s._parse_error }));
    }
    // actions row
    const actions = el("div", { class: "actions" });
    const openBtn = el("button", { class: "btn", text: "↗ Open terminal" });
    openBtn.addEventListener("click", async () => {
      openBtn.disabled = true;
      const r = await postJSON("/api/focus", { pid: s.pid });
      openBtn.disabled = false;
      toast(r.message, r.ok ? "success" : "error");
    });
    actions.appendChild(openBtn);
    const killBtn = el("button", { class: "btn danger", text: "Stop" });
    killBtn.addEventListener("click", async () => {
      const label = s.title || s.project || ("pid " + s.pid);
      if (!confirm("Stop this Claude Code session?\n\n" + label + "\n(SIGTERM)")) return;
      killBtn.disabled = true;
      const r = await postJSON("/api/kill", { pid: s.pid });
      killBtn.disabled = false;
      toast(r.message, r.ok ? "success" : "error");
      if (r.ok) tick();
    });
    actions.appendChild(killBtn);
    card.appendChild(actions);
    root.appendChild(card);
  }
}
// new-session form wiring
function wireNewSessionForm() {
  const toggle = document.getElementById("new-toggle");
  const form = document.getElementById("new-form");
  const cancel = document.getElementById("new-cancel");
  const cwd = document.getElementById("new-cwd");
  const submit = document.getElementById("new-submit");
  function open() {
    toggle.style.display = "none";
    form.classList.add("open");
    cwd.value = "~";
    cwd.focus();
    cwd.select();
  }
  function close() {
    form.classList.remove("open");
    toggle.style.display = "";
  }
  toggle.addEventListener("click", open);
  cancel.addEventListener("click", close);
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    submit.disabled = true;
    const r = await postJSON("/api/new", { cwd: cwd.value || "~" });
    submit.disabled = false;
    toast(r.message, r.ok ? "success" : "error");
    if (r.ok) close();
  });
  // Esc closes
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && form.classList.contains("open")) close();
  });
}
wireNewSessionForm();
let consecutiveFailures = 0;
let lastSuccessTs = null;
async function tick() {
  try {
    const r = await fetch("/api/state", { cache: "no-store" });
    if (!r.ok) throw new Error("HTTP " + r.status);
    const s = await r.json();
    consecutiveFailures = 0;
    lastSuccessTs = Date.now();
    render(s);
  } catch (e) {
    consecutiveFailures++;
    const meta = document.getElementById("meta");
    if (consecutiveFailures <= 2) {
      // brief blip — keep the last-good UI; show a small inline note
      const ago = lastSuccessTs ? Math.round((Date.now() - lastSuccessTs) / 1000) : null;
      meta.textContent = ago != null
        ? `reconnecting… (last update ${ago}s ago)`
        : "connecting…";
    } else {
      meta.textContent = "server unreachable — is `python3 server.py` running?";
    }
  }
}
tick();
setInterval(tick, 2000);
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # quieter logs
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))

    def do_GET(self):  # noqa: N802
        try:
            if self.path == "/" or self.path.startswith("/?"):
                self._send(200, "text/html; charset=utf-8", INDEX_HTML.encode("utf-8"))
                return
            if self.path == "/api/state":
                payload = json.dumps(collect_state(), default=str).encode("utf-8")
                self._send(200, "application/json", payload, no_cache=True)
                return
            if self.path == "/healthz":
                self._send(200, "text/plain", b"ok")
                return
            self._send(404, "text/plain", b"not found")
        except Exception as e:
            import traceback
            sys.stderr.write(f"[handler] do_GET {self.path} crashed: {e}\n{traceback.format_exc()}")
            self._try_error_response(e)

    def _try_error_response(self, e: Exception):
        try:
            payload = json.dumps({"ok": False, "error": str(e)}).encode("utf-8")
            self._send(500, "application/json", payload, no_cache=True)
        except Exception:
            pass  # connection may already be in a bad state — at minimum we logged it

    def do_POST(self):  # noqa: N802
        try:
            length = int(self.headers.get("Content-Length") or "0")
            raw = self.rfile.read(length) if length > 0 else b""
            try:
                body = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                self._send_json(400, {"ok": False, "message": "invalid JSON body"})
                return
            self._dispatch_post(body)
        except Exception as e:
            import traceback
            sys.stderr.write(f"[handler] do_POST {self.path} crashed: {e}\n{traceback.format_exc()}")
            self._try_error_response(e)

    def _dispatch_post(self, body: dict):
        if self.path == "/api/focus":
            pid = body.get("pid")
            if not isinstance(pid, int):
                self._send_json(400, {"ok": False, "message": "missing or invalid pid"})
                return
            ok, msg = focus_terminal_for_pid(pid)
            self._send_json(200 if ok else 400, {"ok": ok, "message": msg})
            return
        if self.path == "/api/kill":
            pid = body.get("pid")
            if not isinstance(pid, int):
                self._send_json(400, {"ok": False, "message": "missing or invalid pid"})
                return
            tracked = {int(m.get("pid")) for m in read_session_files() if m.get("pid")}
            ok, msg = stop_session(pid, tracked)
            self._send_json(200 if ok else 400, {"ok": ok, "message": msg})
            return
        if self.path == "/api/new":
            cwd = body.get("cwd") or "~"
            ok, msg = open_new_session(str(cwd))
            self._send_json(200 if ok else 400, {"ok": ok, "message": msg})
            return
        self._send(404, "text/plain", b"not found")

    def _send_json(self, code: int, obj: dict):
        payload = json.dumps(obj).encode("utf-8")
        self._send(code, "application/json", payload, no_cache=True)

    def _send(self, code: int, ctype: str, body: bytes, no_cache: bool = False):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        if no_cache:
            self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    if not CLAUDE_HOME.exists():
        print(f"warning: {CLAUDE_HOME} does not exist — dashboard will show no sessions.", file=sys.stderr)

    ThreadingHTTPServer.allow_reuse_address = True
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"
    print(f"Claude Agent Watch listening at {url}")
    print(f"Reading state from: {CLAUDE_HOME}")

    def _shutdown(*_):
        print("\nshutting down…")
        srv.shutdown()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)
    try:
        srv.serve_forever()
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
