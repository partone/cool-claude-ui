# Claudius Minimus

A tiny local web dashboard that shows which Claude Code sessions are running
on your machine and what they're working on.

No dependencies — just Python 3.9+ stdlib. Reads `~/.claude/sessions/*.json`
and the per-session JSONL transcripts under `~/.claude/projects/`.

## Run

```sh
python3 server.py
# open http://127.0.0.1:8765
```

Options:

```sh
python3 server.py --host 127.0.0.1 --port 8765
```

Set `CLAUDE_HOME` to point at a non-default Claude state directory.

## What it shows

For each live Claude Code process:

- Project (cwd) and git branch
- Status (`busy` / `idle`) and "sub-agent active" indicator
- Session title (Claude's own summary of the conversation)
- Most recent user prompt
- Last ~12 tool calls (Read / Edit / Bash / Agent / ...)
- Any in-flight sub-agents spawned via the `Agent` tool

The page polls `/api/state` every 2 seconds. No JS framework, no build step.

## Endpoints

- `GET /` — dashboard
- `GET /api/state` — JSON snapshot of all live sessions
- `GET /healthz` — `ok`

## How it detects "live"

It reads every `~/.claude/sessions/*.json`, then filters out any whose `pid`
is no longer running. The remaining entries each map to a JSONL transcript
under `~/.claude/projects/<encoded-cwd>/<sessionId>.jsonl`, which is tailed
to extract the latest activity.
