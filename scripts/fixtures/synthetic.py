#!/usr/bin/env python3
"""Generate synthetic conversations for demos and CI.

Produces ~15 realistic conversations across multiple providers using the same
builder helpers that the test suite uses. Content covers: debugging sessions,
code review, writing assistance, data analysis, and general Q&A.

Usage:
    python -m scripts.fixtures.synthetic --output-dir /tmp/demo-fixtures
    python -m scripts.fixtures.synthetic  # prints to stdout as summary
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Conversation content — realistic enough for demos, provider-diverse
# ---------------------------------------------------------------------------

_CHATGPT_CONVERSATIONS: list[dict] = [
    {
        "id": "chatgpt-debug-oauth",
        "title": "Debugging OAuth Token Refresh",
        "messages": [
            ("user", "My OAuth refresh token keeps expiring after 7 days even though I set it to 30 days. What could cause this?"),
            ("assistant", "There are several common causes for premature OAuth token expiry:\n\n1. **Server-side policy override**: Your authorization server may enforce a maximum lifetime shorter than what you requested. Check the server's token policy configuration.\n\n2. **Token rotation**: If refresh token rotation is enabled, each use of a refresh token invalidates the old one and issues a new one. If your client isn't persisting the new token, it appears to expire.\n\n3. **Clock skew**: If there's significant clock drift between your client and the auth server, tokens may be rejected before their stated expiry.\n\nCheck your auth server logs for `token_revoked` or `token_expired` events around the 7-day mark."),
            ("user", "It was token rotation! The new refresh token wasn't being saved. Thanks!"),
            ("assistant", "That's a very common gotcha. Make sure your token storage callback handles the rotation atomically — if the save fails after using the old token but before persisting the new one, you'll be locked out.\n\nA good pattern:\n```python\ndef on_token_refresh(new_token):\n    # Write to temp file first, then atomic rename\n    tmp = token_path.with_suffix('.tmp')\n    tmp.write_text(json.dumps(new_token))\n    tmp.rename(token_path)\n```"),
        ],
    },
    {
        "id": "chatgpt-data-analysis",
        "title": "Pandas Performance Optimization",
        "messages": [
            ("user", "I have a DataFrame with 50M rows and need to do a groupby + rolling window calculation. It's taking 45 minutes. How can I speed this up?"),
            ("assistant", "For 50M rows, here are the main optimization strategies in order of impact:\n\n1. **Use appropriate dtypes**: Convert object columns to categories, downcast numerics:\n```python\ndf['category'] = df['category'].astype('category')\ndf['value'] = pd.to_numeric(df['value'], downcast='float')\n```\n\n2. **Avoid apply()**: Replace `.apply()` with vectorized operations. The rolling window should use built-in aggregations:\n```python\n# Slow\ndf.groupby('key').apply(lambda g: g['val'].rolling(7).mean())\n# Fast\ndf.sort_values(['key', 'date']).groupby('key')['val'].transform(\n    lambda x: x.rolling(7).mean()\n)\n```\n\n3. **Consider Polars**: For this scale, Polars is often 5-10x faster:\n```python\nimport polars as pl\ndf_pl = pl.from_pandas(df)\nresult = df_pl.sort('date').group_by('key').agg(\n    pl.col('val').rolling_mean(window_size=7)\n)\n```\n\n4. **Chunk processing**: If memory is the bottleneck, process in chunks by group key."),
            ("user", "Switching to Polars brought it down to 3 minutes. The category dtype tip also helped with memory."),
            ("assistant", "That's a great improvement! A few more Polars tips for your workflow:\n\n- Use **lazy evaluation** with `scan_parquet()` instead of reading everything into memory\n- The `.rolling()` in Polars supports `group_by` directly without needing sort + transform\n- For repeated analyses, save intermediate results as Parquet — it preserves dtypes and is much faster than CSV\n\nAt 50M rows, you're in the sweet spot where Polars really shines over pandas."),
        ],
    },
    {
        "id": "chatgpt-writing-assist",
        "title": "Technical Blog Post Review",
        "messages": [
            ("user", 'Can you review the opening paragraph of my blog post about distributed systems?\n\n"Distributed systems are hard. Everyone knows this. But what most people don\'t realize is that the hardest part isn\'t the distributed part — it\'s the systems part. Getting individual components right, with proper error handling, clean interfaces, and well-defined contracts, is the foundation that everything else rests on."'),
            ("assistant", 'The opening is strong — it uses a reversal technique ("not the distributed part — it\'s the systems part") that immediately challenges reader assumptions. A few suggestions:\n\n1. **The first two sentences are too abrupt**. "Everyone knows this" reads as dismissive. Consider:\n   > "Distributed systems are notoriously difficult. This is practically an axiom in our field."\n\n2. **The reversal payoff is good** but could be sharper. The list (error handling, clean interfaces, contracts) dilutes the punch. Pick one vivid example instead.\n\n3. **Add a hook for what follows**: End with a question or promise that makes the reader continue.\n\nRevised version:\n> "Distributed systems are notoriously difficult — this is practically an axiom of software engineering. But after a decade of building them, I\'ve come to believe the hardest part isn\'t the *distributed* part. It\'s the *systems* part. If your individual services can\'t handle a database hiccup gracefully, adding a network between them won\'t make things better. Here\'s what I wish someone had told me about getting the fundamentals right."'),
            ("user", "That revised version is much better. The concrete example about database hiccups really grounds it."),
        ],
    },
    {
        "id": "chatgpt-git-workflow",
        "title": "Git Rebase vs Merge Strategy",
        "messages": [
            ("user", "Our team is debating rebase vs merge for feature branches. What's your recommendation?"),
            ("assistant", "Both strategies have legitimate use cases. Here's my framework for deciding:\n\n**Use rebase when:**\n- You want a clean, linear history\n- The feature branch is personal (not shared with others)\n- You're preparing a PR for review\n- The branch has messy WIP commits you want to clean up\n\n**Use merge when:**\n- Multiple people are working on the same branch\n- You need to preserve the exact sequence of events (audit trail)\n- The branch has been pushed and others have based work on it\n- You want explicit merge points visible in the graph\n\n**My recommendation for most teams:**\n1. Rebase feature branches onto main before merging\n2. Use merge commits (not fast-forward) for the final merge to main\n3. Never rebase anything that's been shared/pushed\n\nThis gives you clean feature branch history AND clear merge points on main:\n```bash\ngit checkout feature-branch\ngit rebase main\ngit checkout main\ngit merge --no-ff feature-branch\n```"),
        ],
    },
]

_CLAUDE_CONVERSATIONS: list[dict] = [
    {
        "id": "claude-api-design",
        "name": "REST API Design Review",
        "messages": [
            ("human", "I'm designing a REST API for a task management system. Here's my initial endpoint design:\n\nPOST /tasks - create task\nGET /tasks - list tasks\nGET /tasks/:id - get task\nPUT /tasks/:id - update task\nDELETE /tasks/:id - delete task\nPOST /tasks/:id/complete - mark complete\nPOST /tasks/:id/assign - assign to user\n\nDoes this look right?"),
            ("assistant", "Your CRUD endpoints follow REST conventions well. A few suggestions:\n\n**1. Use PATCH instead of PUT for partial updates:**\n```\nPATCH /tasks/:id  — partial update (change title only)\nPUT /tasks/:id    — full replacement (must send all fields)\n```\nMost clients want PATCH. Reserve PUT for full replacements.\n\n**2. Model state changes as PATCH, not custom POST endpoints:**\n```\n# Instead of POST /tasks/:id/complete\nPATCH /tasks/:id  { \"status\": \"completed\" }\n\n# Instead of POST /tasks/:id/assign\nPATCH /tasks/:id  { \"assignee_id\": \"user-123\" }\n```\nThis reduces your API surface and is more consistent. The custom endpoints are only needed if the operation has side effects beyond updating a field (e.g., sending notifications).\n\n**3. Add filtering and pagination to the list endpoint:**\n```\nGET /tasks?status=open&assignee=me&page=1&per_page=25\nGET /tasks?sort=created_at&order=desc\n```\n\n**4. Consider bulk operations early:**\n```\nPATCH /tasks  { \"ids\": [...], \"status\": \"archived\" }\n```"),
            ("human", "Good point about PATCH vs PUT. But for 'complete', there's actually a side effect — it triggers a webhook notification and updates project progress. Should I keep the custom endpoint?"),
            ("assistant", "Yes, when an action has significant side effects beyond field mutation, a custom action endpoint is the right call. The REST purist approach of encoding everything as resource state changes breaks down when:\n\n1. The operation triggers workflows (webhooks, notifications)\n2. The state transition has preconditions (can't complete if blocked)\n3. Multiple resources change atomically\n\nI'd keep `POST /tasks/:id/complete` and document it clearly:\n\n```\nPOST /tasks/:id/complete\n\nMarks a task as completed. This will:\n- Set status to 'completed' and completion_date to now\n- Notify assigned user and watchers via webhook\n- Update parent project progress percentage\n- Trigger any automation rules for task completion\n\nPreconditions:\n- Task must not already be completed\n- Task must not have unresolved blockers\n\nReturns: 200 with updated task, or 409 if preconditions fail\n```\n\nThe key distinction: use PATCH for simple field updates, use action endpoints for operations with business logic."),
        ],
    },
    {
        "id": "claude-error-handling",
        "name": "Error Handling Strategy for Python Services",
        "messages": [
            ("human", "What's the best error handling strategy for a Python microservice? I keep going back and forth between exception-based and result-type approaches."),
            ("assistant", "Both approaches have their place. Here's a practical framework:\n\n**Use exceptions for:**\n- Infrastructure failures (DB down, network timeout)\n- Programming errors (assertion failures, type mismatches)\n- Anything the immediate caller can't reasonably handle\n\n**Use result types for:**\n- Expected business failures (validation errors, not-found)\n- Operations where the caller MUST handle both outcomes\n- Public API boundaries where you want exhaustive handling\n\nA pragmatic Python approach that works well:\n\n```python\nfrom dataclasses import dataclass\nfrom typing import TypeVar, Generic\n\nT = TypeVar('T')\n\n@dataclass\nclass Result(Generic[T]):\n    value: T | None = None\n    error: str | None = None\n    \n    @property\n    def ok(self) -> bool:\n        return self.error is None\n    \n    @classmethod\n    def success(cls, value: T) -> 'Result[T]':\n        return cls(value=value)\n    \n    @classmethod  \n    def failure(cls, error: str) -> 'Result[T]':\n        return cls(error=error)\n\n# Service layer returns Result for business operations\ndef create_user(email: str) -> Result[User]:\n    if not validate_email(email):\n        return Result.failure(\"Invalid email format\")\n    if user_exists(email):\n        return Result.failure(\"Email already registered\")\n    user = db.insert_user(email)  # DB errors raise exceptions\n    return Result.success(user)\n```\n\nThe key insight: exceptions flow UP the stack, results flow ACROSS boundaries."),
        ],
    },
    {
        "id": "claude-regex-help",
        "name": "Complex Regex for Log Parsing",
        "messages": [
            ("human", "I need a regex to parse log lines like:\n2025-01-15T10:30:45.123Z [INFO] request.handler: processed request method=GET path=/api/users duration=45ms status=200\n\nI need to extract: timestamp, level, logger, message, and all key=value pairs."),
            ("assistant", 'Here\'s a regex that handles this format robustly:\n\n```python\nimport re\n\nLOG_PATTERN = re.compile(\n    r"(?P<timestamp>\\d{4}-\\d{2}-\\d{2}T[\\d:.]+Z?)\\s+"\n    r"\\[(?P<level>\\w+)\\]\\s+"\n    r"(?P<logger>[\\w.]+):\\s+"\n    r"(?P<message>[^=]+?)\\s+"\n    r"(?P<kvpairs>(?:\\w+=\\S+\\s*)+)$"\n)\n\nKV_PATTERN = re.compile(r"(\\w+)=(\\S+)")\n\ndef parse_log_line(line: str) -> dict | None:\n    m = LOG_PATTERN.match(line.strip())\n    if not m:\n        return None\n    result = {\n        "timestamp": m.group("timestamp"),\n        "level": m.group("level"),\n        "logger": m.group("logger"),\n        "message": m.group("message").strip(),\n    }\n    result["fields"] = dict(KV_PATTERN.findall(m.group("kvpairs")))\n    return result\n```\n\nThis splits the problem into two passes:\n1. The main regex extracts the fixed-format prefix and captures the tail as a blob\n2. A simple `key=value` regex extracts structured fields from the tail\n\nThis is more maintainable than trying to capture variable numbers of k/v pairs in a single regex.'),
        ],
    },
]

_CLAUDE_CODE_CONVERSATIONS: list[dict] = [
    {
        "id": "cc-refactor-session",
        "title": "Refactoring Database Layer",
        "messages": [
            ("human", "Let's refactor the database module to use the repository pattern. Currently all SQL is inline in the handlers."),
            ("assistant", "I'll analyze the current database usage and design the repository layer.\n\nLooking at the codebase, I see SQL scattered across:\n- `handlers/users.py` — 12 queries\n- `handlers/tasks.py` — 18 queries  \n- `handlers/projects.py` — 8 queries\n\nI'll create a repository for each domain:\n\n```python\n# repositories/users.py\nclass UserRepository:\n    def __init__(self, conn: Connection):\n        self.conn = conn\n    \n    def get_by_id(self, user_id: str) -> User | None: ...\n    def get_by_email(self, email: str) -> User | None: ...\n    def create(self, user: CreateUser) -> User: ...\n    def update(self, user_id: str, data: UpdateUser) -> User: ...\n    def list_active(self, limit: int = 50) -> list[User]: ...\n```\n\nLet me start implementing this."),
            ("human", "Looks good. Make sure to keep the transaction boundaries in the handlers, not the repositories."),
            ("assistant", "Agreed — repositories should be unit-of-work agnostic. The handler owns the transaction lifecycle.\n\nI've created the repository files and updated the handlers. Here's the key change:\n\n```python\n# Before (handler)\nasync def create_task(request):\n    async with db.transaction():\n        row = await db.execute(\"INSERT INTO tasks ...\")\n        await db.execute(\"INSERT INTO activity_log ...\")\n    return Task.from_row(row)\n\n# After (handler + repository)\nasync def create_task(request):\n    async with db.transaction() as conn:\n        task = await task_repo(conn).create(request.data)\n        await activity_repo(conn).log('task_created', task.id)\n    return task\n```\n\nThe repository receives the connection, keeping it transaction-unaware."),
        ],
    },
    {
        "id": "cc-test-debugging",
        "title": "Debugging Flaky CI Test",
        "messages": [
            ("human", "This test passes locally but fails in CI about 30% of the time:\n\ndef test_concurrent_writes():\n    results = asyncio.gather(*[write_record(i) for i in range(100)])\n    assert len(results) == 100\n    assert len(set(r.id for r in results)) == 100  # All unique IDs"),
            ("assistant", "Classic race condition pattern. The test is timing-dependent because:\n\n1. **ID generation might not be truly unique under contention** — if using timestamp-based IDs, concurrent writes within the same millisecond can collide\n2. **`asyncio.gather` doesn't await** — you're missing `await`:\n\n```python\n# Bug: results is a coroutine, not a list\nresults = asyncio.gather(...)  # Missing await!\n\n# Fix:\nresults = await asyncio.gather(*[write_record(i) for i in range(100)])\n```\n\nThe test passes locally because the coroutine object happens to be truthy, and `len()` of a coroutine raises `TypeError` which... actually, this should fail everywhere. Let me re-examine.\n\nAh — you're likely running this inside a `pytest-asyncio` fixture that implicitly wraps it. The real issue is probably the gather succeeding but with duplicate IDs due to your ID generation strategy. What does `write_record` use for ID generation?"),
            ("human", "It uses uuid4(). But you're right about the missing await — CI runs Python 3.12 which changed some asyncio behavior. Adding await fixed it."),
            ("assistant", "Good catch on the Python version difference. In Python 3.12, `asyncio.gather()` without `await` behaves differently in some event loop implementations. The key takeaway: always `await` your gather calls, and consider using `TaskGroup` for Python 3.11+:\n\n```python\nasync def test_concurrent_writes():\n    results = []\n    async with asyncio.TaskGroup() as tg:\n        for i in range(100):\n            tg.create_task(write_and_collect(i, results))\n    assert len(results) == 100\n```\n\n`TaskGroup` is more explicit about error handling — if any task fails, all tasks are cancelled and the first exception is raised."),
        ],
    },
]

_CODEX_CONVERSATIONS: list[dict] = [
    {
        "id": "codex-docker-compose",
        "title": "Docker Compose for Dev Environment",
        "messages": [
            ("user", "Create a docker-compose.yml for a Python web app with PostgreSQL, Redis, and a worker process."),
            ("assistant", "```yaml\nservices:\n  web:\n    build: .\n    ports:\n      - \"8000:8000\"\n    environment:\n      DATABASE_URL: postgresql://app:secret@db:5432/appdb\n      REDIS_URL: redis://redis:6379/0\n    depends_on:\n      db:\n        condition: service_healthy\n      redis:\n        condition: service_started\n    volumes:\n      - .:/app\n    command: uvicorn main:app --host 0.0.0.0 --reload\n\n  worker:\n    build: .\n    environment:\n      DATABASE_URL: postgresql://app:secret@db:5432/appdb\n      REDIS_URL: redis://redis:6379/0\n    depends_on:\n      db:\n        condition: service_healthy\n      redis:\n        condition: service_started\n    command: celery -A tasks worker --loglevel=info\n\n  db:\n    image: postgres:16-alpine\n    environment:\n      POSTGRES_USER: app\n      POSTGRES_PASSWORD: secret\n      POSTGRES_DB: appdb\n    volumes:\n      - pgdata:/var/lib/postgresql/data\n    healthcheck:\n      test: pg_isready -U app -d appdb\n      interval: 5s\n      retries: 5\n\n  redis:\n    image: redis:7-alpine\n    volumes:\n      - redisdata:/data\n\nvolumes:\n  pgdata:\n  redisdata:\n```\n\nKey choices:\n- Health check on PostgreSQL prevents the app from starting before the DB is ready\n- Named volumes persist data between restarts\n- `--reload` on uvicorn for development hot-reloading\n- Alpine images for smaller footprint"),
        ],
    },
    {
        "id": "codex-sql-optimization",
        "title": "SQL Query Optimization",
        "messages": [
            ("user", "This query is slow on 10M rows:\nSELECT u.*, COUNT(o.id) as order_count\nFROM users u LEFT JOIN orders o ON u.id = o.user_id\nWHERE u.created_at > '2024-01-01'\nGROUP BY u.id\nORDER BY order_count DESC\nLIMIT 20"),
            ("assistant", "Several optimizations:\n\n**1. Add a covering index:**\n```sql\nCREATE INDEX idx_users_created ON users(created_at) INCLUDE (id, name, email);\nCREATE INDEX idx_orders_user ON orders(user_id);\n```\n\n**2. Use a subquery to reduce the join scope:**\n```sql\nSELECT u.*, COALESCE(oc.cnt, 0) as order_count\nFROM users u\nLEFT JOIN (\n    SELECT user_id, COUNT(*) as cnt\n    FROM orders\n    GROUP BY user_id\n) oc ON u.id = oc.user_id\nWHERE u.created_at > '2024-01-01'\nORDER BY order_count DESC\nLIMIT 20;\n```\n\nThis is faster because:\n- The subquery aggregates orders independently (smaller intermediate result)\n- The join operates on the pre-aggregated data instead of expanding all rows\n- The `COALESCE` handles users with zero orders\n\n**3. If you only need recent orders:**\n```sql\nWHERE o.created_at > '2024-01-01'  -- Add to subquery if applicable\n```\n\nRun `EXPLAIN ANALYZE` on both versions to compare."),
        ],
    },
]

_GEMINI_CONVERSATIONS: list[dict] = [
    {
        "id": "gemini-ml-pipeline",
        "title": "ML Feature Pipeline Design",
        "messages": [
            ("user", "I need to design a feature engineering pipeline for a recommendation system. We have user activity logs, item metadata, and interaction data. What's the best architecture?"),
            ("assistant", "Here's a layered architecture that scales well:\n\n**Layer 1: Raw Feature Store**\n- Ingest raw events into a time-partitioned table (BigQuery/Snowflake)\n- Schema: `(user_id, event_type, item_id, timestamp, metadata_json)`\n- Retention: 2 years raw, aggregated beyond that\n\n**Layer 2: Feature Computation**\n```\nBatch features (daily):\n  - user_activity_30d: count of events per type in last 30 days\n  - item_popularity_7d: interaction count per item, 7-day window\n  - user_category_affinity: normalized interaction counts by category\n  \nReal-time features (streaming):\n  - session_click_count: clicks in current session\n  - time_since_last_interaction: seconds since last event\n  - trending_score: exponentially weighted recent interactions\n```\n\n**Layer 3: Feature Serving**\n- Batch features: precomputed daily, stored in Redis/DynamoDB\n- Real-time features: computed on-the-fly from event stream\n- Feature vector assembly at inference time combines both\n\n**Key design decisions:**\n1. **Point-in-time correctness**: Always join features using timestamp constraints to prevent future data leakage during training\n2. **Feature versioning**: Track feature definitions in code, not notebooks\n3. **Monitoring**: Alert on feature drift (distribution shift > threshold)"),
        ],
    },
]


def _build_chatgpt_export(conv: dict) -> dict:
    """Build ChatGPT wire format from simplified definition."""
    nodes = {}
    prev_id = None
    for i, (role, content) in enumerate(conv["messages"]):
        node_id = f"node-{i+1}"
        node = {
            "id": node_id,
            "message": {
                "id": node_id,
                "author": {"role": role},
                "content": {"parts": [content]},
                "create_time": 1704067200.0 + i * 60,
            },
        }
        if prev_id:
            node["parent"] = prev_id
            nodes[prev_id]["children"] = [node_id]
        nodes[node_id] = node
        prev_id = node_id

    result = {"id": conv["id"], "mapping": nodes}
    if "title" in conv:
        result["title"] = conv["title"]
    return result


def _build_claude_export(conv: dict) -> dict:
    """Build Claude web wire format from simplified definition."""
    messages = []
    for i, (sender, text) in enumerate(conv["messages"]):
        messages.append({
            "uuid": f"msg-{i+1}",
            "sender": sender,
            "text": text,
        })
    conversation = {"id": conv["id"], "chat_messages": messages}
    if "name" in conv:
        conversation["name"] = conv["name"]
    return {"conversations": [conversation]}


def _build_claude_code_export(conv: dict) -> list[dict]:
    """Build Claude Code JSONL wire format from simplified definition."""
    entries = []
    session_id = conv["id"]
    for i, (role, content) in enumerate(conv["messages"]):
        entry = {
            "parentUuid": f"parent-{i}" if i > 0 else None,
            "uuid": f"msg-{i+1}",
            "sessionId": session_id,
            "type": role,
            "message": {"content": [{"type": "text", "text": content}]},
            "timestamp": f"2025-01-15T10:{i:02d}:00.000Z",
        }
        entries.append(entry)
    return entries


def _build_codex_export(conv: dict) -> dict:
    """Build Codex/generic wire format from simplified definition."""
    messages = []
    for i, (role, content) in enumerate(conv["messages"]):
        messages.append({"id": f"m{i+1}", "role": role, "content": content})
    result = {"id": conv["id"], "messages": messages}
    if "title" in conv:
        result["title"] = conv["title"]
    return result


def _build_gemini_export(conv: dict) -> dict:
    """Build Gemini chunkedPrompt wire format from simplified definition.

    Produces the real Drive/Gemini structure that exercises parse_chunked_prompt():
    {"chunkedPrompt": {"chunks": [{"text": ..., "role": "user"|"model"}]}}
    """
    chunks = []
    for role, content in conv["messages"]:
        # Gemini uses "model" for assistant role
        gemini_role = "model" if role in ("assistant", "model") else "user"
        chunks.append({"text": content, "role": gemini_role})
    return {"chunkedPrompt": {"chunks": chunks}}


def generate_synthetic_fixtures(output_dir: Path) -> dict[str, list[Path]]:
    """Write synthetic fixture files to output_dir, organized by provider.

    Returns mapping of provider -> list of written file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, list[Path]] = {}

    # ChatGPT exports
    chatgpt_dir = output_dir / "chatgpt"
    chatgpt_dir.mkdir(exist_ok=True)
    written["chatgpt"] = []
    for conv in _CHATGPT_CONVERSATIONS:
        path = chatgpt_dir / f"{conv['id']}.json"
        data = _build_chatgpt_export(conv)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        written["chatgpt"].append(path)

    # Claude web exports
    claude_dir = output_dir / "claude"
    claude_dir.mkdir(exist_ok=True)
    written["claude"] = []
    for conv in _CLAUDE_CONVERSATIONS:
        path = claude_dir / f"{conv['id']}.json"
        data = _build_claude_export(conv)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        written["claude"].append(path)

    # Claude Code exports (JSONL)
    cc_dir = output_dir / "claude-code"
    cc_dir.mkdir(exist_ok=True)
    written["claude-code"] = []
    for conv in _CLAUDE_CODE_CONVERSATIONS:
        path = cc_dir / f"{conv['id']}.jsonl"
        entries = _build_claude_code_export(conv)
        lines = [json.dumps(e) for e in entries]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        written["claude-code"].append(path)

    # Codex exports
    codex_dir = output_dir / "codex"
    codex_dir.mkdir(exist_ok=True)
    written["codex"] = []
    for conv in _CODEX_CONVERSATIONS:
        path = codex_dir / f"{conv['id']}.json"
        data = _build_codex_export(conv)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        written["codex"].append(path)

    # Gemini exports (simple JSON format — no Drive needed)
    gemini_dir = output_dir / "gemini"
    gemini_dir.mkdir(exist_ok=True)
    written["gemini"] = []
    for conv in _GEMINI_CONVERSATIONS:
        path = gemini_dir / f"{conv['id']}.json"
        data = _build_gemini_export(conv)  # Uses real chunkedPrompt format to exercise Gemini parser
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        written["gemini"].append(path)

    return written


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic demo fixtures")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory (default: temp dir)")
    args = parser.parse_args()

    if args.output_dir:
        output_dir = args.output_dir
    else:
        import tempfile
        output_dir = Path(tempfile.mkdtemp(prefix="polylogue-synthetic-"))

    written = generate_synthetic_fixtures(output_dir)

    total = sum(len(files) for files in written.values())
    print(f"Generated {total} synthetic conversations in {output_dir}")
    for provider, files in written.items():
        print(f"  {provider}: {len(files)} conversations")
        for f in files:
            print(f"    {f.name}")


if __name__ == "__main__":
    main()
