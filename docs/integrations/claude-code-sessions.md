# Claude Code Session Integration

This document describes how Claude Code sessions are stored and how to access them for historical context lookup.

## Session Storage

### Location

Claude Code stores session transcripts as JSONL files:

```
~/.claude/projects/<project-path-encoded>/<session-id>.jsonl
```

Where:
- `<project-path-encoded>` is the working directory with `/` replaced by `-`
- `<session-id>` is a UUID

Example:
```
~/.claude/projects/-realm-project-sinex/72c20600-068d-4158-91a6-47c60399c78b.jsonl
```

### JSONL Format

Each line is a JSON object representing a message or event:

```jsonl
{"type":"user","content":"user message text","timestamp":"..."}
{"type":"assistant","content":"assistant response","timestamp":"..."}
{"type":"tool_use","name":"Read","input":{"file_path":"/path/to/file"},"timestamp":"..."}
{"type":"tool_result","content":"file contents...","timestamp":"..."}
{"type":"system","content":"system message","timestamp":"..."}
```

Key fields:
- `type`: Message type (user, assistant, tool_use, tool_result, system)
- `content`: Text content of the message
- `name`: Tool name (for tool_use type)
- `input`: Tool parameters (for tool_use type)
- `timestamp`: ISO timestamp

### File Sizes

Sessions can be large (1-15MB typical). Tool outputs are often the bulk of the content.

## Querying Sessions

### List Recent Sessions

```bash
# For a specific project
ls -lt ~/.claude/projects/-realm-project-sinex/*.jsonl | head -10

# All projects
find ~/.claude/projects -name "*.jsonl" -mtime -7 -ls | sort -k11 -r
```

### Search Content

```bash
# Find sessions mentioning a topic
grep -l "search term" ~/.claude/projects/*/*.jsonl

# Extract user messages only
cat session.jsonl | jq -r 'select(.type=="user") | .content'

# Extract assistant messages only
cat session.jsonl | jq -r 'select(.type=="assistant") | .content'

# Get conversation without tool outputs (much smaller)
cat session.jsonl | jq -r 'select(.type=="user" or .type=="assistant") | "\(.type): \(.content[:200])"'
```

### Session Metadata

Session start time can be inferred from:
1. File modification time: `stat session.jsonl`
2. First entry timestamp: `head -1 session.jsonl | jq -r '.timestamp'`

## Polylogue Integration

### Current Support

Polylogue's generic JSONL parser can ingest Claude Code sessions:

```bash
# Import sessions (if configured)
polylogue run --source claude-code
```

Current limitations (from `docs/providers/claude-code.md`):
- Full tool pairing and workspace metadata are not implemented yet
- Tool use/result segments are serialized as JSON text

### Recommended Configuration

Configure via `POLYLOGUE_*` environment variables:

```bash
export POLYLOGUE_CLAUDE_CODE_SOURCE="~/.claude/projects/*/*.jsonl"
```

Or configure the source directly in your archive setup.

### Future Improvements Needed

1. **Tool pairing**: Match `tool_use` with corresponding `tool_result`
2. **Metadata extraction**: Project path, session duration, tool usage stats
3. **Embedding support**: Semantic search across sessions
4. **Incremental ingestion**: Watch for new sessions, ingest on close

## Claude Code `/history` Command

Claude Code includes a `/history` command for session lookup:

```
/history recent           # List recent sessions
/history search <query>   # Search across sessions
/history session <id>     # Read specific session
/history summarize <id>   # Generate summary
```

This uses the raw JSONL files directly. For richer search, use polylogue after it ingests the sessions.

## Use Cases

### 1. Continue Interrupted Work

When a session loses context (compaction), find the prior session:

```bash
# Find sessions from today
ls -lt ~/.claude/projects/-realm-project-*/*.jsonl | head -5

# Read the previous session's user messages
cat <session>.jsonl | jq -r 'select(.type=="user") | .content'
```

### 2. Find Prior Solution

Search for how a problem was solved before:

```bash
grep -l "error message" ~/.claude/projects/*/*.jsonl
```

### 3. Session Analytics

Count messages by type:

```bash
cat session.jsonl | jq -r '.type' | sort | uniq -c
```

Tool usage frequency:

```bash
cat session.jsonl | jq -r 'select(.type=="tool_use") | .name' | sort | uniq -c | sort -rn
```

## Integration with Claude Code Setup

The `/history` command and this documentation are part of the Claude Code self-improvement tooling. When Claude accesses history:

1. Search for relevant sessions
2. Extract key information
3. **Write learnings** to `~/.claude/learnings.local.md` to prevent re-discovery

This creates a persistent learning loop where insights from past sessions inform future ones.
