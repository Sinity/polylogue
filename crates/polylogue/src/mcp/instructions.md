Polylogue is the archive of past AI sessions on this machine: Claude Code,
Codex, Hermes, Gemini CLI and other agents' sessions, and ChatGPT, claude.ai,
Grok, Gemini and AI Studio conversations, with every tool call, file change
and verification they made. Use it to start from what is already known instead of
exploring again.

When to use which tool:
- `brief`: what recent sessions in a directory did, what is left open, which
  verifications fail there, and other agents working there now. Take it before
  starting work in a repository you do not know.
- `recall`: past work about a text (words, and similarity when vectors are
  built), ranked, with pages. Use it before solving a problem that may have
  been solved before.
- `takeover`: what you need to continue another session: its prompts, what it
  left open, its last answer.
- `query`: anything else, in the query language. Examples:
  `repo:polylogue since:2026-10-01 parser` (sessions mentioning parser),
  `messages where "exact phrase"`,
  `tool_calls where tool:Bash outcome:error session.repo:polylogue`,
  `sessions where live:waiting`,
  `repo:polylogue | measure usage`.
  An empty answer with `why` set says which condition removed the rows.
- `read`: one session (`claude-code:<id>`), document or event, in a view
  (`dialogue`, `prompts`, `actions`, `full`); `budget` returns a page sized for
  you and a cursor for the next.
- `note_add`: record a lesson, decision or finding on a session, message,
  repository or directory. Your notes are kept as yours; the operator accepts
  or rejects them before briefs carry them.

Rules:
- Session IDs are `origin:native-id` and may contain colons; pass them whole.
- Unknown figures are null, never zero. An empty result is not proof that
  something never happened: check `status` for what the archive holds.
- Cite what you used by its ID (session, message, note), so the operator can
  read it.
- Every context Polylogue gives you names its delivery; it records exactly
  what it gave.
