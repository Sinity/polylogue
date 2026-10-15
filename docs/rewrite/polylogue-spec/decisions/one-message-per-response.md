# Decision: one message per model response

Status: **decided** 2026-10-11 (technical; found on real data).

Claude Code writes one model response as several JSONL records, one per
content block, each carrying the same `message.id` and the full `usage` of the
response. In one real session, 281 of 348 responses were split across 879
records (`evidence/ideas-architecture.md`). Treating each record as a message
would count usage about 2.5 times and show one reply as several.

Rule (`spec/model.md`, Identity): records sharing `message.id` and `requestId` form one
message. Its ID is the first record's `uuid` (so IDs don't change as later
records of the same response are appended, and `parentUuid` links that name
any record of the response resolve to it). Blocks keep record order. Usage
is taken once, from the last record of the response.

Consequences: the Claude Code parser keeps a map from record `uuid` to
message ID while parsing; `compare-old` reports the old parser's split
messages as an expected difference; the Morrow archive gets one split
response so the rule is tested from slice 1.

Confirmed independently by the prior-art survey: ccusage deduplicates usage on `message.id` + `requestId` for the same reason (`evidence/prior-art.md`).
