# Coding-agent master prompt for this package

You are working in the Polylogue repository. Use `urgent_correctness_task_packets.jsonl` or the individual Markdown packet for the bead assigned to you. Do not start by scanning the whole backlog. Start from the packet, confirm the named source anchors, write the smallest failing test for the stated invariant, implement the single choke-point fix, and run the verification lane.

Rules:
- Preserve evidence honesty: unknown is not zero, text-derived is not structured evidence, fallback time is not provider time.
- Preserve data: blob cleanup and attachment acquisition must never invent hashes or delete leased/in-flight blobs.
- Preserve agent-write safety: non-user writes are candidate/non-injected until judged.
- Keep public claims backed by a test, log, report, or durable artifact.
- Add bead notes with the verification command and exact result after implementation.
