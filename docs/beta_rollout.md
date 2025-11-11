# Beta Rollout Checklist

To keep Polylogue's refactor aligned with early adopters, share the highlights below with the beta mailing list / chat and capture the responses before promoting to general availability.

## Message Template

```
Subject: Polylogue registrar + sync refresh ready for beta

Hi everyone,

We've pushed the registrar-backed pipelines into the latest build. Key changes:

- Local codex/Claude Code syncs, ChatGPT/Claude bundle syncs, and Drive renders all feed the same SQLite registrar, so slugs/stats stay consistent.
- `polylogue watch chatgpt|claude` automates bundle imports as soon as a new ZIP or conversations.json lands.
- `polylogue status --providers <list> --summary metrics.json` emits machine-readable rollups for cron/systemd jobs.

Please upgrade (`git pull && nix develop` or `pip install --upgrade polylogue`) and run your usual workflows. Reply in-thread or drop feedback in #polylogue-beta with stack traces, UX hiccups, or migration issues.

Thanks!
```

## Feedback Tracking

- [ ] Collect Drive sync latency feedback after 24h.
- [ ] Confirm automation users tested the new `--summary` / `--providers` switches.

Document findings in `docs/releases/2024-11-refactor.md` before tagging the release.
