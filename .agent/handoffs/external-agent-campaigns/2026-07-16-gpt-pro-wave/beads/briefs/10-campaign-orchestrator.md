Title: "[beads 10] External-agent campaign orchestrator"

Job ID: `beads-10`
Result ZIP: `beads-10-campaign-orchestrator-r01.zip`
Primary Bead: `polylogue-yyvg.6`.

## Mission

Implement a replaceable repository-side orchestrator over generic browser
action and canonical Polylogue capture primitives. It owns campaign/job/
attempt/package-revision identities, prompt and attachment manifests, queue and
backoff policy, same-chat iteration, immutable acquisition, triage, worktree/
agent/PR integration receipts, and the full acquired-to-Bead-closed funnel.
It must not add campaign/work-package concepts to the browser extension,
receiver transport, or product UI.

Use the campaign layout in `.agent/handoffs/external-agent-campaigns/` as input
and result convention. Create/reply only through the generic action API and
observe responses/assets only through canonical Polylogue evidence. Include a
deterministic current-campaign fixture that fails on omitted conversations,
turns, assets, retries, packages, or integration outcomes. Do not grant browser
surfaces Git/Beads mutation authority.
