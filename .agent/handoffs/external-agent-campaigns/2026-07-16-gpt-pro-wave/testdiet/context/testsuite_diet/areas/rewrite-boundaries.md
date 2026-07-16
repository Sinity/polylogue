---
created: 2026-07-16
purpose: Preserve behavioral obligations without porting tests for implementations scheduled for rewrite
status: rewrite-boundary
project: polylogue
---

# MCP and web-reader rewrite boundaries

## MCP

Do not consolidate or improve current MCP tests against the current MCP
implementation. Extract externally meaningful obligations from tests using
real archives, then design the new suite from the rewritten protocol and
production route. Do not port `EXPECTED_TOOL_NAMES`, FastMCP private
registration assumptions, source parsing, test-authored tool classifications,
or same-name mock forwarding.

The rewrite-native suite owns discovery/schema truth, real archive selection
and mutation, stable error envelopes, context budgets, cancellation/deadlines,
and restart behavior. Exact names are pinned only if the new public contract
deliberately promises them.

## Web reader

Do not replace bad current-reader tests with better tests against the current
reader. Preserve HTTP/API substrate behavior the rewrite consumes, then write
Playwright/component tests against the rewritten DOM and interaction contract:
selection, navigation, sanitization/XSS, accessibility, loading/error states,
and representative large-transcript behavior.

Retire JavaScript/CSS/source-spelling tests with the old reader. Exclude both
areas from savings forecasts until rewrite-native suite sizes exist.
