# Validation and public-claims discipline

## Claim classes

Use the same vocabulary in both projects.

- **Proven:** a bounded statement supported by a named artifact with an independent oracle.
- **Capability:** implemented behavior is demonstrated, but no downstream benefit or population claim is made.
- **Field evidence:** measured on a particular private deployment; useful for scale and failure discovery, not generalization.
- **Experimental:** a comparison with declared arms, metrics, and limitations.
- **Aspirational:** a Beads-owned target not yet shipped.
- **Retired:** a claim or architecture statement that no longer describes the intended product.

## Release validation matrix

| Concern | Polylogue gate | Sinex gate | Joint gate |
|---|---|---|---|
| Category consistency | README, package metadata, site, footer agree | README and docs agree | Backend page uses same authority model |
| Demo construct validity | manifest + fixture audit + ref resolution | manifest + product-boundary runner + gap controls | Agent Work Packet legs resolve independently |
| Privacy | demo path/secret scan, no live archive | material and artifact scrub, public corpus decision | raw-text capabilities separated from generic evidence access |
| Determinism | fresh demo root reproduces report | fresh sandbox reproduces packet | fixed fixture and semantics versions |
| Readiness honesty | FTS/projection/embedding state visible | source/material/projection coverage visible | both frontiers and local-replica lag visible |
| Performance | first useful result and web SLA | bounded moment query and replay operation | context/packet budget and latency reported |
| Installation | clean environment matrix | Nix/devshell/operator path proof | documented optional backed mode |
| Evidence reachability | public refs resolve to blocks/raw material | refs resolve to events/material anchors | stable Polylogue IDs survive Sinex replay |
| Deletion | current limits stated | lifecycle/excision status stated | cache/vector/context cascades accounted |
| Planning truth | Beads, not GitHub issues | Beads, not GitHub issues | superseding authority decision recorded |

## Cold-reader protocol

Recruit readers who have not worked on either repository. Give them only the README and hero demo. Ask them to answer, in writing:

1. What category is this product?
2. What does it know that a transcript viewer or activity logger does not?
3. Which result in the demo is direct evidence, which is derived, and which is reviewed?
4. What is one thing the product refuses to claim?
5. Is the demonstrated behavior shipped, experimental, or planned?
6. Where would you inspect the source evidence?

Pass criteria should be specified before the review. Do not coach readers during the first pass. Their errors are product-legibility defects, not reader defects.

## Artifact scrub

Before publication, scan every tracked or generated public artifact for:

- absolute home paths;
- usernames and hostnames;
- tokens, cookies, credentials, and environment values;
- private repository names or URLs;
- private session IDs and titles;
- source text not licensed for publication;
- screenshots containing browser chrome, notifications, or unrelated windows;
- stale GitHub issue links where Beads are authoritative;
- claims lacking ledger entries.

Store the scrub command and result in the launch packet.
