# Fork prompt 06 — Build the public claims and findings lane

Use the uploaded Polylogue repository and the prior analysis. Build a static but rigorous first public slice toward `polylogue-3tl.16` and `polylogue-3tl.4` without pretending to complete their broader blocked architecture.

Deliver:

1. `docs/public-claims.yaml` with statuses `proven`, `capability`, `aspirational`, and `retired`, plus an orthogonal evidence class;
2. a linter that scans declared public surfaces and rejects unknown claim IDs, retired primary wording, missing evidence refs, and present-tense publication of aspirational claims where practical;
3. a generated or checked public finding page for the claim-vs-evidence field observation;
4. a deterministic synthetic reproduction lane separated from the private field observation;
5. a route or page manifest suitable for the docs site;
6. tests for stale artifacts, missing provenance fields, unsupported quantitative claims, and malformed sample/caveat metadata.

The field finding must preserve exact scope: a bounded origin-stratified sample of 5,000 structured failures from a frame of 42,033, with 1,205 silent-proceed next-turn cases, 3,375 ambiguous cases, a 24.1% lower bound, and the existing marker calibration caveats. Do not generalize it to all agents or models.

Inspect `docs/proof-artifacts.md`, `devtools/demo_packet.py`, `.agent/demos`, page generation, docs drift checks, and Beads. Prefer a simple auditable YAML + generator over a large premature finding database. Make future migration to first-class finding objects explicit.

Produce a patch, generated page, claims inventory, and verification receipt under `/mnt/data/polylogue-claims-findings/`. Return links and list claims that remain unsupported.
