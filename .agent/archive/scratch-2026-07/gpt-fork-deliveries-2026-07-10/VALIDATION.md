# Extraction Validation (2026-07-11)

## Result

**PASS after one bookkeeping correction.** The spool contains 22 distinct ChatGPT captures.
Folders 01-22 represent them exactly once; folder 00 is explicitly synthetic and contains the
shared branch prefix once. `INDEX.md` incorrectly called this 21 real conversations and had an
ambiguous summary-only count; the corrected partition is 10 complete-inline + 2 dual + 8
summary-only + 2 needs-regeneration = 22.

## Checks

- Parsed all 22 source JSON files and matched each `session.provider_session_id` to the
  conversation URL and numbered escrow folder. No duplicate provider session IDs or missing
  numbered folders were found.
- Confirmed the 11-member shared-root set named in folder 00: `6a506bcf`, `6a5112e7`,
  `6a5112f3`, `6a5112f5`, `6a5112f9`, `6a5112fd`, `6a5113e8`, `6a511407`, `6a51140b`,
  `6a511416`, and `6a511425`. Folder 00 declares itself non-independent; downstream folders
  retain only their divergent delivery, avoiding false inflation of the capture count.
- Read every escrow Markdown file as UTF-8. All 23 folders contain readable `STATUS.md` and
  `delivery.md`; folders 01-22 also contain `inline-artifacts.md`.
- Enforced the PRIVATE rule for folder 12: it exists only under project scratch and has no copy
  under `/realm/inbox/gpt-pro-sol/sinex-fork-deliveries`. Its status explicitly prohibits
  staging or quotation. No PRIVATE slug or source ID occurs in the Sinex staging tree.
- Compared all nine Sinex-scoped staging pairs (06-11, 18, 21, 22). Every staged `delivery.md`
  and `inline-artifacts.md` is byte-identical to the escrow copy; no `STATUS.md` or Polylogue /
  PRIVATE fork was staged.
- Source provenance remains intact: the source directory, conversation URLs, provider session
  IDs, recovery classification, and lost-artifact caveats are recorded in `INDEX.md` and the
  per-fork statuses. Source SHA-256 prefixes below permit quick identity checks without copying
  source content into the escrow.

## Source Identity

| Folder | Provider session ID | Source SHA-256 prefix |
|---|---|---|
| 01 | `6a4433f6-8604-83eb-a40c-7cc2f641e158` | `5f70ae183eab9a9e` |
| 02 | `6a4ebe8b-8f50-83ed-9867-7b77a8e567b0` | `ed7c4972d4d90650` |
| 03 | `6a506bcf-852c-83eb-82e6-e23ac8a418e1` | `369e28d05bc5a4e8` |
| 04 | `6a506c24-ec1c-83eb-b97d-0b78b8b69833` | `b7b795541c793e7b` |
| 05 | `6a507c1a-940c-83eb-9600-f8449aeda538` | `7770787fec98a8e5` |
| 06 | `6a507e40-46bc-83eb-9c9f-0e0815dc142a` | `c4377216b5ed4b43` |
| 07 | `6a507e8f-49d8-83eb-9fdc-29a4761588a0` | `3e96781678e21df2` |
| 08 | `6a507e9d-c600-83ed-8f9a-3eece5fea417` | `154334dcd40b4ee0` |
| 09 | `6a507eaf-a91c-83eb-afed-39b3c2033f77` | `855028cb76b81a00` |
| 10 | `6a507ebd-5df8-83ed-87f6-5cf829136f00` | `b7b4135403b29435` |
| 11 | `6a507ecd-96b8-83eb-ad31-56a9df0fa81d` | `4e70f48d5d028ae9` |
| 12 | `6a50b7cc-0b24-83eb-bd15-2edadd846f2b` | `cb685b02532cc4b6` |
| 13 | `6a5112e7-46a0-83eb-8486-d0865595094d` | `094164cb763f9d39` |
| 14 | `6a5112f3-7798-83eb-b1ee-96ef29477c12` | `b96928715c264dab` |
| 15 | `6a5112f5-d4c8-83eb-8efd-84aa9ace5836` | `3430d17d20320a8d` |
| 16 | `6a5112f9-f200-83ed-a96e-a083dd7d4a46` | `cc5e2092042f72bc` |
| 17 | `6a5112fd-1ee0-83eb-b12a-27317f452bc5` | `1c259a2b65969c93` |
| 18 | `6a5113e8-3f48-83ed-931a-da42be13baea` | `d99c0cb823efd50b` |
| 19 | `6a511407-631c-83eb-b5e4-9fc3b6852eee` | `d0153ff9f6360421` |
| 20 | `6a51140b-8f70-83eb-bbff-c2e9c1ee77a8` | `e25d3109e782141f` |
| 21 | `6a511416-6d5c-83eb-9a00-159fa983aabf` | `73c1c6baff992cf7` |
| 22 | `6a511425-dc34-83eb-a46b-83d6790f4c90` | `86bbc9bb79c1305b` |

## Suggested Beads Surgery

Do not create implementation beads from lost sandbox claims alone. Existing Beads already own
the renderer (`polylogue-ap7`), cockpit (`polylogue-bby` / `polylogue-37km`), Demo Packet v2
(`polylogue-212.12`), query DSL (`polylogue-fnm`), context-memory loop (`polylogue-37t`), and
launch/claims work (`polylogue-3tl`). Attach the corresponding escrow folder as research input
to those owners only after checking it against current source.

One Polylogue-targeted finding appears unrepresented: fork 02 reports a naming collision with
the active `polylogue.page` agent product and recommends deciding the public name before broad
release. Suggested new decision bead: **"Decide Polylogue public naming before launch"**, owned
under `polylogue-3tl`, with AC to verify current trademark/domain/package/repository collisions,
compare migration cost and discoverability, record keep-vs-rename judgment, and block only the
public-launch slice rather than product development. Treat the fork's collision claim as dated
research evidence, not established current fact.

Fork 05's NARROW verdict and 30-day stop gate should be appended as decision evidence to the
existing strategy/launch owner rather than becoming another feature bead. The remaining
Polylogue findings are overlaps or artifact-loss notices, not new work units.
