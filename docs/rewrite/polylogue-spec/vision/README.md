# Vision: what grows on Polylogue

**In short.** A catalog of uses for Polylogue: questions the operator and
agents ask, acts they take, routines and artifacts that should grow on the
archive of AI conversations and agent sessions. It exists to judge whether
the substrate is worth building and general enough. In the operator's words:
"design / specify things that are supposed to 'grow' on top of this
substrate. So we know the substrate is actually going to be helpful", and
"expand to cover all kinds of possibilities, applications etc. not
necessarily intending to implement everything like that -- but having really
many things designed" (2026-10-12). `uses.toml` is the catalog;
`check_uses.py` (flake check `polylogue-vision`) checks it against the spec
and prints the gaps. The Sinex catalog (`sinex-spec/vision/`) has the same
form, and uses that need both systems go to `polylogue-sinex-joint/vision/`.

## The entry form

| Field | Meaning |
|---|---|
| `id`, `title` | short slug and name |
| `act` | the question asked or the act taken, in the operator's words where seeded |
| `level` | how solid the entry is (below) |
| `query` | level 1 and up: the exact query in Polylogue's query language (`spec/query.md`) |
| `output` | level 1 and up: the exact text or table it prints |
| `source_note` | level 1 and up: where each shown figure comes from |
| `degrades` | level 1 and up: what it shows when data is missing, unknown or excised |
| `expected` | level 2: the answer on the Morrow archive (`kind` = `rows`, `count`, `groups`, `agg` or `notes`, and `value`); `world = "live"` with a `clock` evaluates against the live-state home instead (`examples/archive/FACTS.md`, live set), where sessions are named V1–V3 |
| `reads` | query units it reads (the units `spec/query.md` declares) |
| `needs` | substrate features, from the closed list in `[features]`, each owned by a spec heading |
| `gaps` | what the design lacks for it; each is fixed in `spec/`, set aside, or joint (below) |
| `shows` | what the operator or agent gets |
| `mark` | `intended` (the operator asked for it), `plausible` (follows from what they asked), `speculative` (generated) |
| `origin` | `seeded` (cited: a chat, a workflow, a real question, a scene) or `expanded` (generated beyond the sources) |
| `cite` | where a seeded use comes from |
| `prototype` | a prototype that exercises it, if any |

## Who meets it: the consumer test

An intended use names its consumer: `agent` (an agent asks or acts through it), `push` (a briefing, signal or status brings it), `workflow` (a step in the operator's work), or `view` (an intentionally opened interactive surface). A requested reader, comparison or investigation can qualify through direct use; it does not first need a briefing or notification. `check_uses.py` checks that a consumer is named, separately from the catalog's evidence level.

The May 2026 “Substrate without Surface” diagnosis remains a warning about useful work that nobody encounters. Favor low-effort discovery and contextual links where they help. Demonstrate the intended task through the actual route; neither unsolicited delivery nor a compiling page establishes usefulness. An unused surface calls for reviewing its purpose and discoverability before a recorded retirement, rather than automatic deletion after a deadline.

## Levels

Like the spec, the catalog is written at several levels of solidity:

| Level | What it adds | Checked by |
|---|---|---|
| 0 | the catalog line | the form, the features and the units exist |
| 1 | `query`, `output`, `source_note`, `degrades` | the query parses in the full language: the reference evaluator's parser (`prototypes/query-ref`) with the field names `spec/query.md` declares |
| 2 | `expected` | the query runs under the reference evaluator on the Morrow archive and equals the stated answer |
| 3 | a real-data prototype | `prototype` names it |

Every `intended` use is at level 1 or higher, or carries a gap that begins
"query language cannot express": a question the language cannot write down is
the most valuable kind of gap, so it is recorded rather than covered by a vague
query. Plausible and speculative uses stay at level 0 until they earn more.
The level-2 answers use the reference evaluator's subset of fields; a level-1
query may use any declared field.

## Sources of the seeded entries

- The operator's web chats and notes (`polylogue-sinex-joint/vision/uses-from-chats.md`,
  conversation, agent and context uses; and the two uses tagged `both` in the
  Sinex section).
- `spec/workflows.md`: every workflow.
- `evidence/real-questions.md`: what the operator and agents actually asked of
  past sessions.
- `evidence/meta-vision.md`: ten scenes worked backwards to the spec.

## Gaps and what happens to each

Every gap carries a disposition in `uses.toml` (`[[disposition]]`, the form
the Sinex catalog uses), and `check_uses.py` refuses a gap without one, a
`fixed`, `planned` or `by_design` one that cites no existing heading, a
`set_aside` one that cites no existing `SET-ASIDE.md` row, and a `joint` one
that names no use in the joint catalog. Counts are uses needing the gap;
`python3 vision/check_uses.py --gaps` lists them.

| Gap | Uses | Disposition |
|---|---|---|
| No entity identity: a person, project or concept mentioned across sessions is not an entity | 8 | **set aside** with Sinex's knowledge-graph layer (the operator, 2026-05-25); the way back is identity only from shared identifiers or an operator confirmation; repositories already join through `repo` and `cwd`; a joint gap too |
| Stated plans and todo lists are not units | 5 | **set aside**: a `plans` unit |
| No outgoing surface: nothing is surfaced unasked | 4 | **fixed**: signals, a watch with a delivery rule in the shape both projects share (`spec/user-data.md`; `polylogue-sinex-joint/prototype/signals/`) |
| Generation is outside the substrate | 2 | **by design** (`spec/purpose.md`, not in scope): drafting in the operator's voice is a client; Polylogue supplies the examples |
| No dataset export for training or evaluation | 2 | **set aside** |
| Query language cannot cluster rows by meaning | 1 | **planned**: `cluster by near` (`spec/query.md`, planned extensions) |
| Query language cannot group by a derived key, or by a text fingerprint | 2 | **planned**: derived grouping keys `program(command)`, `day(at)`, `fingerprint(text)` |
| Query language has no series statistics over grouped periods | 1 | **planned**: `| series` (median, percentiles, longest empty run, halves), complete periods only |
| Query language cannot give the distance to the next matching row | 1 | **planned**: `gap_steps` and `gap_time` on `seq` match rows |
| Query language cannot group by commit; session counts measure activity, not velocity | 2 | **planned**: commits as a unit (`spec/analysis.md`, git correlation) |
| Query language cannot read a number inside tool output | 1 | **planned**: `extract number` |
| No Codex prices | 1 | **planned** for slice 3; until then a displayed total leads with its unpriced share (`spec/analysis.md`) |
| A failure rate counting unknown outcomes as success | 1 | **fixed**: failure rates are bounds that count unknown and pending (`spec/analysis.md`, failure statistics) |
| Pasted context counted as authored | 1 | **specified**: separate provider reports and clipboard candidates, with honest byte proxies; text origin remains unknown without causal evidence (`spec/model.md`, Text-origin evidence; joint #30) |
| No index of code facts across sessions | 1 | **set aside** |
| No speech source | 1 | **joint**: Sinex captures speech; Polylogue reads its transcript segments as context (joint catalog `speech-as-source`) |
| Forecasts have no automatic resolution | 1 | **planned**: calibration, a later feature (`spec/analysis.md`, measures) |

Fixed in the spec while building the catalog: the `distillate` note kind
(`spec/user-data.md`, the most repeated request in the chats), a commit's span
(`spec/analysis.md`), and `source` as a notes field (`spec/query.md`; a
level-1 query needed it and the check refused it).
