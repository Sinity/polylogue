# Message rendering and actions

The message card is the most important MK3 component. It must present all useful data without turning the transcript into a debugger.

## Message card anatomy

Header row:

- role chip: user, assistant, tool, system, thinking, protocol.
- message ordinal and timestamp.
- model/token/cost mini chips when known.
- topology chip if this message is a branch/fork/parent target.
- data quality chip: canonical, inferred, heuristic paste, raw-only, partial.
- action rail: copy, mark, annotate, open raw, more.

Body:

- content segments rendered in stable order.
- text segment: normal readable transcript text.
- code segment: folded by default when long; copy block; language if known.
- thinking segment: folded and muted; visible only when provider supports it.
- tool use/result segment: command/operation summary first, raw payload behind fold.
- paste segment: folded block with label, size, confidence, and copy modes.
- attachment segment: inline card with preview state.

Footer:

- source/provenance hint.
- message-level tags/marks/notes indicators.
- “copied”/pending/error ephemeral feedback.

## Copy actions

`y` opens the copy menu for the active target. `Shift-y` copies the default representation directly.

Conversation target defaults:

- copy title + permalink.
- copy Markdown transcript excerpt.
- copy context-image seed when available.
- copy raw provenance pointer.

Message target defaults:

- copy text.
- copy markdown with role header.
- copy prompt only for user messages.
- copy typed-only, if paste spans exist.
- copy paste-only, if paste spans exist.
- copy raw JSON, if raw artifact/message provenance exists.
- copy permalink / anchor.

Content-block target defaults:

- copy code block.
- copy tool input.
- copy tool output.
- copy thinking block, if visible.

Attachment target defaults:

- copy filename.
- copy metadata.
- copy extracted text.
- copy path, gated by privacy/redaction setting.
- open containing raw artifact.

Clipboard states:

- ready: action enabled.
- denied: browser clipboard denied; show fallback textarea.
- partial: selected representation omits folded/hidden blocks; state this in toast.
- unsafe: raw HTML/unsafe attachment cannot be copied as rendered; text/raw only.

## Mark and annotation actions

Marks are quick state changes: star, pin, important, read-later, archive. They sit beside tags but are not tags. Tags classify; marks drive workflow.

Annotations are notes on a target. The first UI should support conversation and message notes; content-block/attachment notes can be disabled with a clear “target identity not implemented” reason.

Optimistic mutation state:

- clicked → pending chip.
- success → settled chip.
- idempotent no-op → no visual jump, but toast says already marked/unmarked if the user explicitly requested it.
- failure → revert and show structured error.

## Folding policies

Fold by default:

- tool output longer than 30 lines.
- thinking blocks.
- paste blocks longer than 20 lines.
- file-read/file-search results.
- assistant messages over configured threshold when in dense mode.

Never fold by default:

- short user prompts.
- message summary headers.
- small assistant answers.
- active search match lines.

Fold affordance text should include why: “tool output · 1,244 lines · copy,” “paste · heuristic · 8,103 chars,” “thinking · Claude · hidden by default.”

## Search match behavior inside messages

Search matches should open the relevant folds enough to show the match, but not expand the whole block. For giant paste/tool blocks, show a match window with 3–5 surrounding lines and a jump-to-full control.

## Accessibility and keyboard

Every message card has a stable focus target. Arrow keys move through messages; `[` and `]` move through search matches; `f` folds/unfolds current segment; `m` marks; `a` annotates; `y` copies; `r` opens raw/provenance; `l` opens lineage context.
