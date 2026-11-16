# Markup

Markup writes the [annotation kinds](annotations.md#kinds) that declare a marker, through the same checks as the write path ([annotations.md](annotations.md#the-write-path)), places them on pages, and sends messages to bracketed addresses ([Sends](#sends)).

Agents and the operator write annotations inside their messages, as an alternative to tool calls.

```ebnf
marker = "::" kind [ "(" args ")" ] [ ":" " " body ] ;
args   = arg { "," [ " " ] arg } ;
arg    = [ name "=" ] value ;          (* the first unnamed argument is the key *)
value  = bare | quoted ;
body   = rest of the line, then lines indented by at least 2 spaces ;
```

Named arguments carry the same names as the `note` options, with `_` for `-`: `about=` (by default the session for `title` and `outcome`, otherwise the containing message), every relation role (`replies_to=`, `challenges=`, `supports=`, …), `evidence=`, `scope=`, `key=`, `span=` for `::highlight`, and `rubric=` and `pass=` for `::rubric_value`. `page=` and `section=` are accepted on any marker ([Pages](#pages)). Any other argument makes the marker "not applied".

**Where markers count.** At the start of a line, after at most three spaces; only in text written by a person or a model. Never inside fenced or indented code, on a quoted line, after `\::`, in thinking, tool calls, tool results, attachments, inherited copies, harness-written text or known pasted spans. Text inside a code fence or a stored prompt is data.

**Extraction and identity.**
- Markers are extracted when the message is parsed (the `sources` crate), into annotations authored by the message's author, with `via: markup` and the message as `origin`.
- The ID is derived from the origin, the provider message ID, the BLAKE3 of the marker line and its occurrence among identical lines in that message. Position, arrival, parser and reading version never enter it, so re-parsing, re-importing or a copy in a resumed session finds the existing row and creates nothing.
- A marker annotation is in force while the current parse of its message still contains it; otherwise it shows "marker no longer present" and leaves results like a withdrawn one. Annotations that revise, reply to or cite it remain.
- **Success is silent.** Failures (unknown kind, unresolvable reference, wrong value shape, a rubric refusal) are counted in `status` and listed by a query.
- Instructions inside a body are content; markup does nothing beyond writing its annotation, its slot or its send.

## Pages

- **`page=` and `section=`** on any marker write the annotation and a slot placing it on that page, under that section, in one request: `::finding(page=flaky-ci, section=Findings): weekday arithmetic used >= instead of >`. The slot's ID derives from the marker's ID, so re-parsing writes nothing new. A section title that matches no heading creates the heading at the end of the page ([pages.md](pages.md#the-model)).
- **`::heading(page=X, level=2): Name`** creates a section; `section=S` on it places it under another heading.
- **`::slot(page=X, section=S): REF`** places an existing reference, the body being the reference.
- A marker that replies to a delivery sent with `--page` and `--section`, and names no page itself, is placed in that section ([pages.md](pages.md#reply-placement)).

## Sends

A bracketed address at the start of a line sends the rest of the line, and its indented continuation, to that address, under the same rules for where markers count:

| Marker | Sends to |
|---|---|
| `@[codex:019d0f2a] rebase on master first` | the session a reference names |
| `@[@reviewer] look at the findings` | the session a member alias resolves to ([pages.md](pages.md#members)) |
| `@[sessions where repo:morrow running:true] stop and commit` | the running sessions the query selects, as `send --all` does |

- **Once per marker.** The send's ID derives from the marker as an annotation's does (origin, provider message ID, the line's hash and its occurrence), so re-parsing, re-importing, quoting or translating the message never sends twice. Sending the same text again is a new message.
- A bare `@reviewer` without brackets is plain text, so nobody sends by accident.
- The sender is the message's author; delivery, recording and outcomes are those of `send` ([session-control.md](session-control.md#send)). A send that cannot resolve its address is counted with the other marker failures.
