# Markup

Markup writes the [annotation kinds](annotations.md#kinds) that declare a marker, through the same checks as the write path ([annotations.md](annotations.md#the-write-path)).

Agents and the operator write annotations inside their messages, as an alternative to tool calls.

```ebnf
marker = "::" kind [ "(" args ")" ] [ ":" " " body ] ;
args   = arg { "," [ " " ] arg } ;
arg    = [ name "=" ] value ;          (* the first unnamed argument is the key *)
value  = bare | quoted ;
body   = rest of the line, then lines indented by at least 2 spaces ;
```

Named arguments carry the same names as the `note` options, with `_` for `-`: `about=` (by default the session for `title` and `outcome`, otherwise the containing message), every relation role (`replies_to=`, `challenges=`, `supports=`, …), `evidence=`, `scope=`, `key=`, `span=` for `::highlight`, and `rubric=` and `pass=` for `::rubric_value`. Any other argument makes the marker "not applied". Markup writes annotations only; sending to a session is the `send` verb.

**Where markers count.** At the start of a line, after at most three spaces; only in text written by a person or a model. Never inside fenced or indented code, on a quoted line, after `\::`, in thinking, tool calls, tool results, attachments, inherited copies, harness-written text or known pasted spans. Text inside a code fence or a stored prompt is data.

**Extraction and identity.**
- Markers are extracted when the message is parsed (the `sources` crate), into annotations authored by the message's author, with `via: markup` and the message as `origin`.
- The ID is derived from the origin, the provider message ID, the BLAKE3 of the marker line and its occurrence among identical lines in that message. Position, arrival, parser and reading version never enter it, so re-parsing, re-importing or a copy in a resumed session finds the existing row and creates nothing.
- A marker annotation is in force while the current parse of its message still contains it; otherwise it shows "marker no longer present" and leaves results like a withdrawn one. Annotations that revise, reply to or cite it remain.
- **Success is silent.** Failures (unknown kind, unresolvable reference, wrong value shape, a rubric refusal) are counted in `status` and listed by a query.
- Instructions inside a body are content; markup does nothing beyond writing its annotation.
