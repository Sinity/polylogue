# Python client

Install with `pip install ./clients/python`, beside an installed `polylogue`
CLI. The client needs only Python's standard library at runtime. It invokes
CLI JSON with an argument list. Generated continuation requests go directly to the same daemon socket, retaining their structured fields.

```python
from polylogue.client import Client, Refused

client = Client()  # Or Client("/path/to/archive", executable="/path/to/polylogue")
page = client.query("messages where parser", page_size=100)
rows = page.data["value"]["reply"]["rows"]
if page.data["value"]["reply"].get("next") is not None:
    following = client.query("messages where parser", cursor=page.data["value"]["reply"]["next"], page_size=100)

try:
    session = client.read("session:claude-code:example")
    report = session.json()["value"]["reply"]["steps"][0]
    page = report["outcome"]["reply"]["reply"]
    ready = page["document"]["footer"].get("next")
    if ready is not None:
        following = client.follow_read(ready)
except Refused as error:
    print(error.kind, error.answer.json())
```

`polylogue.models.Answer` is frozen, including its nested maps and lists.
`answer.json()` returns an independent plain JSON value. All fields, nulls,
integer precision and complete refusals survive. Regular calls keep the whole
shared `outcome`/`value` envelope. Empty and degraded answers keep their payloads;
refusals expose the complete error answer. Query cursors belong to query pages.
Read continuations use `client.follow_read(request)` for generated `rows` and
`attachment` actions, including previous pages. A listing is the `rows` reply with `kind: "listing"`. It preserves the original
arguments, request ID, cursor and omitted-versus-null deadline/budget fields,
and rejects unsupported operation kinds. The daemon checks request options against its declarations. A socket refusal has `returncode=None`; CLI refusals retain their process exit code.
Use `client.read(identity, page_tokens=600)` to choose the number of tokens transferred in one page.

`client.call(*arguments)` takes the same words as a CLI command, without
`polylogue --json`. It covers the other declared operations. A write is sent
once. For retries, give the same `--request-id` and identical arguments; the
client never invents a second write after a lost answer. An explicit archive is
passed as the CLI's `--archive` argument for regular calls and change feeds, and selects `ARCHIVE/daemon.sock` for continuations. Without an explicit archive, the status reply supplies that path.
Relative paths are made absolute when `Client` is constructed; declared path
aliases are preserved, including short aliases used for Unix sockets.

```python
for line in client.feed(previous_position):
    record = line.data
    if record["type"] == "begin" and record["replacement"]:
        clear_local_view()
    elif record["type"] == "change":
        apply_change(record["component"], record["change"])
    elif record["type"] == "complete":
        save_position(record["position"])
```

Save a position only after applying the earlier lines in that batch. An error
or interrupted iterator leaves the previous position in force; a new index
generation asks the consumer to replace its local view. Breaking out early
requires closing the iterator (for example with `contextlib.closing`).

Contract tests run through the repository's `check` operation: a real daemon
and CLI over a temporary archive are compared with the Python client.

`client.stream(*arguments)` yields the CLI's finite line events as plain dicts with
`event` and `data` keys. Each requested read has its own `begin`, `record` and
`complete`; a file or directory delivery also emits a `written` receipt after its validated completion. `steps` carries typed step reports and undo lines once, and `answer`
ends the whole line. A failed step raises `Refused` after the preceding events
have been delivered. Its `answer` retains the complete failed-step report.
Closing the iterator terminates and reaps the CLI process.

`status` returns one text path and count summary per watched root in `store.capture_coverage`. Call `client.call("status", "--details")` for pattern entries and unread field details. `client.call("status", "--log")` also includes grouped `slow_statements`; `--wait D --after-changes N` waits beyond an earlier reply's `changes` count.
