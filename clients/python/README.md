# Python client

Install with `pip install ./clients/python`, beside an installed `polylogue`
CLI. The client needs only Python's standard library at runtime. It invokes
CLI JSON with an argument list and uses the same daemon as other clients.

```python
from polylogue.client import Client, Refused

client = Client()  # Or Client("/path/to/archive", executable="/path/to/polylogue")
page = client.query("messages where parser", page_size=100)
rows = page.data["rows"]
if page.data["next"] is not None:
    following = client.query("messages where parser", cursor=page.data["next"], page_size=100)

try:
    session = client.read("claude-code:example")
except Refused as error:
    print(error.kind, error.answer.json())
```

`polylogue.models.Answer` is frozen, including its nested maps and lists.
`answer.json()` returns an independent ordinary JSON value. All fields, nulls,
integer precision and complete refusals survive. A result page's `next` is the
cursor for its next page; an empty result remains an empty result.

`client.call(*arguments)` takes the same words as a CLI command, without
`polylogue --json`. It covers the other declared operations. A write is sent
once. For retries, give the same `--request-id` and identical arguments; the
client never invents a second write after a lost answer.

```python
for line in client.changes(previous_position):
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
