# Read connections

Session catalogs, Sessions and Messages reuse prepared statements on their
read connections. Every visibility check still runs with current parameters:
generation, identity and removal, plus archive excision for Sessions and
catalogs. No check result is cached, and the connections are not held open in
a read transaction.

The selected content snapshot is separate from these checks. A reader that has
already pinned content still fails if the Message is hidden, the removal epoch
or generation changes, or a new archive excision is recorded.
