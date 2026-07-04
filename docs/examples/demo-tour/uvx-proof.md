# Demo Tour Install-Path Proof

This proof was run from the local package source using `uvx --from <repo>` as
the pre-release equivalent of the public command:

```bash
uvx --from <repo> polylogue demo tour --out-dir <tmp-output> --force --format json
```

Result:

- Status: passed
- First query result: 4.636s
- Full tour: 10.180s
- Installed package set: 82 packages
- Demo archive: 11 sessions, 43 messages, overlays present
- Query/read steps: archive facets, pytest evidence drilldown, direct session
  evidence by id, query facets

The release-path command is:

```bash
uvx polylogue demo tour
```
