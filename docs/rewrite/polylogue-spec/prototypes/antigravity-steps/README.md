# Antigravity steps, decoded with Antigravity's own schema

Antigravity's language server binary embeds the serialized descriptor of
every `.proto` file it was built from. `extract.py` pulls them into a
`FileDescriptorSet`, kept outside the repository because the schema is
Google's. `wire.py` decodes protobuf bytes by a message name from that set,
without a descriptor pool, so files with missing dependencies still decode.
`account.py` maps the CLI trajectory stores' steps to Polylogue messages and
prints counts only.

```sh
nix shell --impure --expr 'with import (builtins.getFlake "nixpkgs") {}; python3.withPackages (p: [p.protobuf])'
python3 -I extract.py <antigravity-ide>/lib/antigravity-ide/resources/app/extensions/antigravity/bin/language_server_linux_x64 $TMPDIR/agy.fds
AGY_FDS=$TMPDIR/agy.fds python3 account.py ~/.gemini/antigravity-cli/conversations
```

There is no flake check: it needs the installed binary and real stores.
Results are in `evidence/antigravity-steps-decoded.md`; the mapping is in
`spec/sources/others.md`, Antigravity.
