# Shared crate releases

Polylogue owns `crates/hashed-blobs` and `crates/credential-output`; Sinex uses both. A release is an immutable Git tag in `https://github.com/Sinity/polylogue` named `<crate>-v<version>`, such as `hashed-blobs-v0.1.1` or `credential-output-v0.1.0`. `publish = false` keeps them off crates.io. Never move a tag once it exists; release a new version instead.

Both crates are independent of the product workspace: edition 2021, explicit dependencies, minimum stable Rust 1.98.1 (the Nix devshell supplies it). `hashed-blobs` uses Unix filesystem APIs, is tested on x86_64 Linux, and needs a C compiler for zstd. Neither needs a Sinex service or database.

`docs/shared-crate-source.json` records where `hashed-blobs` came from in the spec repository. `credential-output` was extracted from Polylogue's masking module at commit `272befd54fc95c3e5c5f9783fabd1b7b176de6ab`; its test catalog records its own source in `tests/fixtures/credential-output-v1`.

## Check a release

On a clean, committed checkout, inside `nix develop`:

```sh
cargo package --locked -p hashed-blobs -p credential-output
```

This packs each crate, unpacks it outside the workspace and builds it on its
own, which catches missing files and undeclared dependencies. The crates'
tests run in the full suite (`cargo run -p xtask -- check`), and
`xtask boundaries` rejects internal or SQLite dependencies in either crate.

## Publish

Merge the release commit through the normal route and note its SHA. If later merges touch only other files, compare the crate's files and rerun the check on the commit you will tag. Check the tag is free right before creating it:

```sh
git ls-remote --tags origin refs/tags/hashed-blobs-v0.1.2 'refs/tags/hashed-blobs-v0.1.2^{}'
git tag -a hashed-blobs-v0.1.2 <release-sha> -m 'hashed-blobs 0.1.2'
git push origin refs/tags/hashed-blobs-v0.1.2
git ls-remote --tags origin refs/tags/hashed-blobs-v0.1.2 'refs/tags/hashed-blobs-v0.1.2^{}'
```

If the tag already exists with different content, pick a new version. Confirm the peeled remote target is the release SHA before updating consumers.

## Update Sinex

Sinex pins `hashed-blobs` by tag in `crate/sinex-store/Cargo.toml` and `credential-output` by the tagged release commit in `crate/sinexd/Cargo.toml`:

```toml
hashed-blobs = { git = "https://github.com/Sinity/polylogue", tag = "hashed-blobs-v0.1.1" }
credential-output = { git = "https://github.com/Sinity/polylogue", rev = "<release sha>" }
```

To move to a new release:

1. Change the tag or rev in the pin. Never add a vendored copy or a path dependency beside it.
2. Run `cargo update -p <crate>`. The lock must record the Git URL and the release SHA.
3. The Sinex Nix package sets `cargoLock.allowBuiltinFetchGit = true`, so Nix fetches Git dependencies at the locked SHA and needs no hash entry.
4. For `hashed-blobs`: keep `Reader` `Send + Sync` (Sinex holds it in an `Arc`), and use `put_plain_checked` wherever a caller promises cancellation.
5. Run the affected Sinex tests and the Sinex `package` operation, then land the change through Sinex's normal route.
