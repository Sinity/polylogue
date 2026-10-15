# Shared file layer release

Polylogue owns `crates/hashed-blobs`. The first proposed release is version `0.1.0`, tagged `hashed-blobs-v0.1.0` in `https://github.com/Sinity/polylogue`. This is a release candidate; the tag and downstream pin have not been published. Releases use owner Git tags, with `publish = false` preventing crates.io publication.

The crate is independent of the product workspace: its package metadata and dependencies are explicit, its edition is 2021, and its declared minimum is stable Rust 1.98.1. The locked Nix development shell supplies that compiler for qualification. This is the supported minimum being tested, not a claim that every earlier compiler fails. Its implementation uses Unix filesystem APIs; qualification covers x86_64 Linux. Native zstd compilation needs a C toolchain. No Sinex service, database, runtime or optional backend is needed.

`docs/shared-crate-source.json` records the original and read-only adaptation bytes. Release preparation changes package metadata only; API, implementation, tests and wire formats remain unchanged. Existing users retain the same `hashed_blobs` import and file encodings.

## Qualification

Run the declared operation on a clean, committed candidate:

```sh
agentctl job start <owner-checkout> shared_file_layer_check --workspace <owner-checkout>
```

It packages the crate, extracts the archive outside the workspace, checks the packaged license and source, rejects backend dependencies in the resolved standalone graph, and runs the extracted crate's tests. A disposable clone receives a local tag at the exact candidate, and a separate Cargo consumer resolves that tag and writes/reads synthetic plain and sealed bytes. It runs with defaults and with defaults disabled. The clone, tag, consumer and absolute local dependency URL are temporary qualification inputs; none becomes a product dependency. This proves package and Git-consumer mechanics. It does not prove that the proposed remote tag exists or replace downstream product checks.

## Publication boundary

The release commit must first pass owner review and the normal protected publication route. Preserve its exact qualified SHA. If integration changes only unrelated product files, compare the crate bytes and rerun the focused check on the resulting release target rather than claiming the earlier Git resolution used that new SHA. Check the remote namespace again immediately before creating the tag:

```sh
git ls-remote --tags origin refs/tags/hashed-blobs-v0.1.0 'refs/tags/hashed-blobs-v0.1.0^{}'
git tag -a hashed-blobs-v0.1.0 <reviewed-owner-release-sha> -m 'hashed-blobs 0.1.0'
git push origin refs/tags/hashed-blobs-v0.1.0
git ls-remote --tags origin refs/tags/hashed-blobs-v0.1.0 'refs/tags/hashed-blobs-v0.1.0^{}'
```

These are future publication commands, not actions performed by preparation. An existing tag with different bytes requires a new version; never move an already consumed tag. Verify the peeled remote target before consumer replacement.

## Sinex replacement

After the real tag resolves to the reviewed owner commit, replace this dependency in `crate/sinex-store/Cargo.toml`:

```toml
hashed-blobs = { git = "https://github.com/Sinity/polylogue", tag = "hashed-blobs-v0.1.0" }
```

In the same downstream change, delete `vendor/hashed-blobs`, remove its root-workspace `exclude`, retire the vendor-source record and update its current replacement documentation. Regenerate the downstream lock with `cargo generate-lockfile`, then add `cargoLock.outputHashes."hashed-blobs-0.1.0"` to the downstream Nix package. Use a temporary `pkgs.lib.fakeHash` and take the expected fixed-output hash from the downstream declared build operation; replace the placeholder and rerun that build. The lock must record the remote Git URL, tag query and exact reviewed owner SHA. Run the existing affected Sinex store/file-layer tests and real downstream Nix build with its infrastructure features disabled by default; do not preserve a path patch or vendor fallback beside the Git dependency. Review and publish that connected downstream change through its normal route.

Until owner publication and this consumer replacement pass, Sinex's qualified vendor copy remains a recorded unresolved release gate. Local clone trials cannot close it.
