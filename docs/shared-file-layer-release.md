# Shared crate releases

Polylogue owns `crates/hashed-blobs` and `crates/credential-output`. Releases use immutable owner Git tags in `https://github.com/Sinity/polylogue`, with `publish = false` preventing crates.io publication. Each crate has its own version and tag namespace. Verify the remote tag and locked consumer commit before claiming a particular release is consumed.

The `hashed-blobs` crate is independent of the product workspace: its package metadata and dependencies are explicit, its edition is 2021, and its declared minimum is stable Rust 1.98.1. The locked Nix development shell supplies that compiler for qualification. This is the supported minimum being tested, not a claim that every earlier compiler fails. Its implementation uses Unix filesystem APIs; qualification covers x86_64 Linux. Native zstd compilation needs a C toolchain. No Sinex service, database, runtime or optional backend is needed.

`docs/shared-crate-source.json` records the original and read-only adaptation bytes. Release 0.1.1 repairs malformed seek-map refusal and existing-destination custody, adds bounded callback verification, and documents partial batch installation. The writer encoding and `hashed_blobs` import remain unchanged. General zstd encodings outside the documented writer layout remain unsupported.

## Qualification

Run the declared operation on a clean, committed candidate:

```sh
agentctl job start <owner-checkout> shared_file_layer_check --workspace <owner-checkout>
```

It packages the crate, extracts the archive outside the workspace, checks the packaged license and source, rejects backend dependencies in the resolved standalone graph, and runs the extracted crate's tests. A disposable clone receives a local tag at the exact candidate, and a separate Cargo consumer resolves that tag and writes/reads synthetic plain and sealed bytes. It runs with defaults and with defaults disabled. The clone, tag, consumer and absolute local dependency URL are temporary qualification inputs; none becomes a product dependency. This proves package and Git-consumer mechanics. It does not prove that the proposed remote tag exists or replace downstream product checks.

## Publication boundary

The release commit must first pass owner review and the normal protected publication route. Preserve its exact qualified SHA. If integration changes only unrelated product files, compare the crate bytes and rerun the focused check on the resulting release target rather than claiming the earlier Git resolution used that new SHA. Check the remote namespace again immediately before creating the tag:

```sh
git ls-remote --tags origin refs/tags/hashed-blobs-v0.1.1 'refs/tags/hashed-blobs-v0.1.1^{}'
git tag -a hashed-blobs-v0.1.1 <reviewed-owner-release-sha> -m 'hashed-blobs 0.1.1'
git push origin refs/tags/hashed-blobs-v0.1.1
git ls-remote --tags origin refs/tags/hashed-blobs-v0.1.1 'refs/tags/hashed-blobs-v0.1.1^{}'
```

These commands describe the release procedure; they are not evidence of publication. An existing tag with different bytes requires a new version; never move an already consumed tag. Verify the peeled remote target before consumer replacement.

## Sinex replacement

After the real tag resolves to the reviewed owner commit, replace this dependency in `crate/sinex-store/Cargo.toml`:

```toml
hashed-blobs = { git = "https://github.com/Sinity/polylogue", tag = "hashed-blobs-v0.1.1" }
```

Sinex already consumes immutable 0.1.0; update that exact tag pin to 0.1.1, preserving its Git source and removing the superseded 0.1.0 Nix output hash. Do not restore a vendor copy or path fallback. Update the selected downstream lock entry through the declared `foundation_lock -- -p hashed-blobs` operation, then add `cargoLock.outputHashes."hashed-blobs-0.1.1"` to the downstream Nix package. Use a temporary `pkgs.lib.fakeHash` and take the expected fixed-output hash from the downstream declared build operation; replace the placeholder and rerun that build. The lock must record the remote Git URL, tag query and exact reviewed owner SHA. Preserve Reader Send + Sync (Sinex holds it in Arc), inspect caller-scoped reuse and current visibility checks, and use the checked plain acquisition route wherever a downstream caller promises cancellation (ordinary put/batch have no such contract), and run the affected Sinex store/file-layer tests plus the real downstream Nix build with its infrastructure features disabled by default; do not preserve a path patch or vendor fallback beside the Git dependency. Review and publish that connected downstream change through its normal route.

For a new release, owner publication and the exact remote consumer replacement must both pass. Local clone trials cannot establish remote dependency delivery. Existing consumers remain pinned until their replacement is qualified and published.

## Credential presentation crate

`credential-output` uses this same extracted-package and disposable Git-tag consumer qualification, selected with `scripts/check-shared-file-layer.py --package credential-output` (declared operation `shared_credential_output_check`). Its independently reviewed release name is `credential-output-v0.1.0`. The existing hashed-blobs-v0.1.0 release must never move. The crate has no product dependencies or runtime configuration. Source recognition and the Unicode table were extracted unchanged from the app masking module at parent commit `272befd54fc95c3e5c5f9783fabd1b7b176de6ab`; the independent catalog and corpus retain their own provenance beside the crate tests.

Publication and downstream immutable pin landing belong to the integration owner. Before publication a downstream candidate may bind the exact reviewed owner commit through a temporary explicit Git source. Such a candidate is local qualification evidence, not a released dependency. Remove that temporary source when landing the remote immutable tag and its verified Nix output hash. No vendor copy, path patch or compatibility engine is shipped.
