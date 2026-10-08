"""Original captured Raw inputs for retained parser controls."""

from __future__ import annotations

import hashlib
import sys
from builtins import BaseExceptionGroup
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Literal, Protocol

from polylogue.archive.revision_authority import (
    RawRevisionAuthority,
    RawRevisionEnvelope,
    RawRevisionKind,
    append_source_revision,
)
from polylogue.core.enums import Provider
from polylogue.core.sources import origin_from_provider
from polylogue.core.stage_admission import admit_stage_write, stage_write_admission
from polylogue.operations.raw_observation_derivation import make_raw_observation_derivation, raw_observation_frame
from polylogue.sources.acquisition_boundary import bound_profile_identity, bound_source_observation, open_bound_path
from polylogue.sources.prepared_jsonl import PreparedJsonl
from polylogue.sources.revision_backfill import (
    PreparedRevisionReplayResult,
    RevisionCensusResult,
    prepare_retained_jsonl_artifact,
)
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.derived.raw import RawObservationReplacement
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.archive_tiers.write import PreparedSessionSourceRead
from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation
from polylogue.storage.sqlite.write_lease import write_lease
from tests.infra.live_ingest import prepared_live_convergence_owner


class RetainedArtifactPreparer(Protocol):
    def __call__(self, reader: PreparedSessionSourceRead, raw_id: str, /, *, directory: Path) -> PreparedJsonl: ...


def acquire_full_revision(
    archive: ArchiveStore,
    *,
    provider: Provider,
    source_path: Path,
    payload: bytes,
    native_id: str,
    generation: int,
    acquired_at_ms: int,
) -> str:
    """Capture one actual rewritten file under its original FULL revision evidence."""
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(payload)
    with open_bound_path(source_path, None) as original:
        profile = bound_profile_identity(original)
        canonical, observation = bound_source_observation(original)
        acquired = original.read()
        assert acquired == payload and canonical is not None and observation is not None
        return archive.write_raw_payload(
            provider=provider,
            payload=acquired,
            source_path=str(source_path),
            canonical_source_path=canonical,
            captured_profile_key=profile.key if profile else None,
            native_id=native_id,
            acquired_at_ms=acquired_at_ms,
            file_mtime_ms=observation[3] // 1_000_000,
            revision=RawRevisionEnvelope(
                f"{origin_from_provider(provider).value}:{native_id}",
                RawRevisionKind.FULL,
                hashlib.sha256(acquired).hexdigest(),
                generation,
                authority=RawRevisionAuthority.BYTE_PROVEN,
            ),
        )


@contextmanager
def retained_raw_fixture(
    *,
    root: Path,
    provider: Provider,
    blob_hash: str,
    source_path: str,
    file_mtime: str | None = None,
    source_index: int = 0,
    acquired_at_ms: int = 1,
    native_id: str | None = None,
    raw_id: str | None = None,
) -> Iterator[tuple[PreparedSessionSourceRead, str]]:
    """Acquire the declared neutral file, then borrow its original retained reader.

    The fixture's archive and bytes remain genuine. Consumers run on retained bytes after acquisition closes, while the
    original preparation seal stays alive. No descriptor or profile receipt is inferred.
    """
    blob_store = BlobStore(root / "blob")
    payload = blob_store.read_all(blob_hash)
    path = Path(source_path)
    if not path.is_absolute():
        path = root / path
    path.parent.mkdir(parents=True, exist_ok=True)
    created = not path.exists()
    if created:
        path.write_bytes(payload)
    else:
        assert path.read_bytes() == payload
    try:
        # None is the explicit unbound acquisition law: malformed bytes must
        # remain available for the retained parser's typed refusal controls.
        with open_bound_path(path, None) as original:
            profile = bound_profile_identity(original)
            canonical, observation = bound_source_observation(original)
            acquired = original.read()
            assert acquired == payload and observation is not None and canonical is not None
            mtime = (
                int(datetime.fromisoformat(file_mtime.replace("Z", "+00:00")).timestamp() * 1000)
                if file_mtime is not None
                else observation[3] // 1_000_000
            )
            with write_lease("test.retained-json.acquire", archive_root=root):
                # The lease is held by this thread; bootstrap on it rather than
                # hopping to a loop-free thread the lease does not authorize.
                initialize_active_archive_root(root)
                with ArchiveStore.open_existing(root, read_only=False) as archive:
                    raw_id = archive.write_raw_payload(
                        provider=provider,
                        payload=acquired,
                        source_path=source_path,
                        canonical_source_path=canonical,
                        captured_profile_key=profile.key if profile else None,
                        acquired_at_ms=acquired_at_ms,
                        source_index=source_index,
                        native_id=native_id,
                        raw_id=raw_id,
                        file_mtime_ms=mtime,
                    )
                    archive.commit()
        del payload, acquired
        with PreparedIndexMutation(root / "index.db", archive_root=root) as seal:
            with seal.original_read_snapshot(), seal.source_producer():
                reader = PreparedSessionSourceRead(seal, blob_store=blob_store)
                assert reader.raw_revision_descriptor(raw_id)[1] == blob_hash
                yield reader, raw_id
    finally:
        if created:
            primary = sys.exception()
            try:
                path.unlink()
            except BaseException as cleanup:
                if primary is not None:
                    raise BaseExceptionGroup("retained input and file close failed", [primary, cleanup]) from None
                raise


@contextmanager
def retained_parser_fixture(
    *,
    root: Path,
    provider: Provider,
    blob_hash: str,
    source_path: str,
    directory: Path,
    file_mtime: str | None = None,
    prepare: RetainedArtifactPreparer = prepare_retained_jsonl_artifact,
) -> Iterator[tuple[PreparedJsonl, PreparedSessionSourceRead]]:
    """Run the actual retained producer on original Raw, closing before its seal."""
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    with retained_raw_fixture(
        root=root,
        provider=provider,
        blob_hash=blob_hash,
        source_path=source_path,
        file_mtime=file_mtime,
    ) as (reader, raw_id):
        artifact = prepare(reader, raw_id, directory=directory)
        primary: BaseException | None = None
        try:
            yield artifact, reader
        except BaseException as failure:
            primary = failure
            raise
        finally:
            try:
                artifact.discard()
            except BaseException as cleanup:
                if primary is not None:
                    raise BaseExceptionGroup("retained fixture and artifact close failed", [primary, cleanup]) from None
                raise


@contextmanager
def prepared_source_fixture(root: Path) -> Iterator[PreparedSessionSourceRead]:
    """Borrow the actual original Source window after fixture initialization."""
    with write_lease("test.retained-source.initialize", archive_root=root):
        # Under an enclosing async lease a running loop is present, and the
        # off-loop hop would run bootstrap on a thread the lease does not own.
        initialize_active_archive_root(root)
    with PreparedIndexMutation.source_only(archive_root=root) as seal:
        with seal.original_read_snapshot(), seal.source_producer():
            yield PreparedSessionSourceRead(seal, blob_store=BlobStore(root / "blob"))


async def run_retained_source_phase(
    root: Path,
    acquired_raw_ids: Sequence[str],
    *,
    replay_current: bool = False,
    before_publication: Callable[[RawObservationReplacement], None] | None = None,
) -> tuple[
    bool,
    Literal["census", "classification", "replay"],
    RevisionCensusResult | PreparedRevisionReplayResult,
]:
    """Publish one actual prepared Source phase and return its original receipt.

    Selection offers genuine acquired IDs; the original reader still expands
    required membership. This helper never reconstructs receipt counts or
    loops independent phases to imitate one atomic publication.
    """
    acquired = tuple(acquired_raw_ids)
    if not acquired or any(not raw_id for raw_id in acquired):
        raise ValueError("Source phase control requires actual acquired Raw identities")
    async with prepared_live_convergence_owner(root) as owner:
        retained: list[RawObservationReplacement] = []

        def run() -> tuple[
            bool,
            Literal["census", "classification", "replay"],
            RevisionCensusResult | PreparedRevisionReplayResult,
        ]:
            adapter = make_raw_observation_derivation(root, compute_adapter=owner._compute_adapter)
            frame = raw_observation_frame(root, raw_ids=acquired)
            # A phase commits in place only under writer admission. Preparing
            # without it hands the phase to publish, the path a refused
            # in-place guard takes, so exactly that one phase is published.
            with stage_write_admission(None):
                replacement = adapter.compute(
                    frame,
                    acquired[0],
                    replay_current=replay_current,
                    select_retained_raw_ids=lambda _original_reader: acquired,
                )
            retained.append(replacement)
            try:
                assert replacement.needs_source_census or replacement.needs_source_classification
                phases: list[
                    tuple[
                        Literal["census", "classification", "replay"],
                        RevisionCensusResult | PreparedRevisionReplayResult,
                    ]
                ] = []

                def receive(
                    phase: Literal["census", "classification", "replay"],
                    receipt: RevisionCensusResult | PreparedRevisionReplayResult,
                ) -> None:
                    phases.append((phase, receipt))

                if before_publication is not None:
                    before_publication(replacement)
                failures: list[BaseException] = []
                published = admit_stage_write(
                    "test.retained.source-phase.publish",
                    lambda: adapter.publish(
                        frame, replacement, phase_receipt=receive, publication_failure=failures.append
                    ),
                )
                if failures:
                    if len(failures) == 1:
                        raise failures[0]
                    raise BaseExceptionGroup("original Source publication failures", failures)
                assert len(phases) == 1
                phase, receipt = phases[0]
                return published, phase, receipt
            finally:
                primary = sys.exception()
                try:
                    replacement.close()
                except BaseException as cleanup:
                    if primary is not None:
                        raise BaseExceptionGroup("Source phase and physical close failed", [primary, cleanup]) from None
                    raise
                else:
                    retained.remove(replacement)

        return await owner.run_prepared_sync(
            "test.retained.source-phase.prepare",
            run,
            settlement_owners=lambda: tuple(retained),
            estimated_bytes=0,
        )


@contextmanager
def retained_append_fixture(
    *,
    root: Path,
    provider: Provider,
    source_path: Path,
    native_id: str,
    logical_source_key: str,
    baseline: bytes,
    delta: bytes,
    baseline_generation: int = 0,
    acquired_at_ms: int = 1,
) -> Iterator[tuple[PreparedSessionSourceRead, str, str, str, str, int, int]]:
    """Capture the actual complete baseline, then the observed forward appended range."""
    source_path.parent.mkdir(parents=True, exist_ok=True)
    if source_path.exists():
        raise FileExistsError(source_path)
    owned_file = source_path.open("xb")
    try:
        with owned_file as writer:
            writer.write(baseline)
        baseline_revision = hashlib.sha256(baseline).hexdigest()
        with open_bound_path(source_path, None) as original:
            profile = bound_profile_identity(original)
            profile_key = profile.key if profile else None
            canonical, observation = bound_source_observation(original)
            assert canonical is not None and observation is not None
            acquired = original.read()
            assert acquired == baseline
            with (
                write_lease("test.retained.append.baseline", archive_root=root),
                ArchiveStore.open_existing(root, read_only=False) as archive,
            ):
                baseline_raw_id = archive.write_raw_payload(
                    provider=provider,
                    payload=acquired,
                    source_path=str(source_path),
                    canonical_source_path=canonical,
                    captured_profile_key=profile_key,
                    native_id=native_id,
                    acquired_at_ms=acquired_at_ms,
                    file_mtime_ms=observation[3] // 1_000_000,
                    revision=RawRevisionEnvelope(
                        logical_source_key,
                        RawRevisionKind.FULL,
                        baseline_revision,
                        baseline_generation,
                        authority=RawRevisionAuthority.BYTE_PROVEN,
                    ),
                )
                archive.commit()
        with source_path.open("ab") as writer:
            writer.write(delta)
        complete = baseline + delta
        append_revision = append_source_revision(baseline_revision, hashlib.sha256(delta).hexdigest())
        with open_bound_path(source_path, None) as original:
            profile = bound_profile_identity(original)
            assert (profile.key if profile else None) == profile_key
            current_canonical, observation = bound_source_observation(original)
            assert current_canonical == canonical and observation is not None
            acquired = original.read()
            assert acquired == complete
            original_delta = acquired[len(baseline) :]
            assert original_delta == delta
            with (
                write_lease("test.retained.append.range", archive_root=root),
                ArchiveStore.open_existing(root, read_only=False) as archive,
            ):
                raw_id = archive.write_raw_payload(
                    provider=provider,
                    payload=original_delta,
                    source_path=str(source_path),
                    canonical_source_path=current_canonical,
                    captured_profile_key=profile_key,
                    native_id=native_id,
                    acquired_at_ms=acquired_at_ms + 1,
                    file_mtime_ms=observation[3] // 1_000_000,
                    revision=RawRevisionEnvelope(
                        logical_source_key,
                        RawRevisionKind.APPEND,
                        append_revision,
                        baseline_generation + 1,
                        predecessor_source_revision=baseline_revision,
                        predecessor_raw_id=baseline_raw_id,
                        baseline_raw_id=baseline_raw_id,
                        append_start_offset=len(baseline),
                        append_end_offset=len(complete),
                        authority=RawRevisionAuthority.BYTE_PROVEN,
                    ),
                )
                archive.commit()
        with prepared_source_fixture(root) as reader:
            assert reader.raw_revision_descriptor(baseline_raw_id)[1] == hashlib.sha256(baseline).hexdigest()
            assert reader.raw_revision_descriptor(raw_id)[1] == hashlib.sha256(delta).hexdigest()
            yield reader, baseline_raw_id, raw_id, baseline_revision, append_revision, len(baseline), len(complete)
    finally:
        primary = sys.exception()
        try:
            source_path.unlink()
        except BaseException as cleanup:
            if primary is not None:
                raise BaseExceptionGroup("append acquisition and owned file close failed", [primary, cleanup]) from None
            raise
