"""Authenticated local-file publication through Sinex's native revision RPC."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import shutil
import ssl
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from polylogue.sinex.models import PublicationOriginal, PublicationReceipt, ReceiptState
from polylogue.storage.blob_store import BlobStore

# Native gateway codes for transient infrastructure, including material CAS
# and raw publication. Shared codes below use the public native kind instead.
_RETRYABLE_RPC_CODES = frozenset(
    {
        -32810,
        -32811,
        -32812,
        -32813,
        -32820,
        -32821,
        -32830,
        -32831,
        -32860,
        -32861,
        -32870,
        -32899,
    }
)


class SinexRPCError(RuntimeError):
    """A native RPC refused or could not observe a publication."""

    def __init__(self, code: int, *, retryable: bool) -> None:
        self.code = code
        self.retryable = retryable
        super().__init__(f"sinex_rpc_error:{code}")


class SinexPublicationPendingError(RuntimeError):
    """Native custody exists without a raw PubAck or settled receipt yet."""


class _RefuseRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request: Any, response: Any, code: int, message: str, headers: Any, newurl: str) -> None:
        raise ValueError("sinex_rpc_redirect_refused")


@dataclass
class SinexHTTPTransport:
    """Explicit endpoint/credential composition; retained CAS supplies originals.

    Native claims bind exact descriptors, not the temporary file locators.
    The blocking worker owns its files until the complete exchange settles,
    even if its asynchronous caller stops waiting. Re-drive keeps request_id.
    """

    endpoint: str
    token: str = field(repr=False)
    blob_store: BlobStore = field(repr=False)
    staging_root: Path
    tls_context: ssl.SSLContext | None = field(default=None, repr=False)
    exchange_timeout_s: float = 30.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.exchange_timeout_s) or self.exchange_timeout_s <= 0:
            raise ValueError("sinex_rpc_timeout_invalid")
        endpoint = urllib.parse.urlsplit(self.endpoint)
        if endpoint.scheme not in {"http", "https"} or endpoint.username or endpoint.password:
            raise ValueError("sinex_rpc_endpoint_invalid")
        if endpoint.scheme == "http" and endpoint.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("sinex_rpc_remote_requires_tls")

    def _rpc(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(
            {"jsonrpc": "2.0", "id": params["request_id"], "method": method, "params": params},
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=body,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.token}"},
            method="POST",
        )
        try:
            opener = urllib.request.build_opener(
                _RefuseRedirects(),
                urllib.request.HTTPSHandler(context=self.tls_context),
                urllib.request.ProxyHandler({}),
            )
            with opener.open(request, timeout=self.exchange_timeout_s) as response:
                payload = json.load(response)
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                raise SinexRPCError(-32002, retryable=False) from None
            raise
        if not isinstance(payload, dict) or payload.get("id") != params["request_id"]:
            raise ValueError("sinex_rpc_response_identity_mismatch")
        error = payload.get("error")
        if isinstance(error, dict):
            code = int(error.get("code", -32603))
            public = error.get("data")
            if isinstance(public, dict) and isinstance(public.get("public"), dict):
                public = public["public"]
            kind = public.get("kind_name") if isinstance(public, dict) else None
            retryable = (
                code in _RETRYABLE_RPC_CODES
                or (code == -32850 and kind in {"processing", "kv"})
                or (code == -32840 and kind == "channel_receive")
            )
            raise SinexRPCError(code, retryable=retryable)
        result = payload.get("result")
        if not isinstance(result, dict):
            raise ValueError("sinex_rpc_result_missing")
        return result

    @staticmethod
    def _receipt(request_id: str, result: dict[str, Any]) -> PublicationReceipt:
        if result.get("request_id") != request_id:
            raise ValueError("sinex_publication_receipt_identity_mismatch")
        if result.get("state") is None:
            raise SinexPublicationPendingError("sinex_publication_pending")
        return PublicationReceipt(request_id, ReceiptState(result["state"]), str(result.get("detail", "")))

    def _status(self, request_id: str) -> dict[str, Any] | None:
        try:
            return self._rpc("polylogue.revision.status", {"request_id": request_id})
        except SinexRPCError as exc:
            if exc.code == -32801:
                return None
            raise

    def _publish(
        self,
        *,
        request_id: str,
        manifest_bytes: bytes,
        segment_bytes: Mapping[str, bytes],
        originals: tuple[PublicationOriginal, ...],
        normalizer_version: str,
    ) -> PublicationReceipt:
        if not any(original.role == "provider_native" for original in originals):
            raise ValueError("publication_provider_original_missing")
        self.staging_root.mkdir(parents=True, exist_ok=True)
        # No source payload, path or credential enters public diagnostics.
        with tempfile.TemporaryDirectory(prefix="sinex-publication-", dir=self.staging_root) as directory:
            root = Path(directory)

            def staged(name: str, payload: bytes) -> dict[str, Any]:
                path = root / name
                path.write_bytes(payload)
                return {
                    "file_path": str(path),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "size_bytes": len(payload),
                }

            manifest = staged("manifest.json", manifest_bytes)
            segments = [
                {"name": name, **staged(f"segment-{index}", payload)}
                for index, (name, payload) in enumerate(segment_bytes.items())
            ]
            roots: list[dict[str, Any]] = []
            for index, original in enumerate(originals):
                path = root / f"original-{index}"
                with self.blob_store.open(original.blob_hash) as source, path.open("wb") as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
                digest = hashlib.sha256()
                size = 0
                with path.open("rb") as source:
                    while chunk := source.read(1024 * 1024):
                        digest.update(chunk)
                        size += len(chunk)
                if digest.hexdigest() != original.blob_hash or size != original.size_bytes:
                    raise ValueError("sinex_original_retained_bytes_mismatch")
                roots.append(
                    {
                        "role": original.role,
                        "source_ref": original.source_ref,
                        "file_path": str(path),
                        "sha256": original.blob_hash,
                        "size_bytes": size,
                        "media_type": original.media_type,
                        "acquisition": original.acquisition.to_dict(),
                    }
                )
            params = {
                "request_id": request_id,
                "normalizer_version": normalizer_version,
                "manifest": manifest,
                "segments": segments,
                "originals": roots,
            }
            try:
                result = self._rpc("polylogue.revision.publish", params)
            except (OSError, SinexRPCError) as exc:
                if isinstance(exc, SinexRPCError) and not exc.retryable:
                    if exc.code in {-32800, -32803}:
                        return PublicationReceipt(request_id, ReceiptState.REJECTED, f"sinex_rpc_refused:{exc.code}")
                    raise
                observed = self._status(request_id)
                if observed is None:
                    raise
                result = observed
            return self._receipt(request_id, result)

    async def publish_revision(
        self,
        *,
        request_id: str,
        manifest_bytes: bytes,
        segment_bytes: Mapping[str, bytes],
        originals: tuple[PublicationOriginal, ...] = (),
        normalizer_version: str = "polylogue.material-protocol/v1;semantics=8;canonicalizer=2",
    ) -> PublicationReceipt:
        return await asyncio.to_thread(
            self._publish,
            request_id=request_id,
            manifest_bytes=manifest_bytes,
            segment_bytes=segment_bytes,
            originals=originals,
            normalizer_version=normalizer_version,
        )
