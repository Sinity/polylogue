#!/usr/bin/env python3
"""Synthetic retained citations and seek-map corruption through packaged CLI/SDK MCP."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location("storage_mcp", Path(__file__).with_name("check-mcp-read.py"))
mcp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mcp)


def verify(run):
    run.start()
    text = "Exact retained citation 東京🙂."
    run.import_rows("retained-proof", [mcp.bounded.row("storage-proof", text)])
    ref = mcp.bounded.REF
    page = run.cli([ref, "read", "--budget", "8000"])
    mcp.bounded.validate_page_citations(page, run.sources)
    assert page["shown"] and "".join(s.get("text", "") for s in page["shown"]) == text, page
    run.client = mcp.Client(run)
    remote = run.read({"ref": ref}, 8000)
    assert remote["shown"] == page["shown"], remote
    run.client.close()
    run.client = None
    run.stop()
    blob, = (run.output / "archive" / "blobs").glob("*/*.zst")
    original = blob.read_bytes()
    count = int.from_bytes(original[-9:-5], "little")
    table = len(original) - 17 - count * 8
    assert count == 1 and int.from_bytes(original[table:table+4], "little") == 0x184D2A5E
    corrupt = bytearray(original)
    corrupt[table + 12] ^= 1
    blob.write_bytes(corrupt)
    run.start()
    refused = run.cli([ref, "read", "--budget", "8000"], allowed=(0, 2, 3, 4, 5, 7))
    assert refused["outcome"] == "error" and not refused.get("shown"), refused
    run.client = mcp.Client(run)
    remote = run.client.tool("read", {"ref": ref, "budget_tokens": 8000})["structuredContent"]
    assert remote["outcome"] == "error" and not remote.get("shown"), remote
    run.client.close()
    run.client = None
    # A fresh custody attempt cannot turn the corrupt name into successful retention.
    args = ["import", run.output / "retained-proof.jsonl", "--format", "claude-code-jsonl", "--context", run.output / "retained-proof.context.json", "--request-id", "storage-corrupt-retry", "--wait", "--wait-timeout", "30"]
    failed = run.cli(args, allowed=(0, 2, 3, 4, 5, 7))
    assert failed["outcome"] == "error" and not failed.get("retained"), failed
    assert blob.read_bytes() == corrupt
    run.stop()
    # Restore only this owned synthetic fixture; product puts never perform this repair.
    blob.write_bytes(original)
    run.start()
    retried = run.cli(args)
    assert retried.get("retained") and retried["outcome"] != "error", retried
    restored = run.cli([ref, "read", "--budget", "8000"])
    assert restored["shown"] == page["shown"], restored
    return {"positive_citations": len(page["shown"]), "cli_corruption_error": refused["error"], "mcp_corruption_error": remote["error"], "corrupt_custody_refused": True, "retry_after_fixture_restoration": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)
    output = args.output.absolute()
    output.mkdir(parents=True, exist_ok=False)
    run = mcp.Run(binary, output)
    report = {"binary": str(binary), "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest()}
    failed = None
    try:
        report.update(verify(run), outcome="ok")
    except BaseException as error:
        failed = error
        report.update(outcome="failed", error=repr(error))
    finally:
        errors = mcp.finish(run, report)
    if failed:
        raise failed
    if errors:
        raise RuntimeError(errors)


if __name__ == "__main__":
    main()
