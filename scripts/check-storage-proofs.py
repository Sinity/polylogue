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
    # Reuse the independent synthetic native-source builder, not a product exporter.
    native_spec = importlib.util.spec_from_file_location("storage_native", Path(__file__).with_name("check-codex-native.py"))
    native = importlib.util.module_from_spec(native_spec)
    native_spec.loader.exec_module(native)
    run.start()
    text = "Exact retained citation 東京🙂."
    source = run.output / "retained-proof.jsonl"
    native.rollout(source, output=text)
    receipt = native.imported(run, source, "codex-rollout")
    raw = source.read_bytes()
    _, calls = native.query(run)
    assert len(calls["rows"]) == 1, calls
    ref = calls["rows"][0]["ref"]
    page = run.cli([ref, "read", "--budget", "20000"])
    shown, = page["shown"]
    assert shown["operation"]["output"] == text, page
    citations = shown["sources"]
    assert citations
    for citation in citations:
        assert citation["input"] == receipt["retained"]["id"] and citation["chain"] == 1, citation
        origin = citation["origin"]
        assert origin["representation"] == "rollout"
        assert (citation["start"], citation["end"]) == (origin["start"], origin["end"])
        record = json.loads(raw[citation["start"]:citation["end"]])
        payload = record["payload"]
        assert payload.get("call_id", payload.get("item", {}).get("id")) == "C", record
    run.client = mcp.Client(run)
    remote = run.client.tool("read", {"ref": ref, "budget_tokens": 20000})["structuredContent"]
    assert remote["shown"] == page["shown"], remote
    run.client.close()
    run.client = None
    run.stop()
    blob, = (run.output / "archive" / "blobs").glob("*/*.zst")
    original = blob.read_bytes()
    count = int.from_bytes(original[-9:-5], "little")
    table = len(original) - 17 - count * 8
    assert count == 1 and int.from_bytes(original[table:table+4], "little") == 0x184D2A5E
    controls = {}
    for jointly_changed_header in [False, True]:
        corrupt = bytearray(original)
        corrupt[table + 12] ^= 1
        if jointly_changed_header:
            # This finite fixture uses single-segment/dictionary-free size encoding.
            assert original[4] in (0x24, 0x64)
            corrupt[5] ^= 1
        blob.write_bytes(corrupt)
        run.start()
        refused = run.cli([ref, "read", "--budget", "20000"], allowed=(0, 2, 3, 4, 5, 7))
        assert refused["outcome"] == "error" and not refused.get("shown"), refused
        assert "kind" in refused["error"], refused
        run.client = mcp.Client(run)
        remote = run.client.tool("read", {"ref": ref, "budget_tokens": 20000})["structuredContent"]
        assert remote["outcome"] == "error" and not remote.get("shown"), remote
        assert "kind" in remote["error"], remote
        run.client.close()
        run.client = None
        run.stop()
        controls["header_and_table" if jointly_changed_header else "table_one_bit"] = {"cli": refused["error"], "mcp": remote["error"]}
    run.start()
    # A fresh custody attempt cannot turn the corrupt name into successful retention.
    args = ["import", source, "--format", "codex-rollout", "--request-id", "019a0000-0000-7000-8000-000000000111", "--seen-at", "1", "--wait", "--wait-timeout", "30"]
    failed = run.cli(args, allowed=(0, 2, 3, 4, 5, 7))
    assert failed["outcome"] == "error" and not failed.get("retained"), failed
    assert "seek" in json.dumps(failed).lower() or "existing" in json.dumps(failed).lower() or "frame" in json.dumps(failed).lower(), failed
    assert blob.read_bytes() == corrupt
    run.stop()
    # Restore only this owned synthetic fixture; product puts never perform this repair.
    blob.write_bytes(original)
    run.start()
    retried = run.cli(args)
    assert retried.get("retained") and retried["outcome"] != "error", retried
    restored = run.cli([ref, "read", "--budget", "20000"])
    assert restored["shown"] == page["shown"], restored
    return {"positive_citations": len(citations), "corruption_controls": controls, "corrupt_custody_refused": True, "retry_after_fixture_restoration": True}


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
