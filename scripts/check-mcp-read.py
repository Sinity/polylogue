#!/usr/bin/env python3
"""Exercise the real SDK stdio process with the independent native read oracle."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import selectors
import socket
import subprocess
import time

spec = importlib.util.spec_from_file_location("bounded", Path(__file__).with_name("check-bounded-read.py"))
bounded = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bounded)

META = {"io.modelcontextprotocol/protocolVersion": "2026-07-28",
        "io.modelcontextprotocol/clientCapabilities": {},
        "io.modelcontextprotocol/clientInfo": {"name": "native-read-oracle", "version": "1"}}


class Client:
    def __init__(self, run, legacy=False):
        self.run, self.counter, self.legacy = run, 0, legacy
        self.log = (run.output / f"mcp-{len(run.clients):03}.stderr").open("wb")
        self.process = subprocess.Popen(run.command(["mcp"]), env=run.env, stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=self.log, bufsize=0)
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
        self.buffer = b""
        run.clients.append(self)
        if legacy:
            self.rpc("initialize", {"protocolVersion": "2025-03-26", "clientInfo": {"name": "legacy-control", "version": "1"}, "capabilities": {}})
            self.send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        else:
            discovery = self.rpc("server/discover", {})
            assert discovery["result"]["resultType"] == "complete"
            assert "2026-07-28" in discovery["result"]["supportedVersions"]
            assert set(discovery["result"]["capabilities"]) == {"tools"}
        listing = self.rpc("tools/list", {})["result"]
        assert [tool["name"] for tool in listing["tools"]] == ["status", "search", "read"]
        for tool in listing["tools"]:
            assert tool["outputSchema"]["required"] == ["outcome"]
            assert tool["inputSchema"]["additionalProperties"] is False
            assert tool["annotations"]["readOnlyHint"]
        assert set(listing["tools"][2]["inputSchema"]["properties"]) == {"ref", "budget_tokens", "continuation", "view"}

    def send(self, message):
        attempt = {"send": message, "status": "attempted"}
        self.run.mcp_calls.append(attempt)
        try:
            self.process.stdin.write((bounded.compact(message) + "\n").encode())
            self.process.stdin.flush()
        except BaseException as error:
            attempt.update(status="failed", error=repr(error))
            raise
        attempt["status"] = "sent"

    def receive(self, timeout=45):
        until = time.monotonic() + timeout
        while b"\n" not in self.buffer:
            if not self.selector.select(max(0, until - time.monotonic())):
                self.receive_error("MCP response timeout")
            chunk = os.read(self.process.stdout.fileno(), 65536)
            if not chunk:
                self.receive_error("MCP stdout ended")
            self.buffer += chunk
        line, self.buffer = self.buffer.split(b"\n", 1)
        try:
            value = json.loads(line)
        except BaseException as error:
            self.run.mcp_calls.append({"receive_error": repr(error), "line_hex": line.hex()})
            raise
        self.run.mcp_calls.append({"receive": value})
        return value

    def receive_error(self, reason):
        self.run.mcp_calls.append({"receive_error": reason, "partial_buffer_hex": self.buffer.hex()})
        raise AssertionError(reason)

    def begin(self, method, parameters):
        self.counter += 1
        parameters = dict(parameters)
        if not self.legacy:
            parameters["_meta"] = META
        self.send({"jsonrpc": "2.0", "id": self.counter, "method": method, "params": parameters})
        return self.counter

    def rpc(self, method, parameters):
        request_id = self.begin(method, parameters)
        response = self.receive()
        assert response["id"] == request_id, response
        return response

    def tool(self, name, arguments):
        response = self.rpc("tools/call", {"name": name, "arguments": arguments})
        assert "error" not in response, response
        result = response["result"]
        assert result.get("resultType", "complete") == "complete"
        assert result["isError"] == (result["structuredContent"]["outcome"] == "error")
        return result

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.close()
            self.process.wait(timeout=8)
        assert self.process.returncode == 0
        self.selector.close()
        self.process.stdout.close()
        self.log.close()


class Run(bounded.Run):
    def __init__(self, binary, output):
        super().__init__(binary, output)
        self.clients, self.mcp_calls, self.client, self.next_calls = [], [], None, {}

    def read(self, call, budget=None):
        # The inherited fixture oracle understands the CLI's reference/cursor
        # form. Save the actual MCP next call and use its returned arguments
        # verbatim, changing only capacity when that test explicitly asks.
        args = dict(self.next_calls.get(bounded.compact(call), {"tool": "read", "arguments": call})["arguments"])
        if budget is not None:
            args["budget_tokens"] = budget
        response = self.client.tool("read", args)
        page = response["structuredContent"]
        bounded.validate_page_citations(page, self.sources)
        budget = args.get("budget_tokens", 4000)
        assert response["content"] and response["content"][0]["text"].startswith(bounded.WARNING)
        total = bounded.estimated({"structuredContent": page, "content": response["content"]})
        assert total <= max(128, budget), (total, budget, page)
        if page.get("next"):
            actual = page["next"]
            assert actual["tool"] == "read"
            assert set(actual["arguments"]) == {"ref", "continuation", "budget_tokens"}
            fixture_call = {key: value for key, value in actual["arguments"].items() if key != "budget_tokens"}
            self.next_calls[bounded.compact(fixture_call)] = actual
            page = dict(page, next=fixture_call)
        return page


def cancellation_controls(run):
    # Real SDK requests go to an intentionally stalled Unix peer. Seeing EOF
    # there proves closure of the downstream socket, not only future abortion.
    run.stop()
    # The daemon removes its owned socket on graceful stop, while kill preserves
    # it. Its path is recorded before this test by main.
    sockpath = run.socket_path
    sockpath.unlink(missing_ok=True)
    listener = socket.socket(socket.AF_UNIX)
    listener.bind(str(sockpath))
    listener.listen(4)
    listener.settimeout(4)
    results = []
    try:
        for disconnect in (False, True):
            client = Client(run)
            request_id = client.begin("tools/call", {"name": "read", "arguments": {"ref": bounded.REF, "budget_tokens": 1800}})
            downstream, _ = listener.accept()
            downstream.settimeout(4)
            data = b""
            while b"\n" not in data:
                data += downstream.recv(4096)
            assert json.loads(data)["command"] == "model-read"
            before = time.monotonic()
            if disconnect:
                client.process.stdin.close()
            else:
                client.send({"jsonrpc": "2.0", "method": "notifications/cancelled", "params": {"requestId": request_id, "reason": "oracle cancellation"}})
            assert downstream.recv(1) == b"", "downstream socket was retained after cancel/disconnect"
            elapsed = time.monotonic() - before
            assert elapsed < 2, elapsed
            downstream.close()
            if not disconnect:
                # The SDK legitimately suppresses responses to cancelled
                # requests. A subsequent discovery response proves liveness.
                assert client.rpc("tools/list", {})["result"]["tools"]
            else:
                client.process.wait(timeout=8)
            client.close()
            results.append({"input_disconnect": disconnect, "downstream_eof_seconds": elapsed})
    finally:
        listener.close()
        sockpath.unlink(missing_ok=True)
    run.start()
    # Retained source work and ordinary CLI survive MCP reader cancellation.
    assert run.cli(["status"])["outcome"] == "ok"
    retained = run.cli(["id:" + bounded.REF, "--json"])
    assert retained["rows"] and retained["rows"][0]["id"] == bounded.REF
    native = Client(run)
    request_id = native.begin("tools/call", {"name": "read", "arguments": {"ref": bounded.REF, "budget_tokens": 1000000}})
    native.send({"jsonrpc": "2.0", "method": "notifications/cancelled", "params": {"requestId": request_id, "reason": "native owner control"}})
    sid = "7f3a2c10-0004-4000-8000-000000000004"
    source = run.import_rows("after-native-cancel", [bounded.row("survives", "Source owner survived the native MCP read cancellation.", sid=sid)], sid=sid)
    read = run.cli(["message:claude-code:" + sid + ":survives", "read", "--budget", "1800"])
    assert read["shown"][0]["text"] == "Source owner survived the native MCP read cancellation."
    source.unlink()
    native.close()
    return results


def finish(run, report):
    steps = []
    for index, client in enumerate(run.clients):
        def kill(client=client):
            if client.process.poll() is None:
                client.process.kill()
        def wait(client=client):
            if client.process.poll() is None:
                client.process.wait(timeout=15)
        steps.extend([
            (f"mcp-{index}-close", client.close),
            (f"mcp-{index}-fallback-kill", kill),
            (f"mcp-{index}-fallback-wait", wait),
            (f"mcp-{index}-selector", client.selector.close),
            (f"mcp-{index}-stdin", client.process.stdin.close),
            (f"mcp-{index}-stdout", client.process.stdout.close),
            (f"mcp-{index}-log", client.log.close),
        ])
    steps.extend([("daemon", run.stop), ("daemon_log", run.daemon_log.close),
                  ("socket_directory", run.socket_owner.cleanup)])
    cleanup_errors = bounded.protected_cleanup(steps)
    if cleanup_errors:
        report.update(outcome="failed", cleanup_errors=cleanup_errors)
    report["commands"] = run.calls
    report["daemon_reaped"] = run.daemon is None
    report["mcp_processes_reaped"] = [client.process.poll() is not None for client in run.clients]
    try:
        (run.output / "mcp-calls.json").write_text(json.dumps(run.mcp_calls, ensure_ascii=False, indent=2) + "\n")
    except BaseException as error:
        cleanup_errors.append({"owner": "mcp-calls.json", "error": repr(error)})
        report.update(outcome="failed", cleanup_errors=cleanup_errors)
    (run.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(bounded.compact({k: v for k, v in report.items() if k != "commands"}))
    return cleanup_errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    binary, output = args.binary.resolve(strict=True), args.output.absolute()
    output.mkdir(parents=True, exist_ok=False)
    run = Run(binary, output)
    report = {"binary": str(binary), "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest()}
    failure = None
    try:
        run.start()
        run.socket_path, = Path(run.env["XDG_RUNTIME_DIR"]).glob("polylogue/*/socket")
        run.client = Client(run)
        empty = run.client.tool("search", {"query": ""})["structuredContent"]
        assert empty["outcome"] == "empty" and empty["count"] == 0 and not empty["rows"]
        assert run.client.tool("read", {"ref": bounded.REF})["structuredContent"]["error"]["kind"] == "not_found"
        report.update(bounded.verify(run))
        native = run.cli(["BLUE", "--json"])
        mcp = run.client.tool("search", {"query": "BLUE", "budget_tokens": 4000})["structuredContent"]
        assert mcp == native and mcp["outcome"] == "degraded" and mcp["rows"]
        oversized = run.client.tool("search", {"query": "", "budget_tokens": 128})
        assert oversized["structuredContent"]["error"]["kind"] == "not_yet_available"
        assert "narrow" in oversized["structuredContent"]["error"]["message"]
        assert bounded.estimated({"structuredContent": oversized["structuredContent"], "content": oversized["content"]}) <= 128
        admitted = run.client.tool("search", {"query": "", "budget_tokens": 65536})
        assert len(admitted["structuredContent"]["rows"]) >= 3
        assert bounded.estimated({"structuredContent": admitted["structuredContent"], "content": admitted["content"]}) <= 65536
        assert run.client.tool("search", {"query": "BLUE", "budget_tokens": 1})["structuredContent"]["error"]["kind"] == "not_yet_available"
        assert run.client.tool("read", {"ref": bounded.REF, "view": "transcript"})["structuredContent"]["error"]["kind"] == "invalid_query"
        assert run.client.tool("search", {"query": "", "cursor": "invented"})["structuredContent"]["error"]["kind"] == "invalid_query"
        legacy = Client(run, legacy=True)
        assert legacy.tool("status", {})["structuredContent"]["outcome"] == "ok"
        legacy.close()
        run.client.close()
        report["cancellation"] = cancellation_controls(run)
        run.client = Client(run)
        run.stop()
        unavailable = run.client.tool("search", {"query": ""})["structuredContent"]
        assert unavailable["outcome"] == "error" and unavailable["error"]["kind"] == "not_running"
        report.update(outcome="ok", scope="native rmcp stdio discovery/search/bounded read; broad privacy/removal products and full M1 remain open")
    except BaseException as error:
        failure = error
        report.update(outcome="failed", error=repr(error))
    finally:
        cleanup_errors = finish(run, report)
    if failure:
        raise failure
    if cleanup_errors:
        raise RuntimeError(cleanup_errors)


if __name__ == "__main__":
    main()
