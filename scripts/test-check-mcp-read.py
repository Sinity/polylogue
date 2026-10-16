#!/usr/bin/env python3
"""Failure controls for retained native read-oracle evidence."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("mcp_oracle", Path(__file__).with_name("check-mcp-read.py"))
mcp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mcp)


class EvidenceTests(unittest.TestCase):
    def test_both_routes_use_the_same_exact_source_selector_validation(self):
        data = json.dumps({"sessionId": "neutral", "uuid": "message", "message": {"content": "Neutral text"}}).encode()
        page = {"shown": [{"message": "message:claude-code:neutral:message", "sources": [
            {"input": "input", "start": 0, "end": len(data)}], "field_origins": [
                {"binding": {"input": "input"}, "selection": {"inputs": [
                    {"record": {"start": 0, "end": len(data)}, "path": [{"Key": "message"}, {"Key": "content"}]}]}}]}]}
        mcp.bounded.validate_page_citations(page, {"input": data})
        wrong = json.loads(json.dumps(page))
        wrong["shown"][0]["message"] = "message:claude-code:neutral:other"
        with self.assertRaises(AssertionError):
            mcp.bounded.validate_page_citations(wrong, {"input": data})
        wrong = json.loads(json.dumps(page))
        wrong["shown"][0]["field_origins"][0]["selection"]["inputs"][0]["path"] = [{"Key": "missing"}]
        with self.assertRaises(KeyError):
            mcp.bounded.validate_page_citations(wrong, {"input": data})
        run = object.__new__(mcp.Run)
        run.sources, run.next_calls = {"input": data}, {}
        run.client = SimpleNamespace(tool=lambda *_: {"structuredContent": wrong,
                                                     "content": [{"type": "text", "text": mcp.bounded.WARNING}]})
        with self.assertRaises(KeyError):
            run.read({"ref": "claude-code:neutral"}, 4000)
        run = object.__new__(mcp.bounded.Run)
        run.sources, run.cli = {"input": data}, lambda *_, **__: wrong
        with self.assertRaises(KeyError):
            run.read({"ref": "claude-code:neutral"}, 4000)

    def test_full_discovery_is_precise_and_does_not_advertise_collapsed_modes(self):
        tool={'inputSchema':{'required':['ref'],'properties':{
            'ref':{},'budget_tokens':{},'continuation':{},'branches':{'type':'boolean'},'view':{'enum':['dialogue','transcript','history','withdrawals','current','value','annotations']},
            **{key:{'enum':['full']} for key in ('tools','thinking','harness')}}},
            'description':'Explicit view:transcript, 65536-byte normalized admission'}
        mcp.validate_read_discovery(tool)
        for mutation in ('missing tools','extra mode','missing transcript','old help','wrong branches type','missing authored page'):
            broken=json.loads(json.dumps(tool))
            if mutation=='missing tools': broken['inputSchema']['properties'].pop('tools')
            elif mutation=='extra mode': broken['inputSchema']['properties']['harness']['enum'].append('collapsed')
            elif mutation=='missing transcript': broken['inputSchema']['properties']['view']['enum']=['dialogue']
            elif mutation=='missing authored page': broken['inputSchema']['properties']['view']['enum'].remove('history')
            elif mutation=='old help': broken['description']='Text only'
            else: broken['inputSchema']['properties']['branches']['type']='string'
            with self.subTest(mutation=mutation),self.assertRaises(AssertionError):
                mcp.validate_read_discovery(broken)

    def test_full_continuation_retains_effective_options_and_budget_verbatim(self):
        call={'ref':'opaque:session','view':'transcript','tools':'full','thinking':'full','harness':'full','branches':True}
        emitted=dict(call,continuation='opaque:actual',budget_tokens=2300)
        replies=[{'outcome':'ok','shown':[],'next':{'tool':'read','arguments':emitted}},
                 {'outcome':'ok','shown':[],'next':None}]
        calls=[]
        def tool(name,args):
            calls.append(dict(args))
            return {'structuredContent':replies[len(calls)-1],'content':[{'type':'text','text':mcp.bounded.WARNING}]}
        run=object.__new__(mcp.Run)
        run.sources,run.next_calls,run.client={}, {}, SimpleNamespace(tool=tool)
        page=run.read(call,1800)
        run.read(page['next'])
        self.assertEqual(calls,[dict(call,budget_tokens=1800),emitted])
        replies[0]['next']['arguments']=dict(emitted)
        replies[0]['next']['arguments'].pop('harness')
        calls.clear()
        with self.assertRaises(AssertionError): run.read(call,1800)

    def test_full_citation_validation_admits_typed_values_but_keeps_exact_selectors(self):
        data=json.dumps({'sessionId':'neutral','uuid':'message','message':{'content':{'command':'neutral'}}}).encode()
        page={'shown':[{'message':'message:claude-code:neutral:message','sources':[
            {'input':'input','start':0,'end':len(data)}],'field_origins':[
                {'binding':{'input':'input'},'selection':{'inputs':[
                    {'record':{'start':0,'end':len(data)},'path':[{'Key':'message'},{'Key':'content'}]}]}}]}]}
        with self.assertRaises(AssertionError): mcp.bounded.validate_page_citations(page,{'input':data})
        mcp.bounded.validate_page_citations(page,{'input':data},text_only=False)
        page['shown'][0]['field_origins'][0]['selection']['inputs'][0]['path'].append({'Key':'missing'})
        with self.assertRaises(KeyError): mcp.bounded.validate_page_citations(page,{'input':data},text_only=False)

    def test_failed_send_retains_the_attempt_before_writing(self):
        def broken(_):
            raise BrokenPipeError("synthetic closed input")
        client = object.__new__(mcp.Client)
        client.run = SimpleNamespace(mcp_calls=[])
        client.process = SimpleNamespace(stdin=SimpleNamespace(write=broken))
        message = {"jsonrpc": "2.0", "id": 1, "method": "server/discover"}
        with self.assertRaises(BrokenPipeError):
            client.send(message)
        self.assertEqual(client.run.mcp_calls[0]["send"], message)
        self.assertEqual(client.run.mcp_calls[0]["status"], "failed")

    def test_receive_timeout_preserves_exact_partial_bytes(self):
        client = object.__new__(mcp.Client)
        client.run = SimpleNamespace(mcp_calls=[])
        client.buffer = b'{"jsonrpc":"2.0","id":1,"result":"\xf0\x9f'
        client.selector = SimpleNamespace(select=lambda _: [])
        with self.assertRaisesRegex(AssertionError, "response timeout"):
            client.receive(.01)
        self.assertEqual(bytes.fromhex(client.run.mcp_calls[0]["partial_buffer_hex"]), client.buffer)

    def test_cleanup_failures_cannot_suppress_protocol_and_report_writes(self):
        visited = []
        def broken(label):
            def fail(*_, **__):
                visited.append(label)
                raise RuntimeError(label)
            return fail
        process = SimpleNamespace(poll=lambda: None, kill=broken("kill"), wait=broken("wait"),
                                  stdin=SimpleNamespace(close=broken("stdin")),
                                  stdout=SimpleNamespace(close=broken("stdout")))
        client = SimpleNamespace(close=broken("close"), process=process,
                                 selector=SimpleNamespace(close=broken("selector")),
                                 log=SimpleNamespace(close=broken("log")))
        with tempfile.TemporaryDirectory(dir="/realm/tmp/work") as directory:
            output = Path(directory)
            run = SimpleNamespace(clients=[client], stop=broken("daemon"), daemon=object(),
                                  daemon_log=SimpleNamespace(close=broken("daemon_log")),
                                  socket_owner=SimpleNamespace(cleanup=broken("socket")),
                                  calls=[{"status": "attempted"}], mcp_calls=[{"send": {"id": 1}}], output=output)
            with contextlib.redirect_stdout(io.StringIO()):
                errors = mcp.finish(run, {"outcome": "ok"})
            self.assertEqual(set(visited), {"close", "kill", "wait", "stdin", "stdout", "selector", "log", "daemon", "daemon_log", "socket"})
            self.assertEqual(len(errors), 10)
            self.assertEqual(json.loads((output / "mcp-calls.json").read_text()), run.mcp_calls)
            report = json.loads((output / "report.json").read_text())
            self.assertEqual(report["outcome"], "failed")
            self.assertFalse(report["daemon_reaped"])
            self.assertEqual(report["mcp_processes_reaped"], [False])
            self.assertEqual(report["commands"], run.calls)
            original = Path.write_text
            def write(path, *args, **kwargs):
                if path.name == "mcp-calls.json":
                    raise OSError("synthetic evidence output failure")
                return original(path, *args, **kwargs)
            with patch.object(Path, "write_text", write), contextlib.redirect_stdout(io.StringIO()):
                mcp.finish(run, {"outcome": "ok"})
            report = json.loads((output / "report.json").read_text())
            self.assertTrue(any(error["owner"] == "mcp-calls.json" for error in report["cleanup_errors"]))


if __name__ == "__main__":
    unittest.main()
