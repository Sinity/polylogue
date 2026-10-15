"""Controller regressions, not canonical-product qualification.

Pure cases supply synthetic captured replies and deliberately broken oracles.
An optional immutable real binary exercises only the archive-file launch refusal.
"""
import argparse
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from html.parser import HTMLParser

from morrow_report import artifact

sys.dont_write_bytecode = True


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checker", type=Path, required=True)
parser.add_argument("--binary", type=Path)
args, remaining = parser.parse_known_args()
driver = load(Path(__file__).with_name("check-morrow.py"), "morrow_driver")
checker = load(args.checker, "fixture_checker")
SID = "claude-code:synthetic"
EXPECTED = {"session_id": SID, "sessions": 1, "messages": 1, "human_prompts": ["retained prompt"],
            "import_context": {"version": 1, "context": ["claude-main", "synthetic"]},
            "message_facts": [{"id": "claude-code:synthetic:prompt", "role": "user", "written_by": "human", "at": "2026-10-15T10:00:00.000Z",
                               "blocks": [{"id": "claude-code:synthetic:prompt:0", "kind": "text", "text": "retained prompt"}]}]}
ANSWER = {"result": {"outcome":"ok","notes":[],"unknown":{}}, "format": "polylogue-session", "version": 1, "session": {"id": SID},
          "messages": [{"id": "claude-code:synthetic:prompt", "role": "user", "written_by": "human", "at": "2026-10-15T10:00:00.000Z",
                        "blocks": [{"id": "claude-code:synthetic:prompt:0", "kind": "text", "text": "retained prompt"}]}]}


def admission(errors=0):
    rows={"scope":"imported_source_rows","basis":{"input":"synthetic-input","length":7},
          "state":"unavailable" if errors else "settled"}
    if errors: rows.update(usable_result=True)
    return {"outcome":"degraded" if errors else "ok","retained":{"id":"synthetic-input","length":7},"source_rows":rows,
            "notes":["one undecodable record"] if errors else [],"unknown":{"record_errors":errors} if errors else {}}


class CapturedReplies(driver.Run):
    def client(self, archive, *argv, **kwargs):
        if "read" in argv:
            return deepcopy(ANSWER)
        return {"outcome": "ok", "notes":[], "unknown":{}, "up_to_date": True, "unit": "sessions",
                "count": 1, "rows": [{"id": SID}], "next_cursor": None}

    def command(self, *argv, **kwargs):
        raise ValueError("unrelated export failure must not qualify a negative control")


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="pl-mc-", dir=os.environ.get("TMPDIR"))
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)

    def run_control(self, oracle):
        run = CapturedReplies(self.output, "corrupt-answer")
        with self.assertRaises(ValueError) as error:
            run.answers(self.output / "unused", EXPECTED, "base", oracle)
        return run, str(error.exception)

    def captured_case(self, errors=1):
        run = driver.Run(self.output, None)
        expected = deepcopy(EXPECTED)
        expected["record_errors"] = errors
        qualifier = {"outcome":"degraded" if errors else "ok",
                     "notes":["one retained undecodable record"] if errors else [],
                     "unknown":{"record_errors":errors} if errors else {}}
        calls=[]
        def client(archive, *argv, **kwargs):
            calls.append((argv,kwargs))
            if "import" in argv:
                return admission(errors)
            if "read" in argv:
                return dict(deepcopy(ANSWER), result=deepcopy(qualifier))
            return dict(deepcopy(qualifier),up_to_date=True,unit="sessions",count=1,rows=[{"id":SID}],next_cursor=None)
        def command(argv, label, **kwargs):
            calls.append((argv,kwargs))
            export=Path(argv[-1]);export.mkdir()
            (export/'session.json').write_text(json.dumps(dict(deepcopy(ANSWER),result=qualifier)))
            (export/'manifest.json').write_text(json.dumps({"format":"polylogue-export-manifest","version":1,"sessions":1,"up_to_date":True,"result":qualifier}))
            return ""
        run.client=client;run.command=command
        return run,expected,calls

    def test_malformed_usable_result_is_degraded_through_wait_read_and_export(self):
        run,expected,calls=self.captured_case()
        run.admit(self.output/'archive',self.output/'input',expected,'malformed',checker)
        self.assertEqual(len(calls),4)
        self.assertTrue(all(kw['expected']==(5,) for _,kw in calls))
        actual=json.loads((self.output/'malformed-actual.json').read_text())
        self.assertEqual(actual['result']['unknown'],{'record_errors':1})
        self.assertEqual(actual['messages'],ANSWER['messages'])

    def test_base_still_requires_zero_exit_and_ok_qualified_artifacts(self):
        run,expected,calls=self.captured_case(0)
        run.admit(self.output/'archive',self.output/'input',expected,'base',checker)
        self.assertTrue(all(kw['expected']==(0,) for _,kw in calls))

    def test_malformed_wait_rejects_false_success_missing_diagnostics_and_wrong_coverage(self):
        for mutation in [dict(outcome='ok'),dict(source_rows={"scope":"imported_source_rows","basis":{"input":"synthetic-input","length":7},"state":"settled"}),dict(source_rows=None),dict(notes=[]),dict(unknown={}),dict(unknown={'record_errors':True})]:
            output=self.output/str(len(list(self.output.iterdir())));output.mkdir()
            run=driver.Run(output,None)
            comparisons=[];run.answers=lambda *a: comparisons.append(a)
            receipt=admission(1)
            receipt.update(mutation);run.client=lambda *a,**kw: receipt
            expected=dict(EXPECTED,record_errors=1)
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                run.admit(output/'archive',output/'input',expected,'malformed',checker)
            self.assertEqual(comparisons,[])

    def test_malformed_cannot_accept_ok_listing_or_unqualified_standalone_document(self):
        for where in ['listing','read']:
            output=self.output/where;output.mkdir()
            run=driver.Run(output,None)
            def client(archive,*argv,**kwargs):
                if 'read' in argv: return deepcopy(ANSWER)
                return {'outcome':'ok' if where=='listing' else 'degraded','up_to_date':True,'notes':['undecodable record'],'unknown':{'record_errors':1},'unit':'sessions','count':1,'rows':[{'id':SID}],'next_cursor':None}
            run.client=client
            with self.subTest(where=where),self.assertRaises(ValueError):
                run.answers(output/'archive',dict(EXPECTED,record_errors=1),'malformed',checker)

    def test_export_cannot_drop_or_upgrade_its_qualifiers(self):
        for target in ['session.json','manifest.json']:
            for replacement in [None,{'outcome':'ok','notes':[],'unknown':{}}]:
                output=self.output/str(len(list(self.output.iterdir())));output.mkdir()
                old=self.output;self.output=output
                run,expected,_=self.captured_case();self.output=old
                command=run.command
                def altered(argv,label,**kwargs):
                    result=command(argv,label,**kwargs)
                    path=Path(argv[-1])/target;value=json.loads(path.read_text())
                    if replacement is None:value.pop('result')
                    else:value['result']=replacement
                    path.write_text(json.dumps(value));return result
                run.command=altered
                with self.subTest(target=target,replacement=replacement),self.assertRaises(ValueError):
                    run.admit(output/'archive',output/'input',expected,'malformed',checker)

    def test_source_rows_settled_cannot_qualify_an_incomplete_session_read(self):
        run,expected,_=self.captured_case(0)
        original=run.client
        def incomplete(archive,*argv,**kwargs):
            answer=original(archive,*argv,**kwargs)
            if 'read' in argv: answer['result']={'outcome':'degraded','notes':['catalog incomplete'],'unknown':{}}
            return answer
        run.client=incomplete
        with self.assertRaisesRegex(ValueError,'expected ok outcome'):
            run.admit(self.output/'archive',self.output/'input',expected,'base',checker)

    def test_snapshot_append_preserves_prefix_and_refuses_replacement(self):
        provider, prepared = self.output / 'provider.jsonl', self.output / 'prepared.jsonl'
        provider.write_bytes(b'{"base":1}\n')
        prepared.write_bytes(b'{"base":1}\n{"next":2}\n')
        inode = provider.stat().st_ino
        change = driver.append_snapshot(provider, prepared)
        self.assertEqual(provider.read_bytes(), b'{"base":1}\n{"next":2}\n')
        self.assertEqual(provider.stat().st_ino, inode)
        self.assertEqual(change['appended_bytes'], len(b'{"next":2}\n'))
        self.assertEqual(change['after_bytes'], prepared.stat().st_size)
        for wrong in (b'{"replacement":1}\n', provider.read_bytes(), b''):
            before = provider.read_bytes()
            prepared.write_bytes(wrong)
            with self.assertRaisesRegex(ValueError, 'strictly extend'):
                driver.append_snapshot(provider, prepared)
            self.assertEqual(provider.read_bytes(), before)

    def test_original_capture_stays_distinct_from_planted_answer(self):
        run, _ = self.run_control(checker)
        self.assertEqual(json.loads((self.output / "base-actual.json").read_text()), ANSWER)
        self.assertEqual(json.loads((self.output / "base-corrupted.json").read_text())["messages"], [])
        self.assertEqual(ANSWER["messages"][0]["blocks"][0]["text"], "retained prompt")

    def test_oracle_accepting_mutation_is_control_failure_even_when_run_is_nonzero(self):
        run, message = self.run_control(SimpleNamespace(check=lambda *a: None))
        self.assertTrue(run.control_exercised)
        self.assertEqual(run.control_result["state"], "oracle_accepted")
        self.assertIn("oracle accepted", message)

    def test_invalid_baseline_never_qualifies_mutation_rejection(self):
        def reject(*args):
            raise ValueError("baseline is invalid")
        run, message = self.run_control(SimpleNamespace(check=reject))
        self.assertFalse(run.control_exercised)
        self.assertEqual(run.control_result["state"], "unrun")
        self.assertFalse((self.output / "base-corrupted.json").exists())
        self.assertEqual(message, "baseline is invalid")

    def test_wrong_identity_is_baseline_failure_not_a_qualified_mutant(self):
        for field in ('message', 'block'):
            output = self.output / field
            output.mkdir()
            run = CapturedReplies(output, "corrupt-answer")
            original = run.client
            def reply(archive, *argv, **kwargs):
                answer = original(archive, *argv, **kwargs)
                if "read" in argv:
                    target = answer['messages'][0] if field == 'message' else answer['messages'][0]['blocks'][0]
                    target['id'] = 'claude-code:foreign:prompt:0'
                return answer
            run.client = reply
            with self.subTest(field=field), self.assertRaises(ValueError):
                run.answers(self.output / "unused", EXPECTED, "base", checker)
            self.assertFalse(run.control_exercised)
            self.assertEqual(run.control_result['state'], 'unrun')
            self.assertFalse((output / 'base-corrupted.json').exists())

    def test_independent_oracle_rejects_only_after_valid_baseline(self):
        run, message = self.run_control(checker)
        self.assertEqual(run.control_result["state"], "oracle_rejected")
        self.assertTrue(run.control_result["baseline_valid"])
        self.assertEqual(run.control_result["reason"], "wrong message count")
        self.assertIn("intentional", message)

    def test_legacy_or_missing_source_settlement_never_reaches_answer_comparison(self):
        run = driver.Run(self.output, None)
        comparisons = []
        run.answers = lambda *a: comparisons.append(a)
        for receipt in [{}, {"canonical_complete": None}, {"canonical_complete": False},
                        {"canonical_complete": 0}, {"canonical_complete": 1},
                        {"canonical_complete": "true"}]:
            with self.subTest(receipt=receipt):
                run.client = lambda *a, **kw: receipt
                with self.assertRaisesRegex(ValueError, "source-row settlement"):
                    run.admit(self.output / "archive", self.output / "input", EXPECTED, "base", checker)
        self.assertEqual(comparisons, [])

    def test_affirmative_completion_proceeds_to_independent_answer_comparison(self):
        run = driver.Run(self.output, None)
        comparisons = []
        run.client = lambda *a, **kw: admission()
        run.answers = lambda *a: comparisons.append(a)
        run.admit(self.output / "archive", self.output / "input", EXPECTED, "base", checker)
        self.assertEqual(len(comparisons), 1)
        self.assertIs(comparisons[0][-1], checker)

    def test_loose_import_retains_explicit_context_and_passes_exact_file(self):
        run = driver.Run(self.output, None)
        calls = []
        run.client = lambda *a, **kw: calls.append(a) or admission()
        run.answers = lambda *a: None
        expected = deepcopy(EXPECTED)
        # The native ID is declared independently; never split the opaque reference.
        expected["session_id"] = "opaque:reference:with:colons"
        expected["import_context"]["context"][1] = "native:id:with:colons"
        for name in ("base", "repeat", "append", "empty", "malformed"):
            run.admit(self.output / "archive", self.output / "loose.jsonl", expected, name, checker)
            context = self.output / f"{name}-import-context.json"
            self.assertEqual(json.loads(context.read_text()), expected["import_context"])
            self.assertEqual(calls[-1], (self.output / "archive", "import", str(self.output / "loose.jsonl"),
                                        "--format", "claude-code-jsonl",
                                        "--context", str(context), "--wait", "--json"))

    def test_missing_or_invalid_context_refuses_before_import(self):
        run = driver.Run(self.output, None)
        calls = []
        run.client = lambda *a: calls.append(a)
        for context in (None, {}, {"version": True, "context": ["claude-main", "s"]},
                        {"version": 1, "context": ["claude-main", ""]},
                        {"version": 1, "context": ["claude-subagent", "s"]}):
            expected = dict(EXPECTED, import_context=context)
            with self.subTest(context=context), self.assertRaisesRegex(ValueError, "explicit v1"):
                run.admit(self.output / "archive", self.output / "loose.jsonl", expected, "base", checker)
        self.assertEqual(calls, [])

    def test_failed_report_is_escaped_portable_and_does_not_rewrite_evidence(self):
        root = self.output / 'run'
        root.mkdir()
        (root / '000.stdout').write_text('retained client output')
        (self.output / 'outside.txt').write_text('outside the report')
        (root / 'escape').symlink_to(self.output / 'outside.txt')
        result = {'success':False, 'failure':'candidate <script>bad()</script> failed',
                  'steps':{'base':{'outcome':'failed'}, 'append':{'outcome':'unrun'}},
                  'commands':[{'argv':['candidate','read'], 'label':'actual read', 'exit':2,
                               'stdout':'000.stdout','stderr':'escape'}],
                  'remaining':['actual MCP transport']}
        (root / 'result.json').write_text(json.dumps(result))
        before = (root / 'result.json').read_bytes()
        original = deepcopy(result)
        page = driver.render_report(result, root)
        rendered = page.read_text()
        self.assertEqual(result, original)
        self.assertEqual((root / 'result.json').read_bytes(), before)
        self.assertIn('candidate &lt;script&gt;bad()&lt;/script&gt; failed', rendered)
        self.assertIn('Not recorded in this run', rendered)
        self.assertIn('Not executed; no answer or acceptance is inferred.', rendered)
        self.assertIn('id="run-status" class="badge bad">failed', rendered)
        self.assertEqual(rendered.count('<script>'), 1)
        class Links(HTMLParser):
            def __init__(self):
                super().__init__(); self.links = []
            def handle_starttag(self, tag, attrs):
                if tag == 'a': self.links.append(dict(attrs)['href'])
        parser = Links(); parser.feed(rendered)
        self.assertIn('000.stdout', parser.links)
        self.assertNotIn('escape', parser.links)
        self.assertTrue(all(h.startswith('#') or (root / h).is_file() for h in parser.links))
        moved = self.output / 'relocated'
        root.rename(moved)
        self.assertIn('href="000.stdout"', driver.render_report(result, moved).read_text())
        for path in ['/etc/passwd', '../outside.txt', 'https://example.com', 'escape']:
            self.assertNotIn('<a ', artifact(moved, path))

    @unittest.skipUnless(args.binary, "supply an immutable real binary for launch refusal")
    def test_real_package_launch_refusal_is_classified_and_reaped(self):
        self.assertTrue(str(args.binary.resolve()).startswith("/nix/store/"))
        run = driver.Run(self.output, "launch-failure")
        run.binary = args.binary
        try:
            with self.assertRaises(ValueError):
                run.start(self.output / "archive")
        finally:
            run.stop(check=False)
        self.assertTrue(run.control_exercised)
        self.assertEqual(run.control_result["state"], "launch_refused")
        self.assertEqual(run.processes[0]["exit"], 4)
        self.assertFalse(run.processes[0].get("forced_kill", False))


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0], *remaining])
