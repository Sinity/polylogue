import unittest
from unittest import mock
from types import SimpleNamespace
from pathlib import Path
import tempfile
import sqlite3
import json
import measure


class CpuAttribution(unittest.TestCase):
    def data(self):
        parts = {name: {'calls': 1, 'seconds': 1.0, 'cpu_seconds': 1.0}
                 for name in measure.REQUIRED_PARTS}
        record = {'first_build_attribution': parts}
        profile = {'workers': {'polylogue-writer': {'cpu_seconds': 33.0}},
                   'pipeline_timings': {'rows': 90.0, 'parse': 80.0}}
        return profile, record

    def test_writer_reconciliation_excludes_other_thread_and_keeps_overlapping_walls(self):
        profile, record = self.data()
        result = measure.summarize(profile, [record])
        self.assertEqual(result['exclusive_writer_cpu_seconds'], 33.0)
        self.assertEqual(result['writer_cpu_coverage'], 1.0)
        self.assertEqual(result['writer_cpu_residual_seconds'], 0.0)
        self.assertEqual(result['overlapping_stage_seconds'], {'rows': 90.0, 'parse': 80.0})

    def test_missing_or_unknown_cpu_cannot_be_a_successful_profile(self):
        profile, record = self.data()
        record['first_build_attribution']['selected_blocks']['cpu_seconds'] = None
        with self.assertRaises(RuntimeError):
            measure.summarize(profile, [record])
        del record['first_build_attribution']['selected_blocks']
        with self.assertRaises(RuntimeError):
            measure.summarize(profile, [record])

    def test_no_profile_cannot_be_accepted(self):
        profile, _ = self.data()
        with self.assertRaises(RuntimeError):
            measure.summarize(profile, [])


class CheckpointPair(unittest.TestCase):
    def full_pair(self, arms):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected = root / 'expected.json'
            expected.write_text(json.dumps({'files': 7773, 'bytes': 8590127331}))
            args = SimpleNamespace(work=str(root / 'pair'), expected=str(expected),
                generator='/synthetic/generator', lengths='/synthetic/lengths',
                pair_files=None, binary='/synthetic/before', revision='before',
                candidate_binary='/synthetic/after', candidate_revision='after')
            files = {'/synthetic/full-input': 8590127331}
            with mock.patch.dict(measure.os.environ, {'GITHUB_ACTIONS': 'true', 'RUNNER_TEMP': temporary}), \
                 mock.patch.object(measure, 'available_space', return_value=64 * 1024**3), \
                 mock.patch.object(measure, 'available_memory', return_value=16 * 1024**3), \
                 mock.patch.object(measure, 'command', return_value=SimpleNamespace(stdout='synthetic generation')), \
                 mock.patch.object(measure, 'generated_population', return_value=files), \
                 mock.patch.object(measure, 'run_arm', side_effect=arms) as run:
                code = measure.main(args)
            report = json.loads((root / 'pair/results/paired-summary.json').read_text())
            return code, run.call_args_list, files, report

    def arm(self):
        return {'complete': True, 'failure': None,
                'search_signature': {'ranked': 'independent-search'},
                'context_and_model_signature': {'responses': {'rows': 3, 'sha256': 'independent-context'}}}

    def test_full_population_runs_both_arms_without_the_small_filter(self):
        code, calls, files, report = self.full_pair([self.arm(), self.arm()])
        self.assertEqual(code, 0)
        self.assertEqual(len(calls), 2)
        self.assertEqual([call.args[0].revision for call in calls], ['before', 'after'])
        self.assertTrue(all(call.args[3] is files for call in calls))
        self.assertEqual([call.args[4] for call in calls], ['baseline', 'candidate'])
        self.assertTrue(report['search_and_context_model_parity'])

    def test_failed_baseline_preserves_failure_without_starting_candidate(self):
        failed = {'complete': False, 'failure': 'incomplete checkpoint'}
        code, calls, _, report = self.full_pair([failed])
        self.assertEqual(code, 1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(report['baseline'], failed)
        self.assertIsNone(report['candidate'])
        self.assertFalse(report['search_and_context_model_parity'])

    def test_changed_response_reference_cannot_pass_the_pair(self):
        changed = self.arm()
        changed['context_and_model_signature']['responses']['sha256'] = 'changed-version'
        code, calls, _, report = self.full_pair([self.arm(), changed])
        self.assertEqual(code, 1)
        self.assertEqual(len(calls), 2)
        self.assertFalse(report['search_and_context_model_parity'])

    def test_worker_cpu_requires_all_four_complete_clocks_and_keeps_wall_separate(self):
        workers = {f'polylogue-parse-{n}': {'cpu_seconds': float(n + 1)} for n in range(4)}
        workers['polylogue-writer'] = {'cpu_seconds': 7.0}
        profile = {'workers': workers, 'pipeline_timings': {'parse': 99.0}}
        actual = measure.worker_cpu(profile)
        self.assertEqual(actual['parser_worker_cpu_seconds'], 10.0)
        self.assertEqual(actual['writer_cpu_seconds'], 7.0)
        self.assertEqual(actual['overlapping_stage_seconds'], {'parse': 99.0})
        workers['polylogue-parse-2']['cpu_seconds'] = None
        with self.assertRaises(RuntimeError):
            measure.worker_cpu(profile)
        del workers['polylogue-parse-2']
        with self.assertRaises(RuntimeError):
            measure.worker_cpu(profile)

    def database(self, directory, shift):
        directory.mkdir()
        conn = sqlite3.connect(directory / 'index.db')
        conn.executescript("""
            CREATE TABLE input_ref(pk INTEGER,id TEXT);
            CREATE TABLE parse_state(input_pk INTEGER,chain INTEGER);
            CREATE TABLE message_version(pk INTEGER,version_hash BLOB);
            CREATE TABLE parser_context_response(input_pk INTEGER,chain INTEGER,namespace TEXT,
                session TEXT,key TEXT,version_pk INTEGER,source_start INTEGER,source_end INTEGER);
            CREATE TABLE parser_context_alias(input_pk INTEGER,chain INTEGER,namespace TEXT,uuid TEXT,target TEXT);
            CREATE TABLE parser_context_value(input_pk INTEGER,chain INTEGER,namespace TEXT,key TEXT,value TEXT NOT NULL);
            CREATE TABLE session(pk INTEGER,id TEXT);
            CREATE TABLE message(pk INTEGER,id TEXT,parent_id TEXT,role TEXT,input_pk INTEGER,
                source_chain INTEGER,source_start INTEGER,source_end INTEGER,session_order TEXT,
                in_progress INTEGER,session_pk INTEGER);
            CREATE TABLE message_source(message_pk INTEGER,input_pk INTEGER,chain INTEGER,version_pk INTEGER);
        """)
        i, c, v, m, session = [shift + n for n in range(1, 6)]
        conn.execute('INSERT INTO input_ref VALUES(?,?)', (i, 'synthetic-input'))
        conn.execute('INSERT INTO parse_state VALUES(?,?)', (i, c))
        conn.execute('INSERT INTO message_version VALUES(?,?)', (v, b'version-identity'))
        conn.execute('INSERT INTO parser_context_response VALUES(?,?,?,?,?,?,?,?)',
                     (i,c,'source','session','response',v,4,19))
        conn.execute('INSERT INTO parser_context_alias VALUES(?,?,?,?,?)', (i,c,'source','alias','response'))
        conn.execute('INSERT INTO parser_context_value VALUES(?,?,?,?,?)', (i,c,'source','state','null'))
        conn.execute('INSERT INTO session VALUES(?,?)', (session,'session'))
        conn.execute('INSERT INTO message VALUES(?,?,?,?,?,?,?,?,?,?,?)',
                     (m,'message',None,'assistant',i,c,4,19,'source-order',0,session))
        conn.execute('INSERT INTO message_source VALUES(?,?,?,?)', (m,i,c,v))
        conn.commit()
        conn.close()

    def test_context_and_citation_parity_uses_natural_keys_and_detects_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            before, after = root / 'before', root / 'after'
            self.database(before, 0)
            self.database(after, 100)
            expected = measure.context_and_model_signature(before)
            self.assertEqual(measure.context_and_model_signature(after), expected)
            for sql, changed in [
                ('UPDATE parser_context_response SET source_end=20', 'responses'),
                ("UPDATE parser_context_alias SET target='another'", 'aliases'),
                ("UPDATE parser_context_value SET value='0'", 'states'),
                ('UPDATE message SET in_progress=1', 'selected_messages'),
            ]:
                conn = sqlite3.connect(after / 'index.db')
                conn.execute('BEGIN')
                conn.execute(sql)
                conn.commit()
                actual = measure.context_and_model_signature(after)
                self.assertNotEqual(actual[changed], expected[changed])
                conn.close()



if __name__ == '__main__':
    unittest.main()
