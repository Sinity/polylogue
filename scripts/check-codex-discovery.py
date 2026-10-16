#!/usr/bin/env python3
"""Synthetic configured acquisition through the packaged CLI and SDK MCP server."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
import subprocess
import time

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location('native', ROOT / 'scripts/check-codex-native.py')
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)
mcp = native.mcp
S = native.S
P = 'prompt-only:synthetic'

class Run(mcp.Run):
    def start(self):
        args = ['daemon', '--config', self.output / 'config.json']
        self.calls.append({'launch': self.command(args)})
        self.daemon = subprocess.Popen(self.command(args), env=self.env,
                                       stdout=subprocess.DEVNULL, stderr=self.daemon_log)
        until = time.monotonic() + 30
        while True:
            value = self.cli(['status'], allowed=(0, 4, 5, 6))
            if value and value.get('acquisition'):
                return value
            assert self.daemon.poll() is None, 'configured daemon exited'
            assert time.monotonic() < until, 'configured startup timeout'
            time.sleep(.02)

def config(output, home):
    value = {'codex_home': {'home': str(home), 'machine': 'synthetic-machine', 'host': 'synthetic-host'}}
    (output / 'config.json').write_text(json.dumps(value))
    return value

def catalog(path, rows=()):
    connection = sqlite3.connect(path)
    connection.execute('CREATE TABLE threads(id TEXT PRIMARY KEY, rollout_path TEXT NOT NULL)')
    connection.executemany('INSERT INTO threads VALUES(?,?)', rows)
    connection.commit()
    connection.close()

def settled(run, session='codex:' + S):
    until = time.monotonic() + 30
    while True:
        discovered = run.cli(['origin:codex', '--json'])
        matches = [row for row in discovered['rows'] if row['id'] == session]
        value = run.cli(['tool_calls where session:' + matches[0]['id'], '--json']) if matches else {'rows': []}
        if value['rows']:
            return value
        assert time.monotonic() < until, value
        time.sleep(.03)

def one_pending(value):
    references = value['pending_references']['references']
    rows = [row for row in references if row['session'] == 'codex:' + P]
    assert len(rows) == 1, references
    pending = rows[0]
    assert pending['state'] == 'pending' and pending['conversation_coverage'] is None, pending
    assert pending['assistant_response'] is None, pending
    source = pending['source']
    assert source['provenance']['machine'] == 'synthetic-machine'
    assert source['provenance']['host'] == 'synthetic-host'
    assert source['provenance']['seen_at'] > 0
    return pending

def lifecycle(binary, output):
    run = Run(binary, output)
    home = output / 'codex-home'
    home.mkdir()
    sessions = home / 'sessions'
    sessions.mkdir()
    rollout = sessions / 'rollout-owned.jsonl'
    native.rollout(rollout, output='independent retained output')
    before_bytes = rollout.read_bytes()
    prompt = b'{"session_id":"prompt-only:synthetic","ts":123,"text":"a prompt is not an assistant response"}\n'
    (home / 'history.jsonl').write_bytes(prompt)
    catalog(home / 'state_5.sqlite')
    unrelated = home / 'unrelated.txt'
    unrelated.write_text('must remain unchanged')
    config(output, home)
    try:
        status = run.start()
        assert status['acquisition']['catalog']['state'] == 'accessible_empty', status
        assert status['acquisition']['rollouts']['entries'] == 1, status
        pending = one_pending(status)
        assert pending['source']['start'] == 0 and pending['source']['end'] == len(prompt)
        assert Path(pending['source']['provenance']['path']) == home / 'history.jsonl'
        calls = settled(run)
        assert len(calls['rows']) == 1, calls
        operation = calls['rows'][0]
        assert operation['invocation'] == 'C' and operation['turn'] == 'T', operation
        read = run.cli([operation['ref'], 'read', '--budget', '20000'])
        body = read['shown'][0]
        assert body['operation']['output'] == 'independent retained output', body
        # Raw source facts are independent of the product's parser output.
        source = body['sources'][0]
        assert before_bytes and source, source
        search = run.cli(['origin:codex', '--json'])
        assert all(row['id'] != 'codex:' + P for row in search['rows']), search
        assert one_pending(search) == pending
        missing = run.cli(['codex:' + P, 'read', '--budget', '20000'], allowed=(2, 3, 4, 5, 7))
        assert missing['outcome'] == 'error' and not missing.get('shown'), missing
        assert one_pending(missing) == pending
        run.client = mcp.Client(run)
        remote = run.client.tool('search', {'query': 'tool_calls where session:codex:' + S, 'budget_tokens': 20000})['structuredContent']
        assert remote['rows'] == calls['rows']
        assert one_pending(run.client.tool('status', {})['structuredContent']) == pending
        remote_read = run.client.tool('read', {'ref': operation['ref'], 'budget_tokens': 20000})['structuredContent']
        assert remote_read['shown'] == read['shown']
        run.client.close()
        run.client = None
        run.stop()
        repeat = run.start()
        assert one_pending(repeat) == pending
        assert settled(run)['rows'][0]['ref'] == operation['ref']
        run.stop()
        rollout.unlink()  # This exact disposable file is the only removal.
        after_removal = run.start()
        assert after_removal['acquisition']['rollouts']['state'] == 'accessible_empty'
        retained = settled(run)
        assert retained['rows'][0]['ref'] == operation['ref']
        assert run.cli([operation['ref'], 'read', '--budget', '20000'])['shown'] == read['shown']
        assert one_pending(after_removal) == pending
        assert unrelated.read_text() == 'must remain unchanged'
        run.stop()
        # A later conversation for P shares the provider identity, retaining
        # prompt observations as source evidence without creating Messages.
        original_s = native.S
        native.S = P
        try:
            native.rollout(rollout, output='later conversation output')
        finally:
            native.S = original_s
        later = run.start()
        p_calls = settled(run, 'codex:' + P)
        assert len(p_calls['rows']) == 1, p_calls
        p_sessions = run.cli(['id:codex:' + P, '--json'])
        assert len(p_sessions['rows']) == 1, p_sessions
        resolved = run.cli(['status'])['pending_references']['references']
        assert len(resolved) == 1 and resolved[0]['state'] == 'conversation_evidence', resolved
        assert resolved[0]['source'] == pending['source']
        run.stop()
        # Changed/malformed prompt bytes become another retained snapshot. The
        # old reference and its provenance survive, with explicit diagnostics.
        (home / 'history.jsonl').write_bytes(b'malformed\n{"session_id":')
        malformed = run.start()
        snapshots = malformed['pending_references']['snapshots']
        assert any(s['malformed_records'] == 1 and s['incomplete_tail'] for s in snapshots), malformed
        assert len(malformed['pending_references']['references']) == 1, malformed
        assert run.cli([operation['ref'], 'read', '--budget', '20000'])['shown'] == read['shown']
        run.stop()
        (sessions / 'rollout-malformed.jsonl').write_bytes(b'not-json\n')
        idle = run.cli(['daemon', '--config', output / 'config.json', '--until-idle', '--json'])
        assert idle['outcome'] == 'degraded'
        assert any(p['source_rows']['state'] == 'unavailable' for p in idle['progress']), idle
        run.start()
        assert run.cli([operation['ref'], 'read', '--budget', '20000'])['shown'] == read['shown']
        return {'operation': operation, 'pending': pending, 'retained_read': read, 'later_session': p_sessions,
                'removed_source_acquisition': after_removal['acquisition'], 'malformed_snapshots': snapshots}
    finally:
        native.close_run(run, output)

def acquisition_controls(binary, output):
    run = Run(binary, output)
    home = output / 'codex-home'
    home.mkdir()
    (home / 'sessions').mkdir()
    (home / 'history.jsonl').write_bytes(b'')
    catalog(home / 'state_5.sqlite')
    config(output, home)
    try:
        empty = run.start()
        assert all(empty['acquisition'][key]['state'] == 'accessible_empty' for key in ('catalog', 'history', 'rollouts')), empty
        run.stop()
        value = config(output, home)
        value['codex_home'].update(rollouts='missing-directory', history='missing-history', catalog='missing-catalog')
        (output / 'config.json').write_text(json.dumps(value))
        missing = run.start()
        assert all(missing['acquisition'][key]['state'] == 'unavailable' for key in ('catalog', 'history', 'rollouts')), missing
        assert missing['pending_references']['references'] == []
        assert not (home / 'missing-catalog').exists()
        run.stop()
        # A corrupt catalog is unavailable evidence, never an empty authority.
        (home / 'corrupt.sqlite').write_bytes(b'not a database')
        value['codex_home']['catalog'] = 'corrupt.sqlite'
        (output / 'config.json').write_text(json.dumps(value))
        corrupt = run.start()
        assert corrupt['acquisition']['catalog']['state'] == 'unavailable'
        assert corrupt['acquisition']['catalog']['entries'] is None
        run.stop()
        config(output, home)
        (home / 'external').mkdir()
        path = home / 'external' / 'rollout-relative.jsonl'
        native.rollout(path, output='catalog relative path output')
        connection = sqlite3.connect(home / 'state_5.sqlite')
        connection.execute('INSERT INTO threads VALUES(?,?)', (S, 'external/rollout-relative.jsonl'))
        connection.commit()
        connection.close()
        relative = run.start()
        assert relative['acquisition']['rollouts']['state'] == 'accessible_empty'
        assert relative['acquisition']['catalog']['entries'] == 1
        result = settled(run)
        assert len(result['rows']) == 1, result
        assert run.cli([result['rows'][0]['ref'], 'read', '--budget', '20000'])['shown'][0]['operation']['output'] == 'catalog relative path output'
        return {'empty': empty['acquisition'], 'missing': missing['acquisition'], 'corrupt': corrupt['acquisition'], 'relative': relative['acquisition']}
    finally:
        native.close_run(run, output)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'binary': str(args.binary.resolve()), 'sha256': hashlib.sha256(args.binary.read_bytes()).hexdigest(), 'cases': {}}
    try:
        for name, function in [('lifecycle', lifecycle), ('acquisition', acquisition_controls)]:
            path = args.output / name
            path.mkdir()
            report['cases'][name] = function(args.binary.resolve(), path)
        report['outcome'] = 'passed'
    except BaseException as error:
        report.update(outcome='failed', error=repr(error))
        raise
    finally:
        (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'outcome': report['outcome'], 'report': str(args.output / 'report.json')}))

if __name__ == '__main__':
    main()
