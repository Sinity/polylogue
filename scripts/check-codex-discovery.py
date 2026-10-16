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
        by_tag = {}
        for source in body['sources']:
            origin = source['origin']
            assert origin['representation'] == 'rollout', source
            assert source['start'] == origin['start'] and source['end'] == origin['end'], source
            by_tag[origin['tag']] = json.loads(before_bytes[origin['start']:origin['end']])
        assert set(by_tag) == {'call', 'result', 'status'}, by_tag
        call, result, completed = (by_tag[tag] for tag in ('call', 'result', 'status'))
        assert call['type'] == 'response_item' and call['payload']['type'] == 'function_call'
        assert call['payload']['call_id'] == 'C' and call['payload']['name'] == 'exec_command'
        assert json.loads(call['payload']['arguments']) == {'cmd': 'same command'}
        assert result['type'] == 'response_item' and result['payload']['type'] == 'function_call_output'
        assert result['payload']['call_id'] == 'C' and result['payload']['output'] == 'independent retained output'
        assert completed['type'] == 'event_msg' and completed['payload']['type'] == 'item_completed'
        assert completed['payload']['thread_id'] == S and completed['payload']['turn_id'] == 'T'
        assert completed['payload']['item']['id'] == 'C' and completed['payload']['item']['aggregated_output'] == 'independent retained output'
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

def growth_controls(binary, output):
    run = Run(binary, output)
    home = output / 'codex-home'
    home.mkdir()
    sessions = home / 'sessions'
    sessions.mkdir()
    original = sessions / 'rollout-owned.jsonl'
    native.rollout(original, output='usable despite optional discovery growth')
    original_bytes = original.read_bytes()
    # Put requested P last, beyond the general bounded metadata sample.
    rows = [{'session_id': 'unrelated-' + str(n), 'text': 'synthetic prompt'} for n in range(63)]
    rows.append({'session_id': P, 'text': 'last pending prompt'})
    (home / 'history.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    catalog(home / 'state_5.sqlite')
    config(output, home)
    try:
        status = run.start()
        summary = status['pending_references']
        assert status['outcome'] != 'error' and summary['reference_observations'] == 64, status
        assert summary['references_shown'] < 64 and summary['references_omitted'] == 64 - summary['references_shown'], summary
        assert summary['conversation_coverage'] is None and not summary['presentation_complete'], summary
        assert all(row['session'] != 'codex:' + P for row in summary['references']), summary
        selected = run.cli(['codex:' + P, 'read', '--budget', '20000'], allowed=(2, 3, 4, 5, 7))
        assert one_pending(selected)['state'] == 'pending', selected
        assert selected['pending_references']['matching_reference_observations'] == 1
        calls = settled(run)
        operation = calls['rows'][0]
        read = run.cli([operation['ref'], 'read', '--budget', '20000'])
        assert read['shown'][0]['operation']['output'] == 'usable despite optional discovery growth'
        run.client = mcp.Client(run)
        # Optional reference arrays must yield to an otherwise fitting response.
        remote = run.client.tool('search', {'query': 'tool_calls where session:codex:' + S, 'budget_tokens': 2000})['structuredContent']
        assert remote['rows'] == calls['rows'] and remote['outcome'] != 'error', remote
        assert remote['pending_references']['references'] == [], remote
        assert remote['pending_references']['references_omitted'] == 64, remote
        run.client.close()
        run.client = None
        run.stop()
        # All300 additional source attempts deduplicate to the known raw input.
        # Completed acquisition must survive its public report sample limit.
        for n in range(300):
            (sessions / ('rollout-extra-' + str(n) + '.jsonl')).write_bytes(original_bytes)
        many = run.start()
        acquisition = many['acquisition']
        assert acquisition['retained_count'] == 302, acquisition
        assert acquisition['retained_count_unit'] == 'successful_source_attempts'
        assert acquisition['retained_omitted'] == 302 - len(acquisition['retained']) > 0
        assert acquisition['rollouts']['entries'] == 301
        assert acquisition['history']['entries'] == 1 and acquisition['history']['count_unit'] == 'retained_inputs'
        assert len(json.dumps(acquisition).encode()) < 16384, acquisition
        assert settled(run)['rows'][0]['ref'] == operation['ref']
        assert run.cli([operation['ref'], 'read', '--budget', '20000'])['shown'] == read['shown']
        return {'references': summary, 'exact_pending': selected['pending_references'],
                'bounded_mcp_search': remote, 'many_rollout_attempts': acquisition, 'operation': operation['ref']}
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
        for name, function in [('lifecycle', lifecycle), ('acquisition', acquisition_controls), ('growth', growth_controls)]:
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
