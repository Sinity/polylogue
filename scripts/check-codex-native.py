#!/usr/bin/env python3
"""Independent synthetic Codex representation facts through the ordinary binary and MCP."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parent.parent

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
mcp=load(ROOT/'scripts/check-mcp-read.py','native_mcp')
compact=mcp.bounded.compact
S='native-synthetic-session'
REF='codex:'+S

def native_item(call='C',turn='T',output=None,source='agent',status='failed',exit_code=1):
    item={'type':'commandExecution','id':call,'command':'same command','cwd':'/synthetic','source':source,'status':status,'exitCode':exit_code,'durationMs':5,'commandActions':[]}
    if output is not None:item['aggregatedOutput']=output
    return turn,item

def history(path,items,started_at_ms=None):
    connection=sqlite3.connect(path)
    connection.executescript((ROOT/'crates/store/tests/fixtures/codex-history-v0162.sql').read_text())
    connection.execute('CREATE TABLE _sqlx_migrations(version BIGINT PRIMARY KEY,success BOOLEAN NOT NULL)')
    connection.executemany('INSERT INTO _sqlx_migrations VALUES(?,1)',[(n,)for n in range(1,9)])
    for ordinal,(turn,item) in enumerate(items,1):
        connection.execute('INSERT INTO thread_items(thread_id,turn_id,item_id,rollout_ordinal,updated_at_ordinal,created_at_ms,item_type,item_json,started_at_ms) VALUES(?,?,?,?,?,?,?,?,?)',(S,turn,item['id'],ordinal,ordinal,1,item['type'],compact(item),started_at_ms))
    connection.commit();connection.close()

def rollout(path,scope=True,output='',call='C',turn='T',wrappers=False):
    rows=[{'type':'session_meta','payload':{'id':S,'source':'cli'}}]
    if scope:rows.append({'type':'turn_context','payload':{'turn_id':turn,'model':'synthetic-model'}})
    if wrappers:
        for wrapper in ['W1','W2']:
            rows.extend([{'type':'response_item','payload':{'type':'function_call','id':'message-'+wrapper,'call_id':wrapper,'name':'exec','arguments':'{}'}},{'type':'event_msg','payload':{'type':'token_count','info':None}}])
    else:
        rows.extend([{'type':'response_item','payload':{'type':'function_call','id':'M','call_id':call,'name':'exec_command','arguments':compact({'cmd':'same command'})}},{'type':'response_item','payload':{'type':'function_call_output','call_id':call,'output':output}}])
    rows.append({'type':'event_msg','payload':{'type':'item_completed','thread_id':S,'turn_id':turn,'item':{'type':'CommandExecution','id':call,'source':'agent','command':['same','command'],'cwd':'/synthetic','status':'failed','exit_code':1,'aggregated_output':output,'duration':{'secs':0,'nanos':5000000}}}})
    path.write_text(''.join(compact(row)+'\n'for row in rows))

def imported(run,path,format):
    reply=run.cli(['import',path,'--format',format,'--wait','--wait-timeout','30'],allowed=(0,2,3,4,5,7))
    assert reply and reply.get('retained'),reply
    assert reply['outcome']!='error',reply
    return reply

def query(run,table='tool_calls',text=None):
    q=table+' where session:'+REF
    if text is not None:q+=' text:'+text
    value=run.cli([q,'--json'],allowed=(0,2,3,4,5,7))
    assert value['outcome']!='error',value
    return q,value

def verify_case(binary,output,order,wrappers=False):
    run=mcp.Run(binary,output)
    try:
        run.start()
        rollout_path=output/'rollout.jsonl';rollout(rollout_path,wrappers=wrappers)
        native=output/'history.sqlite';history(native,[native_item()])
        receipts=[]
        for kind in order:
            receipts.append(imported(run,rollout_path if kind=='R' else native,'codex-rollout'if kind=='R'else'codex-history-sqlite'))
            _,population=query(run,'tool_items')
            assert len(population['rows'])>=1,population
        _,calls=query(run)
        # Both wrappers overlap the same child. They do not add counted commands.
        commands=[row for row in calls['rows']if row['kind']=='CommandExecution']
        assert len(commands)==1,calls
        assert calls['population']['tool_calls'] == 1 and calls['population']['correspondence_complete'], calls
        assert calls['population']['retained_items'] is None, calls
        operation=commands[0]
        result=run.cli([operation['ref'],'read','--budget','20000'])
        facts=result['shown'][0]['operation']
        associations = result['shown'][0]['associations']
        assert result['shown'][0]['selected'], result
        assert all(a['message_body'] > 0 and len(a['body_version']) == 64 and a['index_generation'] for a in associations), associations
        if not wrappers:
            assert len(associations) == 1 and associations[0]['kind'] == 'message_owner', associations
        assert facts['result_present'] and facts['output']=='' and facts['outcome']=='error'and facts['exit_code']==1,facts
        assert facts['invocation']=='C'and facts['turn']=='T',facts
        assert facts['tool'] is None if wrappers else facts['tool'] == 'exec_command', facts
        assert facts['message'] is None if wrappers else facts['message']==REF+':M',facts
        assert facts['wrapper'] is None,facts
        if wrappers:
            assert len(facts['wrapper_overlaps'])==2 and all(ref.startswith('operation:provisional-v1:')for ref in facts['wrapper_overlaps']),facts
            associations = result['shown'][0]['associations']
            assert len(associations) == 2 and all(a['kind'] == 'wrapper_overlap' for a in associations), associations
            assert {a['message'] for a in associations} == {REF + ':message-W1', REF + ':message-W2'}, associations
            assert calls['population']['tool_calls']==1,calls
        # The two representations preserve their different raw text presence.
        assert any('core-to-ui-empty-output-filter'in source['constructor']for source in result['shown'][0]['representations']),result
        assert any(source['constructor'].endswith('/core')for source in result['shown'][0]['representations']),result
        q,hits=query(run,text='same')
        assert hits['count']==0,hits  # Command text is not output text.
        run.client=mcp.Client(run)
        remote=run.client.tool('search',{'query':'tool_calls where session:'+REF,'budget_tokens':20000})['structuredContent']
        assert remote['rows']==calls['rows'],remote
        read=run.client.tool('read',{'ref':operation['ref'],'budget_tokens':20000})['structuredContent']
        assert read['shown']==result['shown'],read
        run.client.close();run.client=None
        run.stop();rollout_path.unlink();native.unlink();run.start()
        replay=query(run)[1]
        assert replay['rows']==calls['rows'],replay
        # A lost original path is not consulted by read/restart or same-request retry.
        for receipt,path,format in zip(receipts,[rollout_path if k=='R'else native for k in order],['codex-rollout'if k=='R'else'codex-history-sqlite'for k in order]):
            repeated=run.cli(['import',path,'--format',format,'--request-id',receipt['request_id'],'--seen-at',str(receipt['seen_at'])])
            assert repeated['retained']==receipt['retained'],repeated
        # The UI snapshot is an alternative witness, not a contributor to the
        # selected core body. Losing it cannot invalidate that independent body.
        native_receipt = next(receipt for receipt, kind in zip(receipts, order) if kind == 'N')
        native_hash = native_receipt['source_rows']['basis']['original_hash']
        native_blob = output / 'archive' / 'blobs' / native_hash[:2] / (native_hash + '.zst')
        run.stop()
        native_blob.rename(native_blob.with_suffix('.unavailable'))
        run.start()
        independent = run.cli([operation['ref'], 'read', '--budget', '20000'])
        assert independent['shown'] == result['shown'], independent
        return {'reference':operation['id'],'selected':operation['ref'],'rows':calls['rows'],'receipts':receipts}
    finally:
        for client in run.clients:
            if client.process.poll()is None:client.close()
        run.stop();run.daemon_log.close();run.socket_owner.cleanup()
        (output/'commands.json').write_text(json.dumps(run.calls,indent=2)+'\n')
        (output/'mcp-calls.json').write_text(json.dumps(run.mcp_calls,indent=2)+'\n')

def close_run(run, output):
    for client in run.clients:
        if client.process.poll() is None:
            client.close()
    run.stop()
    run.daemon_log.close()
    run.socket_owner.cleanup()
    (output / 'commands.json').write_text(json.dumps(run.calls, indent=2) + '\n')
    (output / 'mcp-calls.json').write_text(json.dumps(run.mcp_calls, indent=2) + '\n')


def qualification_controls(binary, output):
    run = mcp.Run(binary, output)
    try:
        run.start()
        path = output / 'history.sqlite'
        history(path, [native_item(), native_item(turn='T2', output='unique-output'),
                       native_item(call='item-7'), native_item(call='user', source='userShell'),
                       native_item(call='preexecution', exit_code=None),
                       native_item(call='missing', status='inProgress', exit_code=None)])
        imported(run, path, 'codex-history-sqlite')
        _, all_items = query(run, 'tool_items')
        assert len(all_items['rows']) == 6, all_items
        user = next(row for row in all_items['rows'] if row['invocation'] == 'user')
        assert run.cli([user['ref'], 'read', '--budget', '20000'])['shown'][0]['operation']['tool'] is None
        _, calls = query(run)
        assert len(calls['rows']) == 2, calls
        assert len({row['id'] for row in calls['rows']}) == 2, calls
        assert not calls['population']['correspondence_complete'], calls
        assert calls['population']['tool_calls'] is None and calls['population']['failed'] is None, calls
        assert query(run, text='unique-output')[1]['count'] == 1
        omitted = next(row for row in calls['rows'] if row['turn'] == 'T')
        facts = run.cli([omitted['ref'], 'read', '--budget', '20000'])['shown'][0]['operation']
        assert facts['result_present'] and facts['output'] is None, facts
        # Identical item JSON, distinct physical-row qualification facts. An
        # immutable ref must never select the first body by a colliding hash.
        _, row_item = native_item(call='row-evidence', status='completed', exit_code=None)
        absent = output / 'row-unknown.sqlite'
        started = output / 'row-started.sqlite'
        history(absent, [('T', row_item)])
        history(started, [('T', row_item)], started_at_ms=1)
        imported(run, absent, 'codex-history-sqlite')
        before = next(row for row in query(run, 'tool_items')[1]['rows'] if row['invocation'] == 'row-evidence')
        old = run.cli([before['ref'], 'read', '--budget', '20000'])['shown'][0]
        assert old['operation']['agent_class'] == 'unknown' and not old['operation']['result_present'], old
        imported(run, started, 'codex-history-sqlite')
        after = next(row for row in query(run, 'tool_items')[1]['rows'] if row['invocation'] == 'row-evidence')
        assert before['id'] == after['id'] and before['ref'] != after['ref'], (before, after)
        current = run.cli([after['ref'], 'read', '--budget', '20000'])['shown'][0]
        assert current['operation']['agent_class'] == 'agent' and current['operation']['result_present'], current
        historical = run.cli([before['ref'], 'read', '--budget', '20000'])['shown'][0]
        assert historical['operation'] == old['operation'] and not historical['selected'], historical
        a = old['sources'][0]['origin']; b = current['sources'][0]['origin']
        assert a['raw_hash'] == b['raw_hash'] and a['row_hash'] != b['row_hash'], (a, b)
        return {'classification': all_items, 'row_evidence': {'before': before['ref'], 'after': after['ref']}}
    finally:
        close_run(run, output)


def refinement_control(binary, output, order):
    run = mcp.Run(binary, output)
    try:
        run.start()
        full = output / 'full.jsonl'
        rollout(full, scope=False)
        rows = [json.loads(line) for line in full.read_text().splitlines()]
        partial = output / 'partial.jsonl'
        partial.write_text(''.join(compact(row) + '\n' for row in rows[:-1]))
        native = output / 'history.sqlite'
        history(native, [native_item()])
        old_ref = None
        for kind in order:
            imported(run, {'P': partial, 'F': full, 'N': native}[kind],
                     'codex-history-sqlite' if kind == 'N' else 'codex-rollout')
            _, items = query(run, 'tool_items')
            provisional = [row for row in items['rows'] if row['id'].startswith('operation:provisional-v1:')]
            if kind == 'P' and 'F' not in order[:order.index(kind)]:
                assert provisional and not items['population']['correspondence_complete'], items
                old_ref = provisional[0]['ref']
        _, calls = query(run)
        assert len(calls['rows']) == 1, calls
        if old_ref:
            historical = run.cli([old_ref, 'read', '--budget', '20000'])
            assert historical['shown'][0]['operation']['turn'] is None, historical
            assert not historical['shown'][0]['selected'] and not historical['shown'][0]['associations'], historical
            resolved = run.cli([old_ref.split('@')[0], 'read', '--budget', '20000'])
            assert resolved['shown'][0]['id'] == calls['rows'][0]['id'], resolved
        return calls
    finally:
        close_run(run, output)


def unavailable_control(binary, output):
    run = mcp.Run(binary, output)
    try:
        run.start()
        path = output / 'history.sqlite'
        history(path, [native_item(output='retained-output')])
        imported(run, path, 'codex-history-sqlite')
        _, calls = query(run)
        reference = calls['rows'][0]['ref']
        run.stop()
        blobs = list((output / 'archive').rglob('*.zst'))
        assert blobs, 'retained blob fixture absent'
        for blob in blobs:
            blob.rename(blob.with_suffix('.unavailable'))
        run.start()
        result = run.cli([reference, 'read', '--budget', '20000'], allowed=(0, 2, 3, 4, 5, 7))
        assert result['outcome'] == 'error', result
        assert not result.get('shown'), result
        return result
    finally:
        close_run(run, output)


def missing_and_wrong_session_controls(binary, output):
    run = mcp.Run(binary, output)
    try:
        run.start()
        path = output / 'missing.jsonl'
        rollout(path)
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        path.write_text(''.join(compact(row) + '\n' for row in rows[:3]))
        imported(run, path, 'codex-rollout')
        _, calls = query(run)
        assert len(calls['rows']) == 1, calls
        item = run.cli([calls['rows'][0]['ref'], 'read', '--budget', '20000'])['shown'][0]['operation']
        assert not item['result_present'] and item['output'] is None, item
        assert calls['population']['tool_calls'] == 1 and calls['population']['failed'] is None, calls
        assert not calls['population']['outcomes_complete'], calls
        late = output / 'missing-native-turn.jsonl'
        rollout(late, call='late')
        late_rows = [json.loads(line) for line in late.read_text().splitlines()]
        del late_rows[-1]['payload']['turn_id']
        # Keep current turn context, omit the ResponseItem call/output so it
        # cannot independently supply scope for this late native completion.
        late.write_text(''.join(compact(row) + '\n' for row in [late_rows[0], late_rows[1], late_rows[-1]]))
        imported(run, late, 'codex-rollout')
        late_item = next(row for row in query(run, 'tool_items')[1]['rows'] if row['invocation'] == 'late')
        assert late_item['turn'] is None and late_item['correspondence'] == 'unresolved', late_item
        assert late_item['id'].startswith('operation:provisional-v1:'), late_item
        wrong = output / 'wrong-session.jsonl'
        rollout(wrong, scope=False, call='wrong')
        rows = [json.loads(line) for line in wrong.read_text().splitlines()]
        rows[-1]['payload']['thread_id'] = 'foreign-session'
        wrong.write_text(''.join(compact(row) + '\n' for row in rows))
        imported(run, wrong, 'codex-rollout')
        _, all_items = query(run, 'tool_items')
        wrong_item = next(row for row in all_items['rows'] if row['invocation'] == 'wrong')
        assert wrong_item['id'].startswith('operation:provisional-v1:') and wrong_item['correspondence'] == 'unresolved', wrong_item
        assert not all_items['population']['correspondence_complete'], all_items
        return all_items
    finally:
        close_run(run, output)


def chunked_output_control(binary, output):
    run = mcp.Run(binary, output)
    try:
        run.start()
        text = 'begin-sentinel ' + 'neutral ' * 20000 + ' end-sentinel'
        path = output / 'history.sqlite'
        history(path, [native_item(output=text)])
        imported(run, path, 'codex-history-sqlite')
        _, hits = query(run, text='end-sentinel')
        assert hits['count'] == 1, hits
        item = run.cli([hits['rows'][0]['ref'], 'read', '--budget', '100000'])['shown'][0]['operation']
        assert item['output'] == text, 'chunked output changed'
        return {'bytes': len(text.encode()), 'reference': hits['rows'][0]['ref']}
    finally:
        close_run(run, output)


def giant_admission_control(binary, output):
    run = mcp.Run(binary, output)
    try:
        run.start()
        path = output / 'giant.jsonl'
        rollout(path, output='neutral ' * 150000)
        reply = run.cli(['import', path, '--format', 'codex-rollout', '--wait', '--wait-timeout', '30'], allowed=(0, 2, 3, 4, 5, 7))
        assert reply['retained'] and reply['source_rows']['state'] == 'unavailable', reply
        assert reply['outcome'] != 'ok', reply
        return reply
    finally:
        close_run(run, output)


def command_representation_control(binary, output):
    # Expected UI facts follow the pinned from_raw constructor, shlex1.3's
    # foo-bar/empty quoting vectors and Codex sanitizer.rs's existing 16-byte
    # Bearer test. This fixture is not a second command renderer.
    argv = ['printf', '', 'white space', 'Bearer abcdefghijklmnop']
    display = "printf '' 'white space' 'Bearer [REDACTED_SECRET]'"
    run = mcp.Run(binary, output)
    try:
        run.start()
        core = output / 'command.jsonl'
        rollout(core)
        rows = [json.loads(line) for line in core.read_text().splitlines()]
        rows[-1]['payload']['item']['command'] = argv
        rows[-1]['payload']['item']['parsed_cmd'] = []
        rows = [rows[0], rows[1], rows[-1]]
        core.write_text(''.join(compact(row) + '\n' for row in rows))
        native = output / 'history.sqlite'
        turn, item = native_item()
        item['command'] = display
        history(native, [(turn, item)])
        imported(run, native, 'codex-history-sqlite')
        imported(run, core, 'codex-rollout')
        _, calls = query(run)
        assert calls['count'] == 1, calls
        selected = calls['rows'][0]
        current = run.cli([selected['ref'], 'read', '--budget', '20000'])
        facts = current['shown'][0]['operation']
        assert facts['command'] == argv and facts['message'] is None and facts['tool'] is None, facts
        ui_hash = next(cite['version_hash'] for cite in current['shown'][0]['representations'] if cite['origin']['representation'] == 'history')
        projected = run.cli([selected['id'] + '@' + ui_hash, 'read', '--budget', '20000'])
        assert projected['shown'][0]['operation']['command'] == display, projected
        assert projected['shown'][0]['operation']['command'] != argv
        assert all(cite['adapter_revision'] == '092d3acd6bec3e3a14bdc7e7a2810ab628ab759d' for cite in current['shown'][0]['representations'])
        return {'reference': selected['id'], 'argv': argv, 'ui_display': display,
                'constructor': 'openai/codex@092d3acd6bec3e3a14bdc7e7a2810ab628ab759d:codex-rs/app-server-protocol/src/protocol/item_builders.rs:58',
                'redaction_vector': 'codex-rs/secrets/src/sanitizer.rs:redacts_supported_bearer_tokens',
                'shlex_version': '1.3.0'}
    finally:
        close_run(run, output)


def unresolved_overlap_control(binary, output):
    run = mcp.Run(binary, output)
    try:
        run.start()
        full = output / 'full.jsonl'
        rollout(full, wrappers=True)
        rows = [json.loads(line) for line in full.read_text().splitlines()]
        prefix = output / 'prefix.jsonl'
        prefix.write_text(''.join(compact(row) + '\n' for row in rows[:-1]))
        native = output / 'history.sqlite'
        history(native, [native_item()])
        imported(run, prefix, 'codex-rollout')
        imported(run, native, 'codex-history-sqlite')
        _, unknown = query(run, 'tool_items')
        assert unknown['count'] == 3, unknown
        assert not unknown['population']['correspondence_complete'] and unknown['population']['tool_calls'] is None, unknown
        wrappers = [row for row in unknown['rows'] if row['kind'] == 'CodeExecution']
        assert len(wrappers) == 2 and all(row['id'].startswith('operation:provisional-v1:') for row in wrappers), wrappers
        imported(run, full, 'codex-rollout')
        _, proved = query(run)
        assert proved['count'] == 1 and proved['population']['tool_calls'] == 1, proved
        assert proved['population']['correspondence_complete'], proved
        return {'unknown': unknown, 'proved': proved}
    finally:
        close_run(run, output)


def conflict_and_privacy_control(binary, output):
    run = mcp.Run(binary, output)
    try:
        run.start()
        secret = 'Bearer ' + 'A' * 20
        first = output / 'first.sqlite'
        history(first, [native_item(output=secret)])
        imported(run, first, 'codex-history-sqlite')
        selected = query(run)[1]['rows'][0]
        before = run.cli([selected['ref'], 'read', '--budget', '20000'])
        assert before['shown'][0]['operation']['output'] != secret and secret not in compact(before), before
        conflict = output / 'conflict.sqlite'
        history(conflict, [native_item(output='different-result', status='completed', exit_code=0)])
        imported(run, conflict, 'codex-history-sqlite')
        calls = query(run)[1]
        assert calls['count'] == 0 and not calls['population']['correspondence_complete'], calls
        after = run.cli([selected['ref'], 'read', '--budget', '20000'])
        shown = after['shown'][0]
        assert shown['conflicting_representations'] and shown['representations_complete'], shown
        assert shown['operation'] == before['shown'][0]['operation'] and secret not in compact(after), after
        run.client = mcp.Client(run)
        remote = run.client.tool('read', {'ref': selected['ref'], 'budget_tokens': 20000})
        assert remote['structuredContent']['shown'] == after['shown'] and secret not in compact(remote), remote
        run.client.close(); run.client = None
        return {'before': before, 'after': after, 'calls': calls}
    finally:
        close_run(run, output)


def operation_growth_control(binary, output, replay=None, expect_readable=False, recover=False):
    """Small native bodies, independently bounded alternative history metadata."""
    report = {'snapshots': [], 'reads': {}, 'replay': str(replay) if replay else None}
    if replay:
        prior = json.loads((replay / 'growth-report.json').read_text())
        report['snapshots'] = prior['snapshots']
        report['reads'] = {k: v for k, v in prior['reads'].items() if k.startswith('baseline-')}
        if recover:
            argv = [str(binary), '--archive', str(replay / 'archive'), 'recover', '--to', str(output / 'archive')]
            result = subprocess.run(argv, capture_output=True, timeout=120)
            (output / 'recovery.stdout').write_bytes(result.stdout)
            (output / 'recovery.stderr').write_bytes(result.stderr)
            report['recovery'] = {'argv': argv, 'exit': result.returncode, 'receipt': json.loads(result.stdout)}
            assert result.returncode == 5, report['recovery']
        else:
            shutil.copytree(replay / 'archive', output / 'archive')
        refs = [report['reads']['baseline-' + str(n)]['shown'][0]['id'] + '@' +
                report['reads']['baseline-' + str(n)]['shown'][0]['version_hash'] for n in range(2)]
    run = mcp.Run(binary, output)
    try:
        run.start()
        if not replay:
            path = output / 'growth.jsonl'
            rollout(path, output='fixed-small-output')
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            wrappers = [{'type': 'response_item', 'payload': {
                'type': 'function_call', 'id': 'message-W' + str(n),
                'call_id': 'W' + str(n), 'name': 'exec', 'arguments': '{}'}} for n in range(16)]
            rows = [rows[0], rows[1], *wrappers, rows[-1]]
            refs = []
            # At most 242 snapshots, each with 17 small own-source citations.
            # Keep an import's pending/error receipt distinct from usability.
            for revision in range(242):
                for number, wrapper in enumerate(rows[2:-1]):
                    wrapper['payload']['id'] = 'message-W' + str(number) + '-snapshot-' + str(revision)
                rows[-1]['payload']['item']['duration'] = {'secs': revision, 'nanos': 0}
                path.write_text(''.join(compact(row) + '\n' for row in rows))
                receipt = run.cli(['import', path, '--format', 'codex-rollout', '--wait', '--wait-timeout', '30'],
                                  allowed=(0, 2, 3, 4, 5, 7))
                report['snapshots'].append(receipt)
                assert receipt and receipt.get('retained'), receipt
                if receipt['outcome'] == 'error':
                    report['admission_stop'] = receipt
                    break
                if revision < 2:
                    selected = query(run)[1]['rows'][0]
                    refs.append(selected['ref'])
                    report['reads']['baseline-' + str(revision)] = run.cli([selected['ref'], 'read', '--budget', '20000'])
        identity = refs[0].split('@')[0]
        targets = {'current': identity, 'exact-first': refs[0], 'exact-later': refs[1]}
        for name, reference in targets.items():
            report['reads'][name] = run.cli([reference, 'read', '--budget', '20000'], allowed=(0, 2, 3, 4, 5, 7))
        # A generous diagnostic read measures materialized metadata separately
        # from admission receipts; it is not the ordinary-budget acceptance.
        report['generous'] = run.cli([refs[0], 'read', '--budget', '1000000'], allowed=(0, 2, 3, 4, 5, 7))
        run.client = mcp.Client(run)
        report['mcp'] = {name: run.client.tool('read', {'ref': ref, 'budget_tokens': 20000}) for name, ref in targets.items()}
        run.client.close(); run.client = None
        report['identity'] = identity
        report['cohorts'] = {
            'retained_requests': sum(bool(r.get('retained')) for r in report['snapshots']),
            'waited_usable_results': sum(r.get('source_rows', {}).get('usable_result', False) for r in report['snapshots']),
            'settled_source_rows': sum(r.get('source_rows', {}).get('state') == 'settled' for r in report['snapshots']),
            'pending_source_rows': sum(r.get('source_rows', {}).get('state') == 'pending' for r in report['snapshots'])}
        if expect_readable:
            for name in targets:
                shown = report['reads'][name]['shown'][0]
                assert report['mcp'][name]['structuredContent']['shown'] == [shown], report['mcp'][name]
                assert shown['operation']['output'] == 'fixed-small-output' and len(shown['sources']) == 17, shown
                assert not shown['conflicting_representations'], shown
                assert len(shown['representations']) == 16 and not shown['representations_complete'], shown
                assert 'operation_representation_history' in report['reads'][name]['missing'], report['reads'][name]
            for n, name in enumerate(['exact-first', 'exact-later']):
                assert report['reads'][name]['shown'][0]['operation'] == report['reads']['baseline-' + str(n)]['shown'][0]['operation']
                assert report['reads'][name]['shown'][0]['sources'] == report['reads']['baseline-' + str(n)]['shown'][0]['sources']
            report['integrity'] = growth_integrity(run, report, refs)
        return report
    finally:
        (output / 'growth-report.json').write_text(json.dumps(report, indent=2) + '\n')
        close_run(run, output)


def growth_integrity(run, report, refs):
    controls = {}
    wrong = refs[0].split('@')[0] + '@' + '0' * 64
    controls['wrong-version'] = run.cli([wrong, 'read', '--budget', '20000'], allowed=(0, 2, 3, 4, 5, 7))
    assert controls['wrong-version']['outcome'] == 'error' and not controls['wrong-version'].get('shown')
    inputs = {r['retained']['id']: r['source_rows']['basis']['original_hash'] for r in report['snapshots']}
    def blob(reference):
        source = reference['shown'][0]['sources'][0]['input']
        digest = inputs[source]
        return run.output / 'archive' / 'blobs' / digest[:2] / (digest + '.zst')
    first = blob(report['reads']['exact-first'])
    later = blob(report['reads']['exact-later'])
    saved = first.with_suffix('.saved')
    run.stop(); first.rename(saved)
    try:
        shutil.copyfile(later, first)
        run.start()
        controls['changed-own-bytes'] = run.cli([refs[0], 'read', '--budget', '20000'], allowed=(0, 2, 3, 4, 5, 7))
        assert controls['changed-own-bytes']['outcome'] == 'error' and not controls['changed-own-bytes'].get('shown')
        run.stop(); first.unlink()
        run.start()
        controls['missing-own-bytes'] = run.cli([refs[0], 'read', '--budget', '20000'], allowed=(0, 2, 3, 4, 5, 7))
        assert controls['missing-own-bytes']['outcome'] == 'error' and not controls['missing-own-bytes'].get('shown')
        controls['independent-later'] = run.cli([refs[1], 'read', '--budget', '20000'])
        assert controls['independent-later']['shown'][0]['operation'] == report['reads']['exact-later']['shown'][0]['operation']
    finally:
        run.stop()
        if first.exists():
            first.unlink()
        saved.rename(first)
    return controls


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--operation-growth', action='store_true')
    parser.add_argument('--growth-replay', type=Path)
    parser.add_argument('--expect-growth-readable', action='store_true')
    parser.add_argument('--growth-recover', action='store_true')
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'binary': str(binary), 'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(), 'outcome': 'failed'}
    try:
        if args.operation_growth:
            report['operation-growth'] = operation_growth_control(binary, args.output, args.growth_replay, args.expect_growth_readable, args.growth_recover)
            report['outcome'] = 'observed'
            return
        results = []
        for name, order, wrappers in [('native-first', 'NR', False), ('rollout-first', 'RN', False), ('unowned', 'NR', True)]:
            path = args.output / name
            path.mkdir()
            results.append(verify_case(binary, path, order, wrappers))
        assert results[0]['reference'] == results[1]['reference']
        assert results[0]['selected'] == results[1]['selected']
        report['cases'] = results
        for name, verifier in [('qualification', qualification_controls), ('refine-PNF', lambda b, p: refinement_control(b, p, 'PNF')), ('refine-NFP', lambda b, p: refinement_control(b, p, 'NFP')), ('unavailable', unavailable_control), ('missing-wrong-session', missing_and_wrong_session_controls), ('chunked-output', chunked_output_control), ('giant-admission', giant_admission_control), ('command-representations', command_representation_control), ('unresolved-overlap', unresolved_overlap_control), ('conflict-privacy', conflict_and_privacy_control)]:
            path = args.output / name
            path.mkdir()
            report[name] = verifier(binary, path)
        report['outcome'] = 'ok'
    except BaseException as error:
        report['error'] = repr(error)
        raise
    finally:
        (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
