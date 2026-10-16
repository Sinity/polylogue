#!/usr/bin/env python3
"""Independent synthetic Codex representation facts through the ordinary binary and MCP."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
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

def history(path,items):
    connection=sqlite3.connect(path)
    connection.executescript((ROOT/'crates/store/tests/fixtures/codex-history-v0162.sql').read_text())
    connection.execute('CREATE TABLE _sqlx_migrations(version BIGINT PRIMARY KEY,success BOOLEAN NOT NULL)')
    connection.executemany('INSERT INTO _sqlx_migrations VALUES(?,1)',[(n,)for n in range(1,9)])
    for ordinal,(turn,item) in enumerate(items,1):
        connection.execute('INSERT INTO thread_items(thread_id,turn_id,item_id,rollout_ordinal,updated_at_ordinal,created_at_ms,item_type,item_json) VALUES(?,?,?,?,?,?,?,?)',(S,turn,item['id'],ordinal,ordinal,1,item['type'],compact(item)))
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
        operation=commands[0]
        result=run.cli([operation['ref'],'read','--budget','20000'])
        facts=result['shown'][0]['operation']
        assert facts['result_present'] and facts['output']=='' and facts['outcome']=='error'and facts['exit_code']==1,facts
        assert facts['invocation']=='C'and facts['turn']=='T',facts
        assert facts['message'] is None if wrappers else facts['message']==REF+':M',facts
        assert facts['wrapper'] is None,facts
        if wrappers:
            assert len(facts['wrapper_overlaps'])==2 and all(ref.startswith('operation:')for ref in facts['wrapper_overlaps']),facts
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
        _, calls = query(run)
        assert len(calls['rows']) == 2, calls
        assert len({row['id'] for row in calls['rows']}) == 2, calls
        assert not calls['population']['correspondence_complete'], calls
        assert query(run, text='unique-output')[1]['count'] == 1
        omitted = next(row for row in calls['rows'] if row['turn'] == 'T')
        facts = run.cli([omitted['ref'], 'read', '--budget', '20000'])['shown'][0]['operation']
        assert facts['result_present'] and facts['output'] is None, facts
        return all_items
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'binary': str(binary), 'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(), 'outcome': 'failed'}
    try:
        results = []
        for name, order, wrappers in [('native-first', 'NR', False), ('rollout-first', 'RN', False), ('unowned', 'NR', True)]:
            path = args.output / name
            path.mkdir()
            results.append(verify_case(binary, path, order, wrappers))
        assert results[0]['reference'] == results[1]['reference']
        assert results[0]['selected'] == results[1]['selected']
        report['cases'] = results
        for name, verifier in [('qualification', qualification_controls), ('refine-PNF', lambda b, p: refinement_control(b, p, 'PNF')), ('refine-NFP', lambda b, p: refinement_control(b, p, 'NFP')), ('unavailable', unavailable_control), ('missing-wrong-session', missing_and_wrong_session_controls), ('chunked-output', chunked_output_control), ('giant-admission', giant_admission_control)]:
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
