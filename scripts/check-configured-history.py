#!/usr/bin/env python3
"""Independent configured physical-history facts through ordinary CLI/SDK reads."""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
import time

ROOT=Path(__file__).resolve().parent.parent
spec=importlib.util.spec_from_file_location('continuation',ROOT/'scripts/check-codex-continuation.py')
continuation=importlib.util.module_from_spec(spec);spec.loader.exec_module(continuation)
discovery,native,mcp=continuation.discovery,continuation.native,continuation.mcp
EXPECTED={'C':'history committed result','D':'WAL-only terminal','E':'later same-daemon result'}
TURNS={'C':'T','D':'T-other','E':'T-later'}

def insert(c,call,turn,output,ordinal):
    _,item=native.native_item(call=call,turn=turn,output=output)
    c.execute('INSERT INTO thread_items(thread_id,turn_id,item_id,rollout_ordinal,updated_at_ordinal,created_at_ms,item_type,item_json,started_at_ms) VALUES(?,?,?,?,?,?,?,?,?)',(native.S,turn,call,ordinal,ordinal,1,item['type'],json.dumps(item,separators=(',',':')),1))

def discover(run,remote=False):
    query=lambda q:run.client.tool('search',{'query':q,'budget_tokens':20000})['structuredContent'] if remote else run.cli([q,'--json'])
    read=lambda ref:run.client.tool('read',{'ref':ref,'budget_tokens':20000})['structuredContent'] if remote else run.cli([ref,'read','--budget','20000'])
    found={}
    for session in query('origin:codex')['rows']:
        for row in query('tool_items where session:'+session['id'])['rows']:
            value=read(row['ref'])
            if value.get('shown'):
                body=value['shown'][0];item=body['operation'];found[item['invocation']]=(row['ref'],body)
    return found

def oracle(found,expected):
    assert 'U' not in found,found
    for call in expected:
        assert call in found,(call,found)
        body=found[call][1];item=body['operation']
        assert item['output']==EXPECTED[call],item
        assert item['invocation']==call and item['result_present'] is True,item
        assert item['outcome']=='error' and item['exit_code']==1,item
        assert item['native_session']==native.S and item['turn']==TURNS[call],item
        origins=[s['origin'] for s in body['sources'] if s['origin']['representation']=='history']
        assert origins,body['sources']
        assert all(s['thread']==native.S and s['turn']==TURNS[call] and s['item']==call for s in origins),origins
    if 'D' in expected:
        assert found['D'][1]['operation']['tool'] is None,found['D']
        assert not found['D'][1].get('associations'),found['D']
        assert found['D'][1]['operation'].get('message') is None,found['D']

def await_facts(run,expected):
    until=time.monotonic()+90
    while True:
        found=discover(run)
        try:oracle(found,expected);return found
        except AssertionError:
            if time.monotonic()>=until:raise
            assert run.daemon.poll() is None,'daemon exited'
            time.sleep(.05)

def run_case(binary,output):
    run=discovery.Run(binary,output);home=output/'codex-home';home.mkdir();sessions=home/'sessions';sessions.mkdir()
    rollout=sessions/'rollout-calls.jsonl'
    # Own response-call identity exists, but no terminal result in this route.
    rows=[{'type':'session_meta','payload':{'id':native.S,'source':'cli'}}, {'type':'turn_context','payload':{'turn_id':'T','model':'synthetic-model'}}, {'type':'response_item','payload':{'type':'function_call','id':'M','call_id':'C','name':'exec_command','arguments':'{"cmd":"same command"}'}}, {'type':'event_msg','payload':{'type':'token_count','info':None}}]
    rollout.write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in rows))
    (home/'history.jsonl').write_text('');discovery.catalog(home/'state_5.sqlite')
    path=home/'thread_history_1.sqlite';c=sqlite3.connect(path)
    c.execute('PRAGMA journal_mode=WAL');c.execute('PRAGMA wal_autocheckpoint=0')
    c.executescript((ROOT/'crates/store/tests/fixtures/codex-history-v0162.sql').read_text())
    c.execute('CREATE TABLE _sqlx_migrations(version INTEGER PRIMARY KEY,success INTEGER NOT NULL)');c.executemany('INSERT INTO _sqlx_migrations VALUES(?,1)',[(i,)for i in range(1,9)]);c.commit();c.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    c.close()
    main_before=hashlib.sha256(path.read_bytes()).hexdigest()
    main_stat=path.stat()
    c=sqlite3.connect(path);c.execute('PRAGMA wal_autocheckpoint=0')
    insert(c,'C','T',EXPECTED['C'],1);insert(c,'D','T-other',EXPECTED['D'],2);c.commit()
    assert (path.stat().st_size,path.stat().st_mtime_ns)==(main_stat.st_size,main_stat.st_mtime_ns),'main file should remain checkpointed at its earlier empty population'
    assert Path(str(path)+'-wal').stat().st_size>0
    c.execute('BEGIN IMMEDIATE');insert(c,'U','T-uncommitted','uncommitted distractor',3)
    cfg=discovery.config(output,home);cfg['codex_home'].update(paginated_history='thread_history_1.sqlite',paginated_interval_seconds=1);(output/'config.json').write_text(json.dumps(cfg))
    report={'checkpointed_main_sha256':main_before,'checkpointed_main_items':0}
    try:
        start=run.start();assert start['acquisition']['paginated_history']['state']=='accessible',start
        first=await_facts(run,{'C','D'});run.client=mcp.Client(run);remote=discover(run,True);oracle(remote,{'C','D'})
        for call in ['C','D']:assert remote[call]==first[call]
        run.client.close();run.client=None
        controls=0
        for mutate in ['wrong-result','wrong-row','wrong-verdict','wrong-session','wrong-turn','wrong-origin-thread','wrong-origin-turn','extra-conflicting-origin','uncommitted','text-only-link']:
            broken=copy.deepcopy(first)
            if mutate=='wrong-result':broken['D'][1]['operation']['output']='wrong snapshot'
            elif mutate=='wrong-row':
                for source in broken['D'][1]['sources']:source['origin']['item']='C'
            elif mutate=='wrong-session':broken['C'][1]['operation']['native_session']='other-session'
            elif mutate=='wrong-turn':broken['C'][1]['operation']['turn']='other-turn'
            elif mutate in ['wrong-origin-thread','wrong-origin-turn']:
                field='thread' if mutate=='wrong-origin-thread' else 'turn'
                for source in broken['C'][1]['sources']:
                    if source['origin']['representation']=='history':source['origin'][field]='other-scope'
            elif mutate=='extra-conflicting-origin':
                source=copy.deepcopy(next(s for s in broken['C'][1]['sources'] if s['origin']['representation']=='history'))
                source['origin']['thread']='other-session';broken['C'][1]['sources'].append(source)
            elif mutate=='wrong-verdict':
                broken['C'][1]['operation'].update(outcome='success',exit_code=0)
            elif mutate=='uncommitted':broken['U']=broken['D']
            else:broken['D'][1]['operation']['tool']='exec_command'
            try:oracle(broken,{'C','D'})
            except AssertionError:controls+=1
            else:raise AssertionError('oracle accepted '+mutate)
        c.rollback();insert(c,'E','T-later',EXPECTED['E'],4);c.commit()
        later=await_facts(run,{'C','D','E'});assert run.daemon.pid is not None
        report.update(initial=start,first=first,later=later,mutants_rejected=controls,stop=continuation.graceful_stop(run))
        run.start();restarted=await_facts(run,{'C','D','E'})
        for call in EXPECTED:assert restarted[call]==later[call]
        continuation.graceful_stop(run);c.close();c=None
        for owned in [rollout,path,Path(str(path)+'-wal'),Path(str(path)+'-shm')]:
            if owned.exists():owned.unlink()
        missing=run.start();assert missing['acquisition']['paginated_history']['state']=='unavailable',missing
        retained=await_facts(run,{'C','D','E'})
        for call in EXPECTED:assert retained[call]==later[call]
        report.update(restarted=restarted,provider_removed=retained,provider_unavailable=missing['acquisition']['paginated_history'])
        continuation.graceful_stop(run)
        unknown=sqlite3.connect(path);unknown.execute('CREATE TABLE unsupported(x)');unknown.commit();unknown.close()
        status=run.start();assert status['acquisition']['paginated_history']['state']=='accessible',status
        source=status['acquisition']['paginated_input']
        assert isinstance(source,str) and source, status
        until=time.monotonic()+30
        while True:
            progress=run.cli(['progress',source])
            assert progress['outcome']=='ok' and progress['input']==source,progress
            assert progress['length']>0 and progress['source_rows']['basis']['input']==source,progress
            failure=progress.get('failure')
            if isinstance(failure,str) and 'no_parser_yet: unsupported Codex history schema;' in failure:
                assert progress['source_rows']['state']=='unavailable',progress
                assert progress['source_rows']['usable_result'] is False,progress
                break
            assert time.monotonic()<until,progress
            time.sleep(.05)
        report['unsupported_retained']=progress
        continuation.graceful_stop(run)
    finally:
        if c is not None:c.close()
        runtime=Path(run.socket_owner.name)
        native.close_run(run,output)
        cleanup={'daemon_reaped':run.daemon is None,'mcp_reaped':all(client.process.poll() is not None for client in run.clients),'runtime_removed':not runtime.exists()}
        (output/'cleanup.json').write_text(json.dumps(cleanup,indent=2)+'\n')
        assert all(cleanup.values()),cleanup
    report['passed']=True
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--binary',required=True,type=Path);parser.add_argument('--output',required=True,type=Path);args=parser.parse_args()
    if args.output.exists():parser.error('output must not exist')
    args.output.mkdir(parents=True)
    run_case(args.binary,args.output)
    print(json.dumps({'passed':True,'report':str(args.output/'report.json')}))
if __name__=='__main__':main()
