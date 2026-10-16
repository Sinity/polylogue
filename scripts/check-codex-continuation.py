#!/usr/bin/env python3
"""Configured Codex continuation through real CLI and official SDK MCP clients."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import time

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location('discovery', ROOT / 'scripts/check-codex-discovery.py')
discovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(discovery)
native, mcp = discovery.native, discovery.mcp

def fixture(path, sid, output):
    previous = native.S
    try:
        native.S = sid
        native.rollout(path, output=output)
    finally:
        native.S = previous
    return path.read_bytes()

def wait(run, condition, timeout=60):
    deadline = time.monotonic() + timeout
    while True:
        value = run.cli(['status'])
        if condition(value):
            return value
        assert run.daemon.poll() is None, 'daemon exited during continuation'
        assert time.monotonic() < deadline, value
        time.sleep(.05)

def resources(run):
    status = Path(f'/proc/{run.daemon.pid}/status').read_text()
    fields = Path(f'/proc/{run.daemon.pid}/stat').read_text().split()
    memory = {key: int(next(line.split()[1] for line in status.splitlines() if line.startswith(key + ':')))
              for key in ['VmRSS', 'VmHWM']}
    return dict(pid=run.daemon.pid, cpu_seconds=(int(fields[13])+int(fields[14]))/os.sysconf('SC_CLK_TCK'),
                rss_kib=memory['VmRSS'], high_water_kib=memory['VmHWM'],
                archive_bytes=sum(p.stat().st_size for p in (run.output/'archive').rglob('*') if p.is_file()),
                open_fds=len(list(Path(f'/proc/{run.daemon.pid}/fd').iterdir())))

def discovered_reads(run, remote=False):
    # Session IDs and operation references come only from ordinary discovery.
    search = (run.client.tool('search', {'query': 'origin:codex', 'budget_tokens': 20000})['structuredContent']
              if remote else run.cli(['origin:codex', '--json']))
    replies = {}
    for row in search['rows']:
        query = 'tool_calls where session:' + row['id']
        operations = (run.client.tool('search', {'query': query, 'budget_tokens': 20000})['structuredContent']
                      if remote else run.cli([query, '--json']))
        for operation in operations['rows']:
            read = (run.client.tool('read', {'ref': operation['ref'], 'budget_tokens': 20000})['structuredContent']
                    if remote else run.cli([operation['ref'], 'read', '--budget', '20000']))
            if read.get('shown'):
                replies[read['shown'][0]['operation']['output']] = (operation['ref'], read)
    return replies

def await_facts(run, expected):
    deadline = time.monotonic()+60
    while True:
        facts = discovered_reads(run)
        if set(expected) <= facts.keys():
            return facts
        assert time.monotonic()<deadline, (expected, facts)
        time.sleep(.05)

def witnesses(read, raw):
    sources = read['shown'][0]['sources']
    values = {source['origin']['tag']: json.loads(raw[source['start']:source['end']]) for source in sources}
    assert set(values)=={'call','result','status'}, values
    assert values['call']['payload']['type']=='function_call'
    assert json.loads(values['call']['payload']['arguments'])=={'cmd':'same command'}
    output=read['shown'][0]['operation']['output']
    assert values['result']['payload']['output']==output
    assert values['status']['payload']['item']['aggregated_output']==output
    return values

def graceful_stop(run):
    pid=run.daemon.pid
    started=time.monotonic()
    run.daemon.send_signal(signal.SIGINT)
    code=run.daemon.wait(timeout=10)
    assert code==0, code
    run.daemon=None
    return {'pid':pid,'signal':'SIGINT','exit':code,'seconds':time.monotonic()-started}

def lifecycle(binary, output):
    run=discovery.Run(binary, output)
    home=output/'codex-home';home.mkdir()
    sessions=home/'sessions';sessions.mkdir()
    discovery.catalog(home/'state_5.sqlite')
    (home/'history.jsonl').write_text(json.dumps({'session_id': discovery.P, 'ts': 123, 'text': 'synthetic pending prompt'})+'\n')
    unrelated=home/'unrelated.txt';unrelated.write_text('preserve this synthetic control')
    discovery.config(output,home)
    try:
        start=run.start();pid=run.daemon.pid
        assert start['acquisition']['rollouts']['state']=='accessible_empty'
        assert start['continuation']['state']=='polling'
        assert start['continuation']['conversation_coverage'] is None
        pending=discovery.one_pending(start)
        initial=resources(run)
        delays=[];raw={};owned=[];expected=[]
        run.client=mcp.Client(run)
        for index, fact in enumerate(['violet batch one independently recorded','amber batch two absent from catalog']):
            path=sessions/f'rollout-batch-{index}.jsonl';owned.append(path)
            began=time.monotonic();raw[fact]=fixture(path,f'continuation:batch:{index}',fact)
            expected.append(fact)
            # Only batch one enters the catalog. Batch two is found independently.
            if index==0:
                import sqlite3
                connection=sqlite3.connect(home/'state_5.sqlite')
                connection.execute('INSERT INTO threads VALUES(?,?)',('batch-one',str(path.relative_to(home))))
                connection.commit();connection.close()
            facts=await_facts(run,expected)
            delay=time.monotonic()-began
            assert run.daemon.pid==pid
            remote=discovered_reads(run,remote=True)
            for value in expected:
                assert remote[value][0]==facts[value][0]
                assert remote[value][1]['shown']==facts[value][1]['shown']
                witnesses(facts[value][1],raw[value])
            delays.append({'batch':index+1,'acquisition_to_cli_seconds':delay,'mcp_confirmed_seconds':time.monotonic()-began})
        state=wait(run,lambda v:v['continuation']['completed_passes']>=3 and v['continuation']['last_pass']['unchanged_files']==3)
        passes=state['continuation']['completed_passes'];counter=state['foundation']['write_counter'];stable_before=resources(run)
        idle_started=time.monotonic()
        time.sleep(3.3)
        stable=run.cli(['status'])
        stable_after=resources(run)
        idle_wall=time.monotonic()-idle_started
        assert stable['continuation']['completed_passes']>=passes+2,stable
        assert stable['foundation']['write_counter']==counter, (state,stable)
        assert discovery.one_pending(stable)==pending
        assert stable['continuation']['successful_snapshots']==2
        assert stable['continuation']['last_pass']['acquired_files']==0
        assert stable_after['archive_bytes']==stable_before['archive_bytes'],(stable_before,stable_after)
        # Distinguish inaccessible enumeration from an accessible empty directory.
        hidden=home/'temporarily-hidden-sessions';sessions.rename(hidden)
        unavailable=wait(run,lambda v:v['continuation']['last_pass']['rollouts']['state']=='unavailable')
        assert unavailable['continuation']['last_pass']['rollouts']['entries'] is None
        hidden.rename(sessions)
        for path in owned:path.unlink()
        empty=wait(run,lambda v:v['continuation']['last_pass']['rollouts']['state']=='accessible_empty')
        assert empty['continuation']['last_pass']['rollouts']['entries']==0
        for fact in expected:
            assert run.cli([facts[fact][0],'read','--budget','20000'])['shown']==facts[fact][1]['shown']
        run.client.close();run.client=None
        stop=graceful_stop(run)
        waiting=sessions/'rollout-waiting.jsonl'
        waiting_fact='teal waiting batch discovered after restart'
        fixture(waiting,'continuation:waiting',waiting_fact)
        began=time.monotonic();run.start()
        restarted=await_facts(run,expected+[waiting_fact])
        restart_delay=time.monotonic()-began
        for fact in expected:assert restarted[fact][1]['shown']==facts[fact][1]['shown']
        final=resources(run)
        final_stop=graceful_stop(run)
        # --until-idle owns only its startup snapshot and terminates normally.
        idle=run.cli(['daemon','--config',output/'config.json','--until-idle','--json'])
        assert idle['outcome']=='degraded' and idle['progress'],idle
        assert unrelated.read_text()=='preserve this synthetic control'
        return dict(delays=delays, restart_delay_seconds=restart_delay, startup=start,
                    unchanged_before=stable_before,unchanged_after=stable_after,
                    unchanged_wall_seconds=idle_wall,unchanged_cli_status_requests=1,
                    initial_resources=initial,final_resources=final,stable_write_counter=counter,
                    unavailable=unavailable['continuation'],empty=empty['continuation'],
                    retained_refs={fact:facts[fact][0] for fact in expected},graceful_stops=[stop,final_stop],
                    until_idle=idle,operator_interventions=0,
                    scripted_interventions=['two batch writes','catalog row for batch one','owned directory rename/restore','only two owned rollout removals','SIGINT','waiting batch write','restart','SIGINT','finite until-idle'])
    finally:native.close_run(run,output)

def bounds(binary,output):
    run=discovery.Run(binary,output)
    home=output/'codex-home';home.mkdir();sessions=home/'sessions';sessions.mkdir()
    discovery.catalog(home/'state_5.sqlite');(home/'history.jsonl').write_bytes(b'')
    discovery.config(output,home)
    try:
        run.start();pid=run.daemon.pid
        expected=[];waves=[];original=None
        # A supported metadata-only rollout supplies directory/observation load,
        # without manufacturing conversational Messages or hundreds of answers.
        filler=b'{"type":"session_meta","payload":{"id":"synthetic-filler","source":"cli"}}\n'
        for wave in range(8):
            began=time.monotonic()
            for index in range(80):(sessions/f'rollout-filler-{wave:02}-{index:03}.jsonl').write_bytes(filler)
            path=sessions/f'rollout-wave-{wave:02}.jsonl'
            fact=f'independently recorded reusable wave {wave}'
            raw=fixture(path,f'continuation:wave:{wave}',fact);expected.append(fact)
            acquired=wait(run,lambda v:v['continuation']['successful_snapshots']>=81*(wave+1),timeout=120)
            facts=await_facts(run,expected)
            assert run.daemon.pid==pid
            witnesses(facts[fact][1],raw)
            if original is None:original=(path,raw,facts[fact])
            waves.append({'wave':wave+1,'owned_rollout_paths':81*(wave+1),'acquisition_to_query_seconds':time.monotonic()-began,'continuation':acquired['continuation']})
        assert waves[-1]['continuation']['successful_snapshots']>512
        complete=wait(run,lambda v:v['continuation']['last_pass'] and v['continuation']['last_pass']['enumeration_cycle_complete'] and v['continuation']['last_pass']['rollouts']['state']=='accessible' and v['continuation']['last_pass']['rollouts']['entries']==648,timeout=120)
        assert complete['continuation']['conversation_coverage'] is None
        # Revisit an earlier path after more than one directory turn. Append a
        # second explicit turn/invocation, conserving the earlier operation.
        earlier,raw,old=original
        appended=output/'append-fixture.jsonl'
        previous=native.S
        try:
            native.S='continuation:wave:0'
            native.rollout(appended,output='later appended invocation after fair revisit',call='C2',turn='T2')
        finally:native.S=previous
        rows=[json.loads(line) for line in appended.read_text().splitlines()]
        for row in rows:
            if row.get('payload',{}).get('type')=='function_call':row['payload']['id']='M2'
        suffix=''.join(json.dumps(row,separators=(',',':'))+'\n' for row in rows if row['type']!='session_meta').encode()
        began=time.monotonic()
        with earlier.open('ab') as target:target.write(suffix)
        revised=await_facts(run,expected+['later appended invocation after fair revisit'])
        revisit_delay=time.monotonic()-began
        witnesses(revised['later appended invocation after fair revisit'][1],raw+suffix)
        assert run.cli([old[0],'read','--budget','20000'])['shown']==old[1]['shown']
        settled=wait(run,lambda v:v['continuation']['successful_snapshots']>=649,timeout=120)
        cycles=settled['continuation']['enumeration_cycles_completed']
        stable=wait(run,lambda v:v['continuation']['enumeration_cycles_completed']>=cycles+2,timeout=120)
        counter=stable['foundation']['write_counter'];before=resources(run);started=time.monotonic()
        time.sleep(3.3)
        after_status=run.cli(['status']);after=resources(run)
        stable_wall=time.monotonic()-started
        assert after_status['foundation']['write_counter']==counter
        assert after['archive_bytes']==before['archive_bytes'],(before,after)
        assert after_status['continuation']['state']=='polling'
        assert after_status['continuation']['successful_snapshots']==649
        budget_samples=[];partial_samples=[]
        for call in run.calls:
            if call.get('arguments')!=['status'] or call.get('status')!='completed':continue
            value=json.loads((output/call['stdout']).read_text()).get('continuation')
            if not value or not value.get('last_pass'):continue
            if value['last_pass']['budget_exhausted']:budget_samples.append(value)
            if value['last_pass']['rollouts']['state']=='partial':partial_samples.append(value)
        assert budget_samples and partial_samples,(budget_samples,partial_samples)
        assert after_status['continuation']['completed_passes']>budget_samples[0]['completed_passes']
        stopped=graceful_stop(run)
        # Startup must not abort when its filesystem and catalog prefixes have
        # a larger union. The lexically late catalog target is deferred to the
        # ordinary running cursor, without caller-supplied answer IDs.
        outside=home/'zz-owned';outside.mkdir()
        waiting=outside/'rollout-catalog-waiting.jsonl'
        waiting_fact='catalog waiting behind a larger startup prefix'
        waiting_raw=fixture(waiting,'continuation:catalog-waiting',waiting_fact)
        import sqlite3
        connection=sqlite3.connect(home/'state_5.sqlite')
        connection.execute('INSERT INTO threads VALUES(?,?)',('outside-waiting',str(waiting.relative_to(home))))
        connection.commit();connection.close()
        restarted=run.start()
        assert restarted['acquisition']['deferred_rollout_paths']==1,restarted
        restarted_facts=await_facts(run,expected+['later appended invocation after fair revisit',waiting_fact])
        witnesses(restarted_facts[waiting_fact][1],waiting_raw)
        restart_state=wait(run,lambda v:v['continuation']['successful_snapshots']>=1)
        restart_stop=graceful_stop(run)
        return dict(waves=waves,reusable_budget_witness=budget_samples[0],partial_enumeration_witness=partial_samples[0],completed_larger_cycle=complete['continuation'],revisit_delay_seconds=revisit_delay,
                    stable_before=before,stable_after=after,stable_wall_seconds=stable_wall,
                    stable_write_counter=counter,final=after_status['continuation'],stopped=stopped,
                    larger_restart=restarted['acquisition'],larger_restart_continuation=restart_state['continuation'],restart_stop=restart_stop,operator_interventions=0)
    finally:native.close_run(run,output)

def cancellation(binary,output):
    run=discovery.Run(binary,output)
    home=output/'codex-home';home.mkdir();sessions=home/'sessions';sessions.mkdir()
    discovery.catalog(home/'state_5.sqlite');(home/'history.jsonl').write_bytes(b'')
    discovery.config(output,home)
    try:
        run.start()
        raw=fixture(sessions/'rollout-0000.jsonl','continuation:cancellation','cancellation workload')
        for index in range(1,512):(sessions/f'rollout-{index:04}.jsonl').write_bytes(raw)
        active=wait(run,lambda v:v['continuation']['pass_active'])
        stopped=graceful_stop(run)
        # Restart remains an ordinary configured acquisition, without repair.
        restarted=run.start()
        reads=await_facts(run,['cancellation workload'])
        return dict(active_at_signal=active['continuation'],stopped=stopped,
                    restart_acquisition=restarted['acquisition'],discovered_refs=[r[0] for r in reads.values()],
                    operator_interventions=0)
    finally:native.close_run(run,output)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--case',choices=['all','continuation','bounds','cancellation'],default='all')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    binary=args.binary.resolve(strict=True)
    report={'binary':str(binary),'sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'cases':{}}
    try:
        for name,function in [('continuation',lifecycle),('bounds',bounds),('cancellation',cancellation)]:
            if args.case not in ['all',name]:continue
            output=args.output/name;output.mkdir();report['cases'][name]=function(binary,output)
        report['outcome']='passed'
    except BaseException as error:
        report.update(outcome='failed',error=repr(error));raise
    finally:(args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'outcome':report['outcome'],'report':str(args.output/'report.json')}))
if __name__=='__main__':main()
