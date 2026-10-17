#!/usr/bin/env python3
"""Bounded stopped-custody restore through one packaged CLI and SDK MCP adapter."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import sqlite3
import subprocess
import time

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location('history', ROOT/'scripts/check-configured-history.py')
history = importlib.util.module_from_spec(spec)
spec.loader.exec_module(history)
continuation, discovery, native, mcp = history.continuation, history.discovery, history.native, history.mcp
A = '019a1100-0000-7000-8000-000000000011'
B = '019a1100-0000-7000-8000-000000000012'
N = '019a1100-0000-7000-8000-000000000021'
R = '019a1100-0000-7000-8000-000000000022'
TEXT = 'A retained unrelated source remains readable.'
CORRECTION = 'The independent correction supersedes the initial interpretation.'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest(root):
    return {str(p.relative_to(root)): {'bytes': p.stat().st_size, 'sha256': digest(p)}
            for p in sorted(root.rglob('*')) if p.is_file()}


def custody(source, destination, evidence):
    # Caller has retired the owner. Copy every durable file/sidecar/lock inode;
    # retain the entire original manifest before opening either copy with SQLite.
    before = manifest(source)
    assert 'writer.lock' in before and 'archive.db' in before and 'embeddings.db' in before
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns('index.db', 'index.db-wal', 'index.db-shm'))
    copied = manifest(destination)
    assert copied == {k:v for k,v in before.items() if k not in {'index.db','index.db-wal','index.db-shm'}}
    for name in copied:
        assert (source/name).stat().st_ino != (destination/name).stat().st_ino
    assert manifest(source) == before
    evidence.write_text(json.dumps({'source': str(source), 'destination': str(destination), 'source_manifest': before, 'copied_manifest': copied}, indent=2)+'\n')
    return copied


def prepare(run, request=A, interrupt=False, mode='restore', extra=()):
    args = ['daemon','--prepare-copy',mode,'--request-id',request,'--retired-writers',*extra]
    name = f'preparation-{len(run.calls):03}'
    p = subprocess.Popen(run.command(args), env=run.env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    first, rest, error = b'', b'', b''
    attempt = {'argv':run.command(args),'pid':p.pid,'stdout':name+'.stdout','stderr':name+'.stderr'}
    run.calls.append(attempt)
    try:
        first = p.stdout.readline()
        if interrupt:
            assert json.loads(first)['outcome'] == 'pending', first
            p.send_signal(signal.SIGINT)
        rest, error = p.communicate(timeout=60)
    finally:
        if p.poll() is None:
            p.kill()
            remainder, diagnostic = p.communicate(timeout=15)
            rest += remainder
            error += diagnostic
        p.wait(timeout=15)
        attempt['exit'] = p.returncode
        (run.output/(name+'.stdout')).write_bytes(first+rest)
        (run.output/(name+'.stderr')).write_bytes(error)
    replies = [json.loads(line) for line in (first+rest).splitlines() if line]
    if interrupt:
        assert p.returncode != 5, (attempt,replies,error)
        assert b'cancelled' in error, error
    elif mode == 'restore' and not extra:
        assert p.returncode == 5 and replies[-1]['latest_acknowledged_head'] == 'unknown', (attempt,replies,error)
    else:
        assert p.returncode != 5, (attempt,replies)
    return {'attempt':attempt,'replies':replies}


def read_clients(run, expected, with_note=True, remote=False):
    query = lambda q: run.client.tool('search',{'query':q,'budget_tokens':20000})['structuredContent'] if remote else run.cli([q,'--json'])
    read = lambda ref,view=None: run.client.tool('read',dict(ref=ref,budget_tokens=20000,**({'view':view} if view else {})))['structuredContent'] if remote else run.cli([ref,'read','--budget','20000',*(['--view',view] if view else [])])
    found = {}
    for session in query('origin:codex')['rows']:
        for row in query('tool_items where session:'+session['id'])['rows']:
            reply = read(row['ref'])
            assert reply['outcome'] != 'error', reply
            if reply.get('shown'):
                body = reply['shown'][0]
                found[body['operation']['invocation']] = {'ref':row['ref'],'body':body}
    history.oracle({k:(v['ref'],v['body']) for k,v in found.items()}, expected)
    # Authored references are discovered from returned source blocks. No answer
    # reference, native ID or provider path seeds the queries above or below.
    unrelated = []
    notes = []
    for session in query('origin:claude-code')['rows']:
        body = read(session['id'])
        unrelated += body.get('shown', [])
        for block in body.get('shown',[]):
            association = read(block['block'],'annotations')
            for family in association.get('authored_notes',{}).get('families',[]):
                for entry in family.get('shown',[]):
                    revision = entry['revision']
                    notes.append(revision)
                    pinned = read('note:'+revision['root']+'@'+revision['id'])
                    assert pinned.get('family',{}).get('shown'), pinned
                    assert pinned['family']['shown'][0]['revision']==revision, pinned
    assert any(block.get('text') == TEXT for block in unrelated), unrelated
    if with_note:
        assert any(n['id']==R and n['value']==CORRECTION and n['author']=='script' and n['machine']=='00112233445566778899aabbccddeeff' for n in notes), notes
    return {'operations':found,'unrelated':unrelated,'notes':notes}


def await_clients(run, expected, with_note=True):
    until = time.monotonic()+90
    while True:
        try:
            return read_clients(run, expected, with_note)
        except AssertionError:
            if time.monotonic() >= until:
                raise
            assert run.daemon.poll() is None
            time.sleep(.05)


def inspect_custody(root, expected_images):
    # Independent finite byte oracle, used only after original custody preservation
    # and native preparation. Stock zstd decodes the plain seekable frames.
    conn = sqlite3.connect('file:'+str(root/'archive.db')+'?mode=ro', uri=True)
    blobs = {}
    for raw, size in conn.execute("SELECT hash,size FROM blob WHERE place='file' AND encoding='plain' AND state='present'"):
        name = raw.hex()
        path = root/'blobs'/name[:2]/(name+'.zst')
        result = subprocess.run(['zstd','-dc',str(path)],capture_output=True,check=True)
        assert len(result.stdout)==size
        blobs[raw] = result.stdout
    assert b'' in blobs.values(), 'valid retained zero-byte object absent'
    images = []
    layouts = []
    for source, length in conn.execute("SELECT r.input,r.length FROM revision r JOIN input i ON i.id=r.input WHERE i.format='codex-history-sqlite' ORDER BY r.local_seq"):
        output = bytearray()
        hashes = set()
        for at,n,blob,start in conn.execute('SELECT offset,length,blob,blob_offset FROM extent WHERE input=? ORDER BY offset',(source,)):
            assert at == len(output)
            output += blobs[blob][start:start+n]
            hashes.add(blob.hex())
        assert len(output)==length
        images.append(hashlib.sha256(output).hexdigest())
        layouts.append({'input':source,'bytes':length,'sha256':images[-1],'blobs':sorted(hashes)})
    assert conn.execute('SELECT count(*) FROM note WHERE id=?',(R,)).fetchone()[0]==1
    conn.close()
    assert images==expected_images, (images,expected_images)
    assert len(layouts)==2 and len(layouts[0]['blobs'])==1 and len(layouts[1]['blobs'])==2, layouts
    assert layouts[0]['blobs'][0] in layouts[1]['blobs'], layouts
    return layouts


def authority(root):
    conn=sqlite3.connect('file:'+str(root/'archive.db')+'?mode=ro',uri=True)
    value=dict(conn.execute("SELECT key,value FROM meta WHERE key IN ('write_counter','acquisition_lineage','copy_preparation')"))
    value['completed_requests']=conn.execute('SELECT count(*) FROM client_request').fetchone()[0]
    conn.close()
    return value


def case(binary, unaware_binary, output):
    usage = shutil.disk_usage(output)
    assert usage.free >= 256*1024*1024, usage
    (output/'storage-before.json').write_text(json.dumps({'free_bytes':usage.free,'logical_provider_bound_bytes':12*1024*1024,'fixture_only':True},indent=2)+'\n')
    run = discovery.Run(binary,output)
    # Deterministic synthetic process attribution, never operator session metadata.
    for key in ['CLAUDE_CODE_SESSION_ID','CODEX_THREAD_ID','CLAUDECODE','CODEX_CI','CODEX_SANDBOX']:
        run.env.pop(key,None)
    machine = output/'machine-id'
    machine.write_text('00112233445566778899aabbccddeeff\n')
    run.env['POLYLOGUE_MACHINE_ID_FILE']=str(machine)
    home=output/'codex-home';home.mkdir();(home/'sessions').mkdir()
    (home/'history.jsonl').write_bytes(b'');discovery.catalog(home/'state_5.sqlite')
    path=home/'thread_history_1.sqlite'
    c=sqlite3.connect(path)
    c.executescript((ROOT/'crates/store/tests/fixtures/codex-history-v0162.sql').read_text())
    c.execute('CREATE TABLE _sqlx_migrations(version INTEGER PRIMARY KEY,success INTEGER NOT NULL)')
    c.executemany('INSERT INTO _sqlx_migrations VALUES(?,1)',[(n,) for n in range(1,9)])
    c.execute('CREATE TABLE retained_padding(value BLOB)')
    c.execute('INSERT INTO retained_padding VALUES(zeroblob(8388608))')
    history.insert(c,'C','T',history.EXPECTED['C'],1);c.commit();c.close()
    image1=path.read_bytes();(output/'expected-history-1.sqlite').write_bytes(image1)
    cfg=discovery.config(output,home);cfg['codex_home'].update(paginated_history='thread_history_1.sqlite',paginated_interval_seconds=1);(output/'config.json').write_text(json.dumps(cfg))
    report={}
    readers=[]
    try:
        run.start();history.await_facts(run,{'C'})
        run.import_rows('unrelated',[mcp.bounded.row('u',TEXT)])
        # A standalone empty input produces the valid retained empty CAS object.
        empty=output/'empty.jsonl';empty.write_bytes(b'');native.imported(run,empty,'codex-rollout')
        first=await_clients(run,{'C'},False)
        target=first['unrelated'][0]['block']
        initial=run.cli([target,'note','Initial interpretation.','--kind','correction','--request-id',N])
        assert initial['commit_state']=='committed',initial
        continuation.graceful_stop(run)
        older=output/'older-custody'
        custody(output/'archive',older,output/'older-custody.json')
        c=sqlite3.connect(path);history.insert(c,'D','T-other',history.EXPECTED['D'],2);c.commit();c.close()
        image2=path.read_bytes();(output/'expected-history-2.sqlite').write_bytes(image2)
        run.start();history.await_facts(run,{'C','D'})
        note=run.cli([initial['ref'],'revise',CORRECTION,'--request-id',R])
        assert note['commit_state']=='committed' and note['revision']['value']==CORRECTION,note
        before=await_clients(run,{'C','D'})
        latest=run.cli(['status'])
        report['independent_latest_acknowledgement']={'note':note,'clients':before,'write_counter':latest['foundation']['write_counter']}
        # Close all clients and retire the original writer without checkpointing;
        # its last acknowledged note must remain in a nonempty archive WAL.
        source_pid=run.daemon.pid;run.stop()
        report['original_writer_retired']={'pid':source_pid,'reaped':True,'old_archive_path':str(output/'archive')}
        assert not Path('/proc/'+str(source_pid)).exists()
        source=output/'retired-original';(output/'archive').rename(source)
        assert not (output/'archive').exists()
        assert (source/'archive.db-wal').stat().st_size>32
        destination=output/'restored';destination.mkdir()
        copied=custody(source,destination/'archive',output/'cold-custody.json')
        assert copied['archive.db-wal']['bytes']>32
        # Immutable main-only inspection cannot consume or rewrite the retained
        # WAL. Its missing acknowledged revision distinguishes custody from a
        # main-files-only copy even when SQLite integrity succeeds.
        main=sqlite3.connect('file:'+str(destination/'archive'/'archive.db')+'?immutable=1',uri=True)
        assert main.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
        assert main.execute('SELECT count(*) FROM note WHERE id=?',(R,)).fetchone()[0]==0
        main.close()
        report['acknowledgement_requires_wal']=True
        shutil.rmtree(home)
        empty.unlink();(output/'unrelated.jsonl').unlink()
        assert not home.exists() and not (output/'archive').exists()
        restored=mcp.Run(binary,destination);restored.env.update(run.env);readers.append(restored)
        report['interruption']=prepare(restored,interrupt=True)
        blocked=subprocess.run(restored.command(['daemon']),env=restored.env,capture_output=True,timeout=15)
        assert blocked.returncode!=0 and b'pending action' in blocked.stderr,blocked.stderr
        report['pending_writer_refusal']={'exit':blocked.returncode,'stderr':blocked.stderr.decode()}
        report['preparation']=prepare(restored)
        report['layouts']=inspect_custody(destination/'archive',[hashlib.sha256(image1).hexdigest(),hashlib.sha256(image2).hexdigest()])
        assert not (destination/'archive'/'index.db').exists()
        unaware=subprocess.run([str(unaware_binary),'--archive',str(destination/'archive'),'daemon'],env=restored.env,capture_output=True,timeout=15)
        assert unaware.returncode!=0 and b'newer than supported 2' in unaware.stderr,unaware.stderr
        report['unaware_writer']={'binary':str(unaware_binary.resolve()),'sha256':digest(unaware_binary),'exit':unaware.returncode,'stderr':unaware.stderr.decode()}
        restored.start();after=await_clients(restored,{'C','D'})
        for k,v in before['operations'].items():
            assert after['operations'][k]['ref']==v['ref']
            assert after['operations'][k]['body']['sources']==v['body']['sources']
        restored_status=restored.cli(['status'])
        assert restored_status['foundation']['latest_acknowledged_head']=='unknown'
        assert restored_status['foundation']['acquisition_lineage']==latest['foundation']['acquisition_lineage']
        restored.client=mcp.Client(restored)
        remote=read_clients(restored,{'C','D'},True,True)
        assert remote==after
        restored.client.close();restored.client=None
        continuation.graceful_stop(restored)
        report.update(restored_clients=after,restored_mcp=remote)
        inherited=output/'inherited';inherited.mkdir()
        custody(destination/'archive',inherited/'archive',inherited/'custody.json')
        clone=mcp.Run(binary,inherited);clone.env.update(run.env);readers.append(clone)
        before_authority=authority(inherited/'archive')
        report['inherited_receipt']=prepare(clone)
        assert report['inherited_receipt']['replies'][-1]['action']=='historical_completion'
        assert authority(inherited/'archive')==before_authority
        # Do not start the inherited instance. Only a new declaration prepares it.
        report['fresh_receipt']=prepare(clone,B)
        assert report['fresh_receipt']['replies'][-1]['action']=='prepared'
        assert authority(inherited/'archive')['acquisition_lineage']==before_authority['acquisition_lineage']
        authority_before=authority(destination/'archive')
        report['unsupported_independent']=prepare(restored,mode='independent')
        report['unsupported_primary']=prepare(restored,extra=['--config',str(output/'config.json')])
        assert authority(destination/'archive')==authority_before
        # Each damaging control begins from a separately preserved stopped copy.
        for label,root in [('older',older),('missing-base',source),('missing-patch',source)]:
            owner=output/label;owner.mkdir(exist_ok=False)
            custody(root,owner/'archive',owner/'custody.json')
            control=mcp.Run(binary,owner);control.env.update(run.env);readers.append(control)
            prepare(control,B)
            if label!='older':
                layout=report['layouts']
                base=layout[0]['blobs'][0]
                patch=next(h for h in layout[1]['blobs'] if h!=base)
                missing=base if label=='missing-base' else patch
                (owner/'archive'/'blobs'/missing[:2]/(missing+'.zst')).unlink()
            control.start()
            if label=='older':
                value=await_clients(control,{'C'},False)
                assert 'D' not in value['operations']
                assert not any(n['id']==R for n in value['notes'])
                assert control.cli(['status'])['foundation']['latest_acknowledged_head']=='unknown'
                report[label]={'valid_integrity_is_not_latest':True,'clients':value}
            else:
                until=time.monotonic()+30
                while True:
                    status=control.cli(['status'])
                    assert status['foundation']['unavailable_blob_files']==1 and status['outcome']=='degraded',status
                    unrelated=control.cli(['origin:claude-code','--json'])
                    if unrelated['rows']:
                        value=control.cli([unrelated['rows'][0]['id'],'read','--budget','20000'],allowed=(0,2,3,4,5,7))
                        if any(b.get('text')==TEXT for b in value.get('shown',[])):break
                    assert time.monotonic()<until,status
                    time.sleep(.05)
                failed=control.cli([before['operations']['D']['ref'],'read','--budget','20000'],allowed=(0,2,3,4,5,7))
                assert failed['outcome']=='error' and not failed.get('shown'),failed
                report[label]={'missing_blob':missing,'status':status,'failed_read':failed,'unrelated_read':value}
            continuation.graceful_stop(control)
        report['passed']=True
    finally:
        for current in [run,*readers]:
            native.close_run(current,current.output)
        cleanup={'daemons_reaped':all(r.daemon is None for r in [run,*readers]),'mcp_reaped':all(c.process.poll() is not None for r in [run,*readers] for c in r.clients),'runtimes_removed':all(not Path(r.socket_owner.name).exists() for r in [run,*readers])}
        (output/'cleanup.json').write_text(json.dumps(cleanup,indent=2)+'\n')
        assert all(cleanup.values()),cleanup
        (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--binary',required=True,type=Path);parser.add_argument('--unaware-binary',required=True,type=Path);parser.add_argument('--output',required=True,type=Path);args=parser.parse_args()
    if args.output.exists():parser.error('output must not exist')
    args.output.mkdir(parents=True)
    (args.output/'binding.json').write_text(json.dumps({'binary':str(args.binary.resolve()),'binary_sha256':digest(args.binary),'source':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'diff_sha256':hashlib.sha256(subprocess.check_output(['git','diff','HEAD'],cwd=ROOT)).hexdigest()},indent=2)+'\n')
    case(args.binary,args.unaware_binary,args.output)
    print(json.dumps({'passed':True,'report':str(args.output/'report.json')}))

if __name__=='__main__':main()
