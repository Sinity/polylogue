"""Finite synthetic predecessor regressions through a pinned native binary and official SDK.

Expected facts are source-derived from the retained predecessor tests, not an
executed old-parser parity result. Large cases preserve their donor defaults.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

DONOR = 'b4b61c7eab8daf740bfdccb146e530906839726d'

def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''): digest.update(chunk)
    return digest.hexdigest()

def main():
    parser = argparse.ArgumentParser()
    for name in ('binary', 'source', 'output', 'sdk-python', 'sdk-client'): parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--baseline-run', type=Path, action='append', default=[], help='Stopped synthetic baseline run; copy full archive custody before opening')
    parser.add_argument('--families-only', action='store_true', help='Exploration only; does not qualify large regressions')
    args = parser.parse_args()
    actual_source = subprocess.check_output(['git','-C',str(args.source),'rev-parse','HEAD'], text=True).strip()
    assert actual_source == args.candidate, (actual_source, args.candidate)
    assert not subprocess.check_output(['git','-C',str(args.source),'status','--porcelain'], text=True), 'candidate must be clean'
    args.output.mkdir(mode=0o700)
    sys.path.insert(0, str(args.source / 'scripts'))
    spec = importlib.util.spec_from_file_location('donor_morrow_run', args.source / 'scripts/check-morrow.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    run = module.Run(args.output, None); run.binary = args.binary
    sdk = args.output / 'frozen-sdk' / 'joint-consumer' / 'mcp_client.py'; sdk.parent.mkdir(parents=True)
    shutil.copyfile(args.sdk_client, sdk)
    for name in ('morrow_mcp.py', 'check_answer.py'): shutil.copyfile(args.sdk_client.parent.parent / name, sdk.parent.parent / name)
    archive = args.output / 'archive'
    corpus = args.source / 'crates/sources/tests/fixtures/donor'
    manifest = json.loads((corpus / 'manifest.json').read_text())
    assert manifest['commit'] == DONOR
    report = {'candidate': args.candidate, 'binary': str(args.binary), 'binary_sha256': sha(args.binary),
              'runner_sha256': sha(Path(__file__)), 'donor': manifest, 'expected_facts': 'source-derived predecessor test literals plus exact current-spec differences',
              'sdk_python': str(args.sdk_python), 'sdk_python_sha256': sha(args.sdk_python), 'sdk_client_sha256': sha(sdk),
              'sdk_collector_sha256': sha(sdk.parent.parent / 'morrow_mcp.py'), 'sdk_checker_sha256': sha(sdk.parent.parent / 'check_answer.py'),
              'commands': run.commands, 'processes': run.processes, 'large': [], 'upgrades': [], 'success': False}
    references, provider_paths, bindings, pages = {}, [], {}, {}
    def cli(argv, label, expected=(0,5), timeout=60):
        return json.loads(run.command([args.binary, '--archive', archive, *argv], label, expected, timeout=timeout))
    def sdk_call(name, arguments, label):
        out = args.output / ('official-sdk-' + label)
        run.command([args.sdk_python, sdk, '--binary', args.binary, '--archive', archive, '--call', name,
                     '--arguments', json.dumps(arguments), '--output', out], label, (0,), timeout=60)
        result = json.loads((out / 'result.json').read_text())
        assert result['success'] and result['sdk_context_closed'], result
        assert any(call['method'] == 'initialize' for call in result['calls'])
        assert any(call['method'] == 'tools/list' for call in result['calls'])
        assert any(call['method'] == 'tools/call' and call['status'] == 'returned' for call in result['calls'])
        return result['tool_result']
    def admit(path, format, context=None, timeout=60, degraded=False):
        argv = ['import', path, '--format', format, '--wait', '--wait-timeout', str(timeout - 5)]
        if context:
            cp = path.with_suffix('.context.json'); cp.write_text(json.dumps({'version': 1, 'context': ['claude-main', context]})); argv += ['--context', cp]
        started = time.monotonic(); value = cli(argv, 'import ' + path.name, (0,5), timeout=timeout)
        assert value['source_rows']['state'] in (('unavailable',) if degraded else ('settled','unavailable')), value
        if value['source_rows']['state'] == 'unavailable':
            assert value['source_rows']['usable_result'], value
        bindings[path.name] = value
        return value, time.monotonic() - started
    def inspect(reference, label):
        document = cli([reference, 'read', '--format', 'json'], label + ' JSON')
        assert document['messages'], document
        export = args.output / (label + '-export')
        cli([reference, 'export', '--format', 'json', '--to', export], label + ' export')
        files = [p for p in export.rglob('*.json') if p.name != 'manifest.json']; assert len(files) == 1
        exported = json.loads(files[0].read_text())
        assert exported['messages'] == document['messages']
        page = sdk_call('read', {'ref': reference, 'budget_tokens': 65000, 'view': 'transcript', 'tools': 'full', 'thinking': 'full', 'harness': 'full', 'branches': True}, label + '-read')
        assert page['outcome'] != 'error' and page['primary_content_complete'] and not page.get('next'), page
        assert page['shown'] and all(item['sources'] for item in page['shown'])
        expected_blocks = [(message, block) for message in document['messages'] for block in message['blocks']]
        assert len(page['shown']) == len(expected_blocks), page
        for item, (message, block) in zip(page['shown'], expected_blocks):
            assert item['message'] == 'message:' + message['id'] and item['block'] == 'block:' + block['id']
            assert item['kind'] == block['kind'] and item['role'] == message['role'] and item['written_by'] == message['written_by']
            assert item['message_metadata'].get('at') == message.get('at')
            for field in ('call_id','outcome','exit_code','duration_ms','tool'):
                assert item['block_fields'].get(field) == block.get(field), (field, item, block)
        if label.startswith('restart-'):
            assert page['shown'] == pages[label.removeprefix('restart-')], 'identities, source citations or fields changed after restart'
        else:
            pages[label.removeprefix('initial-')] = page['shown']
        return document, page
    try:
        run.start(archive)
        for source_path, binding in manifest['files'].items():
            name = Path(source_path).name; original = corpus / name
            assert sha(original) == binding['sha256']
            path = args.output / name; shutil.copyfile(original, path); provider_paths.append(path)
            claude = name.startswith('claude-')
            value, elapsed = admit(path, 'claude-code-jsonl' if claude else 'codex-rollout', 'claude-normalization-main' if claude else None)
            assert value['retained']['length'] == path.stat().st_size
            reference = ('claude-code:claude-normalization-main' if claude else 'codex:legacy-response-session' if name.startswith('legacy') else 'codex:tool-result-headers')
            references[name] = reference
        initial = {name: inspect(ref, 'initial-' + name)[0] for name, ref in references.items()}
        legacy_blocks = [b for m in initial['legacy-response-records.jsonl']['messages'] for b in m['blocks']]
        for call, outcome, exit in [('call_sample_ok','ok',0), ('call_sample_fail','error',1)]:
            assert any(b.get('kind') == 'tool_call' and b.get('call_id') == call and b['input']['command'][0] in ('ls','cat') for b in legacy_blocks)
            assert any(b.get('kind') == 'tool_result' and b.get('call_id') == call and b['outcome'] == outcome and b['exit_code'] == exit and b['duration_ms'] == 100 for b in legacy_blocks)
        assert any(b.get('text') == 'Listing the directory first.' for b in legacy_blocks)
        headers = [b for m in initial['tool-result-headers.jsonl']['messages'] for b in m['blocks'] if b['kind'] == 'tool_result']
        expected = {'call_command_ok':('ok',0),'call_command_fail':('error',2),'call_command_multiline':('unknown',None),'call_stdin_running':('unknown',None),'call_patch_fail':('error',1),'call_shell_ok':('ok',0),'call_script_ok':('ok',None),'call_script_fail':('error',None),'call_prose_decoy':('unknown',None),'call_script_decoy':('unknown',None)}
        assert {b['call_id']:(b['outcome'], b.get('exit_code')) for b in headers} == expected
        main = initial['claude-normalization-main.jsonl']; assert len(main['messages']) == 10
        named = ['main-u1','main-a1','main-bg-start','main-command','other-u1','main-context',None,'main-a2','main-fg-result','other-a1']
        assert [message['id'] if native is None else message['id'].removeprefix('claude-code:claude-normalization-main:') for message,native in zip(main['messages'],named) if native is not None] == [native for native in named if native is not None]
        assert main['messages'][6]['id'].startswith('~i1:')
        assert [m['role'] for m in main['messages']] == ['user','assistant','user','user','user','user','user','assistant','user','assistant']
        assert main['messages'][2]['written_by'] == 'harness' and main['messages'][3]['written_by'] == 'human' and main['messages'][3]['prompt_kind'] == 'command'
        assert main['messages'][5]['written_by'] == 'harness' and main['messages'][6]['written_by'] == 'harness'
        assert main['messages'][1]['at'] == '2026-07-01T10:00:01.000Z'
        assert main['messages'][1]['usage'] == {'input':120,'output':40,'cache_read':900,'cache_write':300,'provider_cost_usd':0.125}
        blocks = [b for m in main['messages'] for b in m['blocks']]
        assert any(b.get('text') == 'Conflicting duplicate that must not survive.' for b in blocks)  # explicit current-spec replacement
        assert any(b.get('text') == '[Request interrupted by user]' for b in blocks)
        assert any(b.get('call_id') == 'tool-bg' and b.get('outcome') == 'error' and b.get('exit_code') == 7 for b in blocks)
        assert any(b.get('call_id') == 'tool-fg' and b.get('outcome') == 'error' for b in blocks)
        assert any(m.get('duration_ms') == 1500 and m.get('usage',{}).get('provider_cost_usd') == 0.125 for m in main['messages'])
        side_query = cli(['Normalization complete.', '--json'], 'discover sidechain')
        subagents = [row['id'] for row in side_query['rows'] if row['id'] != 'claude-code:claude-normalization-main']; assert len(subagents) == 1, side_query
        side, _ = inspect(subagents[0], 'initial-sidechain')
        assert len(side['messages']) == 1 and side['messages'][0]['blocks'][0]['text'] == 'Normalization complete.'
        references['sidechain'] = subagents[0]; initial['sidechain'] = side
        for query, ref in [('directory','codex:legacy-response-session'), ('sample','codex:tool-result-headers'), ('Conflicting','claude-code:claude-normalization-main')]:
            result = cli([query, '--json'], 'search ' + query)
            assert any(row['id'] == ref for row in result['rows']), result
            remote = sdk_call('search', {'query':query,'budget_tokens':65000}, 'search-' + query)
            assert remote['rows'] == result['rows']
        if not args.families_only:
            for session, states, filler, future, text in [('codex-stream-million',2000000,False,False,'sanitized streaming boundary'),('bounded-stream',120000,True,True,'bounded')]:
                path = args.output / (session + '.jsonl'); provider_paths.append(path)
                with path.open('wb') as handle:
                    def write(value): handle.write((json.dumps(value,separators=(',',':'),sort_keys=True)+'\n').encode())
                    payload = {'id':session}
                    if not filler: payload['timestamp'] = '2026-08-06T00:00:00Z'
                    write({'type':'session_meta','payload':payload})
                    write({'type':'response_item','payload':{'type':'message','id':session+'-message','role':'user','content':[{'type':'input_text','text':text}]}})
                    for sequence in range(states):
                        record = {'record_type':'state','sequence':sequence}
                        if filler: record['filler'] = 's'*900 + str(sequence)
                        write(record)
                    if future: write({'type':'future_whale_record'})
                size = path.stat().st_size
                if filler: assert size > 64*1024*1024
                receipt, elapsed = admit(path, 'codex-rollout', timeout=1200, degraded=future)
                progress = cli(['progress',receipt['retained']['id']], session + ' progress')
                counts = progress['parser']['records']; assert counts['framed'] == states+2+int(future)
                assert counts['used'] == 2 and sum(counts['ignored'].values()) == states
                assert len(counts['unrecognized']) == int(future) and not counts['undecodable']
                assert progress['parser']['parsed_upto'] == size
                assert progress['parser']['finalization_pending'] == future
                reference = 'codex:'+session; doc, _ = inspect(reference, 'initial-'+session)
                assert len(doc['messages']) == 1 and doc['messages'][0]['blocks'][0]['text'] == text
                references[session] = reference; initial[session] = doc
                report['large'].append({'session':session,'state_records':states,'input_bytes':size,'elapsed_seconds':elapsed,'progress':progress})
        run.stop()
        for path in provider_paths: path.unlink()
        run.start(archive)
        for name, reference in references.items():
            doc, _ = inspect(reference, 'restart-'+name)
            assert doc['messages'] == initial[name]['messages'] and doc.get('links') == initial[name].get('links')
        run.stop()
        for index, baseline in enumerate(args.baseline_run):
            original_archive = baseline / 'archive'
            before = {str(path.relative_to(original_archive)):sha(path) for path in original_archive.rglob('*') if path.is_file()}
            assert before and 'archive.db' in before and 'index.db' in before
            assert all(not path.is_symlink() for path in original_archive.rglob('*'))
            archive = args.output / ('upgrade-archive-' + str(index))
            shutil.copytree(original_archive, archive)
            assert {str(path.relative_to(archive)):sha(path) for path in archive.rglob('*') if path.is_file()} == before
            upgrade = {'baseline':str(baseline),'custody_before_open':before,'baseline_commands_sha256':sha(baseline/'commands.json'),'inputs':[]}
            report['upgrades'].append(upgrade)
            run.start(archive)
            for command in json.loads((baseline/'commands.json').read_text()):
                if command['arguments'][0] != 'import': continue
                old = json.loads((baseline/command['stdout']).read_text())
                assert old['outcome'] == 'error' and old['source_rows']['basis']['parser_version'] == 8
                name = Path(command['arguments'][1]).name
                assert name in references and old['retained']['length'] > 0
                retained_id = old['retained']['id']
                cli(['retry',retained_id], 'upgrade retry ' + name, (0,))
                deadline = time.monotonic() + 90
                while True:
                    progress = cli(['progress',retained_id], 'upgrade progress ' + name)
                    if progress['source_rows']['basis']['parser_version'] == 9 and progress['source_rows']['state'] != 'pending': break
                    assert time.monotonic() < deadline, progress
                    time.sleep(0.2)
                assert progress['failure'] is None and progress['retained'] == old['retained'], progress
                assert progress['source_rows'].get('usable_result',True), progress
                doc, page = inspect(references[name], 'upgrade-' + str(index) + '-' + name)
                # Observation paths differ between fresh and baseline archives.
                # Content, IDs, versions and original-value fields must agree.
                def semantic(messages): return [{key:value for key,value in message.items() if key not in ('sources','field_origins')} for message in messages]
                assert semantic(doc['messages']) == semantic(initial[name]['messages'])
                upgrade['inputs'].append({'name':name,'retained':old['retained'],'old_failure':old['error'],'progress':progress})
            run.stop()
            after = {str(path.relative_to(original_archive)):sha(path) for path in original_archive.rglob('*') if path.is_file()}
            assert after == before, 'baseline original custody changed'
            upgrade['original_custody_unchanged'] = True
        report.update(success=True,bindings=bindings,reference_count=len(references))
    finally:
        run.stop(check=False); run.close_runtime()
        (args.output / 'result.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'success':report['success'],'reference_count':len(references),'large':[{'session':v['session'],'state_records':v['state_records'],'input_bytes':v['input_bytes'],'elapsed_seconds':v['elapsed_seconds']} for v in report['large']]}))

if __name__ == '__main__': main()
