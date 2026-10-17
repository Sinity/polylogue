"""Neutral old10 retry and CLI/official-SDK witness for unified Message reads."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    for key in ('binary', 'old-binary', 'source', 'output', 'sdk-python', 'sdk-client'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--candidate', required=True)
    args = parser.parse_args()
    assert subprocess.check_output(['git', '-C', args.source, 'rev-parse', 'HEAD'], text=True).strip() == args.candidate
    assert not subprocess.check_output(['git', '-C', args.source, 'status', '--porcelain'])
    args.output.mkdir(mode=0o700)
    sys.path.insert(0, str(args.source / 'scripts'))
    spec = importlib.util.spec_from_file_location('existing_morrow_run', args.source / 'scripts/check-morrow.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    run = module.Run(args.output, None)
    sdk = args.output / 'frozen-sdk/joint-consumer/mcp_client.py'
    sdk.parent.mkdir(parents=True)
    shutil.copyfile(args.sdk_client, sdk)
    for name in ('morrow_mcp.py', 'check_answer.py'):
        shutil.copyfile(args.sdk_client.parent.parent / name, sdk.parent.parent / name)
    report = {'success': False, 'candidate': args.candidate, 'binary_sha256': sha(args.binary),
              'old_binary_sha256': sha(args.old_binary), 'runner_sha256': sha(Path(__file__)),
              'sdk_python_sha256': sha(args.sdk_python), 'sdk_client_sha256': sha(sdk),
              'sdk_collector_sha256': sha(sdk.parent.parent / 'morrow_mcp.py'),
              'sdk_checker_sha256': sha(sdk.parent.parent / 'check_answer.py'),
              'commands': run.commands, 'processes': run.processes}
    archive = args.output / 'archive'
    binary = args.old_binary
    reference = 'codex:unified-neutral'

    def cli(argv, label, expected=(0, 2, 5, 7)):
        return json.loads(run.command([binary, '--archive', archive, *argv], label, expected, timeout=45))

    def sdk_read(label, branches):
        output = args.output / ('sdk-' + label)
        arguments = {'ref': reference, 'budget_tokens': 65000, 'view': 'transcript',
                     'tools': 'full', 'thinking': 'full', 'harness': 'full', 'branches': branches}
        run.command([args.sdk_python, sdk, '--binary', binary, '--archive', archive,
                     '--call', 'read', '--arguments', json.dumps(arguments), '--output', output], label, (0,), timeout=45)
        receipt = json.loads((output / 'result.json').read_text())
        assert receipt['success'] and receipt['sdk_context_closed']
        assert all(any(call['method'] == method and (method != 'tools/call' or call['status'] == 'returned')
                       for call in receipt['calls']) for method in ('initialize', 'tools/list', 'tools/call'))
        return receipt['tool_result']

    full = [reference, 'read', '--budget', '65000', '--view', 'transcript',
            '--tools', 'full', '--thinking', 'full', '--harness', 'full']
    result_text = 'Chunk ID: 123abc\nWall time: 0.001 seconds\nProcess exited with code 0\nFinal output:\nneutral\n'
    records = [
        {'type': 'session_meta', 'payload': {'id': 'unified-neutral', 'source': 'cli'}},
        {'type': 'turn_context', 'payload': {'turn_id': 'T'}},
        {'type': 'response_item', 'payload': {'type': 'message', 'id': 'U', 'role': 'user', 'content': [{'type': 'input_text', 'text': 'A neutral check.'}]}},
        {'type': 'response_item', 'payload': {'type': 'function_call', 'id': 'call-message', 'call_id': 'C', 'name': 'exec_command', 'arguments': '{"cmd":"printf neutral"}'}},
        {'type': 'response_item', 'payload': {'type': 'function_call_output', 'call_id': 'C', 'output': result_text}},
        {'type': 'event_msg', 'payload': {'type': 'item_completed', 'thread_id': 'unified-neutral', 'turn_id': 'T', 'started_at_ms': 1, 'completed_at_ms': 2,
            'item': {'type': 'CommandExecution', 'id': 'C', 'source': 'unified_exec_startup', 'command': ['printf', 'neutral'], 'status': 'completed', 'exit_code': 0, 'aggregated_output': 'not the independent result', 'duration': {'secs': 0, 'nanos': 1000000}}}},
    ]
    path = args.output / 'neutral.jsonl'
    path.write_text(''.join(json.dumps(row, separators=(',', ':')) + '\n' for row in records))
    try:
        run.binary = binary
        run.start(archive, ready_exits=(0, 5))
        retained = cli(['import', path, '--format', 'codex-rollout', '--machine', '00112233445566778899aabbccddeeff', '--host', 'synthetic-collector', '--wait', '--wait-timeout', '35'], 'old10 import')
        input_id = retained['retained']['id']
        old = cli([reference, 'read', '--format', 'json'], 'old10 JSON')
        refused = cli([*full, '--branches'], 'old10 branches')
        assert refused['error']['reason'] == 'current_selected_basis_unavailable'
        assert cli(['progress', input_id], 'old10 progress')['source_rows']['basis']['parser_version'] == 10
        run.stop()
        path.unlink()
        binary = args.binary
        run.binary = binary
        run.start(archive, ready_exits=(0, 5))
        report['retry'] = cli(['retry', input_id], 'retained retry', (0, 7))
        if report['retry']['outcome'] == 'error':
            assert report['retry']['error']['kind'] == 'timeout' and report['retry']['error']['retryable'] is True
            active = cli(['progress', input_id], 'active retry proof')
            assert active['input'] == input_id and active['active'] and active['failure'] is None
        deadline = time.monotonic() + 90
        while True:
            progress = cli(['progress', input_id], 'current progress')
            assert progress['failure'] is None, progress
            if progress['source_rows']['basis']['parser_version'] == 11 and progress['source_rows']['state'] != 'pending':
                break
            assert time.monotonic() < deadline, progress
            time.sleep(0.1)
        report['progress'] = progress
        assert progress['length'] == retained['retained']['length'] and progress['source_rows']['basis']['chain'] == retained['retained']['chain']
        current = cli([reference, 'read', '--format', 'json'], 'current JSON')
        assert current['messages'] == old['messages'], 'known Message values and origins must not change'
        exported = args.output / 'export'
        run.command([binary, '--archive', archive, reference, 'export', '--format', 'json', '--to', exported], 'current export', (0, 5), timeout=45)
        documents = [p for p in exported.rglob('*.json') if p.name != 'manifest.json']
        assert len(documents) == 1 and json.loads(documents[0].read_text())['messages'] == current['messages']
        page = cli([*full, '--branches'], 'current branches')
        assert page['outcome'] != 'error' and page['primary_content_complete']
        remote = sdk_read('current-branches', True)
        assert remote['shown'] == page['shown']
        blocks = [block for message in current['messages'] for block in message['blocks']]
        assert [block['kind'] for block in blocks] == ['text', 'tool_call', 'tool_result']
        assert blocks[-1]['text'] == result_text and blocks[-1]['outcome'] == 'ok' and blocks[-1]['exit_code'] == 0 and blocks[-1]['duration_ms'] == 1
        default = cli(full, 'current default')
        assert default['outcome'] == 'error' and 'shown' not in default
        assert sdk_read('current-default', False)['error'] == default['error']
        run.stop()
        run.start(archive, ready_exits=(0, 5))
        assert cli([reference, 'read', '--format', 'json'], 'restart JSON')['messages'] == current['messages']
        assert sdk_read('restart-branches', True)['shown'] == page['shown']
        report['success'] = True
    finally:
        run.stop(check=False)
        run.close_runtime()
        (args.output / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'success': True, 'sdk_receipts': 3, 'retained_retry': True}))


if __name__ == '__main__':
    main()
