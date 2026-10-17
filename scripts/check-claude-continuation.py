#!/usr/bin/env python3
"""One configured Claude daemon through ordinary CLI and official SDK MCP."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location('bodies', ROOT / 'scripts/check-claude-required-body.py')
bodies = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bodies)
observed = bodies.observed


def wait(run, predicate, timeout=40):
    deadline = time.monotonic() + timeout
    while True:
        value = run.command(['status'], allowed=(0, 4, 6, 7))
        if value is not None and predicate(value):
            return value
        assert run.process.poll() is None, 'configured daemon exited'
        assert time.monotonic() < deadline, value
        time.sleep(.05)


def await_progress(run, source, predicate):
    deadline = time.monotonic() + 40
    while True:
        progress = run.command(['progress', source])
        if predicate(progress):
            return progress
        assert time.monotonic() < deadline, progress
        time.sleep(.05)


def sdk(args, run, method, arguments, name):
    output = run.output / name
    argv = [str(args.sdk_python), str(args.sdk_client), '--binary', run.binary,
            '--archive', str(run.archive), '--call', method, '--arguments', json.dumps(arguments), '--output', str(output)]
    attempt = {'argv': argv, 'status': 'attempted'}
    run.sdk_calls.append(attempt)
    result = subprocess.run(argv, env=run.env, capture_output=True, timeout=40)
    attempt.update(status='completed', exit=result.returncode)
    (run.output / (name + '.stdout')).write_bytes(result.stdout)
    (run.output / (name + '.stderr')).write_bytes(result.stderr)
    assert result.returncode == 0, result.stderr.decode(errors='replace')
    receipt = json.loads((output / 'result.json').read_text())
    assert receipt['success'] and receipt['sdk_context_closed'] and receipt['sdk']['name'] == 'mcp', receipt
    return receipt['tool_result']


def assert_sdk_origin(shown, field, origin):
    decoded = []
    for envelope in shown['field_origins']:
        assert envelope['encoding'] == 'canonical-json-v1' and envelope['masked'] is False, envelope
        entry = json.loads(envelope['json'])
        assert set(entry) == {'field', 'origin'}, entry
        if entry['field'] == field:
            decoded.append(entry)
    assert decoded == [{'field': field, 'origin': origin}], decoded


def document(run, reference):
    value = run.command([reference, 'read', '--format', 'json'], allowed=(0, 2, 3, 4, 5))
    attempt = run.calls[-1]
    if attempt['exit'] == 2:
        diagnostic = (run.output / attempt['stderr']).read_text()
        assert diagnostic.strip() == 'session contains a pending Message', diagnostic
    return value


def await_text(run, reference, text):
    deadline = time.monotonic() + 40
    while True:
        value = document(run, reference)
        if value and any(block.get('text') == text for item in value.get('messages', []) for block in item['blocks']):
            return value
        assert time.monotonic() < deadline, value
        time.sleep(.05)


def start_configured(run, config):
    assert run.process is None
    argv = [run.binary, '--archive', str(run.archive), 'daemon', '--config', str(config), '--json']
    run.process = subprocess.Popen(argv, env=run.env, stdout=run.log, stderr=run.log)
    run.calls.append({'argv': argv, 'status': 'daemon_started', 'pid': run.process.pid})
    return wait(run, lambda value: value.get('claude_continuation') is not None)


def verify(args, run):
    main, body, duplicate, config, call, result, inline = bodies.fixture(run.output)
    settings = json.loads(config.read_text())
    settings['claude_main']['acquisition_quantum_bytes'] = 65536
    config.write_text(json.dumps(settings))
    initial = observed.user('earlier', 'Independent earlier human text.', None, 0)
    main.write_bytes(observed.line(initial))
    start_configured(run, config)
    pid = run.process.pid
    # Ordinary discovery supplies the reference; neither client gets an answer ID.
    deadline = time.monotonic() + 40
    while True:
        found = run.command(['origin:claude-code', '--json'], allowed=(0, 5))
        if found.get('rows'):
            break
        assert time.monotonic() < deadline, found
        time.sleep(.05)
    assert [row['id'] for row in found['rows']] == [observed.REF], found
    reference = found['rows'][0]['id']
    base = await_text(run, reference, initial['message']['content'])
    source = base['messages'][0]['sources'][0]['input']
    correction_text = 'Operator correction: the selected neutral colour is cedar green.'
    correction = observed.user('correction', correction_text, 'earlier', 1)
    correction_bytes = observed.line(correction)
    with main.open('ab') as stream:
        stream.write(correction_bytes[:len(correction_bytes)//2])
    state = wait(run, lambda value: value['claude_continuation']['retained_length'] == main.stat().st_size)
    progress = await_progress(run, source, lambda value: value.get('parser', {}).get('status') == 'waiting_tail' and value['active'] is False)
    before = await_text(run, reference, initial['message']['content'])
    assert all(item['id'] != reference + ':correction' for item in before['messages']), before
    earlier = run.command(['message:' + reference + ':earlier', 'read', '--budget', '20000'], allowed=(0, 5))
    assert earlier['shown'][0]['text'] == initial['message']['content'], earlier
    assert progress['parser']['status'] == 'waiting_tail' and progress['source_rows']['state'] != 'settled', progress
    with main.open('ab') as stream:
        stream.write(correction_bytes[len(correction_bytes)//2:])
    corrected = await_text(run, reference, correction_text)
    search = run.command(['"cedar green"', '--json'], allowed=(0, 5))
    assert [row['id'] for row in search['rows']] == [reference], search
    remote = sdk(args, run, 'search', {'query': '"cedar green"', 'budget_tokens': 20000}, 'sdk-correction-search')
    assert [row['id'] for row in remote['rows']] == [reference], remote
    remote_ref = remote['rows'][0]['id']
    remote = sdk(args, run, 'read', {'ref': remote_ref, 'budget_tokens': 20000, 'view': 'transcript', 'harness': 'full'}, 'sdk-correction-read')
    assert any(item.get('text') == correction_text for item in remote['shown']), remote
    message = next(item for item in corrected['messages'] if item['id'] == reference + ':correction')
    assert message['version_hash'] == bodies.expected_version(args, correction), message
    assert all(origin['origin']['binding'] == {'input': source, 'chain': 1} for origin in message['field_origins']), message
    raw = run.command(['input', 'cat', source], raw=True)
    assert raw == observed.line(initial) + correction_bytes
    for citation in message['sources']:
        assert citation['input'] == source and citation['chain'] == 1
        assert json.loads(raw[citation['start']:citation['end']]) == correction
    for entry in message['field_origins']:
        selection = entry['origin']['selection']
        selected = selection.get('inputs', [])
        if 'input' in selection:
            selected = [selection['input']]
        assert selected
        for selector in selected:
            span = selector['record']
            record = json.loads(raw[span['start']:span['end']])
            assert record == correction
            if entry['field'] == 'blocks[0].text':
                value = record
                for component in selector['path']:
                    value = value[component.get('Key', component.get('Index'))]
                assert value == correction_text
    remote_correction = next(item for item in remote['shown'] if item.get('text') == correction_text)
    assert remote_correction['version_hash'] == message['version_hash']
    assert remote_correction['message'] == 'message:' + message['id']
    assert remote_correction['sources'] == message['sources']
    text_origin = next(entry['origin'] for entry in message['field_origins'] if entry['field'] == 'blocks[0].text')
    assert_sdk_origin(remote_correction, 'blocks[0].text', text_origin)
    # Later required output is absent first, then arrives and changes without a main edit.
    body.unlink()
    call['parentUuid'] = 'correction'
    with main.open('ab') as stream:
        stream.write(observed.line(call) + observed.line(result))
    wait(run, lambda value: value['claude_continuation']['retained_length'] == main.stat().st_size)
    await_progress(run, source, lambda value: value.get('parser') is not None and value['parser']['acquired_upto'] == main.stat().st_size and value['active'] is False)
    assert run.command(['message:' + reference + ':earlier', 'read', '--budget', '20000'], allowed=(0, 5))['shown'][0]['text'] == initial['message']['content']
    body.write_bytes(bodies.PAYLOAD)
    full = await_text(run, reference, bodies.MASKED)
    first_message, first_origin = bodies.assert_result(args, full, result, source, body)
    changed = bodies.PAYLOAD + b'\nBody-only correction.'
    body.write_bytes(changed)
    changed_text = bodies.MASKED + '\nBody-only correction.'
    revised = await_text(run, reference, changed_text)
    revised_message, revised_origin = bodies.assert_result(args, revised, result, source, body, changed_text, changed)
    assert first_message['version_hash'] != revised_message['version_hash']
    assert {item['id']: item['version_hash'] for item in full['messages'] if item['id'] != reference + ':result'} == {item['id']: item['version_hash'] for item in revised['messages'] if item['id'] != reference + ':result'}
    remote = bodies.sdk_read(args, run, 'block:' + remote_ref + ':result:0', 'sdk-body-only')
    assert remote['shown'][0]['text'] == changed_text, remote
    assert_sdk_origin(remote['shown'][0], 'blocks[0].text', revised_origin)
    stable = wait(run, lambda value: value['claude_continuation']['unchanged_passes'] >= 2)
    attempts = stable['claude_continuation']['attempts']
    counter = stable['foundation']['write_counter']
    stable_after = wait(run, lambda value: value['claude_continuation']['attempts'] >= attempts + 2)
    assert stable_after['foundation']['write_counter'] == counter, (stable, stable_after)
    # Missing and denied configured mains are distinct; neither becomes accessible-empty.
    main.chmod(0)
    denied = wait(run, lambda value: value['claude_continuation']['state'] == 'denied')
    main.chmod(0o600)
    main_saved = main.with_suffix('.saved')
    main.rename(main_saved)
    missing = wait(run, lambda value: value['claude_continuation']['state'] == 'missing')
    assert document(run, reference)['messages'] == revised['messages']
    main_saved.rename(main)
    wait(run, lambda value: value['claude_continuation']['state'] == 'polling')
    assert run.process.pid == pid
    # Interrupt with an acknowledged unfinished suffix; plain restart is provider-free.
    tail = observed.line(observed.user('unfinished', 'This record remains owed.', 'result', 4))[:-3]
    with main.open('ab') as stream:
        stream.write(tail)
    wait(run, lambda value: value['claude_continuation']['retained_length'] == main.stat().st_size)
    run.process.send_signal(signal.SIGINT)
    assert run.process.wait(timeout=15) == 0
    run.process = None
    main.unlink(); body.unlink(); duplicate.unlink()
    run.start()
    try:
        progress = await_progress(run, source, lambda value: value.get('parser', {}).get('status') == 'waiting_tail' and value['active'] is False)
        retained = document(run, reference)
        assert next(item for item in retained['messages'] if item['id'] == reference + ':result')['version_hash'] == revised_message['version_hash'], retained
        assert run.command(['input', 'cat', source], raw=True).endswith(tail)
        progress = await_progress(run, source, lambda value: value.get('parser', {}).get('status') == 'waiting_tail')
        assert progress['source_rows']['state'] != 'settled' and progress['parser']['status'] == 'waiting_tail', progress
        remote = bodies.sdk_read(args, run, 'block:' + reference + ':result:0', 'sdk-provider-free')
        assert remote['shown'][0]['text'] == changed_text, remote
        assert_sdk_origin(remote['shown'][0], 'blocks[0].text', revised_origin)
    finally:
        run.stop()
    return {'pid': pid, 'input': source, 'discovered_reference': reference, 'correction_version': message['version_hash'], 'first_body_origin': first_origin, 'revised_body_origin': revised_origin, 'unchanged_write_counter': counter, 'denied': denied['claude_continuation'], 'missing': missing['claude_continuation'], 'restart_pending': progress}


def empty_control(args, output):
    output.mkdir()
    run = observed.Run(args.binary, output)
    main = output / 'empty.jsonl'
    config = output / 'source.json'
    main.write_bytes(b'')
    config.write_text(json.dumps({'claude_main': {'path': str(main), 'machine': 'synthetic-empty', 'host': 'synthetic-host', 'acquisition_quantum_bytes': 65536}}))
    try:
        start_configured(run, config)
        empty = wait(run, lambda value: value['claude_continuation']['state'] == 'accessible_empty')
        assert empty['claude_continuation']['retained_length'] == 0
        assert run.command(['input', 'cat', empty['claude_continuation']['input']], raw=True) == b''
        return empty['claude_continuation']
    finally:
        errors = run.close()
        (output / 'commands.json').write_text(json.dumps(run.calls, indent=2) + '\n')
        assert not errors, errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('binary', 'output', 'sdk-python', 'sdk-client', 'b3sum'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--binary-candidate', required=True)
    args = parser.parse_args()
    args.binary = args.binary.resolve(strict=True)
    args.output.mkdir(parents=True, exist_ok=False)
    run = observed.Run(args.binary, args.output)
    run.sdk_calls = []
    report = {'outcome': 'failed', 'binary': str(args.binary), 'binary_sha256': hashlib.sha256(args.binary.read_bytes()).hexdigest(), 'candidate_sha': args.binary_candidate, 'driver_candidate_sha': subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(), 'candidate_tree': subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', args.binary_candidate + '^{tree}'], text=True).strip(), 'driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'sdk_python': str(args.sdk_python), 'sdk_python_sha256': hashlib.sha256(args.sdk_python.read_bytes()).hexdigest(), 'sdk_client': str(args.sdk_client), 'sdk_client_sha256': hashlib.sha256(args.sdk_client.read_bytes()).hexdigest()}
    try:
        report['continuation'] = verify(args, run)
        report['accessible_empty'] = empty_control(args, args.output / 'empty-control')
        report['outcome'] = 'ok'
    except BaseException as error:
        report['error'] = repr(error)
        raise
    finally:
        report['sdk_calls'] = run.sdk_calls
        report['cleanup_errors'] = run.close()
        (args.output / 'commands.json').write_text(json.dumps(run.calls, indent=2) + '\n')
        (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        assert not report['cleanup_errors'], report['cleanup_errors']


if __name__ == '__main__':
    main()
