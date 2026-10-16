#!/usr/bin/env python3
"""Independent required-Claude-body oracle through packaged CLI and official MCP."""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location('observed', ROOT / 'scripts/check-observed-source.py')
observed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(observed)
SID = observed.SID
REF = observed.REF
SECRET = 'ghp_' + 'A' * 36
PAYLOAD = ('Full retained output 東京🙂.\n' + 'neutral result line\n' * 400 + SECRET + '\nEND full output.').encode()
MASKED = PAYLOAD.decode().replace(SECRET, '[credential: GitHub token, masked]')


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()


def digest(args, data):
    result = subprocess.run([args.b3sum], input=data, capture_output=True, check=True)
    return result.stdout.decode().split()[0]


def expected_version(args, record, payload=None):
    raw = canonical(record)
    if payload is None:
        return digest(args, raw)
    # Owning compound-evidence wire recipe, independently assembled here. The
    # synthetic user result has one record and no differing final metadata.
    units = [(0, b'record/0', raw), (1, b'record/0/tool_result/0/body', len(payload).to_bytes(8, 'big') + bytes.fromhex(digest(args, payload)))]
    framed = b'\0polylogue-message-version\0v1\0' + len(units).to_bytes(8, 'big')
    for kind, tag, value in units:
        framed += bytes([kind]) + len(tag).to_bytes(8, 'big') + tag + len(value).to_bytes(8, 'big') + value
    return digest(args, framed)


def fixture(output):
    main = output / (SID + '.jsonl')
    directory = output / SID / 'tool-results'
    directory.mkdir(parents=True)
    body = directory / 'call.txt'
    duplicate = directory / 'inline.txt'
    call = {'type': 'assistant', 'uuid': 'call', 'sessionId': SID, 'parentUuid': None, 'requestId': 'request', 'timestamp': '2026-10-01T08:00:00Z', 'message': {'id': 'response', 'model': 'synthetic-model', 'role': 'assistant', 'content': [{'type': 'tool_use', 'id': 'call', 'name': 'exec_command', 'input': {'cmd': 'synthetic fixture'}}]}}
    result = {'type': 'user', 'uuid': 'result', 'sessionId': SID, 'parentUuid': 'call', 'timestamp': '2026-10-01T08:01:00Z', 'toolUseResult': {'persistedOutputPath': str(body)}, 'message': {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 'call', 'content': 'Preview; complete output is stored externally.'}]}}
    inline = {'type': 'user', 'uuid': 'inline', 'sessionId': SID, 'parentUuid': 'result', 'timestamp': '2026-10-01T08:02:00Z', 'toolUseResult': {'persistedOutputPath': str(duplicate)}, 'message': {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 'inline', 'content': 'Complete inline result.'}]}}
    config = output / 'source.json'
    config.write_text(json.dumps({'claude_main': {'path': str(main), 'machine': 'synthetic-machine', 'host': 'synthetic-host'}}))
    body.write_bytes(PAYLOAD)
    duplicate.write_text('Complete inline result.')
    return main, body, duplicate, config, call, result, inline


def read_document(run):
    return run.command([REF, 'read', '--format', 'json'], allowed=(0, 3, 4, 5))


def refused_document(run, expected_hash, corrupt=False):
    answer = read_document(run)
    command = run.calls[-1]
    diagnostic = (run.output / command['stderr']).read_text()
    assert command['exit'] in (2, 3, 4, 5), command
    assert answer is None or (answer['outcome'] == 'error' and not answer.get('messages')), answer
    # The streaming document CLI reports a preflight refusal on stderr with no
    # JSON answer. Preserve that actual absence instead of inventing an object.
    if answer is None:
        assert 'blob:' in diagnostic and (corrupt or expected_hash in diagnostic), diagnostic
    return {'answer': answer, 'exit': command['exit'], 'diagnostic': diagnostic, 'stdout': command['stdout'], 'stderr': command['stderr']}


def assert_result(args, document, result, source, body, expected=MASKED, payload=PAYLOAD):
    assert document['session']['id'] == REF, document
    message = next(m for m in document['messages'] if m['id'] == REF + ':result')
    assert message['role'] == 'user' and message['written_by'] == 'harness', message
    assert message['version_hash'] == expected_version(args, result, payload), message
    block = message['blocks'][0]
    assert block['id'] == REF + ':result:0' and block['kind'] == 'tool_result'
    assert block['text'] == expected and block['call_id'] == 'call' and block['outcome'] == 'ok', block
    origin = next(entry['origin'] for entry in message['field_origins'] if entry['field'] == 'blocks[0].text')
    assert origin['format'] == 'claude-code-tool-result-body', origin
    assert origin['binding']['chain'] == 1 and origin['revision'] == 1
    assert origin['path'] == str(body) and origin['start'] == 0 and origin['end'] == len(payload)
    assert origin['blob'] == digest(args, payload)
    assert message['sources'] and all(cite['input'] == source for cite in message['sources'])
    assert SECRET not in json.dumps(document)
    return message, origin


def sdk_read(args, run, reference, name):
    output = run.output / name
    arguments = {'ref': reference, 'budget_tokens': 20000, 'view': 'transcript', 'tools': 'full', 'thinking': 'full', 'harness': 'full', 'branches': True}
    argv = [str(args.sdk_python), str(args.sdk_client), '--binary', run.binary, '--archive', str(run.archive), '--call', 'read', '--arguments', json.dumps(arguments), '--output', str(output)]
    attempt = {'argv': argv, 'status': 'attempted', 'exit': None}
    run.sdk_calls.append(attempt)
    result = subprocess.run(argv, env=run.env, capture_output=True, timeout=40)
    attempt.update(status='completed', exit=result.returncode)
    (run.output / (name + '.stdout')).write_bytes(result.stdout)
    (run.output / (name + '.stderr')).write_bytes(result.stderr)
    assert result.returncode == 0, result.stderr.decode(errors='replace')
    receipt = json.loads((output / 'result.json').read_text())
    assert receipt['success'] and receipt['sdk_context_closed'] and receipt['sdk']['name'] == 'mcp', receipt
    return receipt['tool_result']


def verify(args, run):
    main, body, duplicate, config, call, result, inline = fixture(run.output)
    earlier = observed.user('earlier', 'Independent earlier human text.', None, 0)
    call['parentUuid'] = 'earlier'
    prefix = observed.line(earlier) + observed.line(call) + observed.line(result)
    main.write_bytes(prefix)
    duplicate.chmod(0)  # An unreferenced optional file cannot gate this main.
    initial = run.acquire(config)
    duplicate.chmod(0o600)
    source = initial['input']
    run.start()
    try:
        document = read_document(run)
        message, origin = assert_result(args, document, result, source, body)
        raw_body = run.command(['input', 'cat', origin['binding']['input']], raw=True)
        assert raw_body == MASKED.encode()
        raw_main = run.command(['input', 'cat', source], raw=True)
        assert raw_main == prefix
        cite = message['sources'][0]
        assert json.loads(raw_main[cite['start']:cite['end']]) == result
        remote = sdk_read(args, run, 'block:' + REF + ':result:0', 'sdk-initial')
        shown = remote['shown'][0]
        assert shown['text'] == MASKED and shown['version_hash'] == expected_version(args, result, PAYLOAD), remote
        assert origin in shown['field_origins']
    finally:
        run.stop()
    separator = b'\n \r\n\t\n'
    with main.open('ab') as stream:
        stream.write(separator)
    separated = run.acquire(config)
    assert separated['input'] == source and separated['revision'] == initial['revision'] + 1
    assert separated['progress']['parser']['parsed_upto'] == len(prefix + separator)
    assert separated['progress']['parser']['records']['framed'] == 3
    run.start()
    try:
        assert read_document(run)['messages'] == document['messages']
    finally:
        run.stop()
    suffix = observed.line(inline) + observed.line(observed.user('later', 'Later complete record.', 'inline', 3))
    with main.open('ab') as stream:
        stream.write(suffix)
    final = run.acquire(config)
    assert final['input'] == source and final['chain'] == 1
    run.start()
    try:
        final_doc = read_document(run)
        assert_result(args, final_doc, result, source, body)
        inline_message = next(m for m in final_doc['messages'] if m['id'] == REF + ':inline')
        assert inline_message['version_hash'] == expected_version(args, inline), inline_message
        assert inline_message['blocks'][0]['text'] == 'Complete inline result.'
    finally:
        run.stop()
    # Redundant inline replicas have no required-body dependency.
    duplicate_hash = digest(args, b'Complete inline result.')
    duplicate_blob = run.archive / 'blobs' / duplicate_hash[:2] / (duplicate_hash + '.zst')
    duplicate_saved = duplicate_blob.with_suffix('.saved')
    duplicate_blob.rename(duplicate_saved)
    run.start()
    try:
        assert read_document(run)['messages'] == final_doc['messages']
    finally:
        run.stop(); duplicate_saved.rename(duplicate_blob)
    # Same inputs and path, fresh archive: actual uninterrupted interpretation.
    archive = run.archive
    run.archive = run.output / 'baseline-archive'
    baseline = run.acquire(config)
    run.start()
    try:
        baseline_doc = read_document(run)
        normalize = lambda value, sid: json.loads(json.dumps(value).replace(sid, 'SOURCE'))
        left, right = copy.deepcopy(final_doc), copy.deepcopy(baseline_doc)
        left.pop('result', None); right.pop('result', None)
        # Companion input identities contain main identity; retain each document's
        # independently checked exact origins, then compare semantic Messages.
        for document in (left, right):
            for item in document['messages']:
                item.pop('field_origins', None)
        assert normalize(left, source) == normalize(right, baseline['input'])
    finally:
        run.stop(); run.archive = archive
    # A body-only change invalidates the pinned interpretation without editing
    # the main or disposable index. Its raw-record unit remains exactly equal.
    changed = PAYLOAD + b'\nChanged required body.'
    body.write_bytes(changed)
    changed_reply = run.acquire(config)
    assert changed_reply['revision'] == final['revision']
    run.start()
    try:
        changed_doc = read_document(run)
        changed_message, changed_origin = assert_result(args, changed_doc, result, source, body, MASKED + '\nChanged required body.', changed)
        assert changed_message['version_hash'] != message['version_hash']
    finally:
        run.stop()
    # Provider copies are unnecessary after acquisition and same-state restart.
    main.unlink(); body.unlink(); duplicate.unlink()
    run.start()
    try:
        retained_doc = read_document(run)
        assert retained_doc['messages'] == changed_doc['messages']
        remote = sdk_read(args, run, 'block:' + REF + ':result:0', 'sdk-retained')
        assert remote['shown'][0]['text'] == MASKED + '\nChanged required body.'
    finally:
        run.stop()
    # Actual lost and corrupt retained witnesses must defeat normalized copies.
    blob_hash = changed_origin['blob']
    blob = archive / 'blobs' / blob_hash[:2] / (blob_hash + '.zst')
    saved = blob.with_suffix('.saved')
    blob.rename(saved)
    run.start()
    try:
        missing = refused_document(run, blob_hash)
        remote_missing = sdk_read(args, run, 'block:' + REF + ':result:0', 'sdk-unavailable')
        assert remote_missing['outcome'] == 'error' and not remote_missing.get('shown'), remote_missing
        # A later required body cannot gate independently backed earlier Text.
        # Session context stays explicitly unavailable; selected result proof
        # above remains mandatory despite this narrower successful read.
        independent_reads = []
        for reference in ('message:' + REF + ':earlier', 'block:' + REF + ':earlier:0'):
            independent = run.command([reference, 'read', '--budget', '20000'], allowed=(0, 5))
            assert independent['shown'][0]['text'] == 'Independent earlier human text.', independent
            assert independent['outcome'] == 'degraded' and independent['settlement']['state'] == 'unavailable', independent
            assert independent['unknown']['unavailable_inputs'] > 0, independent
            independent_reads.append(reference)
        remote_independent = sdk_read(args, run, 'message:' + REF + ':earlier', 'sdk-independent')
        assert remote_independent['shown'][0]['text'] == 'Independent earlier human text.', remote_independent
        assert remote_independent['outcome'] == 'degraded' and remote_independent['settlement']['state'] == 'unavailable', remote_independent
    finally:
        run.stop(); saved.rename(blob)
    before = blob.read_bytes()
    blob.write_bytes(b'corrupt retained body')
    run.start()
    try:
        corrupt = refused_document(run, blob_hash, corrupt=True)
    finally:
        run.stop(); blob.write_bytes(before)
    for path in archive.rglob('*'):
        if path.is_file():
            assert not stat.S_IMODE(path.stat().st_mode) & 0o077, path
    return {'input': source, 'initial_version': message['version_hash'], 'changed_version': changed_message['version_hash'], 'body_origin': changed_origin, 'separator_records': 0, 'provider_loss': True, 'unavailable_retained': missing, 'corrupt_retained': corrupt, 'inline_duplicate_version_neutral': True, 'independent_earlier_reads_with_later_body_unavailable': independent_reads}


def missing_and_failed(args, output):
    output.mkdir()
    run = observed.Run(args.binary, output)
    try:
        main, body, duplicate, config, call, result, inline = fixture(output)
        main.write_bytes(observed.line(call) + observed.line(result))
        body.unlink()
        missing = run.acquire(config, allowed=(5,))
        assert missing['progress']['source_rows']['state'] != 'settled', missing
        body.write_bytes(PAYLOAD)
        body.chmod(0)
        try:
            failed_answer = run.acquire(config, allowed=(4,))
            command = run.calls[-1]
            diagnostic = (output / command['stderr']).read_text()
            assert failed_answer is None and 'Permission denied' in diagnostic, (failed_answer, diagnostic)
            failed = {'answer': failed_answer, 'exit': command['exit'], 'diagnostic': diagnostic, 'stdout': command['stdout'], 'stderr': command['stderr']}
        finally:
            body.chmod(0o600)
        recovered = run.acquire(config)
        assert recovered['progress']['source_rows']['state'] == 'settled', recovered
        return {'missing': missing, 'failed': failed, 'recovered': recovered}
    finally:
        errors = run.close()
        (output / 'commands.json').write_text(json.dumps(run.calls, indent=2) + '\n')
        assert not errors, errors


def containment_controls(args, output):
    output.mkdir()
    results = {}
    for component in ('session', 'tool-results', 'result.txt'):
        case = output / component
        case.mkdir()
        run = observed.Run(args.binary, case)
        try:
            main, body, duplicate, config, call, result, inline = fixture(case)
            main.write_bytes(observed.line(call) + observed.line(result))
            target = {'session': case / SID, 'tool-results': body.parent, 'result.txt': body}[component]
            outside = case / 'outside'
            target.rename(outside)
            target.symlink_to(outside, target_is_directory=component != 'result.txt')
            answer = run.acquire(config, allowed=(4,))
            command = run.calls[-1]
            diagnostic = (case / command['stderr']).read_text()
            assert answer is None and ('Not a directory' in diagnostic or 'Too many levels of symbolic links' in diagnostic), (component, answer, diagnostic)
            body_hash = digest(args, PAYLOAD)
            outside_blob_present = (run.archive / 'blobs' / body_hash[:2] / (body_hash + '.zst')).exists()
            assert not outside_blob_present, component
            results[component] = {'exit': command['exit'], 'diagnostic': diagnostic, 'outside_blob_present': outside_blob_present}
        finally:
            errors = run.close()
            (case / 'commands.json').write_text(json.dumps(run.calls, indent=2) + '\n')
            assert not errors, errors
    return results


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
    report = {'outcome': 'failed', 'binary': str(args.binary), 'binary_sha256': hashlib.sha256(args.binary.read_bytes()).hexdigest(), 'candidate_sha': args.binary_candidate, 'driver_candidate_sha': subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(), 'candidate_tree': subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', args.binary_candidate + '^{tree}'], text=True).strip(), 'sdk_python': str(args.sdk_python), 'sdk_python_sha256': hashlib.sha256(args.sdk_python.read_bytes()).hexdigest(), 'sdk_client': str(args.sdk_client), 'sdk_client_sha256': hashlib.sha256(args.sdk_client.read_bytes()).hexdigest(), 'expected_body_sha256': hashlib.sha256(PAYLOAD).hexdigest()}
    try:
        report['required_body'] = verify(args, run)
        report['failures'] = missing_and_failed(args, args.output / 'failure-controls')
        report['containment'] = containment_controls(args, args.output / 'containment-controls')
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
