#!/usr/bin/env python3
"""Synthetic configured Claude projects discovery through CLI and official SDK."""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import shutil
import stat
import subprocess
import time

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location('continuation', ROOT / 'scripts/check-claude-continuation.py')
continuation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(continuation)
observed = continuation.observed
bodies = continuation.bodies


class Run(observed.Run):
    def command(self, *arguments, **keywords):
        if shutil.disk_usage(self.output).free < 768 * 1024 * 1024:
            raise RuntimeError('synthetic qualification stopped at reserved free-space boundary')
        return super().command(*arguments, **keywords)


def record(native, message, text, parent=None, role='user'):
    if role == 'user':
        value = observed.user(message, text, parent)
    else:
        value = observed.assistant(message, text, parent)
    value['sessionId'] = native
    return value


def write(root, relative, records):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b''.join(observed.line(r) for r in records))
    return path


def state(run, predicate, timeout=240):
    return continuation.wait(run, predicate, timeout)


def project_status(value):
    return value['claude_continuation']['projects']


def await_progress(run, source, predicate, timeout=240):
    deadline = time.monotonic() + timeout
    while True:
        value = run.command(['progress', source])
        if predicate(value):
            return value
        assert time.monotonic() < deadline, value
        time.sleep(.1)


def await_text(run, reference, text, timeout=240):
    deadline = time.monotonic() + timeout
    while True:
        value = continuation.document(run, reference)
        if value and any(b.get('text') == text for m in value.get('messages', []) for b in m['blocks']):
            return value
        assert time.monotonic() < deadline, value
        time.sleep(.1)


def discover(run, query, text, timeout=240):
    deadline = time.monotonic() + timeout
    while True:
        answer = run.command([query, '--json'], allowed=(0, 5))
        for row in answer.get('rows', []):
            document = continuation.document(run, row['id'])
            if document and any(block.get('text') == text for m in document.get('messages', []) for block in m['blocks']):
                return row['id'], document
        assert time.monotonic() < deadline, answer
        time.sleep(.05)


def check_text(document, reference, native, message_id, text, role, author, run, expected, args):
    assert document['session']['id'] == reference, document
    messages = [m for m in document['messages'] if m['id'] == reference + ':' + message_id]
    assert len(messages) == 1, messages
    message = messages[0]
    assert message['role'] == role and message['written_by'] == author, message
    assert message['blocks'][0]['id'] == reference + ':' + message_id + ':0'
    assert any(b.get('text') == text for b in message['blocks']), message
    assert message['version_hash'] == bodies.expected_version(args, expected), message
    import datetime
    expected_at = datetime.datetime.fromisoformat(expected['timestamp'].replace('Z', '+00:00')).isoformat(timespec='milliseconds').replace('+00:00', 'Z')
    assert message['at'] == expected_at, message
    assert message.get('parent') == (reference + ':' + expected['parentUuid'] if expected.get('parentUuid') else None), message
    assert message['sources'] and message['field_origins'], message
    raw_sources = {}
    for cite in message['sources']:
        assert cite['chain'] == 1 and cite['preferred'] is True, cite
        raw = run.command(['input', 'cat', cite['input']], raw=True)
        raw_sources[cite['input']] = raw
        spans = []
        offset = 0
        for physical in raw.splitlines(keepends=True):
            if physical.strip() and json.loads(physical) == expected:
                spans.append({'start': offset, 'end': offset + len(physical.rstrip(b'\r\n'))})
            offset += len(physical)
        assert len(spans) == 1 and {'start': cite['start'], 'end': cite['end']} == spans[0], cite
        selected = json.loads(raw[cite['start']:cite['end']])
        assert selected == expected and selected['uuid'] == message_id and selected['sessionId'] == native, selected
        assert selected['message']['content'] == text or any(b.get('text') == text for b in selected['message']['content']), selected
    for entry in message['field_origins']:
        assert entry['origin']['binding'] == {'input': message['sources'][0]['input'], 'chain': 1}, entry
    text_origin = next(e['origin'] for e in message['field_origins'] if e['field'] == 'blocks[0].text')
    selection = text_origin['selection']
    assert selection['adapter_version'] == 2 and selection['codec_version'] == 1
    selected = [selection['input']] if 'input' in selection else selection['inputs']
    assert selected
    for selector in selected:
        span = selector['record']
        assert span in [{'start': c['start'], 'end': c['end']} for c in message['sources']], span
        selected_record = json.loads(raw_sources[text_origin['binding']['input']][span['start']:span['end']])
        assert selected_record == expected, selected_record
        value = expected
        for component in selector['path']:
            value = value[component.get('Key', component.get('Index'))]
        assert value == text
    return message


def check_mutants(args, document, reference, native, text, raw, expected, source):
    class FixtureRead:
        def command(self, arguments, raw=False):
            assert arguments == ['input', 'cat', source] and raw
            return fixture_bytes
    fixture_bytes = raw
    base = copy.deepcopy(document)
    mutants = {}
    def change(name, callback):
        value = copy.deepcopy(base)
        callback(value['messages'][0])
        mutants[name] = value
    change('wrong_time', lambda m: m.update(at='2026-11-01T08:00:00.000Z'))
    change('wrong_version', lambda m: m.update(version_hash='0' * 64))
    change('wrong_author', lambda m: m.update(written_by='model'))
    change('wrong_chain', lambda m: m['sources'][0].update(chain=2))
    change('wrong_span', lambda m: m['sources'][0].update(end=m['sources'][0]['end'] + 1))
    change('wrong_binding', lambda m: m['field_origins'][0]['origin']['binding'].update(input='foreign-source'))
    change('wrong_block_scope', lambda m: m['blocks'][0].update(id='foreign:block'))
    duplicate = copy.deepcopy(base)
    duplicate['messages'].append(copy.deepcopy(duplicate['messages'][0]))
    mutants['duplicate_message'] = duplicate
    for name, mutant in mutants.items():
        try:
            check_text(mutant, reference, native, 'first', text, 'user', 'human', FixtureRead(), expected, args)
        except (AssertionError, ValueError, KeyError):
            continue
        raise AssertionError('accepted damaging oracle mutant: ' + name)
    return list(mutants)


def export_document(run, reference, document, name):
    output = run.output / ('export-' + name)
    run.command([reference, 'export', '--format', 'json', '--to', str(output)], allowed=(0, 5))
    files = [p for p in output.rglob('*.json') if p.name != 'manifest.json']
    assert len(files) == 1, files
    exported = json.loads(files[0].read_text())
    assert exported['messages'] == document['messages'] and exported['session'] == document['session']
    assert exported.get('links') == document.get('links') and exported.get('events') == document.get('events')


def sdk_text(args, run, query, text, expected_reference, name):
    remote = continuation.sdk(args, run, 'search', {'query': query, 'budget_tokens': 20000}, name + '-search')
    references = [r['id'] for r in remote['rows']]
    assert references == [expected_reference], remote
    answer = continuation.sdk(args, run, 'read', {'ref': references[0], 'budget_tokens': 20000, 'view': 'transcript', 'tools': 'full', 'thinking': 'full', 'harness': 'full', 'branches': True}, name + '-read')
    shown = [s for s in answer['shown'] if s.get('text') == text]
    assert len(shown) == 1 and shown[0]['sources'] and shown[0]['field_origins'], answer
    return shown[0]


def body_result(args, document, reference, result, source, path, text, payload, message_id="result", call_id="call"):
    message = next(m for m in document['messages'] if m['id'] == reference + ':' + message_id)
    assert message['role'] == 'user' and message['written_by'] == 'harness', message
    assert message['version_hash'] == bodies.expected_version(args, result, payload), message
    block = message['blocks'][0]
    assert block['id'] == reference + ':' + message_id + ':0' and block['kind'] == 'tool_result'
    assert block['text'] == text and block['call_id'] == call_id and block['outcome'] == 'ok', block
    origin = next(e['origin'] for e in message['field_origins'] if e['field'] == 'blocks[0].text')
    assert origin['format'] == 'claude-code-tool-result-body' and origin['binding']['chain'] == 1
    assert origin['path'] == str(path) and origin['start'] == 0 and origin['end'] == len(payload)
    assert origin['blob'] == bodies.digest(args, payload)
    assert message['sources'] and all(c['input'] == source for c in message['sources'])
    assert bodies.SECRET not in json.dumps(document)
    return message, origin


def configured(run, config):
    return continuation.start_configured(run, config)


def verify(args, run):
    root = run.output / 'projects'
    root.mkdir()
    main_id = 'discovery-main'
    initial_text = 'Aster anchor conversation, independently retained.'
    main = write(root, '-repo-one/' + main_id + '.jsonl', [record(main_id, 'first', initial_text), record(main_id, 'answer', 'Aster model answer.', 'first', 'assistant')])
    second_text = 'Birch second project, separate conversation.'
    write(root, '-repo-two/second.ndjson', [record('second', 'u', second_text)])
    agent_text = 'Cedar separate agent conversation.'
    agent_body = root / '-repo-one' / main_id / 'tool-results' / 'agent-call.txt'
    agent_body.parent.mkdir(parents=True)
    agent_payload = b'Agent output from the shared parent directory.\n'
    agent_body.write_bytes(agent_payload)
    agent_call = {'type': 'assistant', 'uuid': 'agent-call', 'sessionId': 'actual-agent-owner', 'parentUuid': 'au', 'requestId': 'agent-request', 'message': {'id': 'agent-response', 'role': 'assistant', 'model': 'synthetic-model', 'content': [{'type': 'tool_use', 'id': 'agent-call', 'name': 'exec_command', 'input': {'cmd': 'synthetic agent fixture'}}]}}
    agent_result = {'type': 'user', 'uuid': 'agent-result', 'sessionId': 'actual-agent-owner', 'parentUuid': 'agent-call', 'toolUseResult': {'persistedOutputPath': str(agent_body)}, 'message': {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 'agent-call', 'content': 'Preview; the agent body is required.'}]}}
    write(root, '-repo-one/' + main_id + '/subagents/agent-a.jsonl', [record('actual-agent-owner', 'au', agent_text), agent_call, agent_result])
    workflow_text = 'Dahlia workflow agent conversation.'
    write(root, '-repo-two/folder-owner/subagents/workflows/run-one/agent-w.jsonl', [record('workflow-native-owner', 'wu', workflow_text)])
    excluded_text = 'EXCLUDED nested copy must never become a conversation.'
    for relative in ('.claude/worktrees/-repo/copy.jsonl', '-repo-one/.claude/worktrees/copy.jsonl', '-repo-one/analysis/copy.jsonl', '-repo-one/' + main_id + '/subagents/nested/agent-x.jsonl'):
        write(root, relative, [record('excluded', 'x', excluded_text)])
    write(root, '-repo-one/' + main_id + '/subagents/workflows/run/journal.jsonl', [{'type': 'workflow-journal', 'status': 'synthetic'}])
    (root / '-repo-one/sessions-index.json').write_text('{"entries":[]}')
    config = run.output / 'source.json'
    config.write_text(json.dumps({'claude_projects': {'root': str(root), 'machine': 'synthetic-discovery', 'host': 'synthetic-host', 'acquisition_quantum_bytes': 65536}}))
    configured(run, config)
    pid = run.process.pid
    reference, document = discover(run, '"Aster anchor"', initial_text)
    assert reference == 'claude-code:' + main_id, reference
    first = check_text(document, reference, main_id, 'first', initial_text, 'user', 'human', run, record(main_id, 'first', initial_text), args)
    check_text(document, reference, main_id, 'answer', 'Aster model answer.', 'assistant', 'model', run, record(main_id, 'answer', 'Aster model answer.', 'first', 'assistant'), args)
    oracle_controls = check_mutants(args, document, reference, main_id, initial_text, main.read_bytes(), record(main_id, 'first', initial_text), first['sources'][0]['input'])
    references = {'main': reference}
    export_document(run, reference, document, 'main')
    for label, query, text, native, message in [('second', '"Birch second"', second_text, 'second', 'u'), ('agent', 'kind:* "Cedar separate"', agent_text, 'actual-agent-owner', 'au'), ('workflow', 'kind:* "Dahlia workflow"', workflow_text, 'workflow-native-owner', 'wu')]:
        found, doc = discover(run, query, text)
        references[label] = found
        expected_native = native if label == 'second' else native + ('/a' if label == 'agent' else '/w')
        assert found == 'claude-code:' + expected_native, found
        check_text(doc, found, native, message, text, 'user', 'human', run, record(native, message, text), args)
        export_document(run, found, doc, label)
        sdk_text(args, run, query, text, found, 'sdk-' + label)
        if label == 'agent':
            agent_source = next(m for m in doc['messages'] if m['id'] == found + ':au')['sources'][0]['input']
            agent_message, agent_origin = body_result(args, doc, found, agent_result, agent_source, agent_body, agent_payload.decode(), agent_payload, 'agent-result', 'agent-call')
            remote_agent = bodies.sdk_read(args, run, 'block:' + found + ':agent-result:0', 'sdk-agent-shared-body')
            assert remote_agent['shown'][0]['text'] == agent_payload.decode()
            assert remote_agent['shown'][0]['version_hash'] == agent_message['version_hash']
            continuation.assert_sdk_origin(remote_agent['shown'][0], 'blocks[0].text', agent_origin)
    assert len(set(references.values())) == 4, references
    assert not run.command(['"EXCLUDED nested"', '--json'], allowed=(0, 5)).get('rows')
    sdk_text(args, run, '"Aster anchor"', initial_text, reference, 'sdk-main')
    # Runnable directory backlog crosses many scan turns without a new event.
    for i in range(5000):
        name = 'bulk-' + str(i).zfill(5)
        write(root, f'-bulk-{i % 7}/{name}.jsonl', [record(name, 'u', f'Neutral bulk witness {i:05}.')])
    began = time.monotonic()
    before = state(run, lambda v: project_status(v)['scan'] is not None and project_status(v)['scan']['observed_transcripts'] >= 100)
    assert not project_status(before)['scan']['cycle_complete'], before
    during = continuation.document(run, reference)
    assert during['messages'] == document['messages'], during
    bulk = state(run, lambda v: project_status(v)['initial_cycle_finished'] and project_status(v)['scan'] is not None and project_status(v)['scan']['cycle_complete'] and project_status(v)['scan']['observed_transcripts'] == 5004 and project_status(v)['acquired_source_attempts'] >= 5004, timeout=420)
    bulk_elapsed = time.monotonic() - began
    for i in (0, 127, 1024, 4999):
        discover(run, f'"Neutral bulk witness {i:05}"', f'Neutral bulk witness {i:05}.')
    # The default budget is exercised after the full population exists, using
    # fresh SDK discovery and its returned reference rather than a supplied ID.
    default_search = continuation.sdk(args, run, 'search', {'query': '"Aster anchor"'}, 'sdk-default-population-search')
    assert [r['id'] for r in default_search['rows']] == [reference], default_search
    default_read = continuation.sdk(args, run, 'read', {'ref': default_search['rows'][0]['id'], 'view': 'transcript', 'tools': 'full', 'thinking': 'full', 'harness': 'full'}, 'sdk-default-population-read')
    default_first = next(s for s in default_read['shown'] if s.get('text') == initial_text)
    assert default_first['message'] == 'message:' + reference + ':first'
    assert default_first['version_hash'] == bodies.expected_version(args, record(main_id, 'first', initial_text))
    assert default_first['sources'] == first['sources']
    text_origin = next(e['origin'] for e in first['field_origins'] if e['field'] == 'blocks[0].text')
    continuation.assert_sdk_origin(default_first, 'blocks[0].text', text_origin)
    # No user-supplied main path or Session ID is added for the late project.
    late_text = 'Elm late project arrives in the same daemon.'
    write(root, '-late/late.jsonl', [record('late', 'u', late_text)])
    late_ref, _ = discover(run, '"Elm late project"', late_text)
    assert late_ref == 'claude-code:late', late_ref
    sdk_text(args, run, '"Elm late project"', late_text, late_ref, 'sdk-late')
    # An incomplete suffix cannot publish its absent message, then completes.
    correction_text = 'Fir later complete correction.'
    correction = observed.line(record(main_id, 'correction', correction_text, 'answer'))
    with main.open('ab') as stream:
        stream.write(correction[:len(correction)//2])
    source = first['sources'][0]['input']
    await_progress(run, source, lambda p: p.get('parser', {}).get('status') == 'waiting_tail')
    assert all(m['id'] != reference + ':correction' for m in continuation.document(run, reference)['messages'])
    with main.open('ab') as stream:
        stream.write(correction[len(correction)//2:])
    changed = await_text(run, reference, correction_text)
    check_text(changed, reference, main_id, 'correction', correction_text, 'user', 'human', run, record(main_id, 'correction', correction_text, 'answer'), args)
    sdk_text(args, run, '"Fir later complete"', correction_text, reference, 'sdk-correction')
    # Required companions are resolved in the parent's session scope, including
    # a separate agent; unreferenced duplicates never become conversation rows.
    body = root / '-repo-one' / main_id / 'tool-results' / 'call.txt'
    body.parent.mkdir(parents=True, exist_ok=True)
    call = {'type': 'assistant', 'uuid': 'call', 'sessionId': main_id, 'parentUuid': 'correction', 'requestId': 'request-body', 'message': {'id': 'response-body', 'role': 'assistant', 'model': 'synthetic-model', 'content': [{'type': 'tool_use', 'id': 'call', 'name': 'exec_command', 'input': {'cmd': 'synthetic fixture'}}]}}
    result = {'type': 'user', 'uuid': 'result', 'sessionId': main_id, 'parentUuid': 'call', 'toolUseResult': {'persistedOutputPath': str(body)}, 'message': {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 'call', 'content': 'Preview of retained required output.'}]}}
    with main.open('ab') as stream:
        stream.write(observed.line(call) + observed.line(result))
    await_progress(run, source, lambda p: p.get('parser', {}).get('acquired_upto') == main.stat().st_size and p['active'] is False)
    assert run.command(['message:' + reference + ':first', 'read', '--budget', '20000'], allowed=(0, 5))['shown'][0]['text'] == initial_text
    body.write_bytes(bodies.PAYLOAD)
    full = await_text(run, reference, bodies.MASKED)
    message, first_origin = body_result(args, full, reference, result, source, body, bodies.MASKED, bodies.PAYLOAD)
    body.write_bytes(bodies.PAYLOAD + b'\nLater companion-only correction.')
    revised_text = bodies.MASKED + '\nLater companion-only correction.'
    changed = await_text(run, reference, revised_text)
    revised, revised_origin = body_result(args, changed, reference, result, source, body, revised_text, body.read_bytes())
    assert message['version_hash'] != revised['version_hash']
    remote = bodies.sdk_read(args, run, 'block:' + reference + ':result:0', 'sdk-body-correction')
    assert remote['shown'][0]['text'] == revised_text, remote
    continuation.assert_sdk_origin(remote['shown'][0], 'blocks[0].text', revised_origin)
    # The late large source uses many grants, while ordinary reads retain the
    # earlier acknowledged answer. This measures this fixture, not a rate bound.
    large_text = 'Ginkgo large transcript final answer.'
    large = root / '-large/large.jsonl'
    large.parent.mkdir()
    progress = observed.line({'type': 'progress', 'data': {'text': 'neutral progress padding ' * 12}})
    large.write_bytes(progress * 30000 + observed.line(record('large', 'u', large_text)))
    assert large.stat().st_size > 8 * 1024 * 1024
    pending = state(run, lambda v: v['claude_continuation']['state'] == 'acquiring' and project_status(v)['active_path'] == str(large), timeout=240)
    during_large = continuation.document(run, reference)
    assert during_large['messages'] == changed['messages']
    large_ref, _ = discover(run, '"Ginkgo large transcript"', large_text)
    # A whole quiet cycle adds neither a source revision nor canonical writes.
    def quiet_cycle(value):
        p = project_status(value)
        return p['scan'] is not None and p['scan']['cycle_complete'] and p['queued_candidates'] == 0 and p['active_path'] is None
    stable = state(run, quiet_cycle)
    deadline = time.monotonic() + 420
    while True:
        cycle = project_status(stable)['scan']['cycle']
        quiet = state(run, lambda v: quiet_cycle(v) and project_status(v)['scan']['cycle'] > cycle, timeout=420)
        if quiet['foundation']['write_counter'] == stable['foundation']['write_counter']:
            break
        assert time.monotonic() < deadline, (stable, quiet)
        stable = quiet
    original_mode = stat.S_IMODE(root.stat().st_mode)
    try:
        root.chmod(0)
        denied = state(run, lambda v: project_status(v)['availability'] == 'denied')
        assert continuation.document(run, reference)['messages'] == changed['messages']
    finally:
        root.chmod(original_mode)
    state(run, lambda v: project_status(v)['availability'] not in ('denied', 'unavailable', 'missing'))
    absent = run.output / 'temporarily-absent-projects'
    root.rename(absent)
    try:
        missing = state(run, lambda v: project_status(v)['availability'] == 'missing')
        assert continuation.document(run, reference)['messages'] == changed['messages']
    finally:
        absent.rename(root)
    state(run, lambda v: project_status(v)['availability'] not in ('denied', 'unavailable', 'missing'))
    assert run.process.pid == pid
    run.process.send_signal(signal.SIGINT)
    assert run.process.wait(timeout=20) == 0
    run.process = None
    saved = run.output / 'removed-provider'
    root.rename(saved)
    run.start()
    try:
        retained = await_text(run, reference, correction_text)
        assert retained['messages'] == changed['messages'], retained
        sdk_text(args, run, '"Fir later complete"', correction_text, reference, 'sdk-provider-free')
    finally:
        run.stop()
    return {'oracle_controls': oracle_controls, 'denied': project_status(denied), 'missing': project_status(missing), 'large_input_bytes': large.stat().st_size if large.exists() else (saved / '-large/large.jsonl').stat().st_size, 'large_read_service': project_status(pending), 'references': references, 'late_reference': late_ref, 'bulk_source_files': 5000, 'bulk_acquisition_elapsed_seconds': bulk_elapsed, 'initial_document_source': source, 'bulk_status': project_status(bulk), 'ordinary_read_during_backlog': project_status(before), 'pid': pid}


def dimensions(root):
    sizes = [p.stat().st_size for p in root.rglob('*') if p.is_file()]
    return {'files': len(sizes), 'bytes': sum(sizes)}


def migration(args):
    output = args.output / 'v3-preservation'
    output.mkdir()
    run = Run(args.v3_binary, output)
    run.sdk_calls = []
    main = output / 'old-main.jsonl'
    text = 'Hazel preserving version-three citation.'
    raw = observed.line(record('old-main', 'u', text))
    progress = observed.line({'type': 'progress', 'data': {'text': 'neutral progress padding ' * 12}})
    main.write_bytes(raw + progress * 12000)
    config = output / 'source.json'
    config.write_text(json.dumps({'claude_main': {'path': str(main), 'machine': 'synthetic-v3', 'host': 'synthetic-host', 'acquisition_quantum_bytes': 65536}}))
    process = None
    try:
        acquired = run.acquire(config)
        run.start()
        reference, before = discover(run, '"Hazel preserving"', text)
        assert reference == 'claude-code:old-main', reference
        run.stop()
        request = '019a1100-0000-7000-8000-000000000001'
        command = [str(args.v3_binary), '--archive', str(run.archive), 'daemon', '--prepare-copy', 'restore', '--request-id', request, '--retired-writers']
        process = subprocess.Popen(command, env=run.env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        first = process.stdout.readline()
        (output / 'interrupted.stdout').write_bytes(first)
        assert json.loads(first)['outcome'] == 'pending', first
        process.send_signal(signal.SIGINT)
        rest, error = process.communicate(timeout=30)
        assert process.returncode != 5 and b'cancelled' in error, (first, rest, error)
        (output / 'interrupted.stdout').write_bytes(first + rest)
        (output / 'interrupted.stderr').write_bytes(error)
        def snapshot():
            return {str(p.relative_to(run.archive)): hashlib.sha256(p.read_bytes()).hexdigest() for p in run.archive.rglob('*') if p.is_file() and 'migrations' not in p.relative_to(run.archive).parts and not p.name.endswith('-shm')}
        frozen = snapshot()
        refusal = subprocess.run([str(args.binary), '--archive', str(run.archive), 'daemon'], env=run.env, capture_output=True, timeout=15)
        (output / 'new-refusal.stdout').write_bytes(refusal.stdout)
        (output / 'new-refusal.stderr').write_bytes(refusal.stderr)
        assert refusal.returncode != 0 and b'pending action' in refusal.stderr, refusal.stderr
        after_refusal = snapshot()
        assert all(after_refusal.get(path) == digest for path, digest in frozen.items()), 'new writer changed pending v3 authority/custody'
        for path in after_refusal.keys() - frozen.keys():
            created = run.archive / path
            assert path == 'archive.db-wal' and not created.is_symlink() and created.is_file() and created.stat().st_size == 0, path
        complete = subprocess.run(command, env=run.env, capture_output=True, timeout=60)
        (output / 'v3-complete.stdout').write_bytes(complete.stdout)
        (output / 'v3-complete.stderr').write_bytes(complete.stderr)
        assert complete.returncode == 5, (complete.stdout, complete.stderr)
        run.binary = str(args.binary)
        main.unlink()
        run.start()
        after = await_text(run, reference, text)
        assert after['messages'] == before['messages'], (before, after)
        sdk_text(args, run, '"Hazel preserving"', text, reference, 'sdk-v3-preserved')
        assert run.command(['input', 'cat', acquired['input']], raw=True) == raw + progress * 12000
        run.stop()
        unaware = subprocess.run([str(args.v3_binary), '--archive', str(run.archive), 'daemon'], env=run.env, capture_output=True, timeout=15)
        (output / 'unaware.stdout').write_bytes(unaware.stdout)
        (output / 'unaware.stderr').write_bytes(unaware.stderr)
        assert unaware.returncode != 0 and b'newer than supported' in unaware.stderr, unaware.stderr
        return {'reference': reference, 'input': acquired['input'], 'pending_refusal_exit': refusal.returncode, 'pinned_v3_binary': str(args.v3_binary), 'pinned_v3_sha256': hashlib.sha256(args.v3_binary.read_bytes()).hexdigest(), 'pinned_v3_source': args.v3_source, 'v3_finish_exit': complete.returncode}
    finally:
        if process is not None:
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=15)
            process.wait(timeout=15)
        errors = run.close()
        (output / 'commands.json').write_text(json.dumps(run.calls, indent=2) + '\n')
        (output / 'sdk-calls.json').write_text(json.dumps(run.sdk_calls, indent=2) + '\n')
        assert not errors, errors


def empty_control(args):
    output = args.output / 'empty-scope'
    output.mkdir()
    root = output / 'projects'
    root.mkdir()
    config = output / 'source.json'
    config.write_text(json.dumps({'claude_projects': {'root': str(root), 'machine': 'synthetic-empty-scope', 'host': 'synthetic-host', 'acquisition_quantum_bytes': 65536}}))
    run = Run(args.binary, output)
    try:
        configured(run, config)
        value = state(run, lambda v: project_status(v)['availability'] == 'accessible_empty')
        assert project_status(value)['scan']['cycle_complete']
        assert project_status(value)['scan']['observed_transcripts'] == 0
        assert project_status(value)['acquired_source_attempts'] == 0
        assert project_status(value)['scan']['conversation_coverage'] is None
        assert run.command(['origin:claude-code', '--json'], allowed=(0, 5))['rows'] == []
        return project_status(value)
    finally:
        errors = run.close()
        (output / 'commands.json').write_text(json.dumps(run.calls, indent=2) + '\n')
        assert not errors, errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('binary', 'output', 'sdk-python', 'sdk-client', 'b3sum', 'v3-binary'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--binary-candidate', required=True)
    parser.add_argument('--v3-source', required=True)
    args = parser.parse_args()
    args.binary = args.binary.resolve(strict=True)
    args.output.mkdir(parents=True, exist_ok=False)
    free = shutil.disk_usage(args.output).free
    assert free > 1536 * 1024 * 1024, 'retain Realm free-space boundary'
    # Freeze the already reviewed official SDK adapter and its local imports.
    spec = importlib.util.spec_from_file_location('restore', ROOT / 'scripts/check-local-cold-restore.py')
    restore = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(restore)
    sdk_binding = restore.freeze_sdk(args)
    args.sdk_client = args.frozen_sdk_client
    run_output = args.output / 'discovery'
    run_output.mkdir()
    run = Run(args.binary, run_output)
    run.sdk_calls = []
    report = {'outcome': 'failed', 'binary': str(args.binary), 'binary_sha256': hashlib.sha256(args.binary.read_bytes()).hexdigest(), 'candidate': args.binary_candidate, 'driver_candidate': subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(), 'driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'sdk': sdk_binding, 'free_bytes_at_start': free}
    try:
        report['discovery'] = verify(args, run)
        report['dimensions'] = dimensions(run_output)
        report['preserving_upgrade'] = migration(args)
        report['accessible_empty_scope'] = empty_control(args)
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
