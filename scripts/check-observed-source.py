#!/usr/bin/env python3
"""Independent single-location append and retained-byte client demonstration."""
import copy
import argparse, hashlib, json, os, select, signal, subprocess, tempfile, time
from pathlib import Path
SID = '7f3a2c10-0001-4000-8000-000000000001'
REF = 'claude-code:' + SID

EXPECTED_MESSAGES = [('first', 'user', 'human', 'Initial neutral prompt.'), ('answer', 'assistant', 'model', 'Retained answer 東京🙂.'), ('later', 'user', 'human', 'Later complete prompt.'), ('split', 'user', 'human', 'Split record crosses retained extents.')]

def check_document(document, source, raw):
    assert document['session']['id'] == REF
    assert document['session']['title'] == 'Observed neutral source'
    assert document['session']['models'] == {'synthetic-model': 1}
    assert document['session']['started_at'] == '2026-10-01T08:00:00.000Z'
    assert document['session']['last_activity_at'] == '2026-10-01T08:03:00.000Z'
    spans = {}
    offset = 0
    for physical in raw.splitlines(keepends=True):
        if physical.strip():
            record = json.loads(physical)
            if 'uuid' in record:
                spans[record['uuid']] = {'start': offset, 'end': offset + len(physical.rstrip(b'\r\n'))}
        offset += len(physical)
    messages = document['messages']
    assert len(messages) == 4
    assert len(set(m['id'] for m in messages)) == 4
    for index, (message, (native, role, author, text)) in enumerate(zip(messages, EXPECTED_MESSAGES)):
        assert message['id'] == REF + ':' + native
        assert message['role'] == role and message['written_by'] == author
        assert message['blocks'] == [{'id': REF + ':' + native + ':0', 'kind': 'text', 'text': text}]
        assert message['at'] == f'2026-10-01T08:{index:02}:00.000Z'
        fields = {'at', 'blocks[0].text'} | ({'prompt_kind'} if role == 'user' else {'model'}) | ({'parent'} if index else set())
        assert len(message['field_origins']) == len(fields)
        assert {entry['field'] for entry in message['field_origins']} == fields
        assert message['sources'] == [{'input': source, 'chain': 1, **spans[native], 'preferred': True}]

        for citation in message['sources']:
            assert citation['input'] == source and citation['chain'] == 1
            record = json.loads(raw[citation['start']:citation['end']])
            assert record['uuid'] == native and record['sessionId'] == SID
        for entry in message['field_origins']:
            origin = entry['origin']
            assert origin['binding'] == {'input': source, 'chain': 1}
            assert origin['format'] == 'claude-code-field-origin'
            selection = origin['selection']
            assert selection['adapter_version'] == selection['codec_version'] == 1
            field = entry['field']
            path = {'at': [{'Key': 'timestamp'}], 'parent': [{'Key': 'parentUuid'}], 'model': [{'Key': 'message'}, {'Key': 'model'}], 'prompt_kind': [], 'blocks[0].text': [{'Key': 'message'}, {'Key': 'content'}] + ([{'Index': 0}, {'Key': 'text'}] if role == 'assistant' else [])}[field]

            selected = selection.get('inputs', [])
            if 'input' in selection:
                selected = [selection['input']]
            assert selected
            for item in selected:
                assert item['path'] == path
                record_range = item['record']
                assert record_range == spans[native]
                record = json.loads(raw[record_range['start']:record_range['end']])
                assert record['uuid'] == native and record['sessionId'] == SID
                value = record
                for component in item['path']:
                    value = value[component.get('Key', component.get('Index'))]
                if entry['field'] == 'blocks[0].text':
                    assert value == text

def check_mutants(document, source, raw):
    mutants = {}
    base = document
    wrong = copy.deepcopy(base)
    wrong['messages'][0]['id'] = 'claude-code:foreign:first'
    mutants['foreign_session_same_native_suffix'] = wrong
    wrong = copy.deepcopy(base)
    wrong['messages'][0]['field_origins'][0]['origin']['binding']['input'] = 'wrong-source'
    mutants['wrong_origin_binding'] = wrong
    wrong = copy.deepcopy(base)
    wrong['messages'][0]['sources'][0]['end'] += 20
    mutants['wrong_source_span'] = wrong
    wrong = copy.deepcopy(base)
    wrong['messages'].append(copy.deepcopy(wrong['messages'][0]))
    mutants['duplicate_live_message'] = wrong
    for name, mutant in mutants.items():
        try:
            check_document(mutant, source, raw)
        except (AssertionError, ValueError, KeyError):
            continue
        raise AssertionError('accepted mutant: ' + name)
    return list(mutants)

def line(value):
    return (json.dumps(value, ensure_ascii=False, separators=(',', ':')) + '\n').encode()

def user(native, text, parent=None, minute=0):
    return {'type': 'user', 'uuid': native, 'sessionId': SID, 'parentUuid': parent, 'isSidechain': False, 'promptSource': 'typed', 'timestamp': f'2026-10-01T08:{minute:02}:00.000Z', 'message': {'role': 'user', 'content': text}}

def assistant(native, text, parent):
    return {'type': 'assistant', 'uuid': native, 'sessionId': SID, 'parentUuid': parent, 'isSidechain': False, 'requestId': 'request-one', 'timestamp': '2026-10-01T08:01:00.000Z', 'message': {'id': 'response-one', 'role': 'assistant', 'model': 'synthetic-model', 'content': [{'type': 'text', 'text': text}]}}

class Run:

    def __init__(self, binary, output):
        self.binary = str(binary)
        self.output = output
        self.calls = []
        self.archive = output / 'archive'
        self.process = None
        (output / 'tmp').mkdir()
        self.runtime = tempfile.TemporaryDirectory(prefix='observed-socket-', dir='/tmp')
        self.env = dict(os.environ, TMPDIR=str(output / 'tmp'), XDG_RUNTIME_DIR=self.runtime.name)
        self.log = (output / 'daemon.log').open('ab')

    def command(self, args, allowed=(0,), raw=False):
        n = len(self.calls)
        record = {'argv': [self.binary, '--archive', str(self.archive), *map(str, args)], 'stdout': f'{n:04}.stdout', 'stderr': f'{n:04}.stderr', 'status': 'attempted', 'exit': None}
        self.calls.append(record)
        try:
            p = subprocess.run(record['argv'], env=self.env, capture_output=True, timeout=150)
        except subprocess.TimeoutExpired as e:
            (self.output / record['stdout']).write_bytes(e.stdout or b'')
            (self.output / record['stderr']).write_bytes(e.stderr or b'')
            record['status'] = 'timeout'
            raise
        (self.output / record['stdout']).write_bytes(p.stdout)
        (self.output / record['stderr']).write_bytes(p.stderr)
        record.update(status='completed', exit=p.returncode)
        assert p.returncode in allowed, (record, p.stdout.decode(errors='replace'), p.stderr.decode(errors='replace'))
        return p.stdout if raw else json.loads(p.stdout) if p.stdout else None

    def acquire(self, config, allowed=(0,)):
        return self.command(['daemon', '--config', config, '--until-idle', '--json'], allowed)

    def start(self):
        assert self.process is None
        self.process = subprocess.Popen([self.binary, '--archive', str(self.archive), 'daemon'], env=self.env, stdout=self.log, stderr=self.log)
        end = time.monotonic() + 15
        while time.monotonic() < end:
            p = self.command(['status'], allowed=(0, 4, 6, 7))
            if p and p.get('outcome') != 'error':
                return
            assert self.process.poll() is None, 'daemon exited'
            time.sleep(0.02)
        raise AssertionError('daemon startup timed out')

    def stop(self):
        if self.process:
            if self.process.poll() is None:
                self.process.send_signal(signal.SIGINT)
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=15)
                raise
            self.process = None

    def inspect(self, source, expected, read=True):
        self.start()
        try:
            actual = self.command(['input', 'cat', source], raw=True)
            assert actual == expected, (len(actual), len(expected))
            progress = self.command(['progress', source])
            assert progress['parser']['acquired_upto'] == len(expected)
            doc = self.command([REF, 'read', '--format', 'json'], allowed=(0, 5, 3, 4)) if read else None
            return (progress, doc)
        finally:
            self.stop()

    def close(self):
        errors = []
        for action in (self.stop, self.log.close, self.runtime.cleanup):
            try:
                action()
            except Exception as e:
                errors.append(repr(e))
        return errors

def verify(run):
    path = run.output / (SID + '.jsonl')
    config = run.output / 'source.json'
    config.write_text(json.dumps({'claude_main': {'path': str(path), 'machine': 'synthetic-machine', 'host': 'synthetic-host'}}))
    title = line({'type': 'ai-title', 'aiTitle': 'Observed neutral source', 'sessionId': SID})
    first = line(user('first', 'Initial neutral prompt.'))
    answer = line(assistant('answer', 'Retained answer 東京🙂.', 'first'))
    later = line(user('later', 'Later complete prompt.', 'answer', 2))
    tail = line(user('split', 'Split record crosses retained extents.', 'later', 3))
    initial = title + first + answer
    path.write_bytes(initial)
    base = run.acquire(config)
    source = base['input']
    assert base['acquisition'] == 'observed-claude-location-v1'
    p, doc = run.inspect(source, initial)
    assert p['source_rows']['scope'] == 'observed_source_rows'
    assert 'original_hash' not in p['source_rows']['basis']
    assert p['parser']['records']['framed'] == 3
    original_document = doc
    separator = b'\n \r\n\t\n'
    with path.open('ab') as f:
        f.write(separator)
    sep = run.acquire(config)
    assert sep['input'] == source and sep['chain'] == base['chain']
    assert sep['revision'] == base['revision'] + 1
    p, sepdoc = run.inspect(source, initial + separator)
    assert p['parser']['parsed_upto'] == len(initial + separator)
    assert p['parser']['records']['framed'] == 3
    assert sepdoc['messages'] == original_document['messages']
    repeated = run.acquire(config)
    assert repeated['input'] == source and repeated['revision'] == sep['revision']
    assert repeated['progress']['parser']['step'] == sep['progress']['parser']['step']
    with path.open('ab') as f:
        f.write(later + tail[:23])
    partial = run.acquire(config, allowed=(5,))
    assert partial['input'] == source and partial['progress']['parser']['status'] == 'waiting_tail'
    p, partialdoc = run.inspect(source, initial + separator + later + tail[:23])
    assert p['parser']['parsed_upto'] == len(initial + separator + later)
    assert p['parser']['inspected_upto'] == p['length']
    assert p['parser']['records']['framed'] == 4
    assert len(partialdoc['messages']) == 3 and partialdoc['messages'][-1]['id'] == REF + ':later', partialdoc
    repeated_partial = run.acquire(config, allowed=(5,))
    assert repeated_partial['revision'] == partial['revision']
    assert repeated_partial['progress']['parser']['step'] == partial['progress']['parser']['step']
    with path.open('ab') as f:
        f.write(tail[23:])
    final = run.acquire(config)
    assert final['input'] == source and final['chain'] == 1
    expected = initial + separator + later + tail
    p, finaldoc = run.inspect(source, expected)
    assert p['parser']['records']['framed'] == 5
    assert p['parser']['records']['undecodable'] == 0
    assert p['source_rows']['state'] == 'settled'
    check_document(finaldoc, source, expected)
    mutants = check_mutants(finaldoc, source, expected)
    incremental_archive = run.archive
    run.archive = run.output / 'baseline-archive'
    baseline = run.acquire(config)
    _, baseline_doc = run.inspect(baseline['input'], expected)

    check_document(baseline_doc, baseline['input'], expected)

    def normalized(document, input_id):
        document = json.loads(json.dumps(document).replace(input_id, 'SOURCE'))
        document.pop('result', None)
        return document
    assert normalized(finaldoc, source) == normalized(baseline_doc, baseline['input'])
    run.archive = incremental_archive
    with path.open('r+b') as f:
        f.seek(len(title) + 5)
        old = f.read(1)
        f.seek(-1, 1)
        f.write(b'X' if old != b'X' else b'Y')
    run.acquire(config, allowed=(3,))
    p, after = run.inspect(source, expected)
    check_document(after, source, expected)
    assert after['messages'] == finaldoc['messages']
    path.unlink()
    p, lost = run.inspect(source, expected)
    check_document(lost, source, expected)
    assert lost['messages'] == finaldoc['messages']
    assert p['source_rows']['state'] == 'settled'
    run.archive = run.output / 'interrupted-archive'
    interrupted = initial + line(user('large', 'Long retained record ' + 'x' * 200000, 'answer', 4))
    path.write_bytes(interrupted)
    argv = [run.binary, '--archive', str(run.archive), 'daemon', '--config', str(config), '--json']
    run.process = subprocess.Popen(argv, env=run.env, stdout=subprocess.PIPE, stderr=run.log)
    ready, _, _ = select.select([run.process.stdout], [], [], 30)
    assert ready, 'missing raw-retention admission'
    receipt = json.loads(run.process.stdout.readline())
    (run.output / 'interrupted-admission.json').write_text(json.dumps(receipt, indent=2))
    assert receipt['scope'] == 'retention_only' and receipt['retained']['length'] == len(interrupted)
    pending = run.command(['progress', receipt['retained']['id']])
    assert pending['source_rows']['state'] == 'pending', pending
    run.process.kill()
    code = run.process.wait(timeout=15)
    assert code == -signal.SIGKILL
    (run.output / 'interrupted-process.json').write_text(json.dumps({'argv': argv, 'exit': code, 'observed_progress': pending}, indent=2))
    run.process.stdout.close()
    run.process = None
    path.unlink()
    run.start()
    try:
        end = time.monotonic() + 45
        while True:
            recovered = run.command(['progress', receipt['retained']['id']])
            assert recovered['failure'] is None, recovered
            if recovered['source_rows']['state'] == 'settled':
                break
            assert time.monotonic() < end, recovered
            time.sleep(.02)
        assert run.command(['input', 'cat', receipt['retained']['id']], raw=True) == interrupted
        recovered_doc = run.command([REF, 'read', '--format', 'json'])
        assert len(recovered_doc['messages']) == 3 and recovered['parser']['records']['framed'] == 4
    finally:
        run.stop()
    run.archive = run.output / 'interrupted-until-idle-archive'
    path.write_bytes(initial)
    idle_source = run.acquire(config)['input']
    # A known retained target lets the ordinary progress client witness pending
    # source work before the owned process receives SIGINT; no timing-only claim.
    waiting = initial + b''.join(line(user('interrupt-' + str(i), 'Pending ' + 'x' * 200000, 'answer', 4)) for i in range(32))
    path.write_bytes(waiting)
    argv = [run.binary, '--archive', str(run.archive), 'daemon', '--config', str(config), '--until-idle', '--json']
    run.process = subprocess.Popen(argv, env=run.env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    deadline = time.monotonic() + 30
    while True:
        assert run.process.poll() is None, 'until-idle exited before pending control'
        pending = run.command(['progress', idle_source], allowed=(0, 4, 6, 7))
        if pending and pending.get('parser', {}).get('acquired_upto') == len(waiting) and pending['source_rows']['state'] == 'pending':
            break
        assert time.monotonic() < deadline, pending
        time.sleep(.01)
    run.process.send_signal(signal.SIGINT)
    stdout, stderr = run.process.communicate(timeout=30)
    code = run.process.returncode
    (run.output / 'interrupted-until-idle.stdout').write_bytes(stdout)
    (run.output / 'interrupted-until-idle.stderr').write_bytes(stderr)
    assert code != 0 and b'interrupted before target settlement' in stderr, (code, stdout, stderr)
    (run.output / 'interrupted-until-idle.json').write_text(json.dumps({'argv': argv, 'exit': code, 'observed_progress': pending}, indent=2))
    run.process = None
    run.archive = run.output / 'masking-archive'
    secret = 'ghp_' + 'A' * 36
    secret_bytes = title + line(user('credential', 'Neutral ' + secret))
    path.write_bytes(secret_bytes)
    secret_source = run.acquire(config)['input']
    run.start()
    try:
        shown = run.command(['input', 'cat', secret_source], raw=True)
        assert secret.encode() not in shown and b'[credential: GitHub token, masked]' in shown
    finally:
        run.stop()
    run.archive = run.output / 'escaped-archive'
    encoded = secret_bytes.replace(secret.encode(), ''.join(('\\u%04x' % ord(c) for c in secret)).encode())
    path.write_bytes(encoded)
    escaped_source = run.acquire(config)['input']
    run.start()
    try:
        refused = run.command(['input', 'cat', escaped_source], allowed=(2,), raw=True)
        assert not refused
    finally:
        run.stop()
    run.archive = run.output / 'empty-genesis-archive'
    path.write_bytes(b'')
    empty = run.acquire(config)
    assert empty['progress']['parser']['records']['framed'] == 0
    run.start()
    try:
        assert run.command(['input', 'cat', empty['input']], raw=True) == b''
        listing = run.command(['list', '--json'], allowed=(0, 1, 5))
        assert listing['rows'] == [], listing
    finally:
        run.stop()
    path.write_bytes(initial)
    after_empty = run.acquire(config)
    assert after_empty['input'] == empty['input'] and after_empty['chain'] == empty['chain']
    _, after_empty_document = run.inspect(empty['input'], initial)
    assert len(after_empty_document['messages']) == 2
    return {'exact_capture_mutants_rejected': mutants, 'input': source, 'chain': 1, 'revisions': [base['revision'], sep['revision'], partial['revision'], final['revision']], 'records': 5, 'messages': 4, 'retained_bytes': len(expected), 'provider_deleted_read': True, 'separator_zero_records': True, 'split_tail_restart_retry': True, 'empty_genesis_then_append': True, 'waiting_tail_publishes_complete_prefix': True, 'owned_pending_process_kill_recovery': True, 'pending_until_idle_interrupt_non_success': True, 'uninterrupted_equivalence': True, 'input_presentation_masked': True, 'escaped_presentation_refused': True, 'retained_sha256': hashlib.sha256(expected).hexdigest()}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert args.binary.is_absolute() and args.output.is_absolute()
    args.output.mkdir(exist_ok=False, parents=True)
    run = Run(args.binary, args.output)
    report = {'binary': str(args.binary), 'binary_sha256': hashlib.sha256(args.binary.read_bytes()).hexdigest(), 'status': 'failed'}
    error = None
    try:
        report['result'] = verify(run)
        report['status'] = 'passed'
    except BaseException as e:
        error = e
        report['error'] = repr(e)
    finally:
        cleanup = run.close()
        report['cleanup_errors'] = cleanup
        if cleanup:
            report['status'] = 'failed'
        (args.output / 'calls.json').write_text(json.dumps(run.calls, indent=2))
        (args.output / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if error:
        raise error
    if cleanup:
        raise SystemExit(1)
if __name__ == '__main__':
    main()
