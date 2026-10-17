"""Qualify the finite native Claude identity corpus through isolated public clients.

Use a checked package and explicit pinned spec, oracle and SDK inputs. The
runner preserves raw results/failures and owns only its fresh output directory,
provider copies and daemon. This is not installed migration or full acceptance.
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


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--artifact-kind', choices=('checked-package', 'smoke-package'), required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--package-source', required=True,
                        help='native commit used for the checked binary')
    parser.add_argument('--spec-repo', type=Path, required=True)
    parser.add_argument('--spec-revision', required=True)
    parser.add_argument('--corpus-path', required=True,
                        help='repository-relative native-identity corpus directory')
    parser.add_argument('--oracle-path', required=True,
                        help='repository-relative public_oracle.py beside the corpus')
    parser.add_argument('--sdk-python', type=Path, required=True)
    parser.add_argument('--sdk-client', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert args.binary.is_absolute() and args.binary.is_file()
    assert not args.output.exists()
    args.output.mkdir(mode=0o700)
    source_sha = subprocess.run(['git', '-C', str(args.source), 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
    assert source_sha == args.candidate
    assert not subprocess.run(['git', '-C', str(args.source), 'status', '--porcelain'], capture_output=True, text=True, check=True).stdout
    package_source = subprocess.run(
        ['git', '-C', str(args.source), 'rev-parse', '--verify', args.package_source + '^{commit}'],
        capture_output=True, text=True, check=True).stdout.strip()
    nonruntime_changes = subprocess.run(
        ['git', '-C', str(args.source), 'diff', '--name-only', package_source, source_sha],
        capture_output=True, text=True, check=True).stdout.splitlines()
    assert set(nonruntime_changes) <= {'README.md', 'scripts/check-native-identity.py'}, nonruntime_changes
    # Keep the package basis distinct when only this runner or supported-scope
    # documentation was added after its checked build.
    spec_repo = args.spec_repo.resolve()
    spec_pin = subprocess.run(
        ['git', '-C', str(spec_repo), 'rev-parse', '--verify', args.spec_revision + '^{commit}'],
        capture_output=True, text=True, check=True).stdout.strip()
    corpus_path, oracle_path = Path(args.corpus_path), Path(args.oracle_path)
    assert not corpus_path.is_absolute() and '..' not in corpus_path.parts
    assert not oracle_path.is_absolute() and '..' not in oracle_path.parts
    assert oracle_path.parent == corpus_path
    prefix = corpus_path.as_posix() + '/'
    fixtures = {}
    corpus = args.output / 'frozen-corpus'
    corpus.mkdir()
    for name in ['facts.json', 'inputs/a:b.jsonl', 'inputs/a.jsonl', 'inputs/unicode-main.jsonl']:
        raw = subprocess.run(['git', '-C', str(spec_repo), 'show', spec_pin + ':' + prefix + name],
                             capture_output=True, check=True).stdout
        fixtures[name] = raw
        path = corpus / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    oracle_file = corpus / 'public_oracle.py'
    oracle_file.write_bytes(subprocess.run(
        ['git', '-C', str(spec_repo), 'show', spec_pin + ':' + oracle_path.as_posix()],
        capture_output=True, check=True).stdout)
    oracle = load('identity_public_oracle', oracle_file)
    _, source_rows = oracle.inputs()  # Validate source hashes before execution.
    sys.path.insert(0, str(args.source / 'scripts'))
    module = load('native_morrow_run', args.source / 'scripts/check-morrow.py')
    run = module.Run(args.output, None)
    run.binary = args.binary
    sdk_python = args.sdk_python.resolve()
    original_sdk_client = args.sdk_client.resolve()
    original_collector = original_sdk_client.parent.parent / 'morrow_mcp.py'
    original_checker = original_sdk_client.parent.parent / 'check_answer.py'
    assert all(path.is_file() for path in (sdk_python, original_sdk_client, original_collector, original_checker))
    sdk_client = args.output / 'frozen-sdk-client' / 'joint-consumer' / 'mcp_client.py'
    sdk_client.parent.mkdir(parents=True)
    shutil.copyfile(original_sdk_client, sdk_client)
    shutil.copyfile(original_collector, sdk_client.parent.parent / 'morrow_mcp.py')
    shutil.copyfile(original_checker, sdk_client.parent.parent / 'check_answer.py')
    facts = json.loads(fixtures['facts.json'])
    extra_by_message = {}
    for fact in facts['messages']:
        row = source_rows[(fact['session_native'], tuple(fact['message_native_utf16']))][3]
        if 'uninterpreted' in row:
            extra_by_message[fact['message_ref']] = row['uninterpreted']
    assert len(extra_by_message) == 1, 'opaque-extra fixture coverage changed'
    archive = args.output / 'archive'
    report = {'candidate': args.candidate, 'package_source': package_source,
              'artifact_kind': args.artifact_kind,
              'runtime_unchanged': True, 'nonruntime_changes': nonruntime_changes,
              'runner_sha256': sha(Path(__file__).resolve()), 'binary': str(args.binary), 'binary_sha256': sha(args.binary),
              'spec': spec_pin, 'oracle_sha256': sha(Path(oracle.__file__)),
              'sdk_python': str(sdk_python), 'sdk_python_sha256': sha(sdk_python),
              'sdk_client': str(sdk_client), 'sdk_client_original': str(original_sdk_client),
              'sdk_client_sha256': sha(sdk_client),
              'sdk_fact_collector_sha256': sha(sdk_client.parent.parent / 'morrow_mcp.py'),
              'sdk_checker_sha256': sha(sdk_client.parent.parent / 'check_answer.py'),
              'commands': run.commands, 'processes': run.processes, 'phases': [], 'bindings': {},
              'limits': ['No installed archive or migration.', 'No normalized reimport implementation.',
                         'No arbitrary provider/invalid UTF8 qualification.', 'No latency/RSS/model efficiency claim.']}
    report['limits'].append('Direct assembled attrs/data preserves logical values but does not guarantee canonical key ordering; full canonical export conformance remains open.')
    bindings = report['bindings']
    owned_sources = []

    def command(arguments, label, expected=(0, 5)):
        return run.command([args.binary, '--archive', archive, *arguments], label, expected, timeout=60)

    def sdk(reference, budget, label, refused=False):
        output = args.output / ('sdk-' + label)
        arguments = {'ref': reference, 'budget_tokens': budget, 'view': 'transcript',
                     'tools': 'full', 'thinking': 'full', 'harness': 'full', 'branches': True}
        run.command([sdk_python, sdk_client, '--binary', args.binary, '--archive', archive,
                     '--call', 'read', '--arguments', json.dumps(arguments), '--output', output],
                    'official MCP ' + label, (0,), timeout=60)
        result = json.loads((output / 'result.json').read_text())
        assert result['success'] and result['sdk_context_closed'], result
        page = result['tool_result']
        if refused:
            assert page['outcome'] == 'error' and page['error']['kind'] == 'not_yet_available', page
            assert 'shown' not in page, page
            return page
        assert page['outcome'] != 'error', page
        assert not page.get('next'), 'tiny exact fixture must fit the selected complete page'
        assert page['primary_content_complete'] is True
        assert page['shown'], page
        assert (len(json.dumps(page, separators=(',', ':'), sort_keys=True, ensure_ascii=False).encode()) + 3) // 4 <= budget
        return page

    def privacy_controls():
        # These fresh synthetic controls are separate from the pinned eight-fact
        # oracle. Assess native IDs before their reversible public/wire frames.
        token = 'ghp_' + 'A' * 36
        def frame(tag, fields):
            raw = bytes((1, tag)) + len(fields).to_bytes(8, 'little')
            for field in fields:
                raw += len(field).to_bytes(8, 'little') + field
            return raw
        def session_ref(native):
            fields = (b'claude-code', native.encode('utf-8', 'surrogatepass'))
            if native.startswith('~') or ':' in native or any(0xd800 <= ord(c) <= 0xdfff for c in native):
                return '~i1:' + frame(1, fields).hex()
            return 'claude-code:' + native
        def message_ref(session, native):
            sid = session_ref(session)
            if sid.startswith('~i1:') or native.startswith('~') or any(0xd800 <= ord(c) <= 0xdfff for c in native):
                return '~i1:' + frame(3, (frame(1, (b'claude-code', session.encode('utf-8', 'surrogatepass'))),
                                          native.encode('utf-8', 'surrogatepass'))).hex()
            return sid + ':' + native
        cases = [
            ('call-scalar', token, 'answer', 'root', False),
            ('call-encoded', token + '\ud800', 'answer', 'root', False),
            ('call-clean', 'ordinary-\ud800', 'answer', 'root', True),
            ('call-split', 'ghp_\ud800' + 'A' * 36, 'answer', 'root', True),
            ('message-encoded', 'ordinary', token + '\ud800', 'root', False),
            ('parent-encoded', 'ordinary', 'answer', token + '\ud800', False),
            ('session-encoded', 'ordinary', 'answer', 'root', False),
        ]
        report['privacy_controls'] = []
        for label, call, native_message, parent, allowed in cases:
            session = '~' + token if label == 'session-encoded' else 'privacy-' + label
            rows = [{'type': 'ai-title', 'aiTitle': 'Synthetic privacy control'},
                    {'type': 'user', 'uuid': parent, 'sessionId': session, 'promptSource': 'typed',
                     'timestamp': '2026-10-01T08:00:00Z', 'message': {'role': 'user', 'content': 'Synthetic prompt'}},
                    {'type': 'assistant', 'uuid': native_message, 'parentUuid': parent, 'sessionId': session,
                     'timestamp': '2026-10-01T08:01:00Z', 'requestId': 'synthetic-request',
                     'message': {'id': 'synthetic-response', 'role': 'assistant', 'model': 'synthetic-model',
                                 'content': [{'type': 'tool_use', 'id': call, 'name': 'Synthetic', 'input': {}}],
                                 'stop_reason': 'end_turn'}}]
            source = args.output / 'owned-provider' / ('privacy-' + label + '.jsonl')
            source.write_text(''.join(json.dumps(row, ensure_ascii=True) + '\n' for row in rows))
            owned_sources.append(source)
            context = args.output / ('privacy-' + label + '.context.json')
            context.write_text(json.dumps({'version': 1, 'context': ['claude-main', session]}))
            imported = json.loads(command(['import', source, '--format', 'claude-code-jsonl', '--context', context,
                                           '--wait', '--wait-timeout', '30'], 'privacy import ' + label, (0,)))
            assert imported['source_rows']['state'] == 'settled', imported
            reference = session_ref(session)
            direct = command([reference, 'read', '--format', 'json'], 'privacy JSON ' + label, (0,) if allowed else (2,))
            terminal = command([reference, 'read', '--format', 'terminal'], 'privacy terminal ' + label, (0,) if allowed else (2,))
            page = sdk('message:' + message_ref(session, native_message), 16000, 'privacy-' + label, refused=not allowed)
            if allowed:
                document = json.loads(direct)
                wire = {'value': '~u:' + call.encode('utf-8', 'surrogatepass').hex(), 'encoding': 'wtf8'}
                assert document['messages'][1]['blocks'][0]['call_id'] == wire, document
                assert any(fragment.get('block_fields', {}).get('call_id') == wire for fragment in page['shown']), page
            else:
                # No successful partial document, terminal credential or encoded
                # recoverable token may escape a refused operation.
                wire = '~u:' + call.encode('utf-8', 'surrogatepass').hex()
                for presented in (direct, terminal, json.dumps(page)):
                    assert token not in presented and wire not in presented, label
            report['privacy_controls'].append({'name': label, 'expected': 'exact' if allowed else 'refused',
                                               'input_sha256': sha(source), 'input': imported['retained']['id'],
                                               'cli_json': True, 'cli_terminal': True, 'official_sdk': True})

    def inspect(phase):
        captures = {}
        record = {'name': phase, 'mcp': []}
        for name, binding in bindings.items():
            until = time.monotonic() + 30
            while True:
                progress = json.loads(command(['progress', binding['id']], phase + ' retained progress ' + name, (0,)))
                state = progress['source_rows']['state']
                if state == 'settled':
                    break
                assert state == 'pending', progress
                assert time.monotonic() < until, progress
                time.sleep(.05)
        for name, context in facts['contexts'].items():
            expected = next(m for m in facts['messages'] if m['session_native'] == context[1])
            text = command([expected['session_ref'], 'read', '--format', 'json'], phase + ' Session ' + name)
            captures[name] = text
            (args.output / (phase + '-' + name + '.document.json')).write_text(text)
            terminal = command([expected['session_ref'], 'read', '--format', 'terminal'], phase + ' terminal ' + name)
            assert terminal.strip()
            export = args.output / (phase + '-' + name + '-export')
            command([expected['session_ref'], 'export', '--format', 'json', '--to', export], phase + ' export ' + name)
            documents = [p for p in export.rglob('*.json') if p.name != 'manifest.json']
            assert len(documents) == 1
            # Compare lossless normalized values through the independent oracle;
            # wrapper timestamps and export qualification are handled separately.
            record.setdefault('exported', {})[name] = documents[0].read_text()
        oracle.documents(captures, bindings)
        oracle.documents(record['exported'], bindings)
        for group in (captures, record['exported']):
            repaired = oracle.loads(group['unicode-main.jsonl'])
            assert oracle.numeric(repaired['result']['diagnostics']['text_repairs']) == 1, repaired['result']
            for message in repaired['messages']:
                if message['id'] in extra_by_message:
                    assert oracle.equal(message['attrs']['uninterpreted'], extra_by_message[message['id']])
                    assert any(entry['field'] == 'attrs.uninterpreted' for entry in message['field_origins'])
        for index, expected in enumerate(facts['messages']):
            budgets = (4000, 16000) if phase == 'initial' else (16000,)
            for budget in budgets:
                page = sdk('message:' + expected['message_ref'], budget, phase + '-' + str(index) + '-' + str(budget))
                assert page['outcome'] == 'ok' and page['settlement']['state'] == 'settled', page
                assert page['diagnostics']['text_repairs'] == (1 if expected['session_native'] == 'unicode-main' else 0), page
                source_name = next(name for name, context in facts['contexts'].items()
                                   if context[1] == expected['session_native'])
                source_basis = page['settlement']['target_basis']['inputs']
                assert len(source_basis) == 1 and source_basis[0]['input'] == bindings[source_name]['id'], source_basis
                assert source_basis[0]['chain'] == bindings[source_name]['chain']
                assert source_basis[0]['original_hash'] == bindings[source_name]['original_hash']
                shown = page['shown']
                assert all(fragment['message'] == 'message:' + expected['message_ref'] for fragment in shown)
                assert all(fragment['session'] == expected['session_ref'] for fragment in shown)
                assert all(fragment['role'] == expected['role'] and fragment['written_by'] == expected['written_by'] for fragment in shown)
                for fragment in shown:
                    for envelope in fragment.get('field_origins', []):
                        assert envelope['encoding'] == 'canonical-json-v1' and envelope['masked'] is False
                        entry = oracle.loads(envelope['json'])
                        assert set(entry) == {'field', 'origin'}
                        assert entry['origin']['binding']['input'] in {b['id'] for b in bindings.values()}
                    if expected['message_ref'] in extra_by_message:
                        attrs = fragment['message_metadata']['attrs']
                        assert attrs['encoding'] == 'canonical-json-v1' and attrs['masked'] is False
                        assert oracle.equal(oracle.loads(attrs['json'])['uninterpreted'],
                                            extra_by_message[expected['message_ref']])
                    fields = fragment.get('block_fields', {})
                    if 'input' in fields:
                        oracle.model_input(fields['input'])
                if expected['text'] is not None:
                    actual = ''.join(fragment.get('text', '') for fragment in shown)
                    assert actual == expected['text'], (actual, expected)
                # Every offered exact block reference must itself be readable,
                # including tag-5 frames under formerly aliasing/UTF16 Messages.
                if budget == budgets[-1]:
                    for block_index, fragment in enumerate(shown):
                        block_page = sdk(fragment['block'], budget, phase + '-block-' + str(index) + '-' + str(block_index))
                        assert all(x['block'] == fragment['block'] for x in block_page['shown'])
                record['mcp'].append({'message': expected['message_ref'], 'budget': budget, 'shown': len(shown)})
        record.pop('exported')
        report['phases'].append(record)

    try:
        run.command([sdk_python, sdk_client, '--help'], 'official SDK import preflight', (0,), timeout=60)
        run.start(archive)
        for name, context in facts['contexts'].items():
            source = args.output / 'owned-provider' / name
            source.parent.mkdir(exist_ok=True)
            source.write_bytes(fixtures['inputs/' + name])
            owned_sources.append(source)
            context_path = args.output / (name + '.context.json')
            context_path.write_text(json.dumps({'version': 1, 'context': context}))
            reply = json.loads(command(['import', source, '--format', 'claude-code-jsonl', '--context', context_path,
                                        '--wait', '--wait-timeout', '30'], 'import ' + name, (0,)))
            assert reply['source_rows']['state'] == 'settled', reply
            basis = reply['source_rows']['basis']
            bindings[name] = {'id': reply['retained']['id'], 'chain': reply['retained']['chain'],
                              'sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                              'original_hash': basis['original_hash'], 'parser_version': basis['parser_version']}
            if name != 'unicode-main.jsonl':
                presented = command(['input', 'cat', reply['retained']['id']], 'retained original ' + name, (0,))
                assert presented.encode() == fixtures['inputs/' + name]
            else:
                # The existing escaped-JSON presentation guard is intentional;
                # source custody is not permission to bypass its privacy rule.
                command(['input', 'cat', reply['retained']['id']], 'escaped original guard ' + name, (2,))
        inspect('initial')
        run.stop()
        run.start(archive)
        inspect('restart')
        run.stop()
        for source in owned_sources:
            source.unlink()
        run.start(archive)
        inspect('provider-removed')
        run.stop()
        saved_index = args.output / 'preserved-index20'
        saved_index.mkdir()
        for name in ('index.db', 'index.db-wal', 'index.db-shm'):
            path = archive / name
            if path.exists():
                shutil.move(path, saved_index / name)
        run.start(archive)
        # Startup must actually reconstruct from retained owned inputs, rather
        # than the oracle reimporting source bytes or filling SQL rows itself.
        inspect('index-reconstructed')
        privacy_controls()
        report['success'] = True
    except BaseException as error:
        report['success'] = False
        report['failure'] = repr(error)
        raise
    finally:
        try:
            run.stop()
            run.close_runtime()
            report['owned_cleanup'] = 'clean'
        finally:
            (args.output / 'result.json').write_text(json.dumps(report, indent=2, ensure_ascii=True) + '\n')


if __name__ == '__main__':
    main()
