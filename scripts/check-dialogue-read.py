#!/usr/bin/env python3
"""Explicit dialogue oracle over real CLI and SDK stdio clients, with pinned Morrow."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mcp = load(Path(__file__).with_name('check-mcp-read.py'), 'dialogue_mcp')
bounded = mcp.bounded
SID = bounded.OTHER
REF = 'claude-code:' + SID
BEGIN = '-----BEGIN PRIVATE KEY-----'
END = '-----END PRIVATE KEY-----'
MORROW = 'polylogue-spec/examples/archive/claude-code/projects/-home-op-morrow/7f3a2c10-0001-4000-8000-000000000001.jsonl'


def block(native, slot):
    return f'block:{REF}:{native}:{slot}'


def fixture():
    huge = ('BEGIN BLUE\n' + '🙂e\u0301 "\\\x1b repeated neutral passage ' * 800
            + '\nMIDDLE AMBER\n' + 'λ東京 ' * 1500 + '\nEND GREEN')
    secret = 'ghp_' + 'A' * 36
    entries = [
        ('human', 'user', 'Remember the mixed neutral controls.'),
        ('model', 'assistant', [
            {'type': 'thinking', 'thinking': BEGIN + ' hidden opener', 'signature': 'synthetic'},
            {'type': 'text', 'text': 'PRIVATE BODY MUST NOT APPEAR'},
            {'type': 'tool_use', 'id': 'independent-end', 'name': 'Bash', 'input': {'command': END}},
            {'type': 'text', 'text': 'STILL PRIVATE' + END + ' after pem'},
            {'type': 'text', 'text': 'short ghp_'},
            {'type': 'tool_use', 'id': 'independent-start', 'name': 'Write', 'input': {'content': BEGIN}},
            {'type': 'text', 'text': 'A' * 36 + ' separate prefix'},
            {'type': 'text', 'text': huge + ' ' + secret},
        ]),
        ('harness', 'user', [
            {'type': 'tool_result', 'tool_use_id': 'independent-end', 'content': BEGIN + ' hidden', 'is_error': False},
            {'type': 'text', 'text': 'PRIVATE TOOL BODY' + END + ' after tool'},
        ]),
        ('large-harness', 'user', [
            {'type': 'tool_result', 'tool_use_id': 'independent-start', 'content': BEGIN + 'x' * 70000, 'is_error': False},
            {'type': 'text', 'text': 'after oversized output'},
        ]),
        ('repeat', 'assistant', [{'type': 'text', 'text': 'Remember the mixed neutral controls.'}]),
    ]
    rows = [{'type': 'ai-title', 'sessionId': SID, 'aiTitle': 'Mixed neutral dialogue'}]
    parent = None
    for index, (native, role, content) in enumerate(entries):
        row = {'type': role, 'uuid': native, 'sessionId': SID, 'parentUuid': parent, 'isSidechain': False,
               'timestamp': f'2026-10-01T08:{index:02}:00.000Z', 'message': {'role': role, 'content': content}}
        if role == 'assistant':
            row['message'].update(model='synthetic-model', id='response-' + native)
        elif isinstance(content, str):
            row['promptSource'] = 'typed'
        if native == 'harness':
            # Structured result channels are independent strings, even when
            # they contain misleading BEGIN/END text.
            row['toolUseResult'] = {'stdout': BEGIN, 'stderr': '', 'interrupted': False}
        rows.append(row)
        parent = native
    expected = [
        (block('human', 0), 'human', entries[0][2]),
        (block('model', 1), 'model', ''),
        (block('model', 3), 'model', ' after pem'),
        (block('model', 4), 'model', 'short ghp_'),
        (block('model', 6), 'model', 'A' * 36 + ' separate prefix'),
        (block('model', 7), 'model', huge.replace('\x1b', '\\x1b') + ' [credential: GitHub token, masked]'),
        (block('repeat', 0), 'model', entries[0][2]),
    ]
    return rows, expected


def cli_read(run, call, budget):
    args = [call['ref'], 'read', '--budget', str(budget)]
    for option in ('view', 'continuation'):
        if option in call:
            args += ['--' + option, call[option]]
    page = run.cli(args, allowed=(0, 2, 3, 4, 5, 7))
    if page is None:
        return None  # CLI argument refusal is stderr-only, captured by Run.cli.
    bounded.validate_page_citations(page, run.sources)
    assert bounded.estimated(page) <= max(128, budget), (budget, page)
    return page


def read(run, client, call, budget):
    if client is None:
        return cli_read(run, call, budget)
    arguments = dict(call, budget_tokens=budget)
    response = client.tool('read', arguments)
    page = response['structuredContent']
    bounded.validate_page_citations(page, run.sources)
    assert bounded.estimated({'structuredContent': page, 'content': response['content']}) <= max(128, budget)
    return page


def follow(run, client, ref, expected, omissions):
    call = {'ref': ref, 'view': 'dialogue'}
    index, position, reconstructed, pages, resized = 0, 0, '', 0, False
    first_next = None
    while True:
        budget = 1800 if pages % 2 else 2200
        page = read(run, client, call, budget)
        assert page['outcome'] != 'error', page
        assert page['rendered_as'] == 'dialogue-v1'
        assert page['omitted'] == omissions, page['omitted']
        assert page['warning'] == bounded.WARNING and page['settlement']
        shown = page['shown']
        assert shown
        for span in shown:
            identity, author, body = expected[index]
            assert span['block'] == identity, (span, identity)
            assert span['written_by'] == author
            assert span['display_start'] == position
            text = span['text']
            assert span['display_end'] == position + len(text)
            assert span['sources'] and span['field_origins']
            reconstructed += text
            position += len(text)
            assert body.startswith(reconstructed), (identity, reconstructed[-100:])
            if reconstructed == body:
                index, position, reconstructed = index + 1, 0, ''
        pages += 1
        assert pages < 150
        if page['next'] is None:
            break
        if client is not None:
            assert page['next']['tool'] == 'read'
            call = page['next']['arguments']
        else:
            call = page['next']
        assert call['view'] == 'dialogue'
        first_next = first_next or dict(call)
        if span['display_end'] > span['display_start'] + 400 and not resized:
            grow = read(run, client, call, 2600)
            shrink = read(run, client, call, 1700)
            assert grow['shown'][0]['display_start'] == shrink['shown'][0]['display_start']
            assert grow['shown'][0]['display_end'] > shrink['shown'][0]['display_end']
            tiny = read(run, client, call, 128)
            assert tiny['error']['reason'] == 'insufficient_budget' and 'shown' not in tiny
            retry = read(run, client, call, tiny['error']['sufficient_budget'])
            assert retry['shown'][0]['display_start'] == grow['shown'][0]['display_start']
            changed = dict(call)
            changed.pop('view')
            # This selected mixed Session refuses the default included tools;
            # exact block controls below distinguish a true basis mismatch.
            assert read(run, client, changed, 2200)['outcome'] == 'error'
            resized = True
    assert index == len(expected) and position == 0 and not reconstructed
    return {'pages': pages, 'capacity_change': resized}, first_next


def pinned_morrow(run, spec_repo, revision):
    revision = subprocess.check_output(['git', '-C', str(spec_repo), 'rev-parse', revision + '^{commit}'], text=True).strip()
    paths = [MORROW, 'polylogue-sinex-joint/realization/demos/prepare.py', 'polylogue-sinex-joint/realization/demos/check_answer.py']
    hashes = {}
    for relative in paths:
        data = subprocess.check_output(['git', '-C', str(spec_repo), 'show', revision + ':' + relative])
        path = run.output / 'spec' / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        hashes[relative] = hashlib.sha256(data).hexdigest()
    demo = run.output / 'spec/polylogue-sinex-joint/realization/demos'
    prepare = load(demo / 'prepare.py', 'dialogue_prepare')
    checker = load(demo / 'check_answer.py', 'dialogue_checker')
    case = prepare.morrow(run.output / 'prepared')['cases']['base']
    provider = run.output / 'prepared' / case['input']
    context = run.output / 'morrow.context.json'
    context.write_text(bounded.compact({'version': 1, 'context': ['claude-main', bounded.SID]}))
    receipt = run.cli(['import', provider, '--format', 'claude-code-jsonl', '--context', context, '--wait', '--wait-timeout', '30'])
    assert receipt['source_rows']['state'] == 'settled', receipt
    run.sources[receipt['retained']['id']] = provider.read_bytes()
    whole = run.cli([bounded.REF, 'read', '--format', 'json'])
    checker.check('polylogue', case, whole)
    rows = [json.loads(line) for line in provider.read_bytes().splitlines()]
    messages = [row for row in rows if row['type'] in ('user', 'assistant')]
    # Explicit independent selection. Do not ask the production selector which
    # Messages should have survived or weaken the full-document comparator.
    expected = []
    for ordinal in (1, 6, 7, 10, 11, 12):
        row = messages[ordinal - 1]
        content = row['message']['content']
        text = content if isinstance(content, str) else content[0]['text']
        expected.append((f"block:{bounded.REF}:{row['uuid']}:0", 'human' if row['type'] == 'user' else 'model', text))
    return expected, {'revision': revision, 'sha256': hashes, 'whole_document_comparator': 'unchanged passed'}


def verify(run):
    rows, expected = fixture()
    run.import_rows('mixed', rows, sid=SID)
    omitted = {'blocks': 7, 'by_author': {'model': 3, 'harness': 4},
               'by_kind': {'thinking': 1, 'tool_call': 2, 'tool_result': 2, 'text': 2},
               'by_reason': {'thinking': 1, 'tools': 2, 'author': 4}}
    results = {}
    run.client = mcp.Client(run)
    for name, client in [('cli', None), ('mcp', run.client)]:
        results[name], next_call = follow(run, client, REF, expected, omitted)
        assert results[name]['capacity_change'] and results[name]['pages'] > len(expected)
        exact = {'ref': block('harness', 1)}
        page = read(run, client, exact, 2200)
        assert page['shown'][0]['text'] == ' after tool', page
        error = read(run, client, {'ref': block('large-harness', 1)}, 2200)
        assert error['error']['reason'] == 'prior_tool_output_unavailable_or_exceeds_admission' and 'shown' not in error
        thinking = read(run, client, {'ref': block('model', 0), 'view': 'dialogue'}, 2200)
        assert 'masked' in thinking['shown'][0]['text']
        exact = {'ref': block('model', 7), 'view': 'dialogue'}
        first = read(run, client, exact, 2200)
        cursor = first['next']['arguments'] if client else first['next']
        wrong = dict(cursor)
        wrong.pop('view')
        assert read(run, client, wrong, 2200)['error']['reason'] == 'stale_read'
        wrong = dict(cursor, ref=block('model', 6))
        assert read(run, client, wrong, 2200)['error']['reason'] == 'cross_reference_continuation'
        incomplete = read(run, client, {'ref': REF, 'view': 'transcript'}, 2200)
        assert incomplete['error']['kind'] == 'invalid_query', incomplete
        assert incomplete['error']['reason'] == 'transcript_requires_full_tools_thinking_harness', incomplete
    # A changed selected Message invalidates existing dialogue continuation.
    changed = json.loads(json.dumps(rows))
    changed[2]['message']['content'][7]['text'] += ' selected edit'
    # Nonidentical immutable candidates remain unsupported. Retain the edit
    # and assert current-authority refusal, rather than claim a successful
    # canonical replacement that this product does not implement.
    path = run.output / 'selected-edit.jsonl'
    path.write_text(''.join(bounded.compact(row) + '\n' for row in changed))
    context = run.output / 'selected-edit.context.json'
    context.write_text(bounded.compact({'version': 1, 'context': ['claude-main', SID]}))
    edit = run.cli(['import', path, '--format', 'claude-code-jsonl', '--context', context, '--wait', '--wait-timeout', '30'], allowed=(2,))
    assert edit['source_rows']['state'] == 'unavailable' and edit['retained']
    revoked = read(run, run.client, cursor, 2200)
    assert revoked['outcome'] == 'error' and 'shown' not in revoked
    results['retained_conflicting_edit'] = {'import_kind': edit['error']['kind'], 'read_reason': revoked['error']['reason']}
    run.client.close()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--spec-repo', type=Path, required=True)
    parser.add_argument('--spec-revision', required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    run = mcp.Run(args.binary.resolve(), args.output.resolve())
    report = {'outcome': 'failed', 'binary': str(run.binary), 'binary_sha256': hashlib.sha256(run.binary.read_bytes()).hexdigest()}
    failure = None
    cleanup_errors = []
    try:
        run.start()
        expected, report['spec'] = pinned_morrow(run, args.spec_repo, args.spec_revision)
        omitted = {'blocks': 13, 'by_author': {'model': 7, 'harness': 6}, 'by_kind': {'thinking': 1, 'tool_call': 6, 'tool_result': 6}, 'by_reason': {'thinking': 1, 'tools': 6, 'author': 6}}
        run.client = mcp.Client(run)
        report['morrow'] = {}
        for name, client in [('cli', None), ('mcp', run.client)]:
            report['morrow'][name], _ = follow(run, client, bounded.REF, expected, omitted)
        run.client.close()
        report['mixed'] = verify(run)
        report.update(outcome='ok', scope='explicit dialogue and supported exact blocks; included tools, ranges, huge tool output and full Morrow/M1 remain open')
    except BaseException as error:
        failure = error
        report.update(outcome='failed', error=repr(error))
    finally:
        cleanup_errors = mcp.finish(run, report)
    if failure:
        raise failure
    if cleanup_errors:
        raise RuntimeError(cleanup_errors)


if __name__ == '__main__':
    main()
