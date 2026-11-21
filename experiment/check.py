"""Tiny neutral controls. SQL-only execution is not Rust or throughput proof."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import selectors
import shutil
import signal
import sqlite3
import subprocess
import time

HERE = Path(__file__).resolve().parent


def route(session):
    result = 14695981039346656037
    for byte in session.encode():
        result = ((result ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return result % 2


def fixture(root):
    root.mkdir()
    assert route('claude-code:alpha') != route('claude-code:gamma')
    cases = [
        ('alpha-old.jsonl', 'alpha', 'shared', 10, ['old fixture marker']),
        ('alpha-new.jsonl', 'alpha', 'shared', 20, ['current fixture marker', 'second current block']),
        ('gamma.jsonl', 'gamma', 'independent', 10, ['independent fixture marker']),
    ]
    observations = []
    for name, session, message, priority, texts in cases:
        record = dict(type='assistant', uuid=message, parentUuid=None, sessionId=session,
                      timestamp='2026-01-02T03:04:05Z',
                      message=dict(id='api-' + message, type='message', role='assistant',
                                   model='fixture-model', content=[dict(type='text', text=t) for t in texts],
                                   stop_reason='end_turn', usage=dict(input_tokens=1, output_tokens=2)))
        raw = b'\n' + json.dumps(record, ensure_ascii=False, separators=(',', ':')).encode() + b'\n'
        (root / name).write_bytes(raw)
        observations.append(dict(input=name, file=name, priority=priority,
                                 session='claude-code:' + session, message='claude-code:' + message,
                                 start=1, end=len(raw)-1, texts=texts,
                                 sha256=hashlib.sha256(raw).hexdigest()))
    manifest = root / 'inputs.json'
    manifest.write_text(json.dumps(dict(observations=observations), indent=2) + '\n')
    return manifest, observations


def sql_literal(value):
    if value is None:
        return 'NULL'
    if isinstance(value, bytes):
        return "X'" + value.hex() + "'"
    if isinstance(value, int):
        return str(value)
    return "'" + value.replace("'", "''") + "'"


def sql(db, script):
    result = subprocess.run([shutil.which('sqlite3'), '-batch', '-bail', str(db)],
                            input=script, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(dict(exit=result.returncode, stderr=result.stderr, stdout=result.stdout))
    return result.stdout.strip()


def sync(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def publish(root, generation):
    pointer = root / 'CURRENT.tmp'
    pointer.write_text(generation)
    sync(pointer)
    pointer.replace(root / 'CURRENT')
    sync(root)


def sql_control(root, manifest, observations):
    """Use the exact experimental SQL, with declared fixture hashes/payloads."""
    results = []
    for order in ('forward', 'reverse'):
        generation = root / ('sql-' + order)
        generation.mkdir()
        sequence = observations if order == 'forward' else list(reversed(observations))
        transitions = []
        for part in range(2):
            directory = generation / str(part)
            directory.mkdir()
            db = directory / 'index.db'
            statements = ['PRAGMA page_size=8192;', 'PRAGMA foreign_keys=ON;', 'PRAGMA journal_mode=WAL;',
                          'PRAGMA synchronous=FULL;', 'PRAGMA wal_autocheckpoint=0;',
                          (HERE / 'versioned.sql').read_text(), 'BEGIN;']
            raw = bytearray()
            for index, row in enumerate(sequence, 1):
                if route(row['session']) != part:
                    continue
                source = (manifest.parent / row['file']).read_bytes()
                # SQL control identifiers are fixture literals, not product content hashes.
                version = bytes([observations.index(row) + 1]) * 32
                values = [index, row['input'], len(raw), len(source), len(source), hashlib.sha256(source).digest()]
                statements.append('INSERT INTO file VALUES(' + ','.join(map(sql_literal, values)) + ');')
                raw.extend(source)
                values = [index, version, row['session'], row['message'], None, json.dumps(row['texts']).encode()]
                statements.append('INSERT INTO version VALUES(' + ','.join(map(sql_literal, values)) + ');')
                values = [index, index, index, row['start'], row['end'], row['priority']]
                statements.append('INSERT INTO observation VALUES(' + ','.join(map(sql_literal, values)) + ');')
                choose = (HERE / 'choose.sql').read_text().replace('?1', sql_literal(row['session'])).replace('?2', sql_literal(row['message']))
                statements += [choose, "SELECT 'choice:'||f.name FROM selected s JOIN observation o ON o.pk=s.observation JOIN file f ON f.id=o.file ORDER BY s.id;"]
            statements += ['COMMIT;', 'PRAGMA wal_checkpoint(TRUNCATE);', 'PRAGMA integrity_check;',
                           'SELECT count(*) FROM pragma_foreign_key_check;']
            (directory / 'originals.raw').write_bytes(raw)
            sync(directory / 'originals.raw')
            output = sql(db, '\n'.join(statements))
            assert output.splitlines()[-3:] == ['0|0|0', 'ok', '0'], output
            transitions.extend(line for line in output.splitlines() if line.startswith('choice:'))
            sync(db)
            sync(directory)
        sync(generation)
        publish(root, generation.name)
        selected = []
        for row in observations:
            if row['input'] == 'alpha-old.jsonl':
                continue
            db = generation / str(route(row['session'])) / 'index.db'
            query = "SELECT f.name||'|'||o.start||'|'||o.end||'|'||f.offset||'|'||f.bytes FROM selected s JOIN observation o ON o.pk=s.observation JOIN file f ON f.id=o.file WHERE s.session=" + sql_literal(row['session']) + ' AND s.id=' + sql_literal(row['message']) + ';'
            fields = sql(db, query).split('|')
            assert fields[:3] == [row['input'], str(row['start']), str(row['end'])]
            offset, length = map(int, fields[3:])
            retained = (db.parent / 'originals.raw').read_bytes()[offset:offset+length]
            source = (manifest.parent / row['file']).read_bytes()
            assert retained == source
            assert retained[row['start']:row['end']] == source[row['start']:row['end']]
            selected.append(row['input'])
        # History remains addressable independently of the selected observation.
        db = generation / str(route('claude-code:alpha')) / 'index.db'
        assert sql(db, 'SELECT count(*) FROM version;') == '2'
        assert sql(db, "SELECT f.name FROM observation o JOIN version v ON v.pk=o.version JOIN file f ON f.id=o.file WHERE v.hash=" + sql_literal(bytes([1]) * 32) + ';') == 'alpha-old.jsonl'
        results.append(dict(order=order, selected=selected, transitions=transitions,
                            index_bytes=sum(p.stat().st_size for p in generation.glob('*/index.db'))))
    # A committed first partition without publication cannot change the current generation.
    before = (root / 'CURRENT').read_bytes()
    partial = root / 'sql-interrupted'
    (partial / '0').mkdir(parents=True)
    sql(partial / '0/index.db', (HERE / 'versioned.sql').read_text() + '\nBEGIN; COMMIT;')
    assert (root / 'CURRENT').read_bytes() == before
    # The original latest-only schema cannot keep both observations under one message ID.
    original_db = root / 'original-overlap.db'
    sql(original_db, (HERE / 'latest.sql').read_text() + "\nINSERT INTO message VALUES(1,'same','session',NULL,1,1,2);")
    duplicate = subprocess.run([shutil.which('sqlite3'), '-batch', '-bail', str(original_db)],
                               input="INSERT INTO message VALUES(2,'same','session',NULL,2,1,3);", text=True, capture_output=True)
    assert duplicate.returncode != 0 and 'UNIQUE constraint failed: message.id' in duplicate.stderr
    return dict(kind='host_sqlite_cli_control', sqlite_version=subprocess.check_output([shutil.which('sqlite3'), '--version'], text=True).strip(),
                cases=results, publication_pointer_unchanged=True,
                original_overlap_refused=duplicate.stderr.strip(),
                limitations=['fixture-literal version hashes and payloads; no Rust parser/codec executed',
                             'SQL publication control only; native interrupted builder still required',
                             'not speed, ranking or production parity evidence'])


def native_control(binary, root, manifest, observations):
    def call(args, check=True):
        before = resource.getrusage(resource.RUSAGE_CHILDREN)
        start = time.perf_counter()
        result = subprocess.run([str(binary), *map(str, args)], capture_output=True, text=True, timeout=60)
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        result.accounting = dict(wall_seconds=time.perf_counter()-start,
                                 child_user_seconds=after.ru_utime-before.ru_utime,
                                 child_system_seconds=after.ru_stime-before.ru_stime,
                                 child_block_output_bytes=(after.ru_oublock-before.ru_oublock)*512,
                                 child_block_input_bytes=(after.ru_inblock-before.ru_inblock)*512)
        if check and result.returncode:
            raise RuntimeError(dict(args=list(map(str, args)), exit=result.returncode, stdout=result.stdout, stderr=result.stderr))
        return result

    receipts = {}
    current = next(row for row in observations if row['input'] == 'alpha-new.jsonl')
    independent = next(row for row in observations if row['input'] == 'gamma.jsonl')
    old = observations[0]
    hashes = {}
    for mode, order in [('latest', 'forward'), ('versioned', 'forward'), ('versioned', 'reverse'), ('compact', 'forward'), ('compact', 'reverse')]:
        destination = root / (mode + '-' + order)
        destination.mkdir()
        result = call(['build', manifest, destination, 'g1', mode, order, 'complete'])
        report = json.loads(result.stdout)
        lookups = []
        for row in [current, independent]:
            found = json.loads(call(['lookup', destination, row['session'], row['message']]).stdout)
            assert found['citation']['input'] == row['input']
            assert (found['citation']['start'], found['citation']['end']) == (row['start'], row['end'])
            original = (manifest.parent / row['file']).read_bytes()
            assert found['citation']['record'].encode() == original[row['start']:row['end']]
            if mode != 'latest':
                assert [block['text'] for block in found['model']['blocks']] == row['texts']
                if row == current:
                    assert hashes.setdefault(order, found['version']) == found['version']
            lookups.append(found)
        if mode != 'latest':
            transitions = {item['input']: item for part in report['parts']
                           for item in part['selection_transitions']}
            old_hash = transitions[old['input']]['observed_version']
            new_hash = transitions[current['input']]['observed_version']
            assert old_hash != new_hash
            assert transitions[current['input']]['selected_version'] == new_hash
            assert transitions[old['input']]['selected_version'] == (old_hash if order == 'forward' else new_hash)
            version = next(item['observed_version'] for part in report['parts']
                           for item in part['selection_transitions'] if item['input'] == old['input'])
            found = json.loads(call(['lookup', destination, old['session'], old['message'], version]).stdout)
            assert [block['text'] for block in found['model']['blocks']] == old['texts']
            assert found['citation']['input'] == old['input']
            assert found['citation']['record'].encode() == (manifest.parent / old['file']).read_bytes()[old['start']:old['end']]
            lookups.append(found)
        wrong_scope = call(['lookup', destination, current['session'], independent['message']], check=False)
        assert wrong_scope.returncode != 0
        searches = {}
        for term, expected in [('old', []), ('current', [[current['session'], current['message'], 0]]),
                               ('independent', [[independent['session'], independent['message'], 0]])]:
            found = json.loads(call(['search', destination, term]).stdout)
            assert found['hits'] == expected
            searches[term] = found
        receipts[mode + '-' + order] = dict(build=report, build_execution=result.accounting,
                                            fresh_connection_reads=lookups, hit_membership=searches,
                                            unrelated_scope_refused=True)
        (root / ('case-' + mode + '-' + order + '.json')).write_text(
            json.dumps(receipts[mode + '-' + order], indent=2) + '\n')
        print(json.dumps(dict(completed_case=mode + '-' + order,
                              build_execution=result.accounting)), flush=True)
    assert hashes['forward'] == hashes['reverse']

    for mode in ['versioned', 'compact']:
        destination = root / (mode + '-forward')
        before = (destination / 'CURRENT').read_bytes()
        process = subprocess.Popen([str(binary), 'build', str(manifest), str(destination), 'partial', mode, 'forward', 'interrupt'],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            if not selector.select(30):
                process.kill()
                stdout, stderr = process.communicate(timeout=10)
                raise RuntimeError(dict(phase='no committed boundary', stdout=stdout, stderr=stderr))
        boundary = json.loads(process.stdout.readline())
        assert boundary['phase'] == 'one-partition-committed'
        process.send_signal(signal.SIGKILL)
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == -signal.SIGKILL
        assert (destination / 'CURRENT').read_bytes() == before == b'g1'
        assert (destination / 'partial/0/index.db').exists()
        assert not (destination / 'partial/1').exists()
        assert not (destination / 'partial/mode').exists()
        found = json.loads(call(['lookup', destination, current['session'], current['message']]).stdout)
        assert found['generation'] == 'g1' and found['version'] == hashes['forward']
        receipts['interrupted' if mode == 'versioned' else 'interrupted-compact'] = dict(boundary=boundary, exit=process.returncode, prior_generation_read=found,
                                       current_pointer_unchanged=True, stderr=stderr)

    # A later generation selects another body. Old patches retain their own
    # immutable base coordinates, and no published g1 payload/DB is rewritten.
    destination = root / 'compact-forward'
    durable = [path for path in (destination / 'g1').rglob('*')
               if path.name in ('index.db', 'originals.raw', 'bodies.zst')]
    before_hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in durable}
    final = dict(current)
    final.update(input='alpha-final.jsonl', file='alpha-final.jsonl', priority=30, texts=['final fixture marker'])
    value = json.loads((manifest.parent / current['file']).read_bytes())
    value['message']['content'] = [dict(type='text', text='final fixture marker')]
    raw = b'\n' + json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode() + b'\n'
    (manifest.parent / final['file']).write_bytes(raw)
    final.update(end=len(raw)-1, sha256=hashlib.sha256(raw).hexdigest())
    new_manifest = manifest.parent / 'inputs-g2.json'
    new_manifest.write_text(json.dumps(dict(observations=observations + [final])))
    result = call(['build', new_manifest, destination, 'g2', 'compact', 'forward', 'complete'])
    selected = json.loads(call(['lookup', destination, final['session'], final['message']]).stdout)
    assert [block['text'] for block in selected['model']['blocks']] == final['texts']
    assert selected['citation']['record'].encode() == raw[final['start']:final['end']]
    transitions = {item['input']: item for part in receipts['compact-forward']['build']['parts']
                   for item in part['selection_transitions']}
    preserved = []
    for previous in [old, current]:
        version = transitions[previous['input']]['observed_version']
        found = json.loads(call(['lookup', destination, previous['session'], previous['message'], version]).stdout)
        assert found['version'] == version and found['citation']['input'] == previous['input']
        assert [block['text'] for block in found['model']['blocks']] == previous['texts']
        assert found['citation']['record'].encode() == (manifest.parent / previous['file']).read_bytes()[previous['start']:previous['end']]
        preserved.append(found)
    assert before_hashes == {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in durable}
    part = route(current['session'])
    with sqlite3.connect((destination / f'g1/{part}/index.db').as_uri() + '?mode=ro', uri=True) as db:
        g1_bases = {row[0].hex() for row in db.execute('SELECT base_hash FROM older')}
    with sqlite3.connect((destination / f'g2/{part}/index.db').as_uri() + '?mode=ro', uri=True) as db:
        g2_bases = {row[0].hex() for row in db.execute('SELECT base_hash FROM older')}
    assert g1_bases == {hashes['forward']} and g2_bases == {selected['version']}
    receipts['later-selection'] = dict(g1_durable_hashes_unchanged=True, g1_base_hashes=sorted(g1_bases),
                                      g2_base_hashes=sorted(g2_bases), historical_reads=preserved,
                                      current=selected, build=json.loads(result.stdout))
    for row in observations:
        assert hashlib.sha256((manifest.parent / row['file']).read_bytes()).hexdigest() == row['sha256']
    return dict(kind='native_synthetic_version_control', cases=receipts,
                ordinary_ranking_qualified=False, production_parity=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--binary', type=Path)
    parser.add_argument('--sql-control', action='store_true')
    args = parser.parse_args()
    args.root.mkdir()
    manifest, observations = fixture(args.root / 'fixture')
    result = sql_control(args.root, manifest, observations) if args.sql_control else native_control(args.binary, args.root, manifest, observations)
    (args.root / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
