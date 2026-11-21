"""A production-population screen; no timing or format-adoption claim."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import sqlite3
import subprocess


HERE = Path(__file__).resolve().parent


def database(path):
    return sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)


def signature(path):
    result = {}
    with database(path) as db:
        for table, vocabulary in [('text_fts', 'fts_vocab_text'),
                                  ('output_fts', 'fts_vocab_output')]:
            result[vocabulary] = db.execute(
                f'SELECT term,doc,cnt FROM {vocabulary} ORDER BY term').fetchall()
            for term in ['fixture', 'current', 'marker', 'old', 'independent', 'lodz', '"two words"']:
                # Full native scores, not a tolerance or a truncated hit set.
                result[table + ':' + term] = db.execute(
                    f'SELECT hex(t.hash),bm25({table}) FROM {table} '
                    f'JOIN text t ON t.id={table}.rowid WHERE {table} MATCH ? '
                    f'ORDER BY bm25({table}),hex(t.hash)', [term]).fetchall()
    return json.loads(json.dumps(result))


def normal_build(binary, root, observations, manifest):
    home = root / 'home'
    project = home / '.claude/projects/-fixture'
    project.mkdir(parents=True)
    for index, row in enumerate(observations):
        target = project / row['input']
        shutil.copyfile(manifest.parent / row['file'], target)
        # The fixture explicitly makes the higher-priority observation newer.
        stamp = 1_767_326_400_000_000_000 + row['priority'] * 1_000_000_000 + index
        os.utime(target, ns=(stamp, stamp))
    env = dict(HOME=str(home), PATH=os.environ['PATH'], LANG='C.UTF-8',
               USER=os.environ.get('USER', 'runner'))
    for name in ['CONFIG', 'DATA', 'STATE', 'CACHE', 'RUNTIME']:
        directory = root / name.lower()
        directory.mkdir(mode=0o700)
        env['XDG_' + name + ('_DIR' if name == 'RUNTIME' else '_HOME')] = str(directory)
    archive = root / 'archive'
    argv = [str(binary), '--json', '--archive', str(archive), '--listen', '127.0.0.1:0',
            'daemon', '--until-idle']
    with (root / 'production.log').open('wb') as log:
        process = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            code = process.wait(timeout=120)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                code = process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
            raise RuntimeError('the tiny normal production build did not finish')
    if code:
        raise RuntimeError(f'the normal production build exited {code}')
    for name in ['archive', 'index']:
        path = archive / (name + '.db')
        with sqlite3.connect(path) as db:
            assert db.execute('PRAGMA wal_checkpoint(TRUNCATE)').fetchone() == (0, 0, 0)
        with path.open('rb') as stream:
            os.fsync(stream.fileno())
        wal = path.with_name(path.name + '-wal')
        assert not wal.exists() or wal.stat().st_size == 0
    return archive


def main(args):
    assert os.environ.get('GITHUB_ACTIONS') == 'true'
    root = args.root.resolve()
    temporary = Path(os.environ['RUNNER_TEMP']).resolve()
    assert root.is_relative_to(temporary) and root != temporary and not root.exists()
    root.mkdir()
    fixture = runpy.run_path(str(HERE / 'check.py'))['fixture']
    manifest, observations = fixture(root / 'sources')
    # Same complete synthetic bodies cross both physical partitions. Repeated
    # text is one document, not one document for every block or source claim.
    gamma = next(row for row in observations if row['input'] == 'gamma.jsonl')
    source = manifest.parent / gamma['file']
    value = json.loads(source.read_bytes())
    repeated = ['current fixture marker'] * 12 + ['Łódź two words', 'two', 'words']
    value['message']['content'].extend(dict(type='text', text=text) for text in repeated)
    output = dict(type='user', uuid='tool-response', parentUuid='independent', sessionId='gamma',
                  timestamp='2026-01-02T03:04:06Z', message=dict(role='user', content=[
                      dict(type='tool_result', tool_use_id='call1', content='current fixture marker')]))
    raw = (b'\n' + json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode() + b'\n'
           + json.dumps(output, ensure_ascii=False, separators=(',', ':')).encode() + b'\n')
    source.write_bytes(raw)
    gamma.update(end=len(raw)-1, texts=gamma['texts'] + repeated, many=True,
                 sha256=hashlib.sha256(raw).hexdigest())
    manifest.write_text(json.dumps(dict(observations=observations), indent=2) + '\n')
    production_root = root / 'production'
    production_root.mkdir()
    archive = normal_build(args.production, production_root, observations, manifest)
    expected = signature(archive / 'index.db')
    report = dict(production_revision=args.revision, production_version=subprocess.check_output(
        [str(args.production), '--version'], text=True).strip(), production=expected,
        modes={}, timing_qualified=False, ordinary_pages_qualified=False,
        missing_projection_owners=['selected time/source order and field-group routes',
            'session items, block/layout facts and active-path selection',
            'sealed ItemPlace fragment continuation and generation invalidation',
            'tool, event and other query projections'],
        purpose='Native global distinct-text population/rank screen before any timed comparison')
    assert args.revision in report['production_version']
    parity = True
    for mode in ['versioned', 'compact']:
        store = root / mode
        store.mkdir()
        built = subprocess.run([str(args.binary), 'build', str(manifest), str(store),
                                'g1', mode, 'forward', 'complete'],
                               text=True, capture_output=True, timeout=120)
        (root / (mode + '-build.log')).write_text(built.stdout + built.stderr)
        if built.returncode:
            raise RuntimeError(f'{mode} build failed: {built.returncode}')
        actual = signature(store / 'g1/search.db')
        # Independent checks that the screen actually distinguishes block rows
        # from distinct documents, channel populations and block-local phrases.
        prose = {term: (documents, count) for term, documents, count in actual['fts_vocab_text']}
        tool_output = {term: (documents, count) for term, documents, count in actual['fts_vocab_output']}
        assert prose['current'] == (2, 2)
        assert prose['lodz'] == (1, 1)
        assert tool_output['current'] == (1, 1)
        assert len(actual['text_fts:"two words"']) == 1
        mismatches = {key: dict(production=expected[key], experiment=actual[key])
                      for key in expected if expected[key] != actual[key]}
        report['modes'][mode] = dict(signature=actual, mismatches=mismatches,
                                   build=json.loads(built.stdout))
        parity &= not mismatches
    report['exact_distinct_text_parity'] = parity
    (root / 'ranking-screen.json').write_text(json.dumps(report, indent=2) + '\n')
    print('COMPACT_DISTINCT_TEXT_SCREEN=' + json.dumps(report), flush=True)
    return 0 if parity else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--binary', required=True, type=Path)
    parser.add_argument('--production', required=True, type=Path)
    parser.add_argument('--revision', required=True)
    raise SystemExit(main(parser.parse_args()))
