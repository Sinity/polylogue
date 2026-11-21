"""Hosted fixed-population accounting for direct final SQLite writes."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import resource
import runpy
import sqlite3
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROUTE = runpy.run_path(str(HERE / 'check.py'))['route']
BASE_FILES = 497
BASE_BYTES = 533_840_740
BASE_MESSAGES = 176_666
BASE_RECORDS = 326_963
COHORT_SIZE = 8
MEMORY_MAX = 4 * 1024**3
SWAP_MAX = 1024**3


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def hash_file(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def memory_available():
    return next(int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines()
                if line.startswith('MemAvailable:'))


def cgroup_files():
    row = next(line for line in Path('/proc/self/cgroup').read_text().splitlines() if line.startswith('0::'))
    group = Path('/sys/fs/cgroup') / row.split('::', 1)[1].lstrip('/')
    return {name: (group / name).read_text() for name in
            ['memory.peak', 'memory.swap.peak', 'memory.stat', 'memory.events', 'memory.max',
             'memory.swap.max', 'cpu.stat', 'io.stat', 'cpu.max', 'cpuset.cpus.effective']}


def helper(args):
    """The existing wait4 measurement boundary, including the builder's final fsync."""
    phase = args.root
    with (phase / 'builder.log').open('wb') as output:
        started = time.monotonic()
        process = subprocess.Popen([str(args.binary), 'build', str(args.manifest), str(phase / 'store'),
                                    'g1', args.mode, 'forward', 'complete'], stdout=output, stderr=subprocess.STDOUT)
        _, status, usage = os.wait4(process.pid, 0)
        code = os.waitstatus_to_exitcode(status)
        process.returncode = code
        elapsed = time.monotonic() - started
    value = dict(exit=code, seconds=elapsed, user_seconds=usage.ru_utime, system_seconds=usage.ru_stime,
                 peak_rss_bytes=usage.ru_maxrss * 1024,
                 block_input_bytes=usage.ru_inblock * 512, block_output_bytes=usage.ru_oublock * 512,
                 minor_faults=usage.ru_minflt, major_faults=usage.ru_majflt,
                 cgroup=cgroup_files(), actual_cpus=sorted(os.sched_getaffinity(0)),
                 ended_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    save(phase / 'execution.json', value)
    return code if code >= 0 else 128 - code


def prepare(args, root, results):
    started = time.monotonic()
    home = root / 'corpus/home'
    generator = subprocess.run([str(args.generator), str(home), str(HERE / 'synthetic-lengths.txt')],
                               text=True, capture_output=True, check=True, timeout=600)
    (results / 'generation.log').write_text(generator.stdout + generator.stderr)
    project = home / '.claude/projects/-generated'
    expected = json.loads((HERE / 'synthetic-samples.json').read_text())
    files = list(project.iterdir())
    assert len(files) == expected['files'] == 7773
    assert sum(p.stat().st_size for p in files) == expected['bytes'] == 8_590_127_331
    for sample in expected['samples']:
        assert (project / sample['name']).stat().st_size == sample['bytes']
        assert hash_file(project / sample['name']) == sample['sha256']
    selected = json.loads((HERE / 'pair-files.json').read_text())
    assert len(selected) == BASE_FILES and sum(selected.values()) == BASE_BYTES
    for name, length in selected.items():
        assert (project / name).stat().st_size == length
    # Exactly the previously measured filter. Deleting unselected generated files is preparation.
    for path in files:
        if path.name not in selected:
            path.unlink()
    assert {p.name for p in project.iterdir()} == set(selected)
    controls = []
    observations = []
    corpus_hashes = {}
    base_records = 0
    cohort = sorted(selected)[:COHORT_SIZE]
    for name in sorted(selected):
        source = project / name
        raw_hash = hashlib.sha256()
        lines = 0
        with source.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b''):
                raw_hash.update(chunk)
                lines += chunk.count(b'\n')
        corpus_hashes[name] = raw_hash.hexdigest()
        base_records += lines
        session = 'claude-code:' + source.stem
        row = dict(input=name, file=str(source), session=session, priority=10, many=True)
        if name in cohort:
            row['control'] = session + '-u1'
        observations.append(row)
    assert base_records == BASE_RECORDS
    extra_records = extra_messages = extra_bytes = 0
    for name in cohort:
        source = project / name
        destination = project / (source.stem + '.overlap.jsonl')
        with source.open('rb') as stream, destination.open('xb') as output:
            first = stream.readline()
            value = json.loads(first)
            assert value['type'] == 'user' and value['sessionId'] == source.stem
            assert value['uuid'] == source.stem + '-u1'
            assert isinstance(value['message']['content'], str)
            old_text = value['message']['content']
            value['message']['content'] += '\nfixture observation revision two'
            replacement = json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode() + b'\n'
            output.write(replacement)
            users = 1
            responses = set()
            records = 1
            for line in stream:
                output.write(line)
                records += 1
                record = json.loads(line)
                assert record['sessionId'] == source.stem
                if record['type'] == 'user':
                    users += 1
                elif record['type'] == 'assistant':
                    responses.add(record['message']['id'])
            output.flush()
            os.fsync(output.fileno())
        control = dict(session='claude-code:' + source.stem, message='claude-code:' + value['uuid'],
                       old_input=name, new_input=destination.name, old_text=old_text,
                       new_text=value['message']['content'], old_start=0, new_start=0,
                       old_end=len(first)-1, new_end=len(replacement)-1,
                       records=records, messages=users+len(responses))
        controls.append(control)
        observations.append(dict(input=destination.name, file=str(destination), session=control['session'],
                                 priority=20, many=True, control=control['message']))
        corpus_hashes[destination.name] = hash_file(destination)
        extra_records += records
        extra_messages += control['messages']
        extra_bytes += destination.stat().st_size
    manifest = root / 'inputs.json'
    save(manifest, dict(observations=observations))
    for path in project.iterdir():
        with path.open('rb') as stream:
            os.fsync(stream.fileno())
    value = dict(original_manifest_sha256=hash_file(HERE / 'pair-files.json'),
                 full_generation_files=7773, full_generation_bytes=8_590_127_331,
                 base_files=BASE_FILES, base_bytes=BASE_BYTES, base_messages=BASE_MESSAGES, base_records=BASE_RECORDS,
                 overlap_files=COHORT_SIZE, overlap_bytes=extra_bytes, overlap_messages=extra_messages,
                 overlap_records=extra_records, total_bytes=BASE_BYTES+extra_bytes,
                 total_records=BASE_RECORDS+extra_records, total_parsed_messages=BASE_MESSAGES+extra_messages,
                 cohort=controls, source_sha256=corpus_hashes,
                 partition_bytes=[sum(Path(r['file']).stat().st_size for r in observations if ROUTE(r['session']) == p)
                                  for p in range(2)], preparation_seconds=time.monotonic()-started)
    save(results / 'preparation.json', value)
    return project, manifest, value


def readonly(path):
    return sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)


def verify(args, phase, preparation):
    root = phase / 'store'
    assert (root / 'CURRENT').read_text() == 'g1'
    mode = (root / 'g1/mode').read_text()
    tuples = []
    retained = {}
    versions = observations = 0
    for part in range(2):
        directory = root / 'g1' / str(part)
        with readonly(directory / 'index.db') as db:
            if mode == 'versioned':
                query = '''SELECT v.session,v.id,v.parent,f.name,o.start,o.end FROM selected s
                           JOIN observation o ON o.pk=s.observation JOIN version v ON v.pk=o.version
                           JOIN file f ON f.id=o.file'''
                versions += db.execute('SELECT count(*) FROM version').fetchone()[0]
                observations += db.execute('SELECT count(*) FROM observation').fetchone()[0]
            else:
                query = '''SELECT m.session,m.id,m.parent,f.name,m.start,m.end
                           FROM message m JOIN file f ON f.id=m.file'''
            tuples.extend(db.execute(query))
            with (directory / 'originals.raw').open('rb') as raw:
                for name, offset, size in db.execute('SELECT name,offset,bytes FROM file ORDER BY offset'):
                    raw.seek(offset)
                    digest = hashlib.sha256()
                    remaining = size
                    while remaining:
                        chunk = raw.read(min(remaining, 1 << 20))
                        assert chunk
                        digest.update(chunk)
                        remaining -= len(chunk)
                    assert digest.hexdigest() == preparation['source_sha256'][name]
                    assert name not in retained
                    retained[name] = size
    assert len(tuples) == BASE_MESSAGES and len(set((t[0], t[1]) for t in tuples)) == BASE_MESSAGES
    assert sum(retained.values()) == preparation['total_bytes']
    assert len(retained) == BASE_FILES + COHORT_SIZE
    fingerprint = hashlib.sha256()
    for row in sorted(tuples):
        assert ROUTE(row[0]) in (0, 1)
        fingerprint.update(json.dumps(row, ensure_ascii=False, separators=(',', ':')).encode() + b'\n')
    report = json.loads((phase / 'builder.log').read_text())
    parts = report['parts']
    assert sum(p['records'] for p in parts) == preparation['total_records']
    assert sum(p['parsed_messages'] for p in parts) == preparation['total_parsed_messages']
    assert sum(p['selected_messages'] for p in parts) == BASE_MESSAGES
    assert all(p['index_wal_bytes'] == 0 and p['checkpoint'] == [0, 0, 0] for p in parts)
    if mode == 'versioned':
        assert versions == BASE_MESSAGES + COHORT_SIZE
        assert observations == preparation['total_parsed_messages']
    citations = []
    transitions = {t['input']: t for part in parts for t in part['selection_transitions']}
    for control in preparation['cohort']:
        argv = [str(args.binary), 'lookup', str(root), control['session'], control['message']]
        current = json.loads(subprocess.check_output(argv, text=True))
        assert current['citation']['input'] == control['new_input']
        assert (current['citation']['start'], current['citation']['end']) == (control['new_start'], control['new_end'])
        assert json.loads(current['citation']['record'])['message']['content'] == control['new_text']
        if mode == 'versioned':
            old_hash = transitions[control['old_input']]['observed_version']
            new_hash = transitions[control['new_input']]['observed_version']
            assert old_hash != new_hash == current['version']
            old = json.loads(subprocess.check_output(argv + [old_hash], text=True))
            assert old['citation']['input'] == control['old_input']
            assert (old['citation']['start'], old['citation']['end']) == (control['old_start'], control['old_end'])
            assert old['model']['blocks'] == [dict(type='text', text=control['old_text'])]
            assert current['model']['blocks'] == [dict(type='text', text=control['new_text'])]
        citations.append(dict(session=control['session'], message=control['message'], input=current['citation']['input'],
                              start=current['citation']['start'], end=current['citation']['end']))
    searches = {}
    for phrase, count in [('"fixture offset daemon"', 137), ('"cargo index rust"', 120)]:
        found = json.loads(subprocess.check_output([str(args.binary), 'search', str(root), phrase], text=True))
        assert len(found['hits']) == count
        searches[phrase] = hashlib.sha256(json.dumps(found['hits']).encode()).hexdigest()
    return dict(messages=len(tuples), versions=versions if mode == 'versioned' else None,
                observations=observations, retained_files=len(retained), retained_bytes=sum(retained.values()),
                current_tuple_sha256=fingerprint.hexdigest(), phrase_hit_hashes=searches, citations=citations,
                build=report, execution=json.loads((phase / 'execution.json').read_text()))


def main(args):
    assert os.environ.get('GITHUB_ACTIONS') == 'true'
    root = args.root.resolve()
    temp = Path(os.environ['RUNNER_TEMP']).resolve()
    assert root.is_relative_to(temp) and root != temp and not root.exists()
    assert os.statvfs(temp).f_bavail * os.statvfs(temp).f_frsize >= 32 * 1024**3
    assert memory_available() >= 8 * 1024**3
    cpus = sorted(os.sched_getaffinity(0))[:2]
    assert len(cpus) == 2
    root.mkdir()
    results = root / 'results'
    results.mkdir()
    save(results / 'host.json', dict(available_cpus=sorted(os.sched_getaffinity(0)), selected_cpus=cpus,
                                   memory_available=memory_available(), uname=list(os.uname()),
                                   memory_max=MEMORY_MAX, swap_max=SWAP_MAX, cpu_quota='200%',
                                   parser_revision='51cbed4d96c2dc24bff41be87c3a5093eeb854b8'))
    project, manifest, preparation = prepare(args, root, results)
    print('PREPARED=' + json.dumps({k: v for k, v in preparation.items() if k not in ('cohort', 'source_sha256')}), flush=True)
    cases = {}
    for mode in ['latest', 'versioned']:
        phase = root / mode
        phase.mkdir()
        (phase / 'store').mkdir()
        for path in project.iterdir():
            with path.open('rb') as stream:
                os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        unit = 'sqlite-version-scale-' + mode + '-' + str(os.getpid())
        command = ['sudo', '-n', 'systemd-run', '--unit=' + unit, '--wait', '--collect',
                   '--property=User=' + str(os.getuid()), '--property=Group=' + str(os.getgid()),
                   '--property=MemoryMax=' + str(MEMORY_MAX), '--property=MemorySwapMax=' + str(SWAP_MAX),
                   '--property=CPUQuota=200%', '--property=CPUAffinity=' + ' '.join(map(str, cpus)),
                   '--property=RuntimeMaxSec=600', '--property=OOMPolicy=kill', '--property=KillMode=control-group',
                   '--property=StandardOutput=append:' + str(phase / 'helper.log'),
                   '--property=StandardError=append:' + str(phase / 'helper.log'),
                   sys.executable, '-I', str(Path(__file__).resolve()), '--helper', '--root', str(phase),
                   '--binary', str(args.binary), '--manifest', str(manifest), '--mode', mode]
        launched = subprocess.run(command, text=True, capture_output=True, timeout=630)
        (results / (mode + '-unit.log')).write_text(launched.stdout + launched.stderr)
        for name in ['execution.json', 'builder.log', 'helper.log']:
            if (phase / name).exists():
                (results / (mode + '-' + name)).write_bytes((phase / name).read_bytes())
        assert launched.returncode == 0, launched.stdout + launched.stderr
        execution = json.loads((phase / 'execution.json').read_text())
        assert execution['exit'] == 0 and execution['actual_cpus'] == cpus
        assert int(execution['cgroup']['memory.max']) == MEMORY_MAX
        assert int(execution['cgroup']['memory.swap.max']) == SWAP_MAX
        events = dict(line.split() for line in execution['cgroup']['memory.events'].splitlines())
        assert all(int(events.get(key, 0)) == 0 for key in ['oom', 'oom_kill'])
        cases[mode] = verify(args, phase, preparation)
        save(results / (mode + '-result.json'), cases[mode])
        print('COMPLETED=' + json.dumps(dict(mode=mode, execution=cases[mode]['execution'])), flush=True)
    assert cases['latest']['current_tuple_sha256'] == cases['versioned']['current_tuple_sha256']
    assert cases['latest']['phrase_hit_hashes'] == cases['versioned']['phrase_hit_hashes']
    for path in project.iterdir():
        assert hash_file(path) == preparation['source_sha256'][path.name]
    value = dict(cases=cases, preparation=preparation, same_current_tuples=True, same_phrase_hit_sets=True,
                 sources_unchanged=True, target_seconds=preparation['total_bytes'] / 40_000_000,
                 ordinary_ranking_qualified=False, product_parity=False,
                 note='Two final partitions/workers, 4GiB cgroup max/1GiB swap, 2 CPUs. Full generation/filter outside timing. Fixed eight full-file overlaps each alter only their first user message. No speed replication or production credit.')
    save(results / 'result.json', value)
    print('SCALE_RESULT=' + json.dumps(value), flush=True)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--generator', type=Path)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--mode')
    parser.add_argument('--helper', action='store_true')
    args = parser.parse_args()
    sys.exit(helper(args) if args.helper else main(args))
