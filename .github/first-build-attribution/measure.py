#!/usr/bin/env python3
"""One unmerged hosted CPU diagnostic using the existing first-build helper.

The daemon child, final checkpoint/fsync, input coverage and profile functions
below are copied from the existing run-generated.py, not another ingest path.
Only process admission uses the hosted VM's system manager instead of AgentCTL.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from collections.abc import Iterator
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import time

UNIT_PROPERTIES = [
    'Id', 'LoadState', 'ActiveState', 'SubState', 'MainPID', 'ControlGroup',
    'Result', 'ExecMainCode', 'ExecMainStatus', 'MemoryCurrent', 'MemoryPeak',
    'MemorySwapCurrent', 'MemorySwapPeak', 'MemoryMax', 'MemorySwapMax',
    'CPUUsageNSec', 'CPUQuotaPerSecUSec', 'IOReadBytes', 'IOWriteBytes', 'Slice',
    'CPUAffinity',
]

def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + '.new')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)

def read_json(path: Path) -> object | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None

def command(argv: list[str], *, check: bool = True,
            timeout: float | None = None) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(argv, text=True, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, check=check, timeout=timeout)
    except subprocess.CalledProcessError as error:
        if error.stdout:
            print(error.stdout, file=sys.stderr, flush=True)
        raise

def show_unit(unit: str) -> dict[str, str]:
    result = command(['sudo', '-n', 'systemctl', 'show', unit,
                      *['--property=' + item for item in UNIT_PROPERTIES]], check=False)
    return dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)

@contextmanager
def database(path: Path) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(path.as_uri() + '?mode=ro', uri=True, timeout=0)
    try:
        conn.execute('PRAGMA query_only=ON')
        conn.execute('PRAGMA cache_size=-256')
        conn.execute('PRAGMA mmap_size=0')
        yield conn
    finally:
        conn.close()

def newest_inputs(archive: Path) -> dict[str, tuple[int, int]]:
    with database(archive / 'archive.db') as conn:
        return {row[0]: (row[1], row[2]) for row in conn.execute('''
            WITH latest AS (SELECT input,max(number) AS number FROM revision GROUP BY input)
            SELECT r.input,r.chain,r.length FROM latest l
            JOIN revision r ON r.input=l.input AND r.number=l.number
        ''')}

def verify_complete(archive: Path, expected: dict[str, tuple[int, int]]) -> dict[str, object]:
    if (archive / 'index-build' / 'index.db').exists():
        raise RuntimeError('the daemon stopped with its private index still unfinished')
    if (archive / 'index-build' / 'unsynced').exists():
        raise RuntimeError('the daemon stopped with an unsynced build marker')
    index = archive / 'index.db'
    if not index.exists():
        raise RuntimeError('the daemon did not install index.db')
    actual = newest_inputs(archive)
    if actual != expected:
        raise RuntimeError('the generated archive newest-chain inputs changed while parsing')
    with database(index) as conn:
        states = {(row[0], row[1]): (row[2], row[3]) for row in conn.execute('''
            SELECT i.id,p.chain,p.parsed_upto,p.status
            FROM input_ref i JOIN parse_state p ON p.input_pk=i.pk
        ''')}
        missing = [(key, chain, length, states.get((key, chain)))
                   for key, (chain, length) in expected.items()
                   if (key, chain) not in states
                   or states[(key, chain)][0] < length
                   or states[(key, chain)][1] in ('failed', 'refused', 'no_parser_yet')]
        if missing:
            raise RuntimeError(f'{len(missing)} newest chains are not fully parsed: {missing[:3]}')
        page_size = conn.execute('PRAGMA page_size').fetchone()[0]
        page_count = conn.execute('PRAGMA page_count').fetchone()[0]
    return {'newest_chains': len(expected),
            'newest_input_bytes': sum(length for _, length in expected.values()),
            'parsed_bytes': sum(states[(key, chain)][0] for key, (chain, _) in expected.items()),
            'index_bytes': index.stat().st_size,
            'index_wal_bytes': index.with_name('index.db-wal').stat().st_size
                if index.with_name('index.db-wal').exists() else 0,
            'index_page_bytes': page_size * page_count}

def checkpoint_databases(archive: Path) -> dict[str, object]:
    """Pay final WAL copy/sync work while the measurement unit still accounts I/O."""
    started = time.monotonic()
    databases = []
    for name in ('index', 'archive'):
        path = archive / (name + '.db')
        wal = path.with_name(path.name + '-wal')
        before = wal.stat().st_size if wal.exists() else 0
        conn = sqlite3.connect(path.as_uri() + '?mode=rw', uri=True, timeout=0)
        try:
            conn.execute('PRAGMA cache_size=-256')
            conn.execute('PRAGMA mmap_size=0')
            conn.execute('PRAGMA temp_store=FILE')
            status = conn.execute('PRAGMA main.wal_checkpoint(TRUNCATE)').fetchone()
            if status is None or status[0] != 0:
                raise RuntimeError(f'{name} WAL checkpoint is busy: {status}')
        finally:
            conn.close()
        after = wal.stat().st_size if wal.exists() else 0
        if after:
            raise RuntimeError(f'{name} still has {after} WAL bytes after checkpoint')
        for file in (path, wal):
            try:
                fd = os.open(file, os.O_RDONLY)
            except FileNotFoundError:
                if file == path:
                    raise
                continue
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        databases.append({'name': name, 'wal_before': before, 'wal_after': after,
                          'database_bytes': path.stat().st_size})
    directory = os.open(archive, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return {'seconds': time.monotonic() - started, 'databases': databases}

def child(args: argparse.Namespace) -> int:
    """The unit helper keeps exact wait4 RSS, including a daemon's final allocation."""
    phase = Path(args.child)
    argv = [args.binary, '--json', '--archive', args.archive,
            '--listen', '127.0.0.1:0', 'daemon', '--until-idle']
    if args.stored:
        argv.append('--no-sources')
    started = time.monotonic()
    process = subprocess.Popen(argv)
    write_json(phase / 'daemon-pid.json', {'pid': process.pid, 'argv': argv})
    _, status, usage = os.wait4(process.pid, 0)
    code = os.waitstatus_to_exitcode(status)
    process.returncode = code
    daemon_seconds = time.monotonic() - started
    archive = Path(args.archive)
    paid = None
    checkpoint_error = None
    checkpoint_seconds = 0.0
    helper_code = code if code >= 0 else 128 - code
    if code == 0 and not (archive / 'index-build' / 'index.db').exists() and (archive / 'index.db').exists():
        checkpoint_started = time.monotonic()
        try:
            paid = checkpoint_databases(archive)
        except Exception as error:
            checkpoint_error = str(error)
            helper_code = 1
            print('final checkpoint failed: ' + checkpoint_error, file=sys.stderr, flush=True)
        finally:
            checkpoint_seconds = time.monotonic() - checkpoint_started
    write_json(phase / 'daemon-exit.json', {
        'exit': helper_code, 'daemon_exit': code, 'wait_status': status,
        'seconds': time.monotonic() - started, 'daemon_seconds': daemon_seconds,
        'checkpoint': paid, 'checkpoint_error': checkpoint_error,
        'checkpoint_seconds': checkpoint_seconds,
        'peak_rss_bytes': usage.ru_maxrss * 1024,
        'user_seconds': usage.ru_utime, 'system_seconds': usage.ru_stime,
        'block_inputs': usage.ru_inblock, 'block_outputs': usage.ru_oublock,
        'minor_faults': usage.ru_minflt, 'major_faults': usage.ru_majflt,
    })
    return helper_code

def live_inputs(home: Path) -> tuple[dict[str, dict[str, int]], dict[str, int]]:
    population: dict[str, dict[str, int]] = {}
    files: dict[str, int] = {}
    for origin, root, pattern in [
        ('claude-code', home / '.claude' / 'projects', '*.jsonl'),
        ('codex', home / '.codex' / 'sessions', 'rollout-*.jsonl'),
        ('codex', home / '.codex' / 'archived_sessions', 'rollout-*.jsonl'),
    ]:
        for path in root.rglob(pattern):
            if not path.is_file():
                continue
            if not path.resolve().is_relative_to(home):
                raise ValueError(f'input leaves the isolated HOME: {path}')
            counts = population.setdefault(origin, {'files': 0, 'bytes': 0})
            counts['files'] += 1
            length = path.stat().st_size
            counts['bytes'] += length
            files[str(path.resolve())] = length
    return population, files

def captured_locations(archive: Path) -> dict[str, str]:
    with database(archive / 'archive.db') as conn:
        return dict(conn.execute('''
            SELECT path,input FROM observation WHERE rowid IN
            (SELECT max(rowid) FROM observation GROUP BY machine,path)
            ORDER BY rowid
        '''))

def verify_live_sources(archive: Path, files: dict[str, int],
                        expected: dict[str, tuple[int, int]]) -> dict[str, int]:
    locations = captured_locations(archive)
    selected = set()
    for path, length in files.items():
        input_id = locations.get(path)
        if input_id not in expected or expected[input_id][1] != length:
            raise RuntimeError(f'selected source was not retained at its exact length: {path}')
        selected.add(input_id)
    auxiliary = set(expected) - selected
    repo_inputs = {input_id for path, input_id in locations.items()
                   if path.startswith('repo-lookup:')}
    if not auxiliary <= repo_inputs:
        raise RuntimeError('unexpected captured inputs outside the selected files and repo metadata')
    return {'selected_files': len(files), 'selected_bytes': sum(files.values()),
            'auxiliary_inputs': len(auxiliary),
            'auxiliary_bytes': sum(expected[key][1] for key in auxiliary)}

def pipeline_profile(phase: Path) -> dict[str, object] | None:
    records = []
    log = phase / 'daemon.log'
    if not log.exists():
        return None
    with log.open() as source:
        for line in source:
            if not line.lstrip().startswith('{'):
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict) and isinstance(record.get('pipeline_timings'), dict):
                records.append(record)
    if not records:
        return None
    write_json(phase / 'pipeline-profile.json', records)
    return combine_profiles(records)

def combine_profiles(records: list[dict[str, object]]) -> dict[str, object]:
    stages: dict[str, float] = {}
    workers: dict[str, dict[str, float | int | None]] = {}
    for record in records:
        for name, seconds in record.get('pipeline_timings', {}).items():
            stages[name] = stages.get(name, 0.0) + seconds
        entries = record.get('workers', [])
        if isinstance(entries, dict):
            # This is an already combined phase profile.
            entries = [dict(name=name, **entry) for name, entry in entries.items()]
        for worker in entries:
            name = worker['name']
            target = workers.setdefault(name, {'lifetimes': 0, 'seconds': 0.0,
                                               'cpu_seconds': 0.0, 'run_queue_seconds': 0.0})
            target['lifetimes'] += worker.get('lifetimes', 1)
            target['seconds'] += worker['seconds']
            for field in ('cpu_seconds', 'run_queue_seconds'):
                value = worker.get(field)
                target[field] = None if value is None or target[field] is None else target[field] + value
    return {'pipeline_timings': stages, 'workers': workers}

def capture_unit(unit: str, phase: Path) -> dict[str, object]:
    properties = show_unit(unit)
    write_json(phase / 'unit.json', properties)
    group = properties.get('ControlGroup', '')
    counters = {}
    if group:
        for name in ('memory.peak', 'memory.swap.peak', 'memory.events', 'memory.stat', 'cpu.stat', 'io.stat'):
            try:
                counters[name] = (Path('/sys/fs/cgroup') / group.lstrip('/') / name).read_text()
            except FileNotFoundError:
                pass
    write_json(phase / 'cgroup.json', counters)
    return {'unit': unit, 'properties': properties, 'cgroup': counters}

def stop_daemon(pid: int, unit: str) -> None:
    # A pidfd prevents a reused process number from receiving this signal.
    fd = os.pidfd_open(pid)
    try:
        membership = Path(f'/proc/{pid}/cgroup').read_text()
        if not any(Path(line.split('::', 1)[1]).name == unit
                   for line in membership.splitlines() if line.startswith('0::')):
            raise RuntimeError('the daemon PID does not belong to the owned measurement unit')
        signal.pidfd_send_signal(fd, signal.SIGTERM)
    finally:
        os.close(fd)

REQUIRED_PARTS = {
    'writer_other', 'queue_wait', 'search_turn_wait', 'raw_other',
    'segment_append', 'segment_sync_wait', 'register_rows', 'register_commit',
    'row_transaction', 'row_other', 'version_text', 'session_facts',
    'message_dispatch', 'message_header', 'message_claim', 'selected_message',
    'selected_blocks', 'wire_shapes', 'parser_context', 'parse_state', 'counts',
    'links', 'items', 'session_values', 'cleanup', 'index_commit',
    'chain_progress', 'wal_restart', 'search_build', 'trigram_build',
    'secondary_build', 'install', 'derive_read', 'derive_apply',
}


def generated_population(home: Path, expected: dict) -> dict[str, int]:
    population, files = live_inputs(home)
    if population != {'claude-code': {'files': expected['files'], 'bytes': expected['bytes']}}:
        raise RuntimeError(f'the generated population differs: {population}')
    project = home / '.claude/projects/-generated'
    for sample in expected['samples']:
        path = project / sample['name']
        if path.stat().st_size != sample['bytes']:
            raise RuntimeError('a synthetic control file has a different size')
        if hashlib.sha256(path.read_bytes()).hexdigest() != sample['sha256']:
            raise RuntimeError('a synthetic control file has different generated bytes')
    return files


def summarize(profile: dict, records: list[dict]) -> dict:
    if len(records) != 1:
        raise RuntimeError('expected one complete daemon profile')
    parts = records[0].get('first_build_attribution', {})
    if not REQUIRED_PARTS <= parts.keys():
        raise RuntimeError(f'missing attribution scopes: {REQUIRED_PARTS - parts.keys()}')
    if any(part['cpu_seconds'] is None for part in parts.values()):
        raise RuntimeError('an attribution CPU clock is unavailable')
    writer = profile['workers']['polylogue-writer']['cpu_seconds']
    total = sum(value['cpu_seconds'] for name, value in parts.items() if name != 'derive_read')
    return {'parts': parts, 'writer_cpu_seconds': writer,
            'exclusive_writer_cpu_seconds': total,
            'writer_cpu_residual_seconds': writer - total,
            'writer_cpu_coverage': total / writer,
            'worker_cpu_seconds': {name: worker['cpu_seconds']
                                   for name, worker in profile['workers'].items()},
            'overlapping_stage_seconds': profile['pipeline_timings']}


def available_memory() -> int:
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) * 1024
    raise RuntimeError('MemAvailable is unavailable')


def available_space(path: Path) -> int:
    value = os.statvfs(path)
    return value.f_bavail * value.f_frsize


def main(args: argparse.Namespace) -> int:
    if os.environ.get('GITHUB_ACTIONS') != 'true':
        raise RuntimeError('this temporary entry point is for a hosted workflow only')
    work = Path(args.work).resolve()
    runner_temp = Path(os.environ['RUNNER_TEMP']).resolve()
    if not work.is_relative_to(runner_temp) or work == runner_temp or work.exists():
        raise RuntimeError('the diagnostic needs a fresh directory inside RUNNER_TEMP')
    print('HOSTED_PREFLIGHT_JSON=' + json.dumps({'free_bytes': available_space(runner_temp), 'MemAvailable': available_memory(), 'available_cpus': len(os.sched_getaffinity(0))}), flush=True)
    if available_space(runner_temp) < 32 * 1024**3:
        raise RuntimeError('less than 32 GiB is available for generation and build output')
    if available_memory() < 8 * 1024**3:
        raise RuntimeError('less than 8 GiB MemAvailable before preparation')
    command(['sudo', '-n', 'true'])
    binary = Path(args.binary).resolve(strict=True)
    version = command([str(binary), '--version'], timeout=5).stdout.strip()
    if args.revision not in version:
        raise RuntimeError('the binary does not report the selected revision')
    work.mkdir()
    results = work / 'results'
    results.mkdir()
    phase = results / 'phase-1'
    phase.mkdir()
    (results / 'tmp').mkdir()
    (phase / 'runtime').mkdir(mode=0o700)
    home = work / 'corpus/home'
    archive = work / 'archive'
    expected = json.loads(Path(args.expected).read_text())
    generated = command([args.generator, '--synthetic-lengths', args.lengths,
                         '--dir', str(work / 'corpus'), '--generate-only'])
    (results / 'generation.log').write_text(generated.stdout)
    files = generated_population(home, expected)
    for path in files:
        with open(path, 'rb') as source:
            os.fsync(source.fileno())
            os.posix_fadvise(source.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    stamps = {path: (Path(path).stat().st_ino, Path(path).stat().st_size,
                     Path(path).stat().st_mtime_ns) for path in files}
    if available_space(work) < 24 * 1024**3 or available_memory() < 8 * 1024**3:
        raise RuntimeError('insufficient disk or available RAM after generation')
    cpus = min(8, len(os.sched_getaffinity(0)))
    run_id = os.environ['GITHUB_RUN_ID']
    attempt = os.environ['GITHUB_RUN_ATTEMPT']
    if not run_id.isdecimal() or not attempt.isdecimal():
        raise RuntimeError('invalid workflow identity')
    unit = f'polylogue-first-build-cpu-{run_id}-{attempt}.service'
    existing = show_unit(unit)
    if existing.get('LoadState') != 'not-found':
        raise RuntimeError(f'the system manager is unavailable or the exact unit exists: {existing}')
    summary = {'kind': 'hosted_writer_cpu_attribution', 'revision': args.revision,
               'diagnostic_source': command(['git', 'rev-parse', 'HEAD']).stdout.strip(),
               'measurement_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'generator_sha256': hashlib.sha256(Path(args.generator).read_bytes()).hexdigest(),
               'binary': str(binary), 'version': version,
               'input_files': expected['files'], 'input_bytes': expected['bytes'],
               'cpu_quota': cpus, 'memory_max_bytes': 4 * 1024**3,
               'swap_max_bytes': 1024**3, 'timer_overhead_included': True,
               'appliance_timing_or_release_acceptance': False,
               'host_cpu': Path('/proc/cpuinfo').read_text().split('model name', 1)[-1].splitlines()[0],
               'before': {'MemAvailable': available_memory(), 'free_bytes': available_space(work)},
               'source_files_unchanged': False, 'complete': False, 'failure': None}
    env = {'HOME': str(home), 'PATH': os.environ['PATH'], 'LANG': 'C.UTF-8',
           'USER': os.environ.get('USER', 'runner'),
           'XDG_CONFIG_HOME': str(phase / 'config'), 'XDG_DATA_HOME': str(phase / 'data'),
           'XDG_STATE_HOME': str(phase / 'state'), 'XDG_CACHE_HOME': str(phase / 'cache'),
           'XDG_RUNTIME_DIR': str(phase / 'runtime'), 'TMPDIR': str(results / 'tmp'),
           'MIMALLOC_SHOW_STATS': '1', 'POLYLOGUE_TIMING_WORKERS': '1'}
    argv = ['sudo', '-n', 'systemd-run', '--quiet', '--service-type=exec',
            '--remain-after-exit', '--unit=' + unit,
            '--uid=' + str(os.getuid()), '--gid=' + str(os.getgid()),
            '--working-directory=' + str(results),
            '--property=MemoryAccounting=yes', '--property=CPUAccounting=yes',
            '--property=IOAccounting=yes', '--property=MemoryMax=4G',
            '--property=MemorySwapMax=1G', '--property=CPUQuota=' + str(cpus * 100) + '%',
            '--property=TimeoutStopSec=15min', '--property=KillMode=control-group',
            '--property=OOMPolicy=kill', '--property=OOMScoreAdjust=900',
            '--property=StandardOutput=append:' + str(phase / 'daemon.log'),
            '--property=StandardError=append:' + str(phase / 'daemon.log'),
            '/usr/bin/env', '-i', *[f'{key}={value}' for key, value in env.items()],
            sys.executable, '-I', str(Path(__file__).resolve()),
            '--binary', str(binary), '--archive', str(archive), '--child', str(phase)]
    launched = False
    started = time.monotonic()
    summary['started_at_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    try:
        command(argv)
        launched = True
        deadline = started + 7200
        stop_reason = None
        stop_sent = False
        while True:
            receipt = read_json(phase / 'daemon-exit.json')
            if receipt is not None:
                break
            properties = show_unit(unit)
            if properties.get('ActiveState') not in ('active', 'activating'):
                raise RuntimeError(f'daemon unit ended without wait4 receipt: {properties}')
            if stop_reason is None and (time.monotonic() > deadline or available_space(work) < 4 * 1024**3):
                stop_reason = 'diagnostic time or free-space bound'
            if stop_reason is not None and not stop_sent:
                pid = read_json(phase / 'daemon-pid.json')
                if pid:
                    stop_daemon(pid['pid'], unit)
                    stop_sent = True
            if time.monotonic() > deadline + 900:
                raise RuntimeError('the diagnostic did not finish its bounded stop')
            time.sleep(1)
        summary['normal_completion_observed_at_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        summary['durable_seconds_diagnostic_only'] = time.monotonic() - started
        summary['daemon'] = receipt
        if stop_reason or receipt['exit'] != 0 or receipt['daemon_exit'] != 0:
            raise RuntimeError(f'incomplete diagnostic: {stop_reason or receipt}')
        if receipt['checkpoint'] is None or receipt['checkpoint_error'] is not None:
            raise RuntimeError('final checkpoint was not completed')
        if any(row['wal_after'] for row in receipt['checkpoint']['databases']):
            raise RuntimeError('a final WAL is not empty')
        expected_inputs = newest_inputs(archive)
        summary['retention'] = verify_live_sources(archive, files, expected_inputs)
        summary['verification'] = verify_complete(archive, expected_inputs)
        profile = pipeline_profile(phase)
        if profile is None:
            raise RuntimeError('the daemon did not report its worker profile')
        summary['workers'] = profile['workers']
        summary['overlapping_stage_seconds'] = profile['pipeline_timings']
        records = json.loads((phase / 'pipeline-profile.json').read_text())
        if any('first_build_attribution' in record for record in records):
            summary.update(summarize(profile, records))
        summary['source_files_unchanged'] = all(
            (Path(path).stat().st_ino, Path(path).stat().st_size, Path(path).stat().st_mtime_ns) == stamp
            for path, stamp in stamps.items())
        if not summary['source_files_unchanged']:
            raise RuntimeError('generated inputs changed during the run')
        summary['complete'] = True
    except BaseException as error:
        summary['failure'] = repr(error)
    finally:
        if launched:
            summary['unit'] = capture_unit(unit, phase)
            stopped = command(['sudo', '-n', 'systemctl', 'stop', unit], check=False, timeout=910)
            summary['stop_exit'] = stopped.returncode
            summary['child_after_stop'] = show_unit(unit)
            command(['sudo', '-n', 'systemctl', 'reset-failed', unit], check=False)
            if stopped.returncode or summary['child_after_stop'].get('MainPID') != '0':
                summary['failure'] = 'the exact owned unit did not stop'
                summary['complete'] = False
        summary['controller_end_at_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        write_json(results / 'hosted-summary.json', summary)
        print('HOSTED_FIRST_BUILD_JSON=' + json.dumps(summary), flush=True)
    return 0 if summary['complete'] and summary['failure'] is None else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', required=True)
    parser.add_argument('--revision')
    parser.add_argument('--work')
    parser.add_argument('--generator')
    parser.add_argument('--lengths')
    parser.add_argument('--expected')
    parser.add_argument('--archive')
    parser.add_argument('--child', help=argparse.SUPPRESS)
    parser.add_argument('--stored', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    sys.exit(child(args) if args.child else main(args))
