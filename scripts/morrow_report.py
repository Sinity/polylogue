"""Render retained Morrow run data without changing its qualification result."""
import argparse
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
import shlex
from string import Template
from urllib.parse import quote


def text(value):
    return escape(str(value), quote=True)


def artifact(root, name, label=None):
    """Only link existing files in this report tree, including after relocation."""
    if not isinstance(name, str) or not name or '\\' in name or ':' in name:
        return '<span class="muted">No artifact recorded</span>'
    path = Path(name)
    if path.is_absolute() or '..' in path.parts:
        return '<span class="muted">Artifact outside report tree</span>'
    try:
        target = (root / path).resolve(strict=True)
        if not target.is_relative_to(root.resolve()) or not target.is_file():
            raise ValueError('not a report file')
    except (OSError, ValueError):
        return f'<span class="muted">{text(label or name)} (unavailable)</span>'
    return f'<a href="{text(quote(path.as_posix(), safe="/"))}">{text(label or name)}</a>'


def detail(label, value):
    return f'<details><summary>{text(label)}</summary><pre>{text(json.dumps(value, indent=2))}</pre></details>'


def render_report(result, root, source='result.json'):
    root = Path(root)
    generated = datetime.now(timezone.utc).isoformat()
    steps = result.get('steps', {})
    state = 'passed' if result.get('success') is True else 'failed'
    rows = []
    for name, step in steps.items():
        outcome = step.get('outcome', 'unrun')
        reason = step.get('failure')
        if outcome == 'unrun':
            reason = 'Not executed; no answer or acceptance is inferred.'
        rows.append(f'<tr data-outcome="{text(outcome)}"><th scope="row">{text(name)}</th>'
                    f'<td class="{text(outcome)}">{text(outcome)}</td><td>{text(reason or "")}</td></tr>')
    cases = result.get('expected', {}).get('cases', {})
    answers = []
    for name in ('base', 'repeat', 'append', 'restart', 'empty', 'malformed'):
        expected_case = 'base' if name == 'repeat' else 'append' if name == 'restart' else name
        expected = cases.get(expected_case)
        links = ' · '.join(artifact(root, f'{name}-{kind}.json', kind)
                           for kind in ('query', 'actual', 'corrupted'))
        answers.append(f'<article><h3>{text(name)}</h3><p>{links}</p>'
                       + (detail('Independent expectation', expected) if expected is not None
                          else '<p>Expectation not prepared.</p>') + '</article>')
    commands = []
    for index, command in enumerate(result.get('commands', [])):
        code = command.get('exit')
        shown = 'not observed' if code is None else str(code)
        logs = ' · '.join(artifact(root, command.get(k), k) for k in ('stdout', 'stderr'))
        argv = detail('Exact argv', command.get('argv', []))
        failure = f'<p>{text(command["failure"])}</p>' if command.get('failure') else ''
        commands.append(f'<tr><td>{index}</td><td>{text(command.get("label", ""))}{argv}{failure}</td>'
                        f'<td>{text(shown)}</td><td>{logs}</td></tr>')
    processes = []
    for process in result.get('owned_processes', []):
        processes.append(detail(f'Owned PID {process.get("pid", "unknown")}', process))
    failures = []
    for key in ('failure', 'cleanup_failure', 'binding_failure', 'report_failure'):
        if result.get(key):
            failures.append(f'<p><strong>{text(key)}:</strong> {text(result[key])}</p>')
    if not failures:
        failures.append('<p>No failure recorded. The remaining acceptance below still applies.</p>')
    invocation = result.get('invocation')
    cwd = result.get('invocation_cwd')
    if isinstance(invocation, list) and isinstance(cwd, str):
        invocation = ['env', '-C', cwd, *invocation]
    repeat = (f'<pre>{text(shlex.join(invocation))}</pre><p>Choose a new output directory when repeating; the recorded candidate checkout must be available.</p>'
              if isinstance(invocation, list) and all(isinstance(x, str) for x in invocation)
              else '<p>Exact driver invocation was not recorded in this older run.</p>')
    candidate = result.get('candidate', {})
    values = dict(
        state=state, state_class='ok' if state == 'passed' else 'bad',
        generated=text(generated), data_as_of=text(result.get('finished_at') or 'Not recorded in this run'),
        source=artifact(root, source, 'Canonical result JSON'),
        candidate=text(candidate.get('head', 'Not recorded')), spec=text(result.get('spec', {}).get('revision', 'Not recorded')),
        failures=''.join(failures), steps=''.join(rows), answers=''.join(answers),
        commands=''.join(commands), processes=''.join(processes) or '<p>No owned process recorded.</p>',
        remaining=''.join(f'<li>{text(x)}</li>' for x in result.get('remaining', [])),
        control=detail('Recorded negative-control state', {
            'requested':result.get('control'), 'exercised':result.get('control_exercised'),
            'result':result.get('control_result', 'Not recorded in this older run')}),
        binding=detail('Candidate and executable identity', candidate), repeat=repeat,
    )
    template = Template(Path(__file__).with_name('morrow_report.html').read_text())
    output = root / 'report.html'
    output.write_text(template.substitute(values))
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result', type=Path)
    args = parser.parse_args()
    print(render_report(json.loads(args.result.read_text()), args.result.parent, args.result.name))
