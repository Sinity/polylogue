//! `polylogue status` as text: a headline saying what the daemon is doing
//! and what most needs attention, a line of daemon facts, then sections with
//! their labels in a left column, each left out when it has nothing to say.
//! Color marks the state (green current, yellow working or worth a look, red
//! failed), puts the step running in bold and dims what is context. The
//! plain style has the same layout without color, with ASCII bars.
use crate::protocol::status::{Answer, Mode, Notices, Running};
use command_surface::TextStyle;
use polylogue_model::freshness::{CompletenessDimension, CompletenessReason, IndexFreshness};
use polylogue_render::short_duration;
use polylogue_render::terminal::escape as clean;
use polylogue_store::archive::build_progress::CountUnit;
use polylogue_store::log::Priority;

/// The label column's width.
const LABEL: usize = 11;
/// The widest a progress bar gets.
const BAR: usize = 30;
/// Log records the view shows; `--log` prints every kept one.
const RECENT: usize = 5;
/// Seconds without progress after which a busy thread says so.
const PROGRESS_SHOWN_AFTER: f64 = 10.;
/// Seconds without progress after which a busy thread is stalled
/// (docs/operations.md, the halt state).
const STALLED_AFTER: f64 = 600.;

/// How the view is drawn.
pub(crate) struct View<'a> {
    pub style: TextStyle,
    /// Columns available; lines longer than this are cut.
    pub width: usize,
    pub now_ms: i64,
    pub zone: &'a jiff::tz::TimeZone,
    /// Print every log record whole instead of cutting them to the width.
    pub whole_log: bool,
}

/// The view of `status` with `problems`, the answer's degraded reasons and
/// notes.
pub(crate) fn render(status: &Answer, problems: &[String], view: &View<'_>) -> String {
    let building = status.index_build.is_some();
    // While an index is built, its section says what that means for reads,
    // and the last check of the sources is from before it.
    let index_building = CompletenessReason::IndexBuilding.text();
    let notes: Vec<&String> = problems
        .iter()
        .filter(|note| !(building && note.as_str() == index_building))
        .collect();
    let mut out = String::new();
    for (n, line) in header(status, &notes, view).iter().enumerate() {
        let label = if n == 0 {
            view.label(&format!("{:<LABEL$}", "polylogue"))
        } else {
            " ".repeat(LABEL)
        };
        out.push_str(&label);
        out.push_str(&view.cut_to(line, view.width.saturating_sub(LABEL)));
        out.push('\n');
    }
    let sections: [(&str, Vec<String>); 14] = [
        ("ARCHIVE", archive(status, view)),
        (
            "READING",
            if building {
                Vec::new()
            } else {
                reading(status, view)
            },
        ),
        ("INDEX", index(status, view)),
        ("PRIMARY", primary(status, view)),
        (
            "LAST CHECK",
            if building {
                Vec::new()
            } else {
                last_check(status, view)
            },
        ),
        ("RUNNING", running(status, view)),
        ("NOTICES", notices(status, view)),
        ("RULES", rules(status, view)),
        ("PENDING", pending(status, view)),
        ("CAPTURE", capture(status, view)),
        ("UNUSED", unused(status)),
        ("THREADS", threads(status, view)),
        ("RECENT", recent(status, view)),
        (
            "NOTES",
            notes.iter().map(|note| view.warning(note)).collect(),
        ),
    ];
    for (label, lines) in sections {
        if lines.is_empty() {
            continue;
        }
        out.push('\n');
        for (n, line) in lines.iter().enumerate() {
            if line.is_empty() {
                out.push('\n');
                continue;
            }
            let label = if n == 0 { label } else { "" };
            let padded = format!("{label:<LABEL$}");
            let value = if label == "RECENT" && view.whole_log {
                line.clone()
            } else {
                view.cut_to(line, view.width.saturating_sub(LABEL))
            };
            out.push_str(&view.label(&padded));
            out.push_str(&value);
            out.push('\n');
        }
    }
    out
}

/// The headline, then the daemon's facts dimmed: what it is doing (blocked,
/// building the index, warming) or whether the reply is current, and
/// otherwise the first note.
fn header(status: &Answer, notes: &[&String], view: &View<'_>) -> Vec<String> {
    let headline = if let Mode::Blocked { path, error, .. } = &status.mode {
        view.paint(
            "1;31",
            &format!(
                "✖ blocked: {}: {}",
                clean(&path.display().to_string()),
                clean(error)
            ),
        )
    } else if status.index_build.is_some() {
        let progress = status.build_progress.as_ref();
        let mut words = "building the index".to_owned();
        if let Some(progress) = progress
            && progress.current.is_some()
        {
            words.push_str(&format!(
                ", step {} of {}",
                progress.finished.len() + 1,
                progress.finished.len() + 1 + progress.to_come.len()
            ));
        }
        if progress.is_some_and(|progress| progress.failed_attempt.is_some()) {
            words.push_str(", again after a failed attempt");
        }
        view.paint("1;33", &format!("◐ {words}"))
    } else if status.warming {
        view.paint("1;33", "◐ warming up after the index was installed")
    } else if let Some(first) = notes.first() {
        let more = match notes.len() {
            1 => String::new(),
            n => format!("  (+{} more under NOTES)", n - 1),
        };
        view.paint("1;33", &format!("⚠ {}{more}", clean(first)))
    } else {
        view.paint("1;32", "✓ up to date")
    };
    let mode = match &status.mode {
        Mode::Standalone => "standalone",
        Mode::Blocked { .. } => "blocked",
    };
    let mut parts = vec![
        mode.to_owned(),
        format!("daemon {}", status.pid),
        format!(
            "up {}",
            short_duration(i64::try_from(status.uptime_seconds).unwrap_or(i64::MAX) * 1000)
        ),
        if status.loop_status.busy {
            "busy"
        } else {
            "idle"
        }
        .to_owned(),
    ];
    if let Some(archive) = &status.archive {
        parts.push(clean(&archive.display().to_string()));
    }
    vec![headline, view.dim(&parts.join(" · "))]
}

fn archive(status: &Answer, view: &View<'_>) -> Vec<String> {
    let mut lines = Vec::new();
    if let Some(store) = &status.store {
        lines.push(format!(
            "{} sessions   {} messages   {} tool calls   {} documents",
            count(store.sessions),
            count(store.messages),
            count(store.tool_calls),
            count(store.documents)
        ));
        lines.push(format!(
            "{} inputs · {} stored · {} revisions",
            count(store.inputs),
            bytes(store.stored_bytes),
            count(store.revisions)
        ));
    } else if status.index_build.is_none()
        && let Some(error) = &status.totals_error
    {
        lines.push(view.error(&format!("totals unavailable: {error}")));
    }
    if let Some(vectors) = &status.vectors {
        for key in &vectors.keys {
            let mut line = match key.dims {
                Some(dims) => format!(
                    "vectors {}: {} for {} texts, {dims} dimensions",
                    clean(&key.key),
                    count(key.vectors),
                    count(key.texts),
                ),
                None => format!("vectors {}: none stored", clean(&key.key)),
            };
            if key.failures > 0 {
                line.push_str(&format!(" · {} failed", count(key.failures)));
            }
            lines.push(line);
        }
    }
    if status.index_build.is_none()
        && let Some(error) = &status.vectors_error
    {
        lines.push(view.error(&format!("vector counts unavailable: {error}")));
    }
    lines
}

/// The parse states as the view words them, in the order it lists them.
const PARSE_STATES: [(&str, &str); 6] = [
    ("parsing", "parsing"),
    ("waiting", "waiting to grow"),
    ("partial", "partly read"),
    ("refused", "refused"),
    ("failed", "failed"),
    ("no_parser_yet", "without a parser"),
];

fn reading(status: &Answer, view: &View<'_>) -> Vec<String> {
    let Some(store) = &status.store else {
        return Vec::new();
    };
    let mut lines = Vec::new();
    let total: u64 = store.parse_states.values().sum();
    if total > 0 {
        let done = store.parse_states.get("done").copied().unwrap_or(0);
        lines.push(format!(
            "{}  {}   {} of {} files read",
            view.bar(done, total),
            percent(done, total),
            count(done),
            count(total)
        ));
        let mut others: Vec<String> = PARSE_STATES
            .iter()
            .filter_map(|(state, words)| {
                store
                    .parse_states
                    .get(*state)
                    .map(|n| format!("{} {words}", count(*n)))
            })
            .collect();
        others.extend(
            store
                .parse_states
                .iter()
                .filter(|(state, _)| {
                    *state != "done" && !PARSE_STATES.iter().any(|(known, _)| known == state)
                })
                .map(|(state, n)| format!("{} {}", count(*n), clean(state))),
        );
        if !others.is_empty() {
            lines.push(others.join(" · "));
        }
    }
    let local = &status.freshness.local;
    let mut words = Vec::new();
    for (name, dimension) in [
        ("parsing", &local.parsing),
        ("session figures", &local.derivation),
        ("search", &local.search),
        ("Sinex stream", &local.confirmed_stream),
    ] {
        let reasons = |reasons: &[polylogue_model::freshness::CompletenessReason]| {
            reasons
                .iter()
                .map(|reason| reason.text())
                .collect::<Vec<_>>()
                .join(", ")
        };
        match dimension {
            CompletenessDimension::Complete => words.push(format!("{name} current")),
            CompletenessDimension::Incomplete(found) => {
                // The reasons say what is behind.
                words.push(reasons(found));
            }
            CompletenessDimension::Unknown(found) => {
                words.push(format!("{name} unknown: {}", reasons(found)));
            }
            CompletenessDimension::NotApplicable => {}
        }
    }
    if store.search_pending > 0 {
        words.push(format!(
            "{} texts awaiting search",
            count(store.search_pending)
        ));
    }
    lines.push(words.join(" · "));
    lines
}

/// An index build: what it does and why, what it means for reads, the
/// running step's bar, then the steps (finished with their time and count,
/// the running one in bold with its count, rate and estimate, those to come)
/// and the time left.
fn index(status: &Answer, view: &View<'_>) -> Vec<String> {
    use polylogue_store::archive::index_rebuild::IndexRebuildReason;
    let Some(build) = &status.index_build else {
        return Vec::new();
    };
    let progress = status.build_progress.as_ref();
    let why = match progress.and_then(|progress| progress.reason.as_ref()) {
        None => "the archive had no index".to_owned(),
        Some(IndexRebuildReason::SchemaDiffers) => "its tables changed".to_owned(),
        Some(IndexRebuildReason::FormatVersion { found, wanted }) => {
            format!("its format changed ({found} to {wanted})")
        }
        Some(IndexRebuildReason::OlderFormat) => "it was written by an older Polylogue".to_owned(),
    };
    let mut lines = vec![if build.bytes > 0 {
        format!(
            "building the index from {} of stored files: {why}",
            bytes(build.bytes)
        )
    } else {
        format!("building the index: {why}")
    }];
    lines.push(view.dim(if build.serving_previous {
        "reads and searches use the previous index until the new one is installed"
    } else {
        "reads and searches answer \"index building\" until it is installed"
    }));
    let Some(progress) = progress else {
        return lines;
    };
    if let Some(failed) = &progress.failed_attempt {
        let during = failed
            .step
            .as_ref()
            .map(|step| format!(" while {}", clean(step)))
            .unwrap_or_default();
        lines.push(view.error(&format!(
            "an earlier attempt failed{during}: {}; started over",
            failed.error
        )));
    }
    if let Some(current) = &progress.current
        && let Some(count) = &current.count
        && let Some(total) = count.total.filter(|total| *total > 0)
    {
        let mut line = format!(
            "{}  {} of {}",
            view.bar(count.done, total),
            view.bold(&percent(count.done, total)),
            clean(&current.name)
        );
        if let Some(eta) = current.eta_seconds {
            line.push_str(&format!(" · {}", view.bold(&left(eta))));
        }
        lines.push(line);
    }
    lines.push(String::new());
    let width = progress
        .finished
        .iter()
        .map(|step| step.name.chars().count())
        .chain(
            progress
                .current
                .iter()
                .map(|step| step.name.chars().count()),
        )
        .chain(progress.to_come.iter().map(|step| step.chars().count()))
        .max()
        .unwrap_or(0);
    for step in &progress.finished {
        let mut line = format!(
            "{:<width$} {:>6}",
            clean(&step.name),
            short_duration((step.seconds * 1000.) as i64)
        );
        if let Some(count) = &step.count {
            line.push_str(&format!("  {}", amount(count.done, count.unit)));
        }
        lines.push(format!("{} {}", view.paint("32", "✓"), view.dim(&line)));
    }
    if let Some(current) = &progress.current {
        let mut parts = Vec::new();
        if let Some(count) = &current.count {
            parts.push(match count.total {
                Some(total) => format!(
                    "{} of {}",
                    amount_number(count.done, count.unit),
                    amount(total, count.unit)
                ),
                None => format!("{} so far", amount(count.done, count.unit)),
            });
        }
        // Reading counts bytes; the files read go beside them.
        if current
            .count
            .as_ref()
            .is_some_and(|count| count.unit == CountUnit::Bytes)
            && build.chains > 0
        {
            parts.push(format!(
                "{} of {} files",
                count(build.chains_read),
                count(build.chains)
            ));
        }
        let line = format!(
            "{:<width$} {:>6}  {}",
            clean(&current.name),
            short_duration(view.now_ms.saturating_sub(current.since.unix_ms())),
            parts.join(" · ")
        );
        lines.push(format!(
            "{} {}",
            view.paint("1;36", "▸"),
            view.bold(line.trim_end())
        ));
        // The rate and estimate on a line of their own, so a narrow
        // terminal does not cut them.
        let mut parts = Vec::new();
        if let (Some(rate), Some(counted)) = (current.per_second, &current.count) {
            parts.push(if rate > 0. {
                match counted.unit {
                    CountUnit::Bytes => format!("{:.1} MB/s", rate / 1e6),
                    unit if rate >= 1000. => format!("{} {}/s", count(rate as u64), unit.word()),
                    unit => format!("{} {}/s", decimal(rate), unit.word()),
                }
            } else {
                let over = short_duration((current.rate_seconds.unwrap_or(0.) * 1000.) as i64);
                view.warning_text(&match counted.unit {
                    CountUnit::Bytes => format!("nothing read in the last {over}"),
                    unit => format!("no {} done in the last {over}", unit.word()),
                })
            });
        }
        // An estimate is on the bar's line; without one, this says why.
        if current.eta_seconds.is_none() {
            if current.rate_varying {
                parts.push("rate varying, no estimate".into());
            } else if let Some(count) = &current.count {
                if count.total.is_none() {
                    parts.push("total known when the step ends".into());
                } else if current.rate_seconds.is_none_or(|seconds| seconds < 60.) {
                    parts.push("estimate after a minute of progress".into());
                }
            }
        }
        if !parts.is_empty() {
            lines.push(format!("  {:<width$} {:>6}  {}", "", "", parts.join(" · ")));
        }
    }
    for step in &progress.to_come {
        lines.push(view.dim(&format!("· {}", clean(step))));
    }
    lines.push(String::new());
    if let Some(eta) = progress.eta_seconds {
        lines.push(format!("whole build: {}", view.bold(&left(eta))));
    } else if !progress.to_come.is_empty() {
        let n = progress.to_come.len();
        lines.push(format!(
            "whole build: time left known once the last step runs ({n} {} to come)",
            if n == 1 { "step" } else { "steps" },
        ));
    }
    let mut parts = Vec::new();
    if let Some(busy) = progress.writer_busy {
        parts.push(format!(
            "database writer busy {:.0}% of the last minute",
            busy * 100.
        ));
    }
    if !progress.worker_cpu_seconds.is_empty() {
        let workers = progress.worker_cpu_seconds.len();
        let each = progress.worker_cpu_seconds.iter().sum::<f64>() / workers as f64;
        parts.push(format!(
            "{workers} parse {}, {each:.1} s CPU/min each",
            if workers == 1 { "worker" } else { "workers" }
        ));
    }
    if !parts.is_empty() {
        lines.push(view.dim(&parts.join(" · ")));
    }
    while lines.last().is_some_and(String::is_empty) {
        lines.pop();
    }
    lines
}

/// An amount of `unit`: a size for bytes, a count and the unit's word
/// otherwise.
fn amount(n: u64, unit: CountUnit) -> String {
    match unit {
        CountUnit::Bytes => bytes(n),
        unit => format!("{} {}", count(n), unit.word()),
    }
}

/// [`amount`] without the unit's word, which the total after it carries.
fn amount_number(n: u64, unit: CountUnit) -> String {
    match unit {
        CountUnit::Bytes => bytes(n),
        _ => count(n),
    }
}

fn primary(status: &Answer, view: &View<'_>) -> Vec<String> {
    let mut lines = Vec::new();
    if let IndexFreshness::Primary {
        applied_position,
        lag_positions,
        state,
        error,
        ..
    } = &status.freshness.sinex
    {
        let mut parts = vec![format!("Sinex {}", state.as_str())];
        if let Some(applied) = applied_position {
            parts.push(format!("applied position {applied}"));
        }
        if let Some(lag) = lag_positions {
            parts.push(format!(
                "{} positions behind",
                count(u64::try_from(*lag).unwrap_or(0))
            ));
        }
        lines.push(parts.join(" · "));
        if let Some(error) = error {
            lines.push(view.error(error));
        }
    }
    lines
}

/// The daemon's last check of its sources for new and changed files.
fn last_check(status: &Answer, view: &View<'_>) -> Vec<String> {
    let looped = &status.loop_status;
    if looped.passes == 0 {
        return Vec::new();
    }
    let pass = &looped.last_pass;
    let mut parts = Vec::new();
    if let Some(at) = looped.last_pass_at {
        parts.push(view.time(at.unix_ms()));
    }
    parts.push(format!("{} s", decimal(pass.seconds)));
    parts.push(format!(
        "{} of {} files changed",
        count(pass.inputs_changed),
        count(pass.inputs_seen)
    ));
    let mut work = vec![
        format!("{} read", bytes(pass.bytes_read)),
        format!("{} messages written", count(pass.messages_written)),
    ];
    if pass.sessions_derived > 0 {
        work.push(format!(
            "figures computed for {} sessions",
            count(pass.sessions_derived)
        ));
    }
    let mut lines = vec![parts.join(" · "), work.join(" · ")];
    if let Some(first) = pass.errors.first() {
        let more = match pass.errors.len() {
            1 => String::new(),
            n => format!(" (and {} more)", n - 1),
        };
        lines.push(view.error(&format!("{first}{more}")));
    }
    if !pass.unreadable_roots.is_empty() {
        let roots: Vec<String> = pass
            .unreadable_roots
            .iter()
            .map(|root| root.display().to_string())
            .collect();
        lines.push(view.warning(&format!("unreadable: {}", roots.join(", "))));
    }
    lines
}

fn running(status: &Answer, view: &View<'_>) -> Vec<String> {
    match &status.running {
        Running::Measured { sessions, .. } if !sessions.is_empty() => {
            let mut lines = vec![status.running.line(view.now_ms)];
            for session in sessions {
                use polylogue_model::live::LiveState;
                let mark = match session.state {
                    LiveState::Waiting => "◆",
                    LiveState::Working => "●",
                    _ => "○",
                };
                let age = session
                    .since
                    .map(|since| short_duration(view.now_ms.saturating_sub(since)))
                    .unwrap_or_default();
                let mut detail = Vec::new();
                if let Some(on) = session.waiting_on {
                    detail.push(format!("on {}", on.as_str()));
                }
                if let Some(stall) = session.stall {
                    detail.push(format!("stalled: {}", stall.as_str().replace('_', " ")));
                }
                if let Some(tool) = &session.open_tool {
                    detail.push(clean(tool));
                }
                if let Some(task) = &session.task {
                    detail.push(clean(task));
                }
                if session.edited_files > 0 {
                    detail.push(format!(
                        "{} {} edited",
                        session.edited_files,
                        if session.edited_files == 1 {
                            "file"
                        } else {
                            "files"
                        }
                    ));
                }
                let line = format!(
                    "{mark} {:<8} {age:>5}  {}  {}",
                    session.state.as_str(),
                    clean(&session.session),
                    detail.join(" · ")
                );
                lines.push(line.trim_end().to_owned());
            }
            lines
        }
        Running::Unavailable { error } if status.index_build.is_none() => {
            vec![view.error(&format!("unavailable: {error}"))]
        }
        _ => Vec::new(),
    }
}

fn notices(status: &Answer, view: &View<'_>) -> Vec<String> {
    match &status.notices {
        Notices::Measured { last_day, latest } if *last_day > 0 => {
            let mut line = format!("{last_day} in the last day");
            if let Some(newest) = latest.first() {
                line.push_str(&format!(
                    " · {} {}",
                    view.time(newest.at.unix_ms()),
                    clean(&newest.text)
                ));
            }
            vec![line]
        }
        Notices::Unavailable { error } if status.index_build.is_none() => {
            vec![view.error(&format!("unavailable: {error}"))]
        }
        _ => Vec::new(),
    }
}

/// The signal rules in force, the ones refused and the signals held.
fn rules(status: &Answer, view: &View<'_>) -> Vec<String> {
    use crate::protocol::status::Signals;
    let (rules, unusable, held) = match &status.signals {
        Some(Signals::Observed {
            rules,
            unusable,
            held,
        }) => (rules.as_slice(), unusable.as_slice(), held.as_slice()),
        Some(Signals::Held { held }) => (&[][..], &[][..], held.as_slice()),
        None => return Vec::new(),
    };
    let mut lines: Vec<String> = rules
        .iter()
        .map(|rule| {
            let fired = rule.last_fired.map_or_else(
                || "never fired".to_owned(),
                |at| {
                    format!(
                        "last fired {} ago",
                        short_duration(view.now_ms.saturating_sub(at.unix_ms()))
                    )
                },
            );
            format!(
                "{} · by {} · {fired}",
                clean(&rule.name),
                clean(&rule.author)
            )
        })
        .collect();
    lines.extend(
        unusable
            .iter()
            .map(|rule| view.warning(&format!("rule {} not used: {}", rule.id, rule.reason))),
    );
    lines.extend(
        held.iter()
            .map(|signal| view.warning(&format!("signals held: {}", signal.reason))),
    );
    lines
}

fn pending(status: &Answer, view: &View<'_>) -> Vec<String> {
    let pending = &status.pending_writes;
    let mut parts = Vec::new();
    if pending.delivery_outcomes > 0 {
        parts.push(format!(
            "{} delivery outcomes",
            count(pending.delivery_outcomes)
        ));
    }
    if pending.live_event_lines > 0 {
        parts.push(format!(
            "{} live event lines",
            count(pending.live_event_lines)
        ));
    }
    let mut lines = Vec::new();
    if !parts.is_empty() {
        lines.push(format!("{} awaiting a write", parts.join(" · ")));
    }
    if let Some(error) = &pending.last_error {
        lines.push(view.error(error));
    }
    lines
}

fn capture(status: &Answer, view: &View<'_>) -> Vec<String> {
    use polylogue_model::capture::{Accessibility, Classification};
    let Some(store) = &status.store else {
        return Vec::new();
    };
    let mut lines = Vec::new();
    for coverage in &store.capture_coverage {
        let roots = coverage.roots.len();
        let mut parts = vec![format!(
            "{}: {roots} {} walked {}",
            clean(&coverage.machine),
            if roots == 1 { "root" } else { "roots" },
            view.time(coverage.observed_at.unix_ms())
        )];
        for (state, word) in [
            (Accessibility::Absent, "absent"),
            (Accessibility::Unavailable, "unavailable"),
            (Accessibility::Incomplete, "incomplete"),
        ] {
            let n = coverage
                .roots
                .iter()
                .filter(|root| root.status == state)
                .count();
            if n > 0 {
                parts.push(format!("{n} {word}"));
            }
        }
        let mut ignored = std::collections::BTreeMap::<&str, u64>::new();
        let mut unrecognized = 0;
        for entry in coverage.roots.iter().flat_map(|root| &root.entries) {
            match entry.class {
                Classification::Ignored => {
                    *ignored
                        .entry(entry.reason.as_deref().unwrap_or("ignored"))
                        .or_default() += entry.files;
                }
                Classification::Unrecognized => unrecognized += entry.files,
                Classification::Collected => {}
            }
        }
        let ignored_files: u64 = ignored.values().sum();
        if ignored_files > 0 {
            let (top, _) = ignored
                .iter()
                .max_by_key(|(reason, files)| (**files, std::cmp::Reverse(**reason)))
                .expect("ignored entries are present");
            parts.push(format!(
                "{} files ignored, most as {}",
                count(ignored_files),
                clean(top)
            ));
        }
        if unrecognized > 0 {
            lines.push(parts.join(" · "));
            lines.push(view.warning(&format!(
                "{} unrecognized files on {}",
                count(unrecognized),
                coverage.machine
            )));
        } else {
            lines.push(parts.join(" · "));
        }
    }
    lines
}

fn unused(status: &Answer) -> Vec<String> {
    let Some(store) = &status.store else {
        return Vec::new();
    };
    let Some((field, n)) = store.unused.first() else {
        return Vec::new();
    };
    let more = store.unused.len() - 1;
    let mut line = format!("{} ×{} not yet read", clean(field), count(*n));
    if more > 0 {
        line.push_str(&format!(" · {more} more in --json"));
    }
    vec![line]
}

/// Each busy daemon thread, named without the program's prefix: what it is doing, outermost first, for how
/// long, and how long since it last made progress when that is a while
/// (stalled, marked, after [`STALLED_AFTER`]); under it the statements it
/// has run for a few seconds, running time first since the SQL is cut.
fn threads(status: &Answer, view: &View<'_>) -> Vec<String> {
    let seconds = |seconds: f64| short_duration((seconds * 1000.) as i64);
    let mut lines = Vec::new();
    for thread in &status.threads {
        // Every daemon thread's name starts with the program's.
        let name = clean(
            thread
                .name
                .strip_prefix("polylogue-")
                .unwrap_or(&thread.name),
        );
        let doing: Vec<String> = thread
            .doing
            .iter()
            .map(|work| match &work.subject {
                Some(subject) => format!("{} ({})", work.what, clean(subject)),
                None => work.what.clone(),
            })
            .collect();
        let mut line = match thread.doing.last() {
            Some(innermost) => format!(
                "{name}  {}  {}",
                view.bold(&doing.join(" › ")),
                seconds(innermost.seconds)
            ),
            None => format!("{name}  {}", view.dim("no work recorded")),
        };
        match thread.seconds_since_progress {
            Some(since) if since >= STALLED_AFTER => line.push_str(&format!(
                " · {}",
                view.warning_text(&format!("stalled, no progress for {}", seconds(since)))
            )),
            Some(since) if since >= PROGRESS_SHOWN_AFTER => {
                line.push_str(&format!(" · no progress for {}", seconds(since)));
            }
            _ => {}
        }
        lines.push(line);
        for statement in &thread.statements {
            lines.push(format!(
                "  {} on the {} connection: {}",
                view.warning_text(&format!("running {}", seconds(statement.seconds))),
                statement.connection,
                clean(&statement.sql)
            ));
        }
    }
    lines
}

/// The newest log records, warnings and errors marked. While an index is
/// built only warnings and errors are shown: its steps, their times and what
/// waits for it are in `INDEX`. Without `--log` the newest [`RECENT`] are
/// shown, with a pointer to the rest.
fn recent(status: &Answer, view: &View<'_>) -> Vec<String> {
    let building = status.index_build.is_some();
    let shown: Vec<_> = status
        .recent_log
        .iter()
        .filter(|record| !building || record.priority != Priority::Info)
        .collect();
    let skipped = if view.whole_log {
        0
    } else {
        shown.len().saturating_sub(RECENT)
    };
    let mut lines: Vec<String> = shown[skipped..]
        .iter()
        .map(|record| {
            let text = match record.priority {
                Priority::Error => view.error(&record.text),
                Priority::Warning => view.warning(&record.text),
                Priority::Info => clean(&record.text),
            };
            format!("{}  {text}", view.dim(&view.time(record.at.unix_ms())))
        })
        .collect();
    if skipped > 0 {
        lines.insert(
            0,
            view.dim(&format!("{skipped} earlier in `polylogue status --log`")),
        );
    }
    lines
}

impl View<'_> {
    fn paint(&self, code: &str, text: &str) -> String {
        match self.style {
            TextStyle::Ansi => format!("\x1b[{code}m{text}\x1b[0m"),
            _ => text.to_owned(),
        }
    }

    fn label(&self, text: &str) -> String {
        if text.trim().is_empty() {
            text.to_owned()
        } else {
            self.paint("1", text)
        }
    }

    /// Context: what a reader can skip.
    fn dim(&self, text: &str) -> String {
        self.paint("2", text)
    }

    fn bold(&self, text: &str) -> String {
        self.paint("1", text)
    }

    fn warning(&self, text: &str) -> String {
        self.paint("33", &format!("⚠ {}", clean(text)))
    }

    /// Text in the warning color, without the mark.
    fn warning_text(&self, text: &str) -> String {
        self.paint("33", text)
    }

    fn error(&self, text: &str) -> String {
        self.paint("31", &format!("✖ {}", clean(text)))
    }

    /// A bar `done` of `total` long, in eighths with color, `#` without.
    fn bar(&self, done: u64, total: u64) -> String {
        let width = BAR.min(self.width.saturating_sub(LABEL + 50)).max(10);
        let share = if total == 0 {
            0.
        } else {
            (done.min(total) as f64 / total as f64).clamp(0., 1.)
        };
        let eighths = (share * (width * 8) as f64).floor() as usize;
        match self.style {
            TextStyle::Ansi => {
                let mut bar = "█".repeat(eighths / 8);
                if !eighths.is_multiple_of(8) {
                    bar.push(['▏', '▎', '▍', '▌', '▋', '▊', '▉'][eighths % 8 - 1]);
                }
                let empty = width - bar.chars().count();
                format!("{}{}", self.paint("36", &bar), self.dim(&"░".repeat(empty)))
            }
            _ => {
                let full = eighths / 8;
                format!("{}{}", "#".repeat(full), ".".repeat(width - full))
            }
        }
    }

    /// The time of day for today, the date and time before.
    fn time(&self, ms: i64) -> String {
        let (Ok(at), Ok(now)) = (
            jiff::Timestamp::from_millisecond(ms),
            jiff::Timestamp::from_millisecond(self.now_ms),
        ) else {
            return "unknown time".into();
        };
        let at = at.to_zoned(self.zone.clone());
        if at.date() == now.to_zoned(self.zone.clone()).date() {
            at.strftime("%H:%M:%S").to_string()
        } else {
            at.strftime("%Y-%m-%d %H:%M").to_string()
        }
    }

    /// `line` cut to `columns` visible characters, color sequences kept;
    /// the last column of a cut line holds `…`.
    fn cut_to(&self, line: &str, columns: usize) -> String {
        if strip(line).chars().count() <= columns {
            return line.to_owned();
        }
        let mut visible = 0;
        let mut out = String::new();
        let mut chars = line.chars();
        while let Some(c) = chars.next() {
            if c == '\x1b' {
                out.push(c);
                for c in chars.by_ref() {
                    out.push(c);
                    if c == 'm' {
                        break;
                    }
                }
            } else if visible + 1 < columns {
                out.push(c);
                visible += 1;
            } else {
                break;
            }
        }
        out.push('…');
        if self.style == TextStyle::Ansi {
            out.push_str("\x1b[0m");
        }
        out
    }
}

/// `text` without color sequences.
fn strip(text: &str) -> String {
    let mut out = String::new();
    let mut chars = text.chars();
    while let Some(c) = chars.next() {
        if c == '\x1b' {
            for c in chars.by_ref() {
                if c == 'm' {
                    break;
                }
            }
        } else {
            out.push(c);
        }
    }
    out
}

/// Counts with thousands separators below a million, then in millions
/// and billions with two decimals.
fn count(n: u64) -> String {
    match n {
        0..1_000_000 => {
            let digits = n.to_string();
            let mut out = String::new();
            for (i, c) in digits.chars().enumerate() {
                if i > 0 && (digits.len() - i).is_multiple_of(3) {
                    out.push(',');
                }
                out.push(c);
            }
            out
        }
        1_000_000..1_000_000_000 => format!("{:.2}M", n as f64 / 1e6),
        _ => format!("{:.2}B", n as f64 / 1e9),
    }
}

/// Bytes in binary units.
fn bytes(n: u64) -> String {
    const UNITS: [&str; 5] = ["KiB", "MiB", "GiB", "TiB", "PiB"];
    if n < 1024 {
        return format!("{n} B");
    }
    let mut value = n as f64 / 1024.;
    let mut unit = 0;
    while value >= 1024. && unit + 1 < UNITS.len() {
        value /= 1024.;
        unit += 1;
    }
    format!("{value:.1} {}", UNITS[unit])
}

/// One decimal below ten, none above.
fn decimal(value: f64) -> String {
    if value < 10. {
        format!("{value:.1}")
    } else {
        format!("{value:.0}")
    }
}

/// A share as a whole percentage, with a decimal between 99 and 100 so an
/// unfinished share never reads as 100%.
fn percent(done: u64, total: u64) -> String {
    if total == 0 {
        return "0%".into();
    }
    let share = done.min(total) as f64 * 100. / total as f64;
    if done < total && share >= 99. {
        format!("{:.1}%", (share * 10.).floor() / 10.)
    } else {
        format!("{:.0}%", share.floor())
    }
}

/// The estimate in words: `about 26 min left`.
fn left(seconds: u64) -> String {
    let minutes = seconds.div_ceil(60);
    if minutes < 60 {
        format!("about {minutes} min left")
    } else {
        format!("about {} h {} min left", minutes / 60, minutes % 60)
    }
}

#[cfg(test)]
mod tests;
