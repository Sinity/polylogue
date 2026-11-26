//! Answers as text for a person: query results as tables (numbered for
//! `@N`), measurements, sessions, briefs, notes and resume plans. Each
//! command names its printer in the command table; `--json` prints the
//! answer itself.

use crate::protocol;
use command_surface::TextStyle;
use polylogue_render::shell_quote;
use serde::Serialize;
use serde_json::Value;
use std::fmt::Write as _;

/// The serde name of a unit variant (`awaiting_approval`).
fn name_of(value: &impl Serialize) -> String {
    match serde_json::to_value(value) {
        Ok(Value::String(name)) => name,
        _ => String::new(),
    }
}

/// The answer as indented JSON: commands without a text view.
pub fn json<T: Serialize>(answer: &T, _style: TextStyle) -> String {
    let value = serde_json::to_value(answer).expect("typed answer serializes");
    format!("{}\n", polylogue_render::terminal::json(&value, true))
}

/// A query's rows as a table, or a measurement in its
/// own text view.
pub fn query(answer: &protocol::query::Answer, style: TextStyle) -> String {
    use protocol::query::Answer;
    match answer {
        Answer::PagedMeasure(measure) => measurement(
            &measure.measure,
            &measure.data,
            measure.next.as_deref(),
            &measure.resolved.context.clock.day.zone,
        ),
        Answer::Measure(measure) => measurement(
            &measure.measure,
            &measure.data,
            None,
            &measure.resolved.context.clock.day.zone,
        ),
        Answer::Rows(found) => rows(found, style),
    }
}

/// Query rows as a table numbered for `@N`.
pub fn rows(found: &protocol::query::Rows, style: TextStyle) -> String {
    let data = serde_json::to_value(found).expect("query rows serialize");
    let mut out = Vec::new();
    let _ =
        polylogue_render::results::terminal_with_color(&data, style == TextStyle::Ansi, &mut out);
    String::from_utf8_lossy(&out).into_owned()
}

/// A measurement in its measure's text view, with the continuation line.
/// Measure data is JSON: each measure declares its own shape.
/// Its local times are in the answer's day zone; a zone this machine
/// cannot load leaves the data as it came, never shown in another zone.
fn measurement(what: &str, data: &Value, next: Option<&str>, zone: &str) -> String {
    let mut out = match crate::daemon::measure::find(what) {
        None => polylogue_render::measures::json(data),
        Some((measure, _)) => match jiff::tz::TimeZone::get(zone) {
            Ok(zone) => (measure.text)(data, &zone),
            Err(error) => {
                let mut out = polylogue_render::measures::json(data);
                if !out.ends_with('\n') {
                    out.push('\n');
                }
                let _ = writeln!(
                    out,
                    "{}",
                    polylogue_render::terminal::escape(&format!(
                        "the day zone {zone} cannot be loaded here ({error}); the measure is shown as its data"
                    ))
                );
                out
            }
        },
    };
    if let Some(next) = next {
        if !out.ends_with('\n') {
            out.push('\n');
        }
        let _ = writeln!(out, "{}", polylogue_render::results::continuation(next));
    }
    out
}

/// The same typed read document, rendered for an MCP/model-facing transcript.
pub fn read(page: &polylogue_model::read::ReadPage, _style: TextStyle) -> String {
    let mut bytes = Vec::new();
    polylogue_render::encode_document(
        &page.document,
        polylogue_render::Format::Markdown,
        polylogue_render::EncodeOptions {
            commands: Some(crate::commands::declarations()),
            via: command_surface::Via::Mcp,
            model_facing: true,
            ..Default::default()
        },
        &mut bytes,
    )
    .expect("typed read document writes to a byte vector");
    String::from_utf8(bytes).expect("read encoder writes UTF-8")
}

/// A line: its listing, or each step's reply in order with the failed and
/// skipped steps named, then the lines that take its writes back.
pub fn line(answer: &protocol::line::LineAnswer, style: TextStyle) -> String {
    use protocol::line::{LineAnswer, StepOutcome};
    let steps = match answer {
        LineAnswer::Listing(found) => return query(found, style),
        LineAnswer::Explained(explanation) => return json(explanation, style),
        LineAnswer::Steps(steps) => &steps.steps,
    };
    let mut out = String::new();
    let mut undo = Vec::new();
    for step in steps {
        match &step.outcome {
            StepOutcome::Done { reply, undo: lines } => {
                out.push_str(&step_reply(reply, &step.verb, style));
                undo.extend(lines);
            }
            StepOutcome::Failed { error } => {
                let _ = writeln!(
                    out,
                    "{} failed: {}",
                    step.verb,
                    polylogue_render::terminal::escape(&error.message)
                );
            }
            StepOutcome::NotRun => {
                let _ = writeln!(out, "{} not run", step.verb);
            }
            StepOutcome::WouldDo { would, refusal } => {
                let _ = writeln!(out, "{}", polylogue_render::terminal::escape(would));
                if let Some(refusal) = refusal {
                    let _ = writeln!(
                        out,
                        "  refused: {}",
                        polylogue_render::terminal::escape(&refusal.message)
                    );
                }
            }
        }
    }
    for line in undo {
        let _ = writeln!(out, "undo: {}", polylogue_render::terminal::escape(line));
    }
    out
}

/// One finished step's reply as text; the MCP text of a step is this too.
pub fn step_reply(reply: &protocol::line::StepAnswer, verb: &str, style: TextStyle) -> String {
    use protocol::line::StepAnswer;
    match reply {
        StepAnswer::Read(page) => read(page, style),
        StepAnswer::Streamed(_) | StepAnswer::Followed(_) => String::new(),
        StepAnswer::Brief(prepared) => context(prepared, style),
        StepAnswer::Resume(found) => resume(found, style),
        StepAnswer::Saved(report) => saved(report, style),
        StepAnswer::WroteEach(report) => wrote_each(report, verb),
        other => json(other, style),
    }
}

fn wrote_each(report: &polylogue_model::user::AssertQueryOutcome, verb: &str) -> String {
    let mut out = String::new();
    let written_suffix = if report.written == 1 { "" } else { "s" };
    if report.pending {
        let suffix = if report.targets == 1 { "" } else { "s" };
        let _ = writeln!(out, "{verb} pending for {} row{suffix}", report.targets);
    } else if verb == "tag" {
        let _ = writeln!(out, "tagged {} row{written_suffix}", report.written);
    } else {
        let _ = writeln!(out, "wrote {} annotation{written_suffix}", report.written);
    }
    if report.repeat {
        out.push_str("repeat; no new writes\n");
    }
    if !report.already.is_empty() {
        let _ = writeln!(
            out,
            "already tagged: {}",
            report
                .already
                .iter()
                .map(|reference| polylogue_render::terminal::escape(reference))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    out
}

pub fn saved(report: &protocol::line::SaveReport, _style: TextStyle) -> String {
    let mut text = String::new();
    if let Some(saved) = &report.saved {
        let description = if saved.new_revision && !report.repeat {
            "runs again each time you use it"
        } else {
            "same question; no new revision"
        };
        let _ = writeln!(
            text,
            "saved   query:{} (revision {}): {description}",
            saved.name, saved.revision
        );
    }
    let _ = write!(
        text,
        "kept    {} ({} rows, {})",
        report.kept.result, report.kept.rows, report.kept.at
    );
    if let Some(saved) = &report.saved {
        let _ = write!(text, ", now result:{}", saved.name);
    }
    text.push('\n');
    if let Some(watch) = &report.watch {
        if watch.new_rule && !report.repeat {
            let _ = writeln!(
                text,
                "watch   notices when a row starts matching query:{}",
                watch.name
            );
        } else {
            let _ = writeln!(text, "watch   unchanged for query:{}", watch.name);
        }
    }
    if report.pending {
        text.push_str("pending Sinex confirmation\n");
    }
    polylogue_render::terminal::escape(&text)
}

/// A prepared brief, takeover or recall.
pub fn context(answer: &protocol::context::Answer, _style: TextStyle) -> String {
    polylogue_render::terminal::escape(&answer.prepared.text)
}

/// `continue`: the sessions worth resuming here, or one session's plan.
pub fn resume(answer: &protocol::resume::Answer, _style: TextStyle) -> String {
    polylogue_render::terminal::escape(&match answer {
        protocol::resume::Answer::Sessions(sessions) => resumable(sessions),
        protocol::resume::Answer::Plan(found) => plan(found),
    })
}

fn resumable(sessions: &[crate::resume::ResumableSession]) -> String {
    let mut out = String::new();
    for entry in sessions {
        let mut why = entry.open.phrases();
        why.extend(entry.follow_ups.iter().cloned());
        let _ = writeln!(
            out,
            "{}  {}  {}\n    {}",
            entry
                .session
                .last_activity
                .map(|at| at.to_string())
                .unwrap_or_default(),
            entry.session.id,
            entry.session.title.as_deref().unwrap_or_default(),
            why.join("; ")
        );
    }
    out
}

/// A command line as a shell would run it, in its directory.
pub(crate) fn command_line(command: &crate::resume::Command) -> String {
    let mut words = vec![shell_quote(&command.program)];
    words.extend(command.args.iter().map(|arg| shell_quote(arg)));
    let line = words.join(" ");
    match &command.cwd {
        Some(dir) => format!("cd {} && {line}", shell_quote(dir)),
        None => line,
    }
}

/// Why a harness resume cannot run here: it runs only in the directory the
/// session recorded, never in another one.
pub(crate) const DIRECTORY_UNKNOWN: &str =
    "directory_unknown: the session recorded no directory to resume in";

fn plan(found: &crate::resume::Plan) -> String {
    let mut out = String::new();
    if let Some(harness) = &found.harness_command {
        let line = match harness.cwd {
            Some(_) => command_line(harness),
            None => DIRECTORY_UNKNOWN.to_owned(),
        };
        let _ = writeln!(out, "harness command:    {line}");
    }
    let _ = writeln!(
        out,
        "with context:       {}",
        command_line(&found.context_assisted)
    );
    out
}

/// `import`: the files report as JSON; a provider import as its job's line.
pub fn import(answer: &protocol::import::Answer, style: TextStyle) -> String {
    match answer {
        protocol::import::Answer::Files(report) => json(report, style),
        protocol::import::Answer::Provider(job) => import_job_answer(job, style),
    }
}

/// `import PROVIDER`: the job's line, or a write still awaiting
/// confirmation as JSON.
pub fn import_job_answer(answer: &protocol::browser::ImportJobAnswer, style: TextStyle) -> String {
    match answer {
        protocol::browser::ImportJobAnswer::Current(current) => polylogue_render::terminal::escape(
            &import_jobs_text(current.job.as_deref().map_or(&[], std::slice::from_ref)),
        ),
        protocol::browser::ImportJobAnswer::Outcome(_) => json(answer, style),
    }
}

fn import_jobs_text(jobs: &[crate::browser::import_job::JobSummary]) -> String {
    if jobs.is_empty() {
        return "no import jobs\n".into();
    }
    let mut out = String::new();
    for job in jobs {
        let _ = writeln!(
            out,
            "{}  {}  {}  listed {}, captured {}, unchanged {}, failed {}  {}",
            job.id,
            job.provider,
            name_of(&job.status),
            job.listed,
            job.captured,
            job.unchanged,
            job.failed,
            job.updated_at.as_deref().unwrap_or_default()
        );
        if let Some(detail) = &job.detail {
            let _ = writeln!(out, "    {detail}");
        }
    }
    out
}

/// One escaped diagnostic for local and remote CLI failures. JSON uses its own encoder.
pub(crate) fn diagnostic(
    kind: &str,
    message: &str,
    suggestion: Option<&str>,
    color: bool,
) -> String {
    let label = if kind.is_empty() {
        format!("polylogue: {message}\n")
    } else {
        format!("polylogue: {kind}: {message}\n")
    };
    let mut text = polylogue_render::terminal::heading(&label, color);
    if let Some(suggestion) = suggestion {
        text.push_str(&polylogue_render::terminal::escape(&format!(
            "polylogue: try: {suggestion}"
        )));
        text.push('\n');
    }
    text
}

#[cfg(test)]
mod tests;

/// What `daemon --until-idle` did, for a person.
pub fn idle_summary(summary: &polylogue_engine::RunSummary) -> String {
    let work = &summary.work;
    let mut out = format!(
        "idle after {} pass{} and {} import{} ({:.1} s in passes)\n",
        summary.passes,
        if summary.passes == 1 { "" } else { "es" },
        summary.imports,
        if summary.imports == 1 { "" } else { "s" },
        work.seconds,
    );
    for (name, n) in [
        ("full scans", work.full_scans),
        ("inputs seen", work.inputs_seen),
        ("inputs changed", work.inputs_changed),
        ("bytes read", work.bytes_read),
        ("bytes stored", work.bytes_stored),
        ("messages written", work.messages_written),
        ("sessions derived", work.sessions_derived),
    ] {
        let _ = writeln!(out, "  {name:<17} {n}");
    }
    for root in &work.unreadable_roots {
        let _ = writeln!(
            out,
            "  unreadable: {}",
            polylogue_render::terminal::escape(&root.display().to_string())
        );
    }
    for error in &work.errors {
        let _ = writeln!(
            out,
            "  error: {}",
            polylogue_render::terminal::escape(&error.to_string())
        );
    }
    out
}

/// The archive verification counts, then each problem.
pub fn verification(found: &polylogue_store::verify::Verification, _style: TextStyle) -> String {
    let counts = &found.counts;
    let mut out = format!(
        "{} inputs, {} revisions, {} extents, {} observations, {} blobs ({} bytes)\n",
        counts.inputs,
        counts.revisions,
        counts.extents,
        counts.observations,
        counts.blobs,
        counts.blob_bytes
    );
    let _ = writeln!(
        out,
        "blobs checked: {} ({} bytes, {} stored), {} with problems",
        found.blobs.checked, found.blobs.bytes, found.blobs.stored_bytes, found.blobs.mismatched
    );
    for format in &found.formats {
        let _ = writeln!(
            out,
            "  {:<28} {:>8} inputs {:>9} revisions {:>14} bytes",
            format.format, format.inputs, format.revisions, format.revision_bytes
        );
    }
    let classes = |counts: &[polylogue_store::verify::ClassCount]| {
        counts
            .iter()
            .map(|class| format!("{} {}", class.class, class.count))
            .collect::<Vec<_>>()
            .join(", ")
    };
    let _ = writeln!(out, "observations: {}", classes(&found.observation_classes));
    let _ = writeln!(out, "revisions: {}", classes(&found.revision_kinds));
    for database in &found.databases {
        let _ = writeln!(
            out,
            "{}: {}",
            polylogue_render::terminal::escape(&database.file),
            polylogue_render::terminal::escape(&database.result.join("; "))
        );
    }
    if let Some(manifest) = &found.manifest {
        let _ = writeln!(
            out,
            "manifest: {} revisions hashed from {} bytes",
            manifest.revisions, manifest.bytes_read
        );
    }
    if found.problems.is_empty() {
        out.push_str("no problems\n");
    }
    for problem in &found.problems {
        let kind = name_of(&problem.kind);
        let _ = writeln!(
            out,
            "{kind}: {}: {}",
            polylogue_render::terminal::escape(&problem.subject),
            polylogue_render::terminal::escape(&problem.detail)
        );
    }
    out
}
