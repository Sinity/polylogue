use super::*;
use crate::protocol::status::{AnswersFrom, NotReadSince, Notice, PendingWrites, Warm};
use polylogue_model::Timestamp;
use polylogue_model::freshness::{
    CompletenessReason, CountedReason, ReadCompleteness, ReadFreshness,
};
use polylogue_model::live::{LiveState, WaitingOn};
use polylogue_store::log::Record;

/// 2026-11-24 14:05:00 UTC.
const NOW: i64 = 1_795_529_100_000;
const MINUTE: i64 = 60_000;

/// `hh:mm:ss` today, in milliseconds.
fn today(hours: i64, minutes: i64, seconds: i64) -> i64 {
    NOW - 14 * 60 * MINUTE - 5 * MINUTE + hours * 60 * MINUTE + minutes * MINUTE + seconds * 1000
}

fn at(ms: i64) -> Timestamp {
    Timestamp::from_unix_ms(ms).unwrap()
}

fn record(ms: i64, priority: Priority, text: &str) -> Record {
    Record {
        at: at(ms),
        priority,
        text: text.into(),
    }
}

fn answer() -> Answer {
    Answer {
        archive: Some("/home/op/archive".into()),
        pid: 4479,
        uptime_seconds: 17 * 3600 + 35 * 60,
        store: None,
        index_build: None,
        index: None,
        build_progress: None,
        answers_from: AnswersFrom::Snapshot {
            read_at: at(NOW - MINUTE),
        },
        changes: 0,
        caught_up: true,
        totals_error: None,
        quota: Vec::new(),
        quota_error: None,
        vectors: None,
        vectors_error: None,
        verification: None,
        pending_writes: PendingWrites {
            delivery_outcomes: 0,
            live_event_lines: 0,
            last_error: None,
        },
        loop_status: polylogue_engine::LoopStatus::default(),
        signals: None,
        notices: Some(Notices::Measured {
            last_day: 0,
            latest: Vec::new(),
        }),
        running: Some(Running::Measured {
            waiting: 0,
            sessions: Vec::new(),
        }),
        warm: Warm {
            scope_index_loaded: true,
            vectors: None,
            vector_loads: 0,
            query_vectors_reused: 0,
        },
        mode: Mode::Standalone,
        freshness: ReadFreshness {
            local: ReadCompleteness::empty(),
            sinex: IndexFreshness::Standalone,
        },
        recent_log: Vec::new(),
        threads: Vec::new(),
    }
}

fn session(id: &str, state: LiveState, since: i64) -> polylogue_store::RunningSession {
    polylogue_store::RunningSession {
        session: id.into(),
        state,
        since: Some(since),
        waiting_on: None,
        stall: None,
        open_tool: None,
        task: None,
        edited_files: 0,
        cwd: None,
    }
}

fn plain(status: &Answer, problems: &[&str]) -> String {
    let problems: Vec<String> = problems.iter().map(|note| (*note).to_owned()).collect();
    render(
        status,
        &problems,
        &View {
            style: TextStyle::Plain,
            width: 100,
            now_ms: NOW,
            zone: &jiff::tz::TimeZone::UTC,
            whole_log: false,
        },
    )
}

/// An idle daemon over a mostly read archive (docs/operations.md#status).
#[test]
fn an_idle_answer_renders_the_archive_reading_and_activity() {
    use polylogue_model::capture::{Accessibility, Classification, Coverage, Entry, RootCoverage};
    let mut status = answer();
    status.store = Some(polylogue_store::Status {
        sessions: 37_419,
        messages: 11_240_000,
        documents: 288,
        tool_calls: 4_770_000,
        inputs: 132_286,
        revisions: 134_089,
        stored_bytes: 105_871_282_586,
        parse_states: [
            ("done", 109_793),
            ("parsing", 48),
            ("failed", 31),
            ("no_parser", 15),
        ]
        .into_iter()
        .map(|(state, n)| (state.to_owned(), n))
        .collect(),
        unused: vec![
            ("claude-code:message.usage.cache".into(), 11_200_000),
            ("codex:payload.extra".into(), 5),
        ],
        unused_fields: 41,
        search_pending: 0,
        search_pending_is_minimum: false,
        summaries_due: 0,
        capture_coverage: vec![Coverage {
            version: 1,
            machine: "host".into(),
            observed_at: at(today(13, 0, 0)),
            roots: vec![
                RootCoverage {
                    path_bytes: b"/home/op/.claude".to_vec(),
                    status: Accessibility::Accessible,
                    entries: vec![
                        Entry {
                            class: Classification::Collected,
                            pattern: "projects/*/*.jsonl".into(),
                            reason: None,
                            files: 100,
                            bytes: 1 << 20,
                            newest_mtime_ms: None,
                        },
                        Entry {
                            class: Classification::Ignored,
                            pattern: "cache/**".into(),
                            reason: Some("cache".into()),
                            files: 40,
                            bytes: 1 << 10,
                            newest_mtime_ms: None,
                        },
                        Entry {
                            class: Classification::Ignored,
                            pattern: "**/*.lock".into(),
                            reason: Some("lock files".into()),
                            files: 3,
                            bytes: 30,
                            newest_mtime_ms: None,
                        },
                    ],
                    unavailable_paths: Vec::new(),
                },
                RootCoverage {
                    path_bytes: b"/home/op/.codex".to_vec(),
                    status: Accessibility::Absent,
                    entries: Vec::new(),
                    unavailable_paths: Vec::new(),
                },
            ],
        }],
        index_rebuild: None,
    });
    status.freshness.local.parsing =
        CompletenessDimension::Incomplete(vec![CountedReason::inputs(
            CompletenessReason::ParserBehind,
            "codex",
            48,
        )]);
    // docs/operations.md, Status: a failed parse is named as a problem
    // with its count and origin, never as parsing behind.
    status.freshness.local.failed_parses = vec![CountedReason::inputs(
        CompletenessReason::ParseFailed,
        "claude-code",
        31,
    )];
    status.loop_status.passes = 12;
    status.loop_status.last_pass_at = Some(at(today(12, 58, 4)));
    status.loop_status.last_pass.seconds = 4.2;
    status.loop_status.last_pass.inputs_changed = 37;
    status.loop_status.last_pass.inputs_seen = 132_286;
    status.loop_status.last_pass.bytes_read = 13_002_342;
    status.loop_status.last_pass.messages_written = 1_204;
    let mut waiting = session("claude-code:a1", LiveState::Waiting, NOW - 7 * MINUTE);
    waiting.waiting_on = Some(WaitingOn::Permission);
    waiting.open_tool = Some("Bash".into());
    let mut working = session("codex:b2", LiveState::Working, NOW - 2 * MINUTE);
    working.edited_files = 2;
    status.running = Some(Running::Measured {
        waiting: 1,
        sessions: vec![waiting, working],
    });
    status.notices = Some(Notices::Measured {
        last_day: 2,
        latest: vec![Notice {
            id: "notice-1".into(),
            at: at(today(14, 2, 0)),
            text: "synthetic notice".into(),
            events: Vec::new(),
        }],
    });
    status.recent_log = vec![
        record(today(14, 1, 10), Priority::Warning, "scope index: busy"),
        record(today(14, 2, 11), Priority::Info, "search thread: caught up"),
    ];
    assert_eq!(
        plain(
            &status,
            &status
                .freshness
                .status_notes()
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>()
        ),
        "\
polylogue  ⚠ 31 claude-code inputs failed to parse  (+1 more under NOTES)
           standalone · daemon 4479 · up 17h35m · idle · /home/op/archive

ARCHIVE    37,419 sessions   11.24M messages   4.77M tool calls   288 documents
           132,286 inputs (98.6 GiB decoded) · 134,089 revisions

READING    #############################.  99.9%   109,793 of 109,887 files read
           48 parsing · 31 failed · 15 without a parser
           48 codex inputs have bytes not parsed yet · session figures current · search current
           31 claude-code inputs failed to parse

LAST CHECK 12:58:04 · 4.2 s · 37 of 132,286 files changed
           12.4 MiB read · 1,204 messages written

RUNNING    2 running · 1 waiting 7m
           ◆ waiting     7m  claude-code:a1  on permission · Bash
           ● working     2m  codex:b2  2 files edited

NOTICES    2 in the last day · 14:02:00 synthetic notice

SOURCES    walked 13:00:00
           claude-code /home/op/.claude · present · 100 collected · 43 ignored
             ignored: 40 cache, 3 lock files
           /home/op/.codex · absent

UNUSED     claude-code:message.usage.cache ×11.20M not yet read · 40 more, 1 of them in --json

RECENT     14:01:10  ⚠ scope index: busy
           14:02:11  search thread: caught up

NOTES      ⚠ 31 claude-code inputs failed to parse
           ⚠ 48 codex inputs have bytes not parsed yet
"
    );
}

use polylogue_store::archive::build_progress::{
    BuildProgress, CountUnit, CurrentStep, FailedAttempt, FinishedStep, StepCount,
};
use polylogue_store::archive::index_build::IndexBuild;

/// 63.1 GiB.
const STORED: u64 = 67_755_000_000;

fn finished(name: &str, seconds: f64, count: Option<(u64, CountUnit)>) -> FinishedStep {
    FinishedStep {
        name: name.into(),
        seconds,
        count: count.map(|(done, unit)| StepCount {
            done,
            total: None,
            unit,
        }),
    }
}

fn current(name: &str, minutes: i64, count: Option<StepCount>) -> CurrentStep {
    CurrentStep {
        name: name.into(),
        since: at(NOW - minutes * MINUTE),
        count,
        per_second: None,
        rate_seconds: None,
        rate_varying: false,
        eta_seconds: None,
    }
}

fn progress(current: CurrentStep, finished: Vec<FinishedStep>, to_come: &[&str]) -> BuildProgress {
    BuildProgress {
        finished,
        current: Some(current),
        to_come: to_come.iter().map(|step| (*step).to_owned()).collect(),
        eta_seconds: None,
        failed_attempt: None,
        writer_busy: None,
        worker_cpu_seconds: Vec::new(),
    }
}

/// An answer while an index is built and none can be read: the sections
/// the index fills are unavailable, and the last check is from before.
fn building(build: IndexBuild, progress: BuildProgress) -> Answer {
    let mut status = answer();
    status.uptime_seconds = 8 * 3600 + 2 * 60;
    status.loop_status.busy = true;
    status.loop_status.passes = 1;
    status.loop_status.last_pass_at = Some(at(today(2, 32, 32)));
    status.loop_status.last_pass.errors = vec![polylogue_model::Failure::stated(
        "",
        polylogue_model::Cause::Transient,
        polylogue_model::Code::Conflict,
        "the file changed after it was listed",
    )];
    status.answers_from = AnswersFrom::Build;
    status.index_build = Some(build);
    status.build_progress = Some(progress);
    status.running = None;
    status.notices = None;
    status
}

const BUILDING: &str = "the index is being built";

/// Mid-build, in a writer step (docs/operations.md#status): why the index
/// is built, what that means for reads, each step with its time and count,
/// the current one's count of its total, rate and estimate, and the whole
/// job's time unknown while steps remain unmeasured. The stale last check
/// and the note restating the build are left out.
#[test]
fn a_build_lists_its_steps_with_counts_times_and_estimates() {
    let mut assembling = current(
        "assembling sessions",
        86,
        Some(StepCount {
            done: 3_205,
            total: Some(41_822),
            unit: CountUnit::Sessions,
        }),
    );
    assembling.per_second = Some(0.62);
    assembling.rate_seconds = Some(300.);
    assembling.eta_seconds = Some(62_280);
    let mut steps = progress(
        assembling,
        vec![
            finished("finding files", 95., None),
            finished("storing files", 4_800., Some((10_687, CountUnit::Files))),
            finished(
                "reading stored files",
                4_133.5,
                Some((STORED, CountUnit::Bytes)),
            ),
            finished(
                "sorting texts",
                3_476.4,
                Some((9_257_783, CountUnit::Records)),
            ),
            finished(
                "writing tables",
                2_806.7,
                Some((50_252_966, CountUnit::Records)),
            ),
            finished("indexing", 8_801.6, Some((61, CountUnit::Indexes))),
        ],
        &["finishing indexes", "search index", "installing"],
    );
    steps.writer_busy = Some(1.);
    let mut status = building(
        IndexBuild {
            reason: polylogue_store::index::file::RebuildReason::NoIndex,
            chains: 10_687,
            chains_read: 10_687,
            bytes: STORED,
            bytes_read: STORED,
        },
        steps,
    );
    status.recent_log = vec![
        record(
            today(6, 45, 32),
            Priority::Info,
            "index: writing tables took 2806.7 s; 50252966 records",
        ),
        record(
            today(9, 12, 13),
            Priority::Info,
            "index: indexing took 8801.6 s; 61 indexes",
        ),
    ];
    assert_eq!(
        plain(
            &status,
            &[BUILDING, "AI Studio through Drive is not authorized"]
        ),
        "\
polylogue  ◐ building the index, step 7 of 10
           standalone · daemon 4479 · up 8h2m · busy · /home/op/archive

INDEX      building the index from 63.1 GiB of stored files: the archive had no index
           reads and searches answer \"index building\" until it is installed
           ##............................  7% of assembling sessions · about 17 h 18 min left

           ✓ finding files            1m
           ✓ storing files         1h20m  10,687 files
           ✓ reading stored files   1h8m  63.1 GiB
           ✓ sorting texts           57m  9.26M records
           ✓ writing tables          46m  50.25M records
           ✓ indexing              2h26m  61 indexes
           ▸ assembling sessions   1h26m  3,205 of 41,822 sessions
                                          0.6 sessions/s
           · finishing indexes
           · search index
           · installing

           whole build: time left known once the last step runs (3 steps to come)
           database writer busy 100% of the last minute

NOTES      ⚠ AI Studio through Drive is not authorized
"
    );
}

/// Reading the stored files counts bytes and files; with the previous
/// index readable, reads use it meanwhile.
#[test]
fn reading_stored_files_shows_bytes_and_files() {
    let mut reading = current(
        "reading stored files",
        34,
        Some(StepCount {
            done: 39_513_699_123,
            total: Some(70_437_463_654),
            unit: CountUnit::Bytes,
        }),
    );
    reading.per_second = Some(18_400_000.);
    reading.rate_seconds = Some(300.);
    reading.eta_seconds = Some(1_560);
    let mut steps = progress(reading, Vec::new(), &["sorting texts", "installing"]);
    steps.worker_cpu_seconds = vec![3.0, 3.2];
    let status = building(
        IndexBuild {
            reason: polylogue_store::index::file::RebuildReason::FormatVersion {
                found: 41,
                wanted: 42,
            },
            chains: 21_678,
            chains_read: 4_120,
            bytes: 70_437_463_654,
            bytes_read: 39_513_699_123,
        },
        steps,
    );
    assert_eq!(
        plain(&status, &[BUILDING]),
        "\
polylogue  ◐ building the index, step 1 of 3
           standalone · daemon 4479 · up 8h2m · busy · /home/op/archive

INDEX      building the index from 65.6 GiB of stored files: its format changed (41 to 42)
           reads and searches answer \"index building\" until it is installed
           ################..............  56% of reading stored files · about 26 min left

           ▸ reading stored files    34m  36.8 GiB of 65.6 GiB · 4,120 of 21,678 files
                                          18.4 MB/s
           · sorting texts
           · installing

           whole build: time left known once the last step runs (2 steps to come)
           2 parse workers, 3.1 s CPU/min each
"
    );
}

/// Where no estimate can be given the view says why: a step whose total is
/// known only when it ends, and one measured for less than a minute.
#[test]
fn a_step_without_an_estimate_says_why() {
    let mut sorting = current(
        "sorting texts",
        20,
        Some(StepCount {
            done: 2_100_000,
            total: None,
            unit: CountUnit::Records,
        }),
    );
    sorting.per_second = Some(2_650.);
    sorting.rate_seconds = Some(300.);
    let build = IndexBuild {
        reason: polylogue_store::index::file::RebuildReason::NoIndex,
        chains: 0,
        chains_read: 0,
        bytes: 0,
        bytes_read: 0,
    };
    let status = building(build.clone(), progress(sorting, Vec::new(), &[]));
    assert_eq!(
        plain(&status, &[BUILDING]),
        "\
polylogue  ◐ building the index, step 1 of 1
           standalone · daemon 4479 · up 8h2m · busy · /home/op/archive

INDEX      building the index: the archive had no index
           reads and searches answer \"index building\" until it is installed

           ▸ sorting texts    20m  2.10M records so far
                                   2,650 records/s · total known when the step ends
"
    );
    let mut storing = current(
        "storing files",
        1,
        Some(StepCount {
            done: 120,
            total: Some(559),
            unit: CountUnit::Files,
        }),
    );
    storing.per_second = Some(2.);
    storing.rate_seconds = Some(30.);
    let status = building(
        build,
        progress(storing, Vec::new(), &["reading stored files"]),
    );
    assert!(
        plain(&status, &[BUILDING]).contains(
            "▸ storing files            1m  120 of 559 files\n                                          2.0 files/s · estimate after a minute of progress\n"
        ),
        "{}",
        plain(&status, &[BUILDING])
    );
}

/// An attempt that failed is shown, with its step and error, while the
/// build that replaced it runs.
#[test]
fn a_failed_attempt_is_shown_while_the_build_runs_again() {
    let mut steps = progress(current("reading stored files", 2, None), Vec::new(), &[]);
    steps.failed_attempt = Some(FailedAttempt {
        step: Some("assembling sessions".into()),
        error: "disk full".into(),
    });
    let status = building(
        IndexBuild {
            reason: polylogue_store::index::file::RebuildReason::NoIndex,
            chains: 10,
            chains_read: 0,
            bytes: 0,
            bytes_read: 0,
        },
        steps,
    );
    let text = plain(&status, &[BUILDING]);
    assert!(
        text.starts_with(
            "polylogue  ◐ building the index, step 1 of 1, again after a failed attempt\n"
        ),
        "{text}"
    );
    assert!(
        text.contains(
            "\n           ✖ an earlier attempt failed while assembling sessions: disk full; started over\n"
        ),
        "{text}"
    );
    assert!(
        text.contains("\n\n           ▸ reading stored files     2m\n"),
        "{text}"
    );
}

/// Once the index is installed the build's section is gone; the last
/// check and the step lines of the log remain.
#[test]
fn a_finished_build_leaves_its_steps_in_the_log() {
    let mut status = answer();
    status.loop_status.passes = 3;
    status.loop_status.last_pass_at = Some(at(today(13, 50, 0)));
    status.loop_status.last_pass.seconds = 1.5;
    status.loop_status.last_pass.inputs_changed = 2;
    status.loop_status.last_pass.inputs_seen = 10_687;
    status.recent_log = vec![
        record(
            today(13, 40, 0),
            Priority::Info,
            "index: installing took 12.0 s; 0 items",
        ),
        record(
            today(13, 40, 1),
            Priority::Info,
            "index: installed; installing took 12 s",
        ),
    ];
    assert_eq!(
        plain(&status, &[]),
        "\
polylogue  ✓ up to date
           standalone · daemon 4479 · up 17h35m · idle · /home/op/archive

LAST CHECK 13:50:00 · 1.5 s · 2 of 10,687 files changed
           0 B read · 0 messages written

RECENT     13:40:00  index: installing took 12.0 s; 0 items
           13:40:01  index: installed; installing took 12 s
"
    );
}

#[test]
fn numbers_bytes_and_estimates_read_as_words() {
    assert_eq!(count(999_999), "999,999");
    assert_eq!(count(1_234_567), "1.23M");
    assert_eq!(bytes(1023), "1023 B");
    assert_eq!(bytes(1536), "1.5 KiB");
    assert_eq!(percent(999, 1000), "99.9%");
    assert_eq!(percent(9_999, 10_000), "99.9%");
    assert_eq!(percent(1000, 1000), "100%");
    assert_eq!(left(59), "about 1 min left");
    assert_eq!(left(3_700), "about 1 h 2 min left");
}

#[test]
fn a_long_line_is_cut_to_the_width_with_a_mark() {
    let view = View {
        style: TextStyle::Plain,
        width: 10,
        now_ms: NOW,
        zone: &jiff::tz::TimeZone::UTC,
        whole_log: false,
    };
    assert_eq!(view.cut_to("0123456789", 10), "0123456789");
    assert_eq!(view.cut_to("0123456789abc", 10), "012345678…");
}

/// Archived text reaches the view escaped: a rule named with terminal
/// controls prints them as `\x..` sequences.
#[test]
fn archived_text_is_escaped_in_the_view() {
    let mut status = answer();
    status.signals = Some(crate::protocol::status::Signals::Observed {
        rules: vec![crate::protocol::status::SignalRule {
            id: "rule-1".into(),
            name: "\x1b]0;title\x07\x7f".into(),
            query: "since:1d".into(),
            author: "operator".into(),
            last_fired: None,
        }],
        unusable: Vec::new(),
        held: Vec::new(),
    });
    let text = plain(&status, &[]);
    assert!(
        text.contains("RULES      \\x1b]0;title\\x07\\x7f · by operator · never fired\n"),
        "{text}"
    );
}

/// A rule shows how long ago it last fired, or that it never has.
#[test]
fn a_rule_shows_when_it_last_fired() {
    let mut status = answer();
    status.signals = Some(crate::protocol::status::Signals::Observed {
        rules: vec![crate::protocol::status::SignalRule {
            id: "authored:one".into(),
            name: "stalled daemon".into(),
            query: "since:1d".into(),
            author: "operator".into(),
            last_fired: polylogue_model::Timestamp::from_unix_ms(NOW - 5 * 60_000),
        }],
        unusable: Vec::new(),
        held: Vec::new(),
    });
    let text = plain(&status, &[]);
    assert!(
        text.contains("RULES      stalled daemon · by operator · last fired 5m ago\n"),
        "{text}"
    );
}

/// A blocked daemon says so in its headline, in red on a color terminal.
#[test]
fn a_blocked_daemon_says_so_first() {
    let mut status = answer();
    status.mode = Mode::Blocked {
        configured: None,
        path: "/home/op/archive".into(),
        error: "the archive is locked by another daemon".into(),
    };
    assert_eq!(
        plain(&status, &[]),
        "\
polylogue  ✖ blocked: /home/op/archive: the archive is locked by another daemon
           blocked · daemon 4479 · up 17h35m · idle · /home/op/archive
"
    );
}

/// The view shows the newest five log records and how many more `--log`
/// prints; while an index is built only warnings and errors are shown.
#[test]
fn recent_shows_the_newest_records_and_points_at_the_rest() {
    let mut status = answer();
    status.recent_log = (0..7)
        .map(|n| record(today(13, n, 0), Priority::Info, &format!("record {n}")))
        .collect();
    assert_eq!(
        plain(&status, &[]),
        "\
polylogue  ✓ up to date
           standalone · daemon 4479 · up 17h35m · idle · /home/op/archive

RECENT     2 earlier in `polylogue status --log`
           13:02:00  record 2
           13:03:00  record 3
           13:04:00  record 4
           13:05:00  record 5
           13:06:00  record 6
"
    );
    let mut status = building(
        IndexBuild {
            reason: polylogue_store::index::file::RebuildReason::NoIndex,
            chains: 0,
            chains_read: 0,
            bytes: 0,
            bytes_read: 0,
        },
        progress(current("indexing", 1, None), Vec::new(), &[]),
    );
    status.recent_log = (0..3)
        .map(|n| record(today(13, n, 0), Priority::Info, "index: a step"))
        .chain([record(
            today(13, 7, 0),
            Priority::Warning,
            "a source root is unreadable",
        )])
        .collect();
    let text = plain(&status, &[BUILDING]);
    assert!(
        text.ends_with("RECENT     13:07:00  ⚠ a source root is unreadable\n"),
        "{text}"
    );
}

/// On a color terminal the headline carries the state's color, the running
/// step is bold and the finished ones are dimmed.
#[test]
fn color_marks_the_state_and_the_running_step() {
    let mut assembling = current(
        "assembling sessions",
        10,
        Some(StepCount {
            done: 50,
            total: Some(100),
            unit: CountUnit::Sessions,
        }),
    );
    assembling.per_second = Some(5.);
    assembling.rate_seconds = Some(300.);
    let status = building(
        IndexBuild {
            reason: polylogue_store::index::file::RebuildReason::NoIndex,
            chains: 0,
            chains_read: 0,
            bytes: 0,
            bytes_read: 0,
        },
        progress(
            assembling,
            vec![finished("indexing", 60., Some((3, CountUnit::Indexes)))],
            &[],
        ),
    );
    let text = render(
        &status,
        &[],
        &View {
            style: TextStyle::Ansi,
            width: 100,
            now_ms: NOW,
            zone: &jiff::tz::TimeZone::UTC,
            whole_log: false,
        },
    );
    let first = text.lines().next().unwrap();
    assert_eq!(
        first,
        "\x1b[1mpolylogue  \x1b[0m\x1b[1;33m◐ building the index, step 2 of 2\x1b[0m"
    );
    assert!(
        text.contains("\x1b[32m✓\x1b[0m \x1b[2mindexing                1m  3 indexes\x1b[0m\n"),
        "{text:?}"
    );
    assert!(
        text.contains(
            "\x1b[1;36m▸\x1b[0m \x1b[1massembling sessions    10m  50 of 100 sessions\x1b[0m\n"
        ),
        "{text:?}"
    );
}

/// A step whose count has not moved over the measured span says so in its
/// own unit, beside its own count.
#[test]
fn a_step_without_progress_says_what_did_not_move() {
    let mut joining = current(
        "joining sessions across files",
        70,
        Some(StepCount {
            done: 1_024,
            total: Some(3_310),
            unit: CountUnit::Sessions,
        }),
    );
    joining.per_second = Some(0.);
    joining.rate_seconds = Some(900.);
    let status = building(
        IndexBuild {
            reason: polylogue_store::index::file::RebuildReason::NoIndex,
            chains: 0,
            chains_read: 0,
            bytes: 0,
            bytes_read: 0,
        },
        progress(
            joining,
            vec![finished(
                "assembling sessions",
                600.,
                Some((33_261, CountUnit::Sessions)),
            )],
            &["finishing indexes"],
        ),
    );
    let text = plain(&status, &[BUILDING]);
    assert!(
        text.contains(
            "\n           ▸ joining sessions across files  1h10m  1,024 of 3,310 sessions\n                                                   no sessions done in the last 15m\n"
        ),
        "{text}"
    );
}

/// `THREADS` gives each busy thread's work, outermost first, with the time
/// of the innermost, says when it has gone a while without progress
/// (stalled after ten minutes), and puts each long statement under its
/// thread with its running time before the SQL (docs/operations.md#status).
#[test]
fn threads_show_their_work_progress_and_long_statements() {
    use polylogue_store::activity::{Statement, Thread, Work};
    let work = |what: &str, subject: Option<&str>, seconds: f64| Work {
        what: what.into(),
        subject: subject.map(Into::into),
        seconds,
    };
    let mut status = answer();
    status.threads = vec![
        Thread {
            worker: None,
            name: "polylogue-parse-0".into(),
            doing: vec![work("parsing a file", Some("/home/op/a.jsonl"), 4.)],
            seconds_since_progress: Some(4.),
            statements: Vec::new(),
        },
        Thread {
            worker: None,
            name: "polylogue-request".into(),
            doing: Vec::new(),
            seconds_since_progress: None,
            statements: vec![Statement {
                connection: "reader".into(),
                sql: "SELECT 1".into(),
                seconds: 5.,
            }],
        },
        Thread {
            worker: None,
            name: "polylogue-search-fill".into(),
            doing: vec![work("reading texts for the search index", None, 40.)],
            seconds_since_progress: Some(30.),
            statements: Vec::new(),
        },
        Thread {
            worker: None,
            name: "polylogue-writer".into(),
            doing: vec![
                work("building the index", None, 2.),
                work("indexing sessions", None, 3720.),
            ],
            seconds_since_progress: Some(3500.),
            statements: vec![Statement {
                connection: "writer".into(),
                sql: "UPDATE transcript_line SET class = 1".into(),
                seconds: 3500.,
            }],
        },
    ];
    let text = plain(&status, &[]);
    let section: Vec<&str> = text
        .lines()
        .skip_while(|line| !line.starts_with("THREADS"))
        .take_while(|line| !line.is_empty())
        .collect();
    assert_eq!(
        section,
        [
            "THREADS    parse-0  parsing a file (/home/op/a.jsonl)  4s",
            "           request  no work recorded",
            "             running 5s on the reader connection: SELECT 1",
            "           search-fill  reading texts for the search index  40s · no progress for 30s",
            "           writer  building the index › indexing sessions  1h2m · stalled, no progress for 58m",
            "             running 58m on the writer connection: UPDATE transcript_line SET class = 1",
        ]
    );
}

fn coverage(
    roots: Vec<polylogue_model::capture::RootCoverage>,
) -> polylogue_model::capture::Coverage {
    polylogue_model::capture::Coverage {
        version: 1,
        machine: "host".into(),
        observed_at: at(today(14, 4, 0)),
        roots,
    }
}

fn root(
    path: &str,
    status: polylogue_model::capture::Accessibility,
    entries: Vec<polylogue_model::capture::Entry>,
) -> polylogue_model::capture::RootCoverage {
    polylogue_model::capture::RootCoverage {
        path_bytes: path.as_bytes().to_vec(),
        status,
        entries,
        unavailable_paths: Vec::new(),
    }
}

fn entry(
    class: polylogue_model::capture::Classification,
    pattern: &str,
    reason: Option<&str>,
    files: u64,
) -> polylogue_model::capture::Entry {
    polylogue_model::capture::Entry {
        class,
        pattern: pattern.into(),
        reason: reason.map(str::to_owned),
        files,
        bytes: files * 1024,
        newest_mtime_ms: None,
    }
}

/// A new archive answers from its first snapshot: up to date, zero counts,
/// its sources watched though none is present yet (docs/operations.md,
/// Status, edge cases).
#[test]
fn a_new_archive_is_up_to_date_with_its_sources_watched() {
    use polylogue_model::capture::Accessibility;
    let mut status = answer();
    status.store = Some(polylogue_store::Status {
        capture_coverage: vec![coverage(vec![
            root("/home/op/.claude", Accessibility::Absent, Vec::new()),
            root("/home/op/.codex", Accessibility::Absent, Vec::new()),
        ])],
        ..polylogue_store::Status::default()
    });
    assert_eq!(
        plain(&status, &[]),
        "\
polylogue  ✓ up to date
           standalone · daemon 4479 · up 17h35m · idle · /home/op/archive

ARCHIVE    0 sessions   0 messages   0 tool calls   0 documents
           0 inputs (0 B decoded) · 0 revisions

READING    parsing current · session figures current · search current

SOURCES    walked 14:04:00
           /home/op/.claude · absent
           /home/op/.codex · absent
"
    );
    assert_eq!(line(&status, &[], NOW), "up to date");
}

/// Before any snapshot the reply has no figures; the headline, the note and
/// `--line` say since when in the same words, and nothing is printed as an
/// error (docs/operations.md#status).
#[test]
fn a_reply_without_figures_says_since_when() {
    for (since, words) in [
        (
            NotReadSince::Start,
            "figures not read yet since the daemon started",
        ),
        (
            NotReadSince::Install,
            "figures not read yet since the new index was installed",
        ),
    ] {
        let mut status = answer();
        status.answers_from = AnswersFrom::NotReadYet { since };
        status.notices = None;
        status.running = None;
        assert_eq!(
            plain(&status, &[words]),
            format!(
                "\
polylogue  ◐ {words}
           standalone · daemon 4479 · up 17h35m · idle · /home/op/archive

NOTES      ⚠ {words}
"
            )
        );
        assert_eq!(line(&status, &[words.to_owned()], NOW), words);
    }
}

/// An index rebuilt beside a readable one: the reply answers from the
/// snapshot, the build shown beside it, and the archive's figures are
/// labelled as the installed index's (docs/operations.md#status).
#[test]
fn a_rebuild_beside_a_readable_index_labels_the_installed_figures() {
    let mut reading = current(
        "reading stored files",
        3,
        Some(StepCount {
            done: 1 << 30,
            total: Some(4 << 30),
            unit: CountUnit::Bytes,
        }),
    );
    reading.rate_seconds = Some(30.);
    let mut status = answer();
    status.store = Some(polylogue_store::Status {
        sessions: 12,
        messages: 340,
        tool_calls: 56,
        inputs: 14,
        revisions: 15,
        stored_bytes: 5 << 30,
        ..polylogue_store::Status::default()
    });
    status.index_build = Some(IndexBuild {
        reason: polylogue_store::index::file::RebuildReason::SchemaDiffers,
        chains: 14,
        chains_read: 3,
        bytes: 4 << 30,
        bytes_read: 1 << 30,
    });
    status.build_progress = Some(progress(reading, Vec::new(), &["installing"]));
    assert_eq!(
        plain(&status, &[]),
        "\
polylogue  ◐ building the index, step 1 of 2
           standalone · daemon 4479 · up 17h35m · idle · /home/op/archive

ARCHIVE    12 sessions   340 messages   56 tool calls   0 documents
           14 inputs (5.0 GiB decoded) · 15 revisions
           from the installed index, until the build replaces it

INDEX      building the index from 4.0 GiB of stored files: its tables changed
           reads and searches use the installed index until the rebuilt one replaces it
           #######.......................  25% of reading stored files

           ▸ reading stored files     3m  1.0 GiB of 4.0 GiB · 3 of 14 files
                                          estimate after a minute of progress
           · installing

           whole build: time left known once the last step runs (1 step to come)
"
    );
    assert_eq!(line(&status, &[], NOW), "building the index, step 1 of 2");
}

fn window(
    tool: &str,
    limit: &str,
    minutes: i64,
    state: polylogue_store::measure::quota::CurrentUse,
) -> polylogue_store::measure::quota::CurrentWindow {
    polylogue_store::measure::quota::CurrentWindow {
        tool: tool.into(),
        limit: limit.into(),
        window_minutes: minutes,
        state,
    }
}

/// `--line`: the headline's words, the counts, then each measured window
/// grouped by tool; a window not measured is left out of the line and is
/// `not_measured` with its reason in `--json` (docs/operations.md#status).
#[test]
fn the_line_carries_measured_quota_windows_after_the_counts() {
    use polylogue_store::measure::quota::{CurrentUse, NotMeasured};
    let measured = |used_percent| CurrentUse::Measured {
        used_percent,
        measured_at: at(NOW - 2 * MINUTE),
        resets_at: Some(at(NOW + 90 * MINUTE)),
        status: None,
    };
    let mut status = answer();
    status.running = Some(Running::Measured {
        waiting: 1,
        sessions: vec![
            session("claude-code:a1", LiveState::Waiting, NOW - 7 * MINUTE),
            session("codex:b2", LiveState::Working, NOW - 2 * MINUTE),
        ],
    });
    status.quota = vec![
        window("claude-code", "five_hour", 300, measured(61.4)),
        window("claude-code", "seven_day", 10_080, measured(20.0)),
        window(
            "codex",
            "codex",
            300,
            CurrentUse::NotMeasured {
                reason: NotMeasured::NoReading,
            },
        ),
    ];
    assert_eq!(
        line(&status, &[], NOW),
        "up to date · 2 running · 1 waiting 7m · claude-code 5h 61% 7d 20%"
    );
    assert_eq!(
        serde_json::to_value(&status.quota[2]).unwrap(),
        serde_json::json!({
            "tool": "codex",
            "limit": "codex",
            "window_minutes": 300,
            "state": "not_measured",
            "reason": "no_reading",
        })
    );
    status.quota.clear();
    status
        .quota
        .push(window("codex", "codex", 300, measured(12.0)));
    assert_eq!(
        line(
            &status,
            &[
                "parsing is behind the stored bytes".into(),
                "synthetic note".into()
            ],
            NOW
        ),
        "parsing is behind the stored bytes (+1 more) · 2 running · 1 waiting 7m · codex 5h 12%"
    );
}

/// SOURCES: a root that is absent, and one whose walk found files no
/// declared pattern names, counted and marked (docs/operations.md#status).
#[test]
fn sources_show_an_absent_root_and_unrecognized_files() {
    use polylogue_model::capture::{Accessibility, Classification};
    let mut status = answer();
    let mut incomplete = root(
        "/home/op/.gemini",
        Accessibility::Incomplete,
        vec![entry(
            Classification::Ignored,
            "**",
            Some("symbolic link is not followed"),
            1,
        )],
    );
    incomplete.unavailable_paths = vec![b"/home/op/.gemini/tmp".to_vec()];
    status.store = Some(polylogue_store::Status {
        capture_coverage: vec![coverage(vec![
            root(
                "/home/op/.codex",
                Accessibility::Accessible,
                vec![
                    entry(
                        Classification::Collected,
                        "sessions/**/rollout-*.jsonl",
                        None,
                        20,
                    ),
                    entry(Classification::Collected, "history.jsonl", None, 1),
                    entry(
                        Classification::Unrecognized,
                        "**",
                        Some("no declared collection or ignore pattern matched"),
                        3,
                    ),
                ],
            ),
            root("/home/op/.claude", Accessibility::Absent, Vec::new()),
            incomplete,
        ])],
        ..polylogue_store::Status::default()
    });
    let text = plain(&status, &[]);
    let sources = text
        .split("\n\n")
        .find(|section| section.starts_with("SOURCES"))
        .unwrap_or_else(|| panic!("{text}"));
    assert_eq!(
        sources,
        "\
SOURCES    walked 14:04:00
           codex /home/op/.codex · present · 21 collected · 3 unrecognized
           /home/op/.claude · absent
           /home/op/.gemini · incomplete, 1 path unreadable · 1 ignored
             ignored: 1 symbolic link is not followed
"
    );
}

/// Every chain state the census counts (`model::chain::ChainStatusName`)
/// is worded in the READING section, in the view's order.
#[test]
fn every_chain_state_the_census_counts_has_its_words() {
    use polylogue_model::chain::ChainStatusName;
    let mut status = answer();
    status.store = Some(polylogue_store::Status {
        parse_states: ChainStatusName::ALL
            .iter()
            .map(|state| (state.as_str().to_owned(), 1))
            .collect(),
        ..Default::default()
    });
    let view = plain(&status, &[]);
    let reading = view
        .lines()
        .find(|line| line.contains("parsing ·"))
        .unwrap_or_else(|| panic!("{view}"));
    assert_eq!(
        reading.trim(),
        "1 parsing · 1 waiting to grow · 1 refused · 1 failed · 1 without a parser · 1 repeated"
    );
}
