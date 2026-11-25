use super::*;
use crate::protocol::status::{Notice, PendingWrites, Warm};
use polylogue_model::Timestamp;
use polylogue_model::freshness::{CompletenessReason, ReadCompleteness, ReadFreshness};
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
        build_progress: None,
        warming: false,
        totals_error: None,
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
        notices: Notices::Measured {
            last_day: 0,
            latest: Vec::new(),
        },
        running: Running::Measured {
            waiting: 0,
            sessions: Vec::new(),
        },
        warm: Warm {
            vectors: None,
            vector_loads: 0,
            query_vectors_reused: 0,
        },
        mode: Mode::Standalone,
        freshness: ReadFreshness {
            local: ReadCompleteness::empty(),
            sinex: IndexFreshness::Standalone,
        },
        #[cfg(feature = "sinex")]
        primary: None,
        recent_log: Vec::new(),
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
            ("no_parser_yet", 15),
        ]
        .into_iter()
        .map(|(state, n)| (state.to_owned(), n))
        .collect(),
        unused: vec![
            ("claude-code:message.usage.cache".into(), 11_200_000),
            ("codex:payload.extra".into(), 5),
        ],
        search_pending: 0,
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
                            pattern: "projects/**/*.jsonl".into(),
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
        CompletenessDimension::Incomplete(vec![CompletenessReason::ParserBehind]);
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
    status.running = Running::Measured {
        waiting: 1,
        sessions: vec![waiting, working],
    };
    status.notices = Notices::Measured {
        last_day: 2,
        latest: vec![Notice {
            id: "notice-1".into(),
            at: at(today(14, 2, 0)),
            text: "synthetic notice".into(),
            events: Vec::new(),
        }],
    };
    status.recent_log = vec![
        record(today(14, 1, 10), Priority::Warning, "scope index: busy"),
        record(today(14, 2, 11), Priority::Info, "search thread: caught up"),
    ];
    assert_eq!(
        plain(&status, &["parsing is behind the stored bytes"]),
        "\
polylogue  standalone · daemon 4479 · up 17h35m · /home/op/archive

ARCHIVE    37,419 sessions   11.24M messages   4.77M tool calls   288 documents
           132,286 inputs · 98.6 GiB stored · 134,089 revisions

READING    #############################.  99.9%   109,793 of 109,887 chains read
           48 parsing · 31 failed · 15 without a parser
           parsing behind: parsing is behind the stored bytes · derivation current · search current

LAST PASS  12:58:04 · 4.2 s · 37 of 132,286 inputs changed
           12.4 MiB read · 1,204 messages written

RUNNING    2 running · 1 waiting 7m
           ◆ waiting     7m  claude-code:a1  on permission · Bash
           ● working     2m  codex:b2  2 files edited

NOTICES    2 in the last day · 14:02:00 synthetic notice

CAPTURE    host: 2 roots walked 13:00:00 · 1 absent · 43 files ignored, most as cache

UNUSED     claude-code:message.usage.cache ×11.20M not yet read · 1 more in --json

RECENT     14:01:10  ⚠ scope index: busy
           14:02:11  search thread: caught up

NOTES      ⚠ parsing is behind the stored bytes
"
    );
}

/// A first build in progress: its bar, rate, estimate and phase; the
/// sections the unreadable index cannot fill are left out.
#[test]
fn a_building_answer_renders_the_build_progress() {
    let mut status = answer();
    status.uptime_seconds = 300;
    status.loop_status.busy = true;
    let build = polylogue_store::archive::index_build::IndexBuild {
        serving_previous: false,
        chains: 21_678,
        chains_read: 4_120,
        bytes: 70_437_463_654,
        bytes_read: 39_513_699_123,
    };
    let unread = build.to_string();
    status.index_build = Some(build);
    status.build_progress = Some(polylogue_store::archive::build_progress::BuildProgress {
        phase: Some("read chains".into()),
        phase_since: Some(at(NOW - 34 * MINUTE)),
        bytes_per_second: Some(18_400_000.),
        chains_per_second: Some(2.4),
        rate_seconds: Some(300.),
        rate_varying: false,
        eta_seconds: Some(1_560),
        writer_busy: Some(0.97),
        worker_cpu_seconds: vec![3.0, 3.2],
        phase_progress: None,
    });
    status.totals_error = Some(unread.clone());
    status.running = Running::Unavailable {
        error: unread.clone(),
    };
    status.notices = Notices::Unavailable { error: unread };
    status.recent_log = vec![record(
        today(13, 31, 0),
        Priority::Info,
        "build: read chains: each stored chain read whole into runs",
    )];
    assert_eq!(
        plain(&status, &["the index is being built"]),
        "\
polylogue  standalone · daemon 4479 · up 5m · busy · /home/op/archive

BUILD      ################..............  56%   36.8 GiB of 65.6 GiB · 4,120 of 21,678 chains
           18.4 MB/s over the last 5m · 2.4 chains/s · about 26 min left
           phase read chains for 34m
           writer busy 97% · 2 workers, 3.1 s CPU/min each

RECENT     13:31:00  build: read chains: each stored chain read whole into runs

NOTES      ⚠ the index is being built
"
    );
}

/// Once the chains are read the writer's phase says how far it is; a round
/// with no chain to read shows no reading bar.
#[test]
fn a_writer_phase_renders_its_own_progress() {
    let mut status = answer();
    status.uptime_seconds = 300;
    status.loop_status.busy = true;
    let build = polylogue_store::archive::index_build::IndexBuild {
        serving_previous: false,
        chains: 0,
        chains_read: 0,
        bytes: 0,
        bytes_read: 0,
    };
    let unread = build.to_string();
    status.index_build = Some(build);
    let progress = |phase: &str, done, total, unit: &str| {
        Some(polylogue_store::archive::build_progress::BuildProgress {
            phase: Some(phase.into()),
            phase_since: Some(at(NOW - 12 * MINUTE)),
            bytes_per_second: None,
            chains_per_second: None,
            rate_seconds: None,
            rate_varying: false,
            eta_seconds: None,
            writer_busy: Some(1.),
            worker_cpu_seconds: Vec::new(),
            phase_progress: Some(polylogue_store::archive::build_progress::PhaseProgress {
                done,
                total,
                unit: unit.into(),
            }),
        })
    };
    status.build_progress = progress("compose", 2_556, None, "sessions");
    status.totals_error = Some(unread.clone());
    status.running = Running::Unavailable {
        error: unread.clone(),
    };
    status.notices = Notices::Unavailable { error: unread };
    assert_eq!(
        plain(&status, &["the index is being built"]),
        "\
polylogue  standalone · daemon 4479 · up 5m · busy · /home/op/archive

BUILD      phase compose for 12m · 2,556 sessions
           writer busy 100%

NOTES      ⚠ the index is being built
"
    );
    status.build_progress = progress("indexes", 12, Some(40), "indexes");
    assert!(
        plain(&status, &["the index is being built"])
            .contains("BUILD      phase indexes for 12m · 12 of 40 indexes\n")
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
    status.warming = true;
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
    assert!(
        text.lines().next().unwrap().contains(" · warming"),
        "{text}"
    );
}
