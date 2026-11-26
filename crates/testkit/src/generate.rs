//! Proptest strategies for activities and schedules, written from the cases
//! that break the stored view: parent links and forks, sidechains, equal
//! times and times out of file order, split responses, tool results in the
//! next record and later, error results, copied prefixes exact, edited and
//! cut short, chained resumes, compactions, non-ASCII text; and for
//! schedules files above the chunk size, cuts inside a line and inside a
//! multi-byte character, any arrival order, restarts and rebuilds.
//!
//! An activity is built from a stream of choices, each a small number where
//! zero is the simplest case, so shrinking a failure gives a smaller
//! activity.

use crate::File;
use crate::activity::{
    Activity, Body, Change, ChangeHow, Copy, Entry, Part, ResponseRecord, ToolResult, Transcript,
};
use crate::schedule::{Schedule, Step};
use proptest::prelude::*;

/// Words of generated text, some of them outside ASCII and some with
/// characters of three and four bytes.
const WORDS: &[&str] = &[
    "parse",
    "the",
    "dates",
    "grün",
    "naïve",
    "日本語",
    "🙂",
    "ß",
    "Ωmega",
    "ok",
    "réglé",
    "tests",
    "fails",
    "again",
];
const COMMANDS: &[&str] = &["cargo test", "cargo build", "ls src", "git status"];
const DIRECTORIES: &[&str] = &["/home/op/morrow", "/home/op/kestrel"];
/// 2026-03-01T10:00:00Z.
const START: i64 = 1_772_359_200_000;

/// Activities of one to three sessions and up to two resumes.
pub fn activity() -> impl Strategy<Value = Activity> {
    prop::collection::vec(any::<u8>(), 0..512).prop_map(|choices| build(&mut Choices::new(choices)))
}

/// An activity with its Claude Code files and a schedule for them.
pub fn delivered_activity() -> impl Strategy<Value = (Activity, Vec<File>, Schedule)> {
    activity().prop_flat_map(|activity| {
        let files = activity.claude_code_files();
        let schedule = schedule(&files);
        (Just(activity), Just(files), schedule)
    })
}

/// A schedule for `files`: each file whole or cut at up to two bytes
/// anywhere, the pieces of all files written in any order that keeps each
/// file's pieces in order, a store after every write, and after a store
/// sometimes a restart or a rebuild. The chunk is either larger than any
/// file or small enough that most files take several.
pub fn schedule(files: &[File]) -> impl Strategy<Value = Schedule> + use<> {
    let lens: Vec<usize> = files.iter().map(|f| f.bytes.len()).collect();
    let cuts = lens
        .iter()
        .map(|len| prop::collection::vec(0..=*len, 0..=2))
        .collect::<Vec<_>>();
    (cuts, prop_oneof![Just(crate::WHOLE_FILES), 128u64..2048])
        .prop_flat_map(move |(cuts, chunk)| {
            let mut pieces = Vec::new();
            for (file, mut ends) in cuts.into_iter().enumerate() {
                ends.push(lens[file]);
                ends.sort_unstable();
                ends.dedup();
                pieces.extend(ends.into_iter().map(|end| (file, end)));
            }
            let n = pieces.len();
            (
                Just(pieces).prop_shuffle(),
                prop::collection::vec(0u8..6, n),
                Just(chunk),
            )
        })
        .prop_map(|(shuffled, between, chunk)| {
            // A shuffle may put a file's later piece before an earlier one;
            // the k-th write of a file takes its k-th end.
            let mut ends: Vec<Vec<usize>> = Vec::new();
            for (file, end) in &shuffled {
                if ends.len() <= *file {
                    ends.resize(*file + 1, Vec::new());
                }
                ends[*file].push(*end);
            }
            for list in &mut ends {
                list.sort_unstable();
            }
            let mut taken = vec![0; ends.len()];
            let mut steps = Vec::new();
            for ((file, _), between) in shuffled.iter().zip(between) {
                let end = ends[*file][taken[*file]];
                taken[*file] += 1;
                steps.push(Step::Write { file: *file, end });
                steps.push(Step::Store);
                match between {
                    4 => steps.push(Step::Restart),
                    5 => steps.push(Step::Rebuild),
                    _ => {}
                }
            }
            Schedule { chunk, steps }
        })
}

/// Reads choices in order; past the end every choice is zero.
struct Choices {
    values: Vec<u8>,
    next: usize,
}

impl Choices {
    fn new(values: Vec<u8>) -> Self {
        Self { values, next: 0 }
    }

    /// A number below `n`.
    fn below(&mut self, n: usize) -> usize {
        let value = self.values.get(self.next).copied().unwrap_or(0);
        self.next += 1;
        value as usize % n.max(1)
    }

    /// True one time in `n`.
    fn one_in(&mut self, n: usize) -> bool {
        self.below(n) == n - 1
    }

    fn pick<'a>(&mut self, list: &[&'a str]) -> &'a str {
        list[self.below(list.len())]
    }

    fn text(&mut self) -> String {
        let n = 1 + self.below(4);
        (0..n)
            .map(|_| self.pick(WORDS))
            .collect::<Vec<_>>()
            .join(" ")
    }
}

/// Names and times as an activity is built.
struct Builder<'c> {
    choices: &'c mut Choices,
    records: u32,
    calls: u32,
}

impl Builder<'_> {
    /// A record uuid. IDs are scrambled so their order is not file order.
    fn uuid(&mut self) -> String {
        self.records += 1;
        let scrambled = self.records.wrapping_mul(2_654_435_761);
        format!("{scrambled:08x}-0000-4000-8000-{:012x}", self.records)
    }

    fn call_id(&mut self) -> String {
        self.calls += 1;
        format!("toolu_{:04}", self.calls)
    }

    /// The next time of a session: the same as the last, later, or earlier.
    fn time(&mut self, last: &mut i64) -> i64 {
        *last += match self.choices.below(5) {
            0 | 1 => 0,
            2 => 1_000,
            3 => 90_000,
            _ => -30_000,
        };
        *last
    }
}

/// Where a session's next entries attach.
struct Ends {
    /// The last record of each earlier main entry, oldest first.
    main: Vec<String>,
    side: Option<String>,
    clock: i64,
}

fn build(choices: &mut Choices) -> Activity {
    let mut b = Builder {
        choices,
        records: 0,
        calls: 0,
    };
    let mut activity = Activity::default();
    let originals = 1 + b.choices.below(3);
    for n in 0..originals {
        let cwd = b.choices.pick(DIRECTORIES);
        let mut transcript = Transcript::new(&session_id(n), cwd);
        let mut ends = Ends {
            main: Vec::new(),
            side: None,
            clock: START + n as i64 * 3_600_000,
        };
        let turns = 1 + b.choices.below(4);
        for _ in 0..turns {
            turn(&mut b, &mut transcript, &mut ends);
        }
        activity.sessions.push(transcript);
    }
    let resumes = b.choices.below(3);
    for n in 0..resumes {
        let from = b.choices.below(activity.sessions.len());
        let source = activity.lines(from);
        // Whole entries, and none from the first sidechain entry on.
        let copyable = source
            .iter()
            .take_while(|line| !line.entry.sidechain)
            .count();
        if copyable == 0 {
            continue;
        }
        let lines = 1 + b.choices.below(copyable);
        let changeable: Vec<usize> = source[..lines]
            .iter()
            .enumerate()
            .filter(|(_, line)| line.change.is_none() && changeable(line.entry))
            .map(|(i, _)| i)
            .collect();
        let change = match b.choices.below(3) {
            1 if !changeable.is_empty() => Some(ChangeHow::Edited),
            2 if !changeable.is_empty() => Some(ChangeHow::CutShort),
            _ => None,
        }
        .map(|how| Change {
            line: changeable[b.choices.below(changeable.len())],
            how,
        });
        let last = source[lines - 1].entry;
        let clock = source[..lines]
            .iter()
            .map(|line| line.entry.at)
            .max()
            .unwrap_or(START);
        let cwd = activity.sessions[from].cwd.clone();
        let mut transcript = Transcript::new(&session_id(originals + n), &cwd);
        transcript.copy = Some(Copy {
            from,
            lines,
            change,
        });
        let mut ends = Ends {
            main: vec![last.last_record().to_owned()],
            side: None,
            clock: clock + 86_400_000,
        };
        let turns = 1 + b.choices.below(2);
        for _ in 0..turns {
            turn(&mut b, &mut transcript, &mut ends);
        }
        activity.sessions.push(transcript);
    }
    activity
}

fn session_id(n: usize) -> String {
    format!("5e551000-0000-4000-8000-{:012x}", n + 1)
}

/// A one-record response with a text part of at least two characters: what
/// a copy can catch in progress.
fn changeable(entry: &Entry) -> bool {
    match &entry.body {
        Body::Response { records, .. } => {
            records.len() == 1
                && records[0]
                    .parts
                    .iter()
                    .find_map(|part| match part {
                        Part::Text(text) => Some(text.chars().count() >= 2),
                        _ => None,
                    })
                    .unwrap_or(false)
        }
        _ => false,
    }
}

/// A prompt and what follows it: a response, perhaps split, perhaps with
/// calls whose results come next and a closing response. Sometimes the turn
/// is a sidechain, forks from an earlier entry, or follows a compaction.
fn turn(b: &mut Builder<'_>, transcript: &mut Transcript, ends: &mut Ends) {
    let sidechain = b.choices.one_in(5);
    let mut parent = if sidechain {
        ends.side.clone()
    } else if ends.main.len() >= 2 && b.choices.one_in(4) {
        // A fork: a reply to an entry that already has one.
        let n = b.choices.below(ends.main.len() - 1);
        Some(ends.main[n].clone())
    } else {
        ends.main.last().cloned()
    };
    let mut entry = |b: &mut Builder<'_>, parent: &mut Option<String>, uuid: String, body: Body| {
        let at = b.time(&mut ends.clock);
        let entry = Entry {
            uuid,
            parent: parent.clone(),
            at,
            sidechain,
            body,
        };
        *parent = Some(entry.last_record().to_owned());
        transcript.entries.push(entry);
    };
    if !sidechain && parent.is_some() && b.choices.one_in(6) {
        let boundary = b.uuid();
        let uuid = b.uuid();
        let summary = format!("Summary: {}", b.choices.text());
        entry(b, &mut parent, uuid, Body::Compaction { boundary, summary });
    }
    let uuid = b.uuid();
    let prompt = b.choices.text();
    entry(b, &mut parent, uuid, Body::Prompt(prompt));
    let (records, calls) = response(b, true);
    let uuid = records[0].uuid.clone();
    entry(
        b,
        &mut parent,
        uuid.clone(),
        Body::Response {
            key: format!("msg_{uuid}"),
            records,
        },
    );
    if !calls.is_empty() {
        let results: Vec<ToolResult> = calls
            .iter()
            .map(|call| {
                let output = b.choices.text();
                if b.choices.one_in(3) {
                    ToolResult::error(call, &output)
                } else {
                    ToolResult::ok(call, &output)
                }
            })
            .collect();
        if b.choices.one_in(2) {
            for result in results {
                let uuid = b.uuid();
                entry(b, &mut parent, uuid, Body::Results(vec![result]));
            }
        } else {
            let uuid = b.uuid();
            entry(b, &mut parent, uuid, Body::Results(results));
        }
        let (records, _) = response(b, false);
        let uuid = records[0].uuid.clone();
        entry(
            b,
            &mut parent,
            uuid.clone(),
            Body::Response {
                key: format!("msg_{uuid}"),
                records,
            },
        );
    }
    let last = parent.expect("a turn writes entries");
    if sidechain {
        ends.side = Some(last);
    } else {
        ends.main.push(last);
    }
}

/// A response's records and its call IDs: one to three records, each with
/// a part or two, calls only when `calls` allows them.
fn response(b: &mut Builder<'_>, calls: bool) -> (Vec<ResponseRecord>, Vec<String>) {
    let count = if b.choices.one_in(3) {
        2 + b.choices.below(2)
    } else {
        1
    };
    let mut ids = Vec::new();
    let records = (0..count)
        .map(|_| {
            let mut parts = Vec::new();
            if b.choices.one_in(4) {
                parts.push(Part::Thinking(b.choices.text()));
            }
            let kinds = 1 + b.choices.below(2);
            for _ in 0..kinds {
                if calls && b.choices.one_in(2) {
                    let id = b.call_id();
                    let command = b.choices.pick(COMMANDS);
                    parts.push(Part::call(&id, command));
                    ids.push(id);
                } else {
                    parts.push(Part::Text(b.choices.text()));
                }
            }
            ResponseRecord {
                uuid: b.uuid(),
                parts,
            }
        })
        .collect();
    (records, ids)
}

#[cfg(test)]
mod tests;
