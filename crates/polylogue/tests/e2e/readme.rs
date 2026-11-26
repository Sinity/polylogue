//! Every example line in the README, run as written against the Morrow
//! archive that `polylogue demo` builds, with a hand-written expectation.
//! The examples name fixed dates, so their answers do not depend on the clock.

use rustix::pty::{OpenptFlags, grantpt, ioctl_tiocgptpeer, openpt, unlockpt};
use std::fs::File;
use std::io::Read;
use std::process::Stdio;

/// The `sh` examples between the tour heading and the reading list, one
/// command per line, without their trailing comments.
fn readme_examples() -> Vec<String> {
    let readme = include_str!("../../../../README.md");
    let section = readme
        .split_once("## A tour on the Morrow archive")
        .unwrap()
        .1
        .split_once("## Where to read next")
        .unwrap()
        .0;
    let mut lines = Vec::new();
    let mut in_block = false;
    for line in section.lines() {
        if line.starts_with("```") {
            in_block = !in_block;
        } else if in_block {
            let command = line.split_once("  #").map_or(line, |(command, _)| command);
            lines.push(command.trim_end().to_owned());
        }
    }
    lines
}

/// One terminal that lasts the whole test: row numbers belong to a terminal.
struct Terminal {
    master: File,
    slave: File,
}

impl Terminal {
    fn new() -> Self {
        let flags = OpenptFlags::RDWR | OpenptFlags::NOCTTY | OpenptFlags::CLOEXEC;
        let master = openpt(flags).unwrap();
        grantpt(&master).unwrap();
        unlockpt(&master).unwrap();
        let slave = ioctl_tiocgptpeer(&master, flags).unwrap();
        rustix::fs::fcntl_setfl(&master, rustix::fs::OFlags::NONBLOCK).unwrap();
        Self {
            master: File::from(master),
            slave: File::from(slave),
        }
    }

    /// Runs `line` in a shell where `polylogue` is the binary under test,
    /// standard output on the terminal; returns what it printed. The
    /// terminal is read while the command runs: a terminal holds only what
    /// its buffers fit, and a command writing more waits until it is read.
    fn run(&mut self, env: &crate::Fixture, line: &str) -> String {
        use std::os::fd::AsRawFd;
        let mut shell = std::process::Command::new("sh");
        env.environment(&mut shell);
        let script = format!("polylogue() {{ \"$POLYLOGUE_TEST_BINARY\" \"$@\"; }}\n{line}");
        let mut child = shell
            .env("POLYLOGUE_TEST_BINARY", crate::BIN)
            .env_remove("NO_COLOR")
            .env("NO_COLOR", "")
            .args(["-c", &script])
            .stdin(self.slave.try_clone().unwrap())
            .stdout(self.slave.try_clone().unwrap())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let mut stderr = child.stderr.take().unwrap();
        let errors = std::thread::spawn(move || {
            let mut bytes = Vec::new();
            stderr.read_to_end(&mut bytes).map(|_| bytes)
        });
        let mut bytes = Vec::new();
        let status = loop {
            self.read_available(&mut bytes);
            if let Some(status) = child.try_wait().unwrap() {
                break status;
            }
            let mut event = libc::pollfd {
                fd: self.master.as_raw_fd(),
                events: libc::POLLIN,
                revents: 0,
            };
            // SAFETY: event is writable, and the master stays open through poll.
            #[allow(unsafe_code)]
            let _ready = unsafe { libc::poll(&mut event, 1, 20) };
        };
        // What the command wrote before it ended.
        self.read_available(&mut bytes);
        let stderr = errors.join().unwrap().unwrap();
        let text = String::from_utf8(bytes).unwrap().replace("\r\n", "\n");
        assert!(
            status.success(),
            "{line}: {status}; stderr:\n{}\nterminal:\n{text}",
            String::from_utf8_lossy(&stderr)
        );
        text
    }

    /// Appends what the terminal holds now; the master never blocks.
    fn read_available(&mut self, bytes: &mut Vec<u8>) {
        let mut chunk = [0; 4096];
        loop {
            match self.master.read(&mut chunk) {
                Ok(0) => return,
                Ok(n) => bytes.extend_from_slice(&chunk[..n]),
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => return,
                Err(error) => panic!("read the terminal: {error}"),
            }
        }
    }
}

#[test]
fn readme_examples_run_against_morrow() {
    assert_eq!(
        readme_examples(),
        [
            "polylogue demo ./morrow",
            "polylogue daemon",
            "polylogue",
            "polylogue 'repo:morrow since:2026-10-05'",
            "polylogue chrono",
            "polylogue @1 read",
            "polylogue @1 note 'chrono parsing fixed' --as decision",
        ]
    );
    let temp = tempfile::tempdir().unwrap();
    let operator = crate::Fixture::new(
        &temp.path().join("operator-archive"),
        &temp.path().join("operator-home"),
        &temp.path().join("config"),
    );
    let mut terminal = Terminal::new();
    let examples = readme_examples();
    let demo = terminal.run(&operator, &examples[0]);
    assert!(demo.starts_with("# Morrow tour\n"), "{demo}");
    let directory = temp.path().join("morrow");
    assert!(directory.join("transcript.md").is_file());
    let env = crate::Fixture::new(
        &directory.join("archive"),
        &directory.join("home"),
        &directory.join("config"),
    );
    // The daemon line: the daemon serves the archive the demo built.
    let _daemon = super::cli::start_daemon(&env);
    let status = super::cli::wait_status(&env, |_| true);
    // Logical sessions as `sessions` lists them, and every session.
    assert_eq!(status["store"]["sessions"], 6);
    assert_eq!(status["store"]["sessions_physical"], 8);

    // The orientation screen counts the eight sessions FACTS.md lists, six
    // of them logical sessions, and the messages they count, 12 + 18 + 6 +
    // 11 + 13 + 10 + 6 + 4 = 80 (#7's abandoned branch of 2 is stored but off
    // the active path).
    let orientation = terminal.run(&env, &examples[2]);
    assert!(
        orientation.contains(
            "ARCHIVE    6 sessions, 8 with subagents, forks, continuations and resumes   80 messages"
        ),
        "{orientation}"
    );

    // Only session #5 is in the repository the morrow remote names and
    // started on or after 2026-10-05 (expected.json examples 5 and 26).
    let recent = terminal.run(&env, &examples[3]);
    assert_eq!(
        listed_ids(&recent),
        ["codex:01a20000-0005-7000-8000-000000000005"]
    );
    assert!(recent.contains("Review parser"), "{recent}");

    // The word "chrono" occurs in the sessions of README wording (#8), Date
    // parsing (#2 and #4) and Date format edge cases (#7); the strongest
    // match is the README question that is only about chrono.
    let hits = terminal.run(&env, &examples[4]);
    let mut ids = listed_ids(&hits);
    assert_eq!(
        ids[0], "claude-ai:0b9d4e6c-0008-4000-8000-000000000008",
        "{hits}"
    );
    ids.sort();
    assert_eq!(
        ids,
        [
            "chatgpt:68f0a1b2-0007-8000-8000-000000000007",
            "claude-ai:0b9d4e6c-0008-4000-8000-000000000008",
            "claude-code:7f3a2c10-0002-4000-8000-000000000002",
            "claude-code:7f3a2c10-0004-4000-8000-000000000004",
        ]
    );

    // Row 1 is that README session, read in full.
    let read = terminal.run(&env, &examples[5]);
    assert!(read.starts_with("README wording\n"), "{read}");
    assert!(
        read.contains(
            "morrow parses human-written dates into chrono types and rejects ambiguous input."
        ),
        "{read}"
    );

    // The note lands on row 1 as the operator's decision.
    terminal.run(&env, &examples[6]);
    let notes = super::cli::cli(&env, &["--json", "annotations"]);
    // Beside the decision the tour recorded on session #2.
    let mut written: Vec<_> = super::cli::query_rows(&notes)["rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| {
            (
                row["kind"].as_str().unwrap().to_owned(),
                row["value"].as_str().unwrap().to_owned(),
                row["about"].as_str().unwrap().to_owned(),
            )
        })
        .collect();
    written.sort();
    assert_eq!(
        written,
        [
            (
                "decision".to_owned(),
                "Use chrono for date parsing.".to_owned(),
                "session:claude-code:7f3a2c10-0002-4000-8000-000000000002".to_owned()
            ),
            (
                "decision".to_owned(),
                "chrono parsing fixed".to_owned(),
                "session:claude-ai:0b9d4e6c-0008-4000-8000-000000000008".to_owned()
            ),
        ]
    );
}

/// The session IDs in a printed list, in row order.
fn listed_ids(list: &str) -> Vec<&str> {
    list.lines()
        .filter_map(|line| line.strip_prefix('@')?.split_whitespace().nth(1))
        .collect()
}
