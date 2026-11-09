use super::read_current::Items;
use super::*;
use polylogue_model::read::ReadPage;

fn next_arguments(call: &polylogue_model::read::ReadyCall) -> Vec<String> {
    let stages = polylogue_model::command::effective_stages(&call.cli);
    assert_eq!(stages.len(), 1);
    assert_eq!(stages[0].program, "polylogue");
    let mut args = vec!["--json".to_owned()];
    args.extend(stages[0].args.clone());
    args
}
#[test]
fn cli_read_pages_cover_utf8_once_and_refuse_malformed_and_revoked_cursors() {
    let dir = tempfile::tempdir().unwrap();
    let home = dir.path().join("home");
    let archive = dir.path().join("archive");
    let project = home.join(".claude/projects/-pages");
    std::fs::create_dir_all(&project).unwrap();
    let expected = "λ🙂 synthetic prompt with several pages";
    std::fs::write(project.join("pages.jsonl"), line("u1", "pages", expected)).unwrap();
    let env = crate::Fixture::new(&archive, &home, &dir.path().join("config"));
    let daemon = start_daemon(&env);
    wait_status(&env, |status| {
        status["store"]["messages"] == 1 && status["store"]["search_pending"] == 0
    });
    let whole = cli(
        &env,
        &[
            "--deadline",
            "none",
            "--json",
            "read",
            "session:claude-code:pages",
            "--page-tokens",
            "100000",
        ],
    );
    assert!(whole.status.success(), "{whole:?}");
    let whole: ReadPage = serde_json::from_value(json(&whole)).unwrap();
    let finite_source = serde_json::to_vec(&whole.document.items).unwrap().len();
    let mut args = vec![
        "--deadline".into(),
        "none".into(),
        "--json".into(),
        "read".into(),
        "session:claude-code:pages".into(),
        "--page-tokens".into(),
        "1".into(),
    ];
    let mut joined = Items::default();
    let mut seen = std::collections::BTreeSet::new();
    let mut first_next = None;
    let mut complete = false;
    for _ in 0..=finite_source {
        let out = cli(&env, &args.iter().map(String::as_str).collect::<Vec<_>>());
        assert!(out.status.success(), "{out:?}");
        let page: ReadPage = serde_json::from_value(json(&out)).unwrap();
        assert!(
            !page.document.items.is_empty(),
            "continuation made no item progress"
        );
        // Items validates distinct complete items and contiguous, advancing
        // fragment coordinates; every page consumes the finite source.
        joined.push(&page);
        let Some(next) = page.document.footer.next else {
            complete = true;
            break;
        };
        assert!(seen.insert(next.cli.clone()), "repeated continuation");
        first_next.get_or_insert_with(|| next.clone());
        args = next_arguments(&next);
    }
    assert!(complete, "finite source delivery did not terminate");
    assert_eq!(joined.text(), expected);
    let next = first_next.expect("one-token page must continue");
    let mut invalid = next_arguments(&next);
    let cursor = invalid.iter().position(|word| word == "--cursor").unwrap() + 1;
    assert!(!invalid[cursor].is_empty());
    invalid[cursor] = "broken".into();
    let invalid = cli(
        &env,
        &invalid.iter().map(String::as_str).collect::<Vec<_>>(),
    );
    assert!(!invalid.status.success());
    assert_eq!(json(&invalid)["error"]["kind"], "invalid_query");
    drop(daemon);
    polylogue_testkit::delete_index(&archive);
    let _rebuilt = start_daemon(&env);
    wait_status(&env, |status| {
        status["store"]["messages"] == 1 && status["store"]["search_pending"] == 0
    });
    // A cursor from an earlier index generation is `stale_cursor`, never
    // restarted and never confused with a broken one (docs/query.md, Paging).
    let args = next_arguments(&next);
    let stale = cli(&env, &args.iter().map(String::as_str).collect::<Vec<_>>());
    assert!(!stale.status.success());
    assert_eq!(json(&stale)["error"]["kind"], "stale_cursor");
}

/// A session of three short prompts, read in pages too small for all three.
fn three_prompts(dir: &std::path::Path) -> (crate::Fixture, super::Daemon) {
    let home = dir.join("home");
    let project = home.join(".claude/projects/-pager");
    std::fs::create_dir_all(&project).unwrap();
    let lines: String = [
        ("u1", "opening question"),
        ("u2", "second marker beta"),
        ("u3", "third marker gamma"),
    ]
    .iter()
    .map(|(uuid, text)| line(uuid, "pager", text))
    .collect();
    std::fs::write(project.join("pager.jsonl"), lines).unwrap();
    let env = crate::Fixture::new(&dir.join("archive"), &home, &dir.join("config"));
    let daemon = start_daemon(&env);
    wait_status(&env, |status| {
        status["store"]["messages"] == 3 && status["store"]["search_pending"] == 0
    });
    (env, daemon)
}

const SMALL_PAGE: &[&str] = &[
    "--deadline",
    "none",
    "read",
    "session:claude-code:pager",
    "--page-tokens",
    "12",
];

/// The continue command a piped read printed on standard error.
fn continue_command(out: &std::process::Output) -> Option<Vec<String>> {
    let stderr = String::from_utf8(out.stderr.clone()).unwrap();
    let command = stderr
        .lines()
        .find_map(|line| line.strip_prefix("continue: "))?;
    let stages = polylogue_model::command::effective_stages(command);
    assert_eq!(stages.len(), 1, "{command}");
    assert_eq!(stages[0].program, "polylogue");
    Some(stages[0].args.clone())
}

#[test]
fn read_without_json_on_pipe_prints_continue_command_on_stderr() {
    let dir = tempfile::tempdir().unwrap();
    let (env, _daemon) = three_prompts(dir.path());
    let out = cli(&env, SMALL_PAGE);
    assert!(out.status.success(), "{out:?}");
    let stdout = String::from_utf8(out.stdout.clone()).unwrap();
    assert!(stdout.contains("opening question"), "{stdout}");
    assert!(!stdout.contains("third marker gamma"), "{stdout}");
    assert!(!stdout.contains("--cursor"), "the page alone is on stdout");
    let next = continue_command(&out).expect("a continue command on stderr");
    assert!(next.iter().any(|word| word == "--cursor"), "{next:?}");
}

#[test]
fn read_pages_tile_with_no_gap_or_overlap() {
    let dir = tempfile::tempdir().unwrap();
    let (env, _daemon) = three_prompts(dir.path());
    let mut args: Vec<String> = SMALL_PAGE.iter().map(|word| (*word).to_owned()).collect();
    args.extend(["--format".to_owned(), "json".to_owned()]);
    let mut joined = Items::default();
    let mut pages = 0;
    loop {
        let out = cli(&env, &args.iter().map(String::as_str).collect::<Vec<_>>());
        assert!(out.status.success(), "{out:?}");
        let document: polylogue_model::read::ReadDocument =
            serde_json::from_slice(&out.stdout).unwrap();
        assert!(!document.items.is_empty(), "a page made no progress");
        joined.push_document(&document);
        pages += 1;
        assert!(pages <= 20, "the reading did not end");
        match continue_command(&out) {
            Some(next) => args = next,
            None => break,
        }
    }
    assert!(pages > 1, "the small budget must page");
    assert_eq!(
        joined.text(),
        "opening questionsecond marker betathird marker gamma"
    );
}

#[test]
fn read_on_terminal_shows_whole_session_through_pager() {
    use rustix::pty::{OpenptFlags, grantpt, ioctl_tiocgptpeer, openpt, unlockpt};
    use std::io::Read;
    let dir = tempfile::tempdir().unwrap();
    let (env, _daemon) = three_prompts(dir.path());
    let flags = OpenptFlags::RDWR | OpenptFlags::NOCTTY | OpenptFlags::CLOEXEC;
    let master = openpt(flags).unwrap();
    grantpt(&master).unwrap();
    unlockpt(&master).unwrap();
    let slave = ioctl_tiocgptpeer(&master, flags).unwrap();
    let mut command = env.command();
    command
        .env("PAGER", "cat")
        .env("NO_COLOR", "")
        .args(SMALL_PAGE)
        .stdin(std::process::Stdio::null())
        .stdout(std::fs::File::from(slave))
        .stderr(std::process::Stdio::piped());
    let child = command.spawn().unwrap();
    drop(command);
    let reader = std::thread::spawn(move || {
        let mut bytes = Vec::new();
        if let Err(error) = std::fs::File::from(master).read_to_end(&mut bytes) {
            assert_eq!(error.raw_os_error(), Some(5));
        }
        bytes
    });
    let output = child.wait_with_output().unwrap();
    assert!(output.status.success(), "{output:?}");
    let shown = String::from_utf8(reader.join().unwrap()).unwrap();
    // One document under one header, every message once.
    assert_eq!(
        shown
            .matches("session:claude-code:pager · transcript")
            .count(),
        1,
        "{shown}"
    );
    assert!(shown.contains("opening question"), "{shown}");
    for marker in ["second marker beta", "third marker gamma"] {
        assert_eq!(shown.matches(marker).count(), 1, "{marker} in {shown}");
    }
    assert!(!String::from_utf8_lossy(&output.stderr).contains("continue:"));
}
