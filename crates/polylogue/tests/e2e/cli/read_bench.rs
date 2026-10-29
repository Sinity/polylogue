//! The opt-in native benchmark uses this existing CLI/daemon test owner.
use super::*;
use serde_json::{Value, json};
use std::collections::HashSet;

const MAIN_MESSAGES: usize = 256;
const BODY_BYTES: usize = 64 * 1024;
const SMALL_SESSIONS: usize = 8;
const SMALL_MESSAGES: usize = 4;
const PAGE_BUDGET: usize = 32_768;
const WINDOW_PAGES: usize = 2;
const LARGE: &str = "claude-code:read-bench-large";

fn body(n: usize) -> String {
    let mut text = format!("readbenchneedle beginmsg{n:04} Łódź 🙂\n");
    let end = format!("\nendmsg{n:04}");
    while text.len() + end.len() < BODY_BYTES {
        text.push_str("steady river amber stone archive page visible content ");
    }
    text.truncate(BODY_BYTES - end.len());
    text.push_str(&end);
    text
}

fn request(env: &crate::Fixture, args: &[&str]) -> (Value, f64) {
    let began = Instant::now();
    let output = cli(env, args);
    let seconds = began.elapsed().as_secs_f64();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    (json(&output), seconds)
}

fn memory(pid: u32) -> Value {
    let text = std::fs::read_to_string(format!("/proc/{pid}/status")).unwrap();
    let bytes = |field: &str| {
        text.lines()
            .find_map(|line| line.strip_prefix(field))
            .and_then(|value| value.split_whitespace().next())
            .and_then(|value| value.parse::<u64>().ok())
            .map(|kb| kb * 1024)
            .expect("daemon memory counter unavailable")
    };
    json!({"rss_bytes":bytes("VmRSS:"),"peak_rss_bytes":bytes("VmHWM:")})
}

fn summary(samples: &[f64]) -> Value {
    assert!(!samples.is_empty());
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    json!({"samples":samples,"count":sorted.len(),"median":sorted[(sorted.len()-1)/2],
        "p95":sorted[(sorted.len()*95).div_ceil(100)-1],"max":sorted.last().unwrap()})
}

fn assert_status(status: &Value) {
    assert_eq!(status["store"]["sessions"], SMALL_SESSIONS + 1);
    assert_eq!(
        status["store"]["messages"],
        MAIN_MESSAGES + SMALL_SESSIONS * SMALL_MESSAGES
    );
    assert_eq!(status["store"]["search_pending"], 0);
    assert!(
        status["last_pass"]["errors"].as_array().unwrap().is_empty(),
        "{status}"
    );
    assert!(
        status["last_pass"]["unreadable_roots"]
            .as_array()
            .unwrap()
            .is_empty(),
        "{status}"
    );
}

fn assert_sessions(answer: &Value) {
    let actual: Vec<_> = answer["rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| row["id"].as_str().unwrap().to_owned())
        .collect();
    let expected: Vec<_> = std::iter::once(LARGE.to_owned())
        .chain((0..SMALL_SESSIONS).map(|n| format!("claude-code:read-bench-small-{n}")))
        .collect();
    assert_eq!(actual, expected);
    assert_eq!(answer["count"], SMALL_SESSIONS + 1);
}

fn verify_text(text: &str, expected: &[String]) {
    let mut previous = 0;
    for body in expected {
        let position = text[previous..]
            .find(body)
            .map(|found| previous + found)
            .expect("read omitted or changed a source body");
        assert!(
            !text[previous..position].contains("readbenchneedle"),
            "read duplicated a source body"
        );
        previous = position + body.len();
    }
    assert!(
        !text[previous..].contains("readbenchneedle"),
        "read duplicated a source body"
    );
}

fn walk(env: &crate::Fixture, expected: &[String]) -> (Value, String) {
    let began = Instant::now();
    let mut cursor = None;
    let mut seen = HashSet::new();
    let mut text = String::new();
    let mut samples = Vec::new();
    let budget = PAGE_BUDGET.to_string();
    // A page limit from this fixed population, so a broken cursor cannot loop.
    let max_pages = expected.iter().map(String::len).sum::<usize>() / PAGE_BUDGET + MAIN_MESSAGES;
    for _ in 0..max_pages {
        let mut args = vec!["--json", "read", LARGE, "--budget", &budget];
        if let Some(token) = cursor.as_deref() {
            args.extend(["--cursor", token]);
        }
        let (page, seconds) = request(env, &args);
        let body = page["text"].as_str().expect("read page has no text");
        assert!(!body.is_empty(), "empty page cannot establish progress");
        assert!(
            body.chars().count() <= PAGE_BUDGET * 4,
            "page exceeded its requested character budget"
        );
        text.push_str(body);
        samples.push(seconds);
        cursor = page["next"].as_str().map(str::to_owned);
        if let Some(token) = &cursor {
            assert!(seen.insert(token.clone()), "read cursor repeated");
        }
        if cursor.is_none() {
            break;
        }
    }
    assert!(
        cursor.is_none(),
        "read did not finish the declared population"
    );
    let seconds = began.elapsed().as_secs_f64();
    let verification = Instant::now();
    verify_text(&text, expected);
    let digest = blake3::hash(text.as_bytes()).to_hex().to_string();
    (
        json!({"seconds":seconds,"first_page_seconds":samples[0],"page_seconds":summary(&samples),
        "pages":samples.len(),"rendered_bytes":text.len(),"rendered_blake3":digest,
        "source_messages_verified":expected.len(),"verification_seconds":verification.elapsed().as_secs_f64()}),
        digest,
    )
}

struct WindowStart {
    name: &'static str,
    byte: usize,
    cursor: Option<String>,
}

fn reference_windows(env: &crate::Fixture, bodies: &[String]) -> (String, Vec<WindowStart>, Value) {
    let began = Instant::now();
    let budget = bodies.iter().map(String::len).sum::<usize>().to_string();
    eprintln!("native read bench: starting complete reference read");
    let (whole, seconds) = request(env, &["--json", "read", LARGE, "--budget", &budget]);
    assert!(
        whole["next"].is_null(),
        "reference read must include the whole declared population"
    );
    let reference = whole["text"].as_str().unwrap().to_owned();
    verify_text(&reference, bodies);
    let characters = reference.chars().count();
    let pages = characters.div_ceil(PAGE_BUDGET * 4);
    let mut starts = Vec::new();
    for (name, page) in [
        ("first", 0),
        ("middle", pages / 2),
        ("tail", pages - WINDOW_PAGES),
    ] {
        let character = page * PAGE_BUDGET * 4;
        let byte = reference.char_indices().nth(character).unwrap().0;
        let cursor = if page == 0 {
            None
        } else {
            eprintln!("native read bench: preparing {name} reference cursor");
            let budget = (page * PAGE_BUDGET).to_string();
            let (prefix, _) = request(env, &["--json", "read", LARGE, "--budget", &budget]);
            assert_eq!(prefix["text"].as_str().unwrap(), &reference[..byte]);
            Some(
                prefix["next"]
                    .as_str()
                    .expect("reference prefix must supply a real continuation")
                    .to_owned(),
            )
        };
        starts.push(WindowStart { name, byte, cursor });
    }
    let facts = json!({"seconds":seconds,"setup_seconds":began.elapsed().as_secs_f64(),"source_messages_verified":bodies.len(),"rendered_bytes":reference.len(),"rendered_characters":characters,"rendered_blake3":blake3::hash(reference.as_bytes()).to_hex().to_string()});
    (reference, starts, facts)
}

fn window(env: &crate::Fixture, start: &WindowStart, reference: &str) -> Value {
    let began = Instant::now();
    let mut cursor = start.cursor.clone();
    let mut text = String::new();
    let mut samples = Vec::new();
    let budget = PAGE_BUDGET.to_string();
    for _ in 0..WINDOW_PAGES {
        let mut args = vec!["--json", "read", LARGE, "--budget", &budget];
        if let Some(token) = cursor.as_deref() {
            args.extend(["--cursor", token]);
        }
        let (page, seconds) = request(env, &args);
        let body = page["text"].as_str().unwrap();
        assert!(!body.is_empty());
        assert!(body.chars().count() <= PAGE_BUDGET * 4);
        text.push_str(body);
        samples.push(seconds);
        cursor = page["next"].as_str().map(str::to_owned);
        if cursor.is_none() {
            break;
        }
    }
    let seconds = began.elapsed().as_secs_f64();
    let expected: String = reference[start.byte..]
        .chars()
        .take(WINDOW_PAGES * PAGE_BUDGET * 4)
        .collect();
    assert_eq!(
        text, expected,
        "bounded pages changed or omitted the requested reference slice"
    );
    let reached_end = expected.len() == reference.len() - start.byte;
    assert_eq!(
        cursor.is_none(),
        reached_end,
        "continuation must describe remaining content honestly"
    );
    json!({"seconds":seconds,"first_page_seconds":samples[0],"page_seconds":summary(&samples),"pages":samples.len(),"rendered_bytes":text.len(),"rendered_blake3":blake3::hash(text.as_bytes()).to_hex().to_string(),"reference_slice_verified":true,"reached_end":reached_end,"complete_session_walk":false})
}

fn cancelled_client(env: &crate::Fixture) {
    let budget = PAGE_BUDGET.to_string();
    let mut command = super::command(env);
    command
        .args(["--json", "read", LARGE, "--budget", &budget])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = Daemon(polylogue_testkit::spawn_daemon(&mut command).unwrap());
    let mut first = [0];
    child
        .0
        .stdout
        .as_mut()
        .unwrap()
        .read_exact(&mut first)
        .unwrap();
    assert_eq!(first[0], b'{');
    assert!(
        child.0.try_wait().unwrap().is_none(),
        "the cancelled read produced no pending output"
    );
    child.0.kill().unwrap();
    assert!(!child.0.wait().unwrap().success());
    let (status, _) = request(env, &["--json", "status"]);
    assert_status(&status);
}

fn check_cancelled_client(env: &crate::Fixture, daemon: &Daemon, expected: &[String]) -> Value {
    eprintln!("native read bench: starting client abandonment");
    cancelled_client(env);
    eprintln!("native read bench: abandoned client and status verified");
    let (after, _) = request(
        env,
        &[
            "--json",
            "read",
            LARGE,
            "--budget",
            &PAGE_BUDGET.to_string(),
        ],
    );
    assert!(after["text"].as_str().unwrap().contains(&expected[0]));
    eprintln!("native read bench: recovery read verified");
    memory(daemon.0.id())
}

#[test]
#[ignore = "opt-in measurement: xtask bench reads"]
fn read_benchmark() {
    let report =
        std::env::var_os("POLYLOGUE_READ_BENCH_REPORT").expect("run through xtask bench reads");
    let repetitions: usize = std::env::var("POLYLOGUE_READ_BENCH_REPETITIONS")
        .unwrap()
        .parse()
        .unwrap();
    assert!(repetitions > 0);
    let full_walk = std::env::var("POLYLOGUE_READ_BENCH_FULL_WALK").as_deref() == Ok("1");
    let setup = Instant::now();
    let home = polylogue_testkit::ClaudeHome::new();
    let config = home.dir.path().join("config");
    let env = crate::Fixture::new(home.root.as_path(), home.home.as_path(), config.as_path());
    let expected: Vec<_> = (0..MAIN_MESSAGES).map(body).collect();
    let mut input_bytes = 0;
    let mut identity = blake3::Hasher::new();
    for (name, lines) in std::iter::once((
        "large.jsonl".to_owned(),
        expected
            .iter()
            .enumerate()
            .map(|(n, text)| {
                let id = format!("main-{n:04}");
                if n % 2 == 0 {
                    polylogue_testkit::claude_user(&id, "read-bench-large", text)
                } else {
                    polylogue_testkit::claude_assistant(&id, "read-bench-large", text)
                }
            })
            .collect::<String>(),
    ))
    .chain((0..SMALL_SESSIONS).map(|session| {
        (
            format!("small-{session}.jsonl"),
            (0..SMALL_MESSAGES)
                .map(|n| {
                    polylogue_testkit::claude_user(
                        &format!("small-{session}-{n}"),
                        &format!("read-bench-small-{session}"),
                        &format!("readbenchneedle small session{session} message{n}"),
                    )
                })
                .collect::<String>(),
        )
    })) {
        identity.update(name.as_bytes());
        identity.update(&[0]);
        identity.update(lines.as_bytes());
        input_bytes += lines.len();
        home.append(&name, &lines);
    }
    let population_seconds = setup.elapsed().as_secs_f64();
    let began = Instant::now();
    let initial = start_daemon(&env);
    let initial_daemon_start_seconds = began.elapsed().as_secs_f64();
    let began = Instant::now();
    let status = wait_status(&env, |s| {
        s["store"]["messages"].as_u64()
            == Some((MAIN_MESSAGES + SMALL_SESSIONS * SMALL_MESSAGES) as u64)
            && s["store"]["search_pending"] == 0
            && s["store"]["parse_states"]["done"] == SMALL_SESSIONS + 1
    });
    assert_status(&status);
    let archive_setup_seconds = began.elapsed().as_secs_f64();
    eprintln!("native read bench: population and archive ready");
    let prepared = (!full_walk).then(|| reference_windows(&env, &expected));
    drop(initial);
    let mut operations = serde_json::Map::new();
    for (name, args) in [
        ("status", vec!["--json", "status"]),
        ("query", vec!["--json", "sessions | sort by id | limit 20"]),
        (
            "search",
            vec!["--json", "readbenchneedle | sort by id | limit 20"],
        ),
    ] {
        let began = Instant::now();
        let daemon = start_daemon(&env);
        let start_seconds = began.elapsed().as_secs_f64();
        let before = memory(daemon.0.id());
        let mut samples = Vec::new();
        for _ in 0..=repetitions {
            let (answer, seconds) = request(&env, &args);
            if name == "status" {
                assert_status(&answer)
            } else {
                assert_sessions(&answer)
            }
            samples.push(seconds);
        }
        eprintln!("native read bench: {name} answers verified");
        operations.insert(name.into(),json!({"daemon_start_seconds":start_seconds,"cold_seconds":samples[0],"warm_seconds":summary(&samples[1..]),"rss_before":before,"rss_after":memory(daemon.0.id())}));
    }
    let mut help = Vec::new();
    for _ in 0..=repetitions {
        let began = Instant::now();
        let answer = cli(&env, &["--help"]);
        help.push(began.elapsed().as_secs_f64());
        assert!(answer.status.success());
        let text = String::from_utf8(answer.stdout).unwrap();
        assert!(text.contains("read") && text.contains("status"));
    }
    let read = if let Some((reference, starts, reference_facts)) = prepared {
        let mut windows = serde_json::Map::new();
        for start in starts {
            eprintln!("native read bench: starting {} window", start.name);
            let began = Instant::now();
            let daemon = start_daemon(&env);
            let daemon_start = began.elapsed().as_secs_f64();
            let before = memory(daemon.0.id());
            let mut walks = Vec::new();
            for iteration in 0..=repetitions {
                let mut result = window(&env, &start, &reference);
                result["rss_after"] = memory(daemon.0.id());
                eprintln!(
                    "native read bench: {} window {iteration} verified at {:.3}s: {result}",
                    start.name,
                    setup.elapsed().as_secs_f64()
                );
                walks.push(result);
            }
            let after_cancel = if start.name == "tail" {
                Some(check_cancelled_client(&env, &daemon, &expected))
            } else {
                None
            };
            let warm: Vec<_> = walks[1..]
                .iter()
                .map(|walk| walk["first_page_seconds"].as_f64().unwrap())
                .collect();
            windows.insert(start.name.into(),json!({"start_byte":start.byte,"daemon_start_seconds":daemon_start,"rss_before":before,"cold_first_page_seconds":walks[0]["first_page_seconds"],"warm_first_page_seconds":summary(&warm),"walks":walks,"rss_after_cancelled_client":after_cancel}));
        }
        json!({"reference":reference_facts,"window_pages":WINDOW_PAGES,"budget_tokens":PAGE_BUDGET,"windows":windows,"complete_session_walk":false})
    } else {
        let began = Instant::now();
        let daemon = start_daemon(&env);
        let daemon_start = began.elapsed().as_secs_f64();
        let before = memory(daemon.0.id());
        let mut walks = Vec::new();
        let mut digest = None;
        for iteration in 0..=repetitions {
            eprintln!("native read bench: starting full walk {iteration}");
            let (mut result, current) = walk(&env, &expected);
            if let Some(previous) = &digest {
                assert_eq!(previous, &current)
            } else {
                digest = Some(current)
            }
            result["rss_after"] = memory(daemon.0.id());
            walks.push(result);
        }
        let after_cancel = check_cancelled_client(&env, &daemon, &expected);
        json!({"daemon_start_seconds":daemon_start,"budget_tokens":PAGE_BUDGET,"rss_before":before,"walks":walks,"rss_after_cancelled_client":after_cancel,"complete_session_walk":true})
    };
    let result = json!({"population":{"id":"native-reads-v1","sessions":SMALL_SESSIONS+1,"messages":MAIN_MESSAGES+SMALL_SESSIONS*SMALL_MESSAGES,"large_messages":MAIN_MESSAGES,"body_bytes":BODY_BYTES,"input_bytes":input_bytes,"blake3":identity.finalize().to_hex().to_string()},
        "mode":if full_walk {"full_walk"}else{"windows"},"profile":"release","debug_assertions":cfg!(debug_assertions),"repetitions":repetitions,"population_setup_seconds":population_seconds,"initial_daemon_start_seconds":initial_daemon_start_seconds,"archive_setup_seconds":archive_setup_seconds,
        "cold_definition":"first request after daemon restart; filesystem cache is not flushed; every CLI call starts a new client process",
        "operations":operations,"help":{"first_seconds":help[0],"repeated_seconds":summary(&help[1..])},
        "read":read,
        "cancellation":"native client killed after its first output byte; a subsequent status and bounded read verified daemon responsiveness",
        "answers_verified":true,"cancelled_client_recovered":true});
    eprintln!(
        "native read bench: saving verified report at {:.3}s",
        setup.elapsed().as_secs_f64()
    );
    std::fs::write(report, serde_json::to_vec_pretty(&result).unwrap()).unwrap();
    eprintln!(
        "native read bench: cleaning synthetic population at {:.3}s",
        setup.elapsed().as_secs_f64()
    );
    drop(home);
    eprintln!(
        "native read bench: finished at {:.3}s",
        setup.elapsed().as_secs_f64()
    );
}

#[test]
fn source_body_checks_refuse_missing_or_duplicate_work() {
    let expected = vec![
        "readbenchneedle first body".into(),
        "readbenchneedle second body".into(),
    ];
    verify_text(
        "heading readbenchneedle first body readbenchneedle second body",
        &expected,
    );
    assert!(
        std::panic::catch_unwind(|| verify_text("readbenchneedle first body", &expected)).is_err()
    );
    assert!(
        std::panic::catch_unwind(|| verify_text(
            "readbenchneedle first body readbenchneedle first body readbenchneedle second body",
            &expected
        ))
        .is_err()
    );
}
