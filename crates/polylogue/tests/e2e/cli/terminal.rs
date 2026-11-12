use super::*;
use rustix::pty::{OpenptFlags, grantpt, ioctl_tiocgptpeer, openpt, unlockpt};
use serde_json::json;
use std::fs::File;

const HOSTILE: &str = "\x1b]0;title\x07\x1b]2;title\x1b\\\x1b]52;c;YQ==\x07\x1b]8;;https://example.test\x1b\\link\x1b]8;;\x1b\\\x7f\x1b[2J\x1b\u{9b}2J\u{202e}";

fn terminal(env: &crate::Fixture, args: &[&str], no_color: bool) -> String {
    let flags = OpenptFlags::RDWR | OpenptFlags::NOCTTY | OpenptFlags::CLOEXEC;
    let master = openpt(flags).unwrap();
    grantpt(&master).unwrap();
    unlockpt(&master).unwrap();
    let slave = ioctl_tiocgptpeer(&master, flags).unwrap();
    let mut command = env.command();
    command.env_remove("NO_COLOR");
    if no_color {
        command.env("NO_COLOR", "");
    }
    let child = command
        .args(args)
        .stdin(Stdio::null())
        .stdout(File::from(slave))
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    drop(command);
    let reader = std::thread::spawn(move || {
        let mut bytes = Vec::new();
        if let Err(error) = File::from(master).read_to_end(&mut bytes) {
            assert_eq!(error.raw_os_error(), Some(5));
        }
        bytes
    });
    let output = child.wait_with_output().unwrap();
    assert!(output.status.success(), "{output:?}");
    let bytes = reader.join().unwrap();
    String::from_utf8(bytes).unwrap().replace("\r\n", "\n")
}

fn safe(text: &str) {
    assert!(
        !text
            .chars()
            .any(|c| (c.is_control() && c != '\n' && c != '\t') || c == '\u{202e}'),
        "{text:?}"
    );
}

#[test]
fn archived_controls_are_data_in_pipes_and_terminals() {
    let dir = tempfile::tempdir().unwrap();
    let home = dir.path().join("home");
    let archive = dir.path().join("archive");
    let config = dir.path().join("config");
    std::fs::create_dir(&home).unwrap();
    let env = crate::Fixture::new(archive.as_path(), home.as_path(), config.as_path());
    let _daemon = start_daemon(&env);
    use polylogue_model::bundle::{self, Record};
    let session = serde_json::from_value(json!({"id":"claude-code:hostile","kind":"main","title":{"text":HOSTILE,"source":"custom"}})).unwrap();
    let message: polylogue_model::Message = serde_json::from_value(json!({"id":"claude-code:hostile/1","session":"claude-code:hostile","role":"user","prompt_kind":"prose","blocks":[{"type":"text","text":HOSTILE}]})).unwrap();
    let manifest = bundle::Manifest {
        format: bundle::FORMAT.into(),
        version: bundle::VERSION,
        query: format!("query:{}", "a".repeat(64)),
        query_text: "sessions".into(),
        ran_at: polylogue_model::Timestamp::from_unix_ms(1).unwrap(),
        read_at: polylogue_model::ChangePosition {
            generation: "synthetic".into(),
            index: 1,
            archive: 1,
        },
        freshness: polylogue_model::currency::ReadFreshness {
            local: polylogue_model::currency::ReadCompleteness::empty(),
            owner: polylogue_model::currency::IndexCurrency::Standalone,
        },
        counts: bundle::Counts {
            sessions: 1,
            messages: 1,
            ..Default::default()
        },
        flags: bundle::Flags {
            format: bundle::Format::Jsonl,
            with_blobs: false,
            read: Default::default(),
        },
        enabled_variants: vec![],
        author: polylogue_model::Author::Operator,
        notes: vec![],
        incomplete: vec![],
        inventory: None,
        coverage: vec![],
    };
    let records = [
        Record::Manifest(Box::new(manifest)),
        Record::Session(bundle::SessionRecord {
            copied_prefixes: Vec::new(),
            reference: "session:claude-code:hostile".into(),
            commit: 1,
            session,
        }),
        Record::Message(Box::new(bundle::MessageRecord {
            reference: format!("message:{}", message.id),
            commit: 1,
            order: 0,
            source_version: message.content_hash().unwrap(),
            block_count: 1,
            block_ordinals: vec![0],
            injections: vec![],
            bindings: vec![],
            copied_for: Vec::new(),
            message,
        })),
    ];
    let mut original = Vec::new();
    for record in records {
        record.validate().unwrap();
        serde_json::to_writer(&mut original, &record).unwrap();
        original.push(b'\n');
    }
    let source = dir.path().join("seed.jsonl");
    std::fs::write(&source, &original).unwrap();
    let imported = cli(&env, &["import", source.to_str().unwrap()]);
    assert!(imported.status.success(), "{imported:?}");
    wait_status(&env, |status| status["store"]["search_pending"] == 0);
    let watched = cli(&env, &["watch", "--name", HOSTILE, "sessions where kind:*"]);
    assert!(watched.status.success(), "{watched:?}");
    let status = wait_status(&env, |status| {
        status["signals"]["rules"]
            .as_array()
            .is_some_and(|rules| rules.iter().any(|rule| rule["name"] == HOSTILE))
    });
    assert_eq!(status["signals"]["rules"][0]["name"], HOSTILE);
    for args in [
        vec!["read", "session:claude-code:hostile"],
        vec![
            "read",
            "session:claude-code:hostile",
            "--format",
            "markdown",
        ],
        vec!["read", "session:claude-code:hostile", "--page-tokens", "10"],
        vec!["sessions where kind:*"],
        vec!["\"title\""],
        vec!["status"],
    ] {
        let output = cli(&env, &args);
        assert!(output.status.success(), "{output:?}");
        let text = std::str::from_utf8(&output.stdout).unwrap();
        safe(text);
        if args == ["status"] {
            assert!(text.contains("\\u007f"), "{text:?}");
        }
        safe(&terminal(&env, &args, true));
    }
    let colored = terminal(&env, &["read", "session:claude-code:hostile"], false);
    assert!(colored.contains("\x1b[1m"), "{colored:?}");
    safe(&colored.replace("\x1b[1m", "").replace("\x1b[0m", ""));
    assert!(colored.contains("\\x1b]52;c;YQ==\\x07"));
    let matches = terminal(&env, &["\"title\""], false);
    assert!(matches.contains("\x1b[1mtitle\x1b[0m"), "{matches:?}");
    safe(&matches.replace("\x1b[1m", "").replace("\x1b[0m", ""));
    let text = terminal(
        &env,
        &[
            "--json",
            "read",
            "session:claude-code:hostile",
            "--format",
            "json",
        ],
        false,
    );
    safe(&text);
    let answer: command_surface::Answer<polylogue_model::read::ReadPage> =
        serde_json::from_str(&text).unwrap();
    let page = answer.value().expect("typed read answer");
    let mut items = super::read_current::Items::default();
    items.push(page);
    assert!(
        page.document.footer.next.is_none(),
        "short source must fit this root"
    );
    assert_eq!(items.text(), HOSTILE);
    assert_eq!(std::fs::read(&source).unwrap(), original);
    let error = cli(&env, &["read", HOSTILE]);
    assert!(!error.status.success());
    safe(std::str::from_utf8(&error.stderr).unwrap());
}
