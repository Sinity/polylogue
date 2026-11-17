use super::*;
use polylogue_model::LockExt as _;
use rustix::pty::{OpenptFlags, grantpt, ioctl_tiocgptpeer, openpt, unlockpt};
use serde_json::{Value, json};
use std::{
    fs::File,
    io::{BufRead, BufReader, Read, Write},
    os::unix::net::{UnixListener, UnixStream},
    process::{Child, Command as Process, Stdio},
    sync::Mutex,
    time::Instant,
};

fn row(name: &str) -> Candidate {
    Candidate {
        reference: format!("session:codex:{name}"),
        title: format!("{name} title"),
        first_prompt: format!("Discuss {name}"),
        repository: "morrow".into(),
        description: "Nov 12 · codex".into(),
    }
}

struct Fixture {
    case: String,
    events: Arc<Mutex<UnixStream>>,
    release: Arc<tokio::sync::Semaphore>,
    started: Instant,
}
impl Fixture {
    fn emit(&self, event: Value) {
        emit(&self.events, event);
    }
    fn rows(&self) -> Vec<Candidate> {
        match self.case.as_str() {
            "single" => vec![row("Alpha")],
            "pages" => ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"]
                .map(row)
                .to_vec(),
            _ => vec![row("Alpha"), row("Beta")],
        }
    }
    async fn delayed(&self, kind: &str, name: &str) {
        let events = self.events.clone();
        let kind = kind.to_owned();
        let name = name.to_owned();
        // A remote service may finish even after its caller drops the socket.
        let (send, receive) = tokio::sync::oneshot::channel();
        let release = self.release.clone();
        self.emit(json!({"started":kind,"name":name}));
        let task = tokio::spawn(async move {
            release.acquire().await.unwrap().forget();
            emit(&events, json!({"completed":kind,"name":name}));
            let _ = send.send(());
        });
        let _ = receive.await;
        task.abort();
    }
}
fn emit(events: &Mutex<UnixStream>, event: Value) {
    writeln!(events.locked(), "{event}").unwrap();
}
impl Source for Fixture {
    async fn page(&self, request: PageRequest) -> Result<Page> {
        self.emit(json!({"page":request.selection,"fuzzy":request.fuzzy,"cursor":request.cursor}));
        if request.fuzzy.is_some() {
            return fuzzy_page(
                self.rows(),
                &request,
                if self.case == "pages" { 2 } else { 64 },
            );
        }
        if request.selection == "older" {
            self.delayed("query", "older").await;
            return Ok(Page {
                rows: vec![row("Old")],
                count: Some(1),
                next: None,
            });
        }
        if request.selection == "newer" {
            self.release.add_permits(1);
            return Ok(Page {
                rows: vec![row("New"), row("Newest")],
                count: Some(2),
                next: None,
            });
        }
        if request.selection == "invalid" {
            return Err(Failure::invalid_query("invalid query text"));
        }
        if self.case == "query" && matches!(request.selection.as_str(), "" | "sessions") {
            std::future::pending::<()>().await;
        }
        Ok(Page {
            count: Some(self.rows().len() as u64),
            rows: self.rows(),
            next: None,
        })
    }
    async fn preview(&self, request: PreviewRequest) -> Result<String> {
        self.emit(json!({"preview":request.reference,"width":request.width,"deadline":request.deadline.as_millis()}));
        let name = request.reference.rsplit(':').next().unwrap();
        if self.case == "preview" && name == "Alpha" {
            self.delayed("preview", "Alpha").await;
        }
        if name == "Beta" {
            self.release.add_permits(1);
        }
        let text = if self.case == "controls" {
            "overview: \x1b[2J\x07\u{202e}\ttext".to_owned()
        } else {
            format!("overview: {name} at {} columns", request.width)
        };
        Ok(text)
    }
    fn act(&self, action: Action, references: &[String]) -> Result<()> {
        self.emit(
            json!({"action":format!("{action:?}"),"references":references,
            "raw":crossterm::terminal::is_raw_mode_enabled().unwrap()}),
        );
        Ok(())
    }
    fn shown(&self, question: &str, rows: &[Candidate]) -> Result<()> {
        self.emit(json!({"shown":rows.iter().map(|row| &row.reference).collect::<Vec<_>>(),"question":question,"since_open_us":self.started.elapsed().as_micros()}));
        Ok(())
    }
}

#[test]
#[ignore = "runs inside the PTY created by its parent test"]
fn terminal_child() {
    let case = std::env::var("POLYLOGUE_PICKER_CASE").unwrap();
    let events = Arc::new(Mutex::new(
        UnixStream::connect(std::env::var_os("POLYLOGUE_PICKER_EVENTS").unwrap()).unwrap(),
    ));
    let source = Arc::new(Fixture {
        case: case.clone(),
        events: events.clone(),
        release: Arc::new(tokio::sync::Semaphore::new(0)),
        started: Instant::now(),
    });
    let selection = if case == "ambiguous" {
        Selection::Ambiguous {
            question: "codex".into(),
            candidates: source.rows(),
        }
    } else {
        Selection::Query("sessions".into())
    };
    let result = open(
        source,
        Options {
            selection,
            enter: if case == "print" {
                String::new()
            } else {
                "tag reviewed".into()
            },
            multiple: true,
            json: case == "json",
            format: match case.as_str() {
                "jsonl" => Some("jsonl".into()),
                "format-json" => Some("json".into()),
                _ => None,
            },
        },
    );
    let result = match result {
        Ok(Choice::Selected(references)) => json!({"choice":references}),
        Ok(Choice::Cancelled) => json!({"cancelled":true}),
        Err(error) => json!({"error":error.code(),"candidates":error.candidates()}),
    };
    emit(
        &events,
        json!({"result":result,"raw":crossterm::terminal::is_raw_mode_enabled().unwrap()}),
    );
}

struct Terminal {
    child: Child,
    input: Option<File>,
    output: mpsc::Receiver<Vec<u8>>,
    events: mpsc::Receiver<Value>,
    bytes: Vec<u8>,
    history: Vec<Value>,
    _dir: tempfile::TempDir,
}
impl Terminal {
    fn new(case: &str, stdin_tty: bool, stdout_tty: bool) -> Self {
        let dir = tempfile::tempdir().unwrap();
        let events = dir.path().join("events");
        let listener = UnixListener::bind(&events).unwrap();
        let flags = OpenptFlags::RDWR | OpenptFlags::NOCTTY | OpenptFlags::CLOEXEC;
        let master = openpt(flags).unwrap();
        grantpt(&master).unwrap();
        unlockpt(&master).unwrap();
        let slave = File::from(ioctl_tiocgptpeer(&master, flags).unwrap());
        // Set the PTY size without unsafe ioctl calls.
        let status = Process::new("stty")
            .args(["rows", "24", "cols", "100"])
            .stdin(slave.try_clone().unwrap())
            .status()
            .unwrap();
        assert!(status.success());
        let master = File::from(master);
        let mut command = Process::new(std::env::current_exe().unwrap());
        command
            .args([
                "--exact",
                "cli::picker::tests::terminal_child",
                "--nocapture",
                "--ignored",
            ])
            .env("POLYLOGUE_PICKER_CASE", case)
            .env("POLYLOGUE_PICKER_EVENTS", &events)
            .stdin(if stdin_tty {
                Stdio::from(slave.try_clone().unwrap())
            } else {
                Stdio::null()
            })
            .stdout(if stdout_tty {
                Stdio::from(slave.try_clone().unwrap())
            } else {
                Stdio::piped()
            })
            .stderr(Stdio::piped());
        let mut child = command.spawn().unwrap();
        drop(command);
        drop(slave);
        let (send_output, output) = mpsc::channel();
        if stdout_tty {
            let mut reader = master.try_clone().unwrap();
            std::thread::spawn(move || {
                let mut buffer = [0; 8192];
                while let Ok(count) = reader.read(&mut buffer) {
                    if count == 0 {
                        break;
                    }
                    if send_output.send(buffer[..count].to_vec()).is_err() {
                        break;
                    }
                }
            });
        } else {
            let mut reader = child.stdout.take().unwrap();
            std::thread::spawn(move || {
                let mut bytes = vec![];
                reader.read_to_end(&mut bytes).unwrap();
                let _ = send_output.send(bytes);
            });
        }
        let (socket, _) = listener.accept().unwrap();
        let (send_events, event_stream) = mpsc::channel();
        std::thread::spawn(move || {
            for line in BufReader::new(socket).lines() {
                let Ok(line) = line else {
                    break;
                };
                if send_events
                    .send(serde_json::from_str(&line).unwrap())
                    .is_err()
                {
                    break;
                }
            }
        });
        Self {
            child,
            input: Some(master),
            output,
            events: event_stream,
            bytes: vec![],
            history: vec![],
            _dir: dir,
        }
    }
    fn send(&mut self, text: &[u8]) {
        self.input.as_mut().unwrap().write_all(text).unwrap();
    }
    fn paste(&mut self, text: &str) {
        self.send(b"\x15\x1b[200~");
        self.send(text.as_bytes());
        self.send(b"\x1b[201~");
    }
    fn until(&mut self, text: &str) {
        self.until_after(0, text);
    }
    fn until_after(&mut self, start: usize, text: &str) {
        while !String::from_utf8_lossy(&self.bytes[start..]).contains(text) {
            self.bytes.extend(
                self.output
                    .recv_timeout(Duration::from_secs(5))
                    .unwrap_or_else(|error| {
                        panic!(
                            "waiting for {text:?}: {error}; output: {}",
                            String::from_utf8_lossy(&self.bytes)
                        )
                    }),
            );
        }
    }
    fn event(&mut self, matches: impl Fn(&Value) -> bool) -> Value {
        loop {
            let event = self
                .events
                .recv_timeout(Duration::from_secs(5))
                .unwrap_or_else(|error| {
                    panic!("waiting for event: {error}; history: {:?}", self.history)
                });
            self.history.push(event.clone());
            if matches(&event) {
                return event;
            }
        }
    }
    fn finish(mut self) -> (Value, String, Vec<Value>) {
        let result = self.event(|event| event.get("result").is_some());
        assert!(self.child.wait().unwrap().success(), "child failed");
        self.input = None;
        for bytes in self.output.iter() {
            self.bytes.extend(bytes);
        }
        assert_eq!(result["raw"], false);
        (
            result["result"].clone(),
            String::from_utf8(self.bytes.clone()).unwrap(),
            self.history.clone(),
        )
    }
}
impl Drop for Terminal {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn lines(output: &str) -> Vec<&str> {
    output
        .split("\x1b[1G\x1b[2K")
        .skip(1)
        .filter_map(|line| line.split_once("\r\n").map(|(line, _)| line))
        .collect()
}

#[test]
fn picker_footer_shows_keys_and_enter_action() {
    let mut terminal = Terminal::new("normal", true, true);
    terminal.until("Esc cancel");
    terminal.send(b"\t\x19\x0f\x12\x10\r");
    let (result, output, events) = terminal.finish();
    assert_eq!(result, json!({"choice":["session:codex:Alpha"]}));
    let footer: Vec<_> = lines(&output)
        .into_iter()
        .filter(|line| line.contains("Enter tag") || line.contains("^p print"))
        .take(2)
        .collect();
    assert_eq!(
        footer.join(" "),
        "↑ up · ↓ down · Tab mark · Enter tag reviewed · Esc cancel · ^j down · ^k up · ^y copy · ^r read · ^o browser · ^p print · ^e query/fuzzy · ^/ preview"
    );
    let actions: Vec<_> = events
        .iter()
        .filter(|event| event.get("action").is_some())
        .cloned()
        .collect();
    assert_eq!(
        actions,
        vec![
            json!({"action":"Copy","references":["session:codex:Beta"],"raw":false}),
            json!({"action":"Browser","references":["session:codex:Beta"],"raw":false}),
            json!({"action":"Read","references":["session:codex:Beta"],"raw":false}),
            json!({"action":"Enter","references":["session:codex:Alpha"],"raw":false}),
        ]
    );
    let mut terminal = Terminal::new("normal", true, true);
    terminal.until("Esc cancel");
    terminal.send(b"\x0a\t\x0b\t\r");
    assert_eq!(
        terminal.finish().0,
        json!({"choice":["session:codex:Beta","session:codex:Alpha"]})
    );
    let mut terminal = Terminal::new("normal", true, true);
    terminal.until("overview: Alpha");
    let overview = String::from_utf8_lossy(&terminal.bytes)
        .rfind("overview: Alpha")
        .unwrap();
    terminal.until_after(overview, "^/ preview");
    let start = terminal.bytes.len();
    terminal.send(b"\x1f");
    terminal.until_after(start, "^/ preview");
    let output = String::from_utf8_lossy(&terminal.bytes[start..]);
    let body: Vec<_> = lines(&output)
        .into_iter()
        .filter(|line| !line.is_empty())
        .take_while(|line| !line.contains("Enter tag"))
        .collect();
    assert_eq!(
        body,
        [
            "sessions ›  [fuzzy] → 2 rows",
            "›  1 Alpha title  morrow  Nov 12 · codex",
            "   2 Beta title  morrow  Nov 12 · codex",
        ]
    );
    terminal.send(b"\x1f\x0a");
    terminal.until("overview: Beta");
    terminal.send(b"\r");
    assert_eq!(
        terminal.finish().0,
        json!({"choice":["session:codex:Beta"]})
    );
}

#[test]
fn picker_query_mode_reruns_and_drops_stale() {
    let mut terminal = Terminal::new("query", true, true);
    terminal.until("Esc cancel");
    terminal.send(b"\x05");
    terminal.paste("older");
    terminal.event(|event| event["started"] == "query");
    terminal.paste("newer");
    terminal.event(|event| event["completed"] == "query");
    terminal.until("New title");
    terminal.send(b"\r");
    let (result, _, events) = terminal.finish();
    assert_eq!(result, json!({"choice":["session:codex:New"]}));
    let shown: Vec<_> = events
        .iter()
        .filter(|event| event.get("shown").is_some())
        .map(|event| event["shown"].clone())
        .filter(|rows| !rows.as_array().unwrap().is_empty())
        .collect();
    assert_eq!(
        shown,
        vec![
            json!(["session:codex:Alpha", "session:codex:Beta"]),
            json!(["session:codex:New", "session:codex:Newest"])
        ]
    );
}

#[test]
fn picker_preview_drops_stale_reply() {
    let mut terminal = Terminal::new("preview", true, true);
    terminal.event(|event| event["started"] == "preview");
    terminal.send(b"\x1b[B");
    terminal.event(|event| event["completed"] == "preview");
    terminal.until("overview: Beta at 99 columns");
    terminal.send(b"\r");
    let (result, output, events) = terminal.finish();
    assert_eq!(result, json!({"choice":["session:codex:Beta"]}));
    let readings: Vec<_> = lines(&output)
        .into_iter()
        .filter(|line| line.starts_with("overview:"))
        .collect();
    assert_eq!(readings, vec!["overview: Beta at 99 columns"]);
    let requests: Vec<_> = events
        .iter()
        .filter(|event| event.get("preview").is_some())
        .cloned()
        .collect();
    assert_eq!(
        requests,
        vec![
            json!({"preview":"session:codex:Alpha","width":99,"deadline":100}),
            json!({"preview":"session:codex:Beta","width":99,"deadline":100})
        ]
    );
    let mut terminal = Terminal::new("preview", true, true);
    terminal.until("overview exceeded its 100 ms deadline");
    terminal.send(b"\x1b");
    assert_eq!(terminal.finish().0, json!({"cancelled":true}));
}

#[test]
fn picker_never_opens_off_terminal_or_with_json() {
    for (case, input, output) in [
        ("normal", false, true),
        ("normal", true, false),
        ("json", true, true),
        ("jsonl", true, true),
        ("format-json", true, true),
    ] {
        let terminal = Terminal::new(case, input, output);
        let (result, _, events) = terminal.finish();
        assert_eq!(result["error"], "ambiguous");
        assert_eq!(
            result["candidates"],
            json!([
                "session:codex:Alpha  Nov 12 · codex  Alpha title\npolylogue session:codex:Alpha tag reviewed",
                "session:codex:Beta  Nov 12 · codex  Beta title\npolylogue session:codex:Beta tag reviewed",
            ])
        );
        assert_eq!(
            events
                .iter()
                .filter(|event| event.get("shown").is_some())
                .count(),
            0
        );
    }
}

#[test]
fn single_match_never_opens_picker() {
    for tty in [true, false] {
        let terminal = Terminal::new("single", tty, tty);
        let (result, _, events) = terminal.finish();
        assert_eq!(result, json!({"choice":["session:codex:Alpha"]}));
        let actions: Vec<_> = events
            .iter()
            .filter(|event| event.get("action").is_some())
            .cloned()
            .collect();
        assert_eq!(
            actions,
            vec![json!({"action":"Enter","references":["session:codex:Alpha"],"raw":false})]
        );
    }
}

#[test]
fn fuzzy_candidates_continue_across_pages() {
    let mut rows = vec![row("Alpha"), row("Beta"), row("Gamma")];
    rows[0].title = "Retry".into();
    rows[1].first_prompt = "retry".into();
    rows[2].repository = "retry".into();
    let mut request = PageRequest {
        selection: "sessions".into(),
        fuzzy: Some("retry".into()),
        cursor: None,
    };
    let mut references = vec![];
    loop {
        let mut matched = FuzzyRows::new(&request, 1).unwrap();
        for chunk in rows.chunks(1) {
            matched.extend(chunk.iter().cloned());
        }
        let page = matched.finish().unwrap();
        assert_eq!(page.count, Some(3));
        references.extend(page.rows.into_iter().map(|row| row.reference));
        request.cursor = page.next;
        if request.cursor.is_none() {
            break;
        }
    }
    assert_eq!(
        references,
        [
            "session:codex:Alpha",
            "session:codex:Beta",
            "session:codex:Gamma"
        ]
    );
    let mut terminal = Terminal::new("pages", true, true);
    terminal.until("Esc cancel");
    terminal.event(|event| event["shown"] == json!(["session:codex:Alpha", "session:codex:Beta"]));
    terminal.send(b"\x1b[B\t");
    terminal.until("Gamma title");
    terminal.event(|event| event["shown"] == json!(["session:codex:Gamma", "session:codex:Delta"]));
    terminal.send(b"\x1b[A");
    terminal.event(|event| event["shown"] == json!(["session:codex:Alpha", "session:codex:Beta"]));
    terminal.send(b"\r");
    let (result, _, events) = terminal.finish();
    assert_eq!(result, json!({"choice":["session:codex:Beta"]}));
    let cursors: Vec<_> = events
        .iter()
        .filter(|event| event.get("page").is_some())
        .map(|event| event["cursor"].clone())
        .collect();
    assert_eq!(cursors.len(), 3);
    assert_eq!(cursors[0], Value::Null);
    assert_eq!(cursors[2], Value::Null);
}

#[test]
fn ambiguity_filtering_printing_and_cancellation() {
    let mut terminal = Terminal::new("ambiguous", true, true);
    terminal.until("Esc cancel");
    terminal.paste("Beta");
    terminal.until("→ 1 rows");
    terminal.send(b"\r");
    let (result, _, _) = terminal.finish();
    assert_eq!(result, json!({"choice":["session:codex:Beta"]}));
    let mut terminal = Terminal::new("print", true, true);
    terminal.until("Enter print references");
    terminal.send(b"\r");
    let (result, output, _) = terminal.finish();
    assert_eq!(result, json!({"choice":["session:codex:Alpha"]}));
    let mut plain = String::new();
    let mut characters = output.chars();
    while let Some(character) = characters.next() {
        if character == '\x1b' {
            assert_eq!(characters.next(), Some('['));
            for character in characters.by_ref() {
                if ('@'..='~').contains(&character) {
                    break;
                }
            }
        } else if character != '\r' {
            plain.push(character);
        }
    }
    let printed: Vec<_> = plain
        .lines()
        .filter(|line| line.starts_with("session:"))
        .collect();
    assert_eq!(printed, ["session:codex:Alpha"]);
    let mut terminal = Terminal::new("normal", true, true);
    terminal.until("Esc cancel");
    terminal.send(b"\x1b");
    assert_eq!(terminal.finish().0, json!({"cancelled":true}));
}

#[test]
fn archived_controls_are_escaped_in_preview() {
    let mut terminal = Terminal::new("controls", true, true);
    terminal.until("overview:");
    terminal.send(b"\r");
    let (_, output, _) = terminal.finish();
    let readings: Vec<_> = lines(&output)
        .into_iter()
        .filter(|line| line.starts_with("overview:"))
        .collect();
    assert_eq!(readings, vec!["overview: \\x1b[2J\\x07\\u{202e}\\ttext"]);
}

#[test]
fn picker_latency_samples() {
    let mut samples = vec![];
    for _ in 0..5 {
        let first = Instant::now();
        let mut terminal = Terminal::new("normal", true, true);
        terminal.until("Esc cancel");
        let process = first.elapsed();
        let first = terminal.event(|event| event.get("shown").is_some())["since_open_us"].clone();
        let pressed = Instant::now();
        terminal.paste("Beta");
        terminal.until("→ 1 rows");
        let key = pressed.elapsed();
        terminal.until("overview: Beta");
        let preview = pressed.elapsed();
        terminal.send(b"\r");
        let result = terminal.finish().0;
        assert_eq!(result, json!({"choice":["session:codex:Beta"]}));
        samples.push(json!({"first_candidates_us":first,"fixture_process_us":process.as_micros(),"keystroke_us":key.as_micros(),"preview_us":preview.as_micros()}));
    }
    eprintln!(
        "picker latency, real PTY, fresh process per sample: {}",
        json!(samples)
    );
    for text in ["", "retry"] {
        let started = Instant::now();
        let page = fuzzy_page(
            (0..10_000).map(|number| row(&format!("Retry {number}"))),
            &PageRequest {
                selection: "sessions".into(),
                fuzzy: Some(text.into()),
                cursor: None,
            },
            64,
        )
        .unwrap();
        assert_eq!(page.count, Some(10_000));
        assert_eq!(page.rows.len(), 64);
        eprintln!(
            "picker catalog scan: {}",
            json!({"rows":10000,"filter":text,"elapsed_us":started.elapsed().as_micros()})
        );
    }
}
