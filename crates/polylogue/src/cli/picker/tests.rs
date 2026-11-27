use super::*;
use polylogue_model::LockExt as _;
use rustix::pty::{OpenptFlags, grantpt, ioctl_tiocgptpeer, openpt, unlockpt};
use serde_json::{Value, json};
use std::{
    fs::File,
    io::{BufRead, BufReader, Read, Write},
    os::unix::net::{UnixListener, UnixStream},
    process::{Child, Command as Process, Stdio},
    sync::{
        Mutex,
        atomic::{AtomicUsize, Ordering},
    },
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
    /// Requests so far for the unfiltered first page.
    first_pages: AtomicUsize,
}
/// Reports that the preview request holding it was dropped.
struct DroppedPreview(Arc<Mutex<UnixStream>>, String);
impl Drop for DroppedPreview {
    fn drop(&mut self) {
        emit(&self.0, json!({"dropped":"preview","name":self.1}));
    }
}
impl Fixture {
    fn page_size(&self) -> usize {
        if matches!(self.case.as_str(), "pages" | "fails" | "held") {
            2
        } else {
            64
        }
    }

    fn emit(&self, event: Value) {
        emit(&self.events, event);
    }
    fn rows(&self) -> Vec<Candidate> {
        match self.case.as_str() {
            "empty" => vec![],
            "single" => vec![row("Alpha")],
            "pages" | "fails" | "held" => ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"]
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
        if self.case == "fails"
            && request.cursor.is_none()
            && request.fuzzy.as_deref() == Some("")
            && self.first_pages.fetch_add(1, Ordering::SeqCst) == 1
        {
            return Err(Failure::not_yet_available("the first page is unavailable"));
        }
        if self.case == "held" && request.cursor.is_some() {
            std::future::pending::<()>().await;
        }
        if request.fuzzy.is_some() {
            return fuzzy_page(self.rows(), &request, self.page_size());
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
        if self.case == "held" && name == "Beta" {
            let _dropped = DroppedPreview(self.events.clone(), name.to_owned());
            std::future::pending::<()>().await;
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
        first_pages: AtomicUsize::new(0),
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
        },
    );
    let result = match result {
        Ok(Choice::Selected(references)) => {
            if case == "print" {
                for reference in &references {
                    println!("{reference}");
                }
            }
            json!({"choice":references})
        }
        Ok(Choice::Cancelled) => json!({"cancelled":true}),
        Ok(Choice::Empty) => json!({"empty":true}),
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
    assert_eq!(
        printed_references(&output),
        ["session:codex:Beta"],
        "Ctrl-p prints the highlighted reference"
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
}

#[test]
fn a_preview_past_its_deadline_says_so() {
    let mut terminal = Terminal::new("preview", true, true);
    terminal.until("overview exceeded its 100 ms deadline");
    terminal.send(b"\x1b");
    assert_eq!(terminal.finish().0, json!({"cancelled":true}));
}

#[test]
fn picker_never_opens_off_terminal() {
    for (case, input, output) in [("normal", false, true), ("normal", true, false)] {
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
fn empty_selection_never_opens_picker() {
    for tty in [true, false] {
        let terminal = Terminal::new("empty", tty, tty);
        let (result, _, events) = terminal.finish();
        assert_eq!(result, json!({"empty":true}));
        assert_eq!(
            events
                .iter()
                .filter(|event| event.get("action").is_some())
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
    let mut rows = [row("Alpha"), row("Beta"), row("Gamma")];
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

/// The reference lines the picker printed, with terminal control sequences
/// and carriage returns removed.
fn printed_references(output: &str) -> Vec<String> {
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
    plain
        .lines()
        .filter(|line| line.starts_with("session:"))
        .map(str::to_owned)
        .collect()
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
    let printed = printed_references(&output);
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
fn a_ten_thousand_row_catalog_filters_to_one_page() {
    for text in ["", "retry"] {
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
    }
}

/// The first drawn line after `start` that contains `mark`, once its frame is
/// complete.
fn drawn_line(terminal: &mut Terminal, start: usize, mark: &str) -> String {
    terminal.until_after(start, mark);
    let marked = start
        + String::from_utf8_lossy(&terminal.bytes[start..])
            .find(mark)
            .unwrap();
    terminal.until_after(marked, "^/ preview");
    let output = String::from_utf8_lossy(&terminal.bytes[start..]).into_owned();
    lines(&output)
        .into_iter()
        .find(|line| line.contains(mark))
        .unwrap()
        .to_owned()
}

#[test]
fn a_new_filter_after_a_failed_page_highlights_the_first_row() {
    let mut terminal = Terminal::new("fails", true, true);
    terminal.until("Esc cancel");
    terminal.event(|event| event["shown"] == json!(["session:codex:Alpha", "session:codex:Beta"]));
    terminal.send(b"\x1b[B\x1b[B");
    terminal.event(|event| event["shown"] == json!(["session:codex:Gamma", "session:codex:Delta"]));
    // The earlier page is asked for with its last row to be highlighted, and fails.
    terminal.send(b"\x1b[A");
    terminal.until("the first page is unavailable");
    let start = terminal.bytes.len();
    terminal.send(b"w");
    assert_eq!(
        drawn_line(&mut terminal, start, "→ 5 rows"),
        "sessions › w [fuzzy] → 5 rows"
    );
    terminal.send(b"\r");
    assert_eq!(
        terminal.finish().0,
        json!({"choice":["session:codex:Alpha"]})
    );
}

#[test]
fn a_page_on_its_way_does_not_replace_the_row_list() {
    let mut terminal = Terminal::new("query", true, true);
    terminal.until("Esc cancel");
    let start = terminal.bytes.len();
    // Query mode asks for a page that never arrives.
    terminal.send(b"\x05");
    drawn_line(&mut terminal, start, "[query]");
    terminal.send(b"\x1b");
    let (result, _, events) = terminal.finish();
    assert_eq!(result, json!({"cancelled":true}));
    let shown: Vec<_> = events
        .iter()
        .filter(|event| event.get("shown").is_some())
        .map(|event| json!({"shown":event["shown"],"question":event["question"]}))
        .collect();
    assert_eq!(
        shown,
        vec![json!({"shown":["session:codex:Alpha","session:codex:Beta"],"question":"sessions"})]
    );
}

#[test]
fn a_page_on_its_way_and_a_page_without_rows_are_drawn_differently() {
    let mut terminal = Terminal::new("query", true, true);
    terminal.until("Esc cancel");
    let start = terminal.bytes.len();
    terminal.send(b"\x05");
    assert_eq!(
        drawn_line(&mut terminal, start, "[query]"),
        "sessions › sessions [query] → loading"
    );
    terminal.send(b"\x05");
    let start = terminal.bytes.len();
    terminal.paste("zzz");
    assert_eq!(
        drawn_line(&mut terminal, start, "zzz [fuzzy] → 0"),
        "sessions › zzz [fuzzy] → 0 rows"
    );
    terminal.send(b"\x1b");
    assert_eq!(terminal.finish().0, json!({"cancelled":true}));
}

#[test]
fn a_preview_of_a_row_on_a_page_that_was_left_is_never_drawn() {
    let mut terminal = Terminal::new("held", true, true);
    terminal.until("overview: Alpha at 99 columns");
    terminal.send(b"\x1b[B");
    terminal.event(|event| event["preview"] == "session:codex:Beta");
    // The next page is asked for while Beta's preview is still on its way.
    terminal.send(b"\x1b[B");
    terminal.event(|event| event["dropped"] == "preview");
    let start = terminal.bytes.len();
    terminal.send(b"\x1b[B");
    terminal.until_after(start, "^/ preview");
    terminal.send(b"\x1b");
    let (result, output, _) = terminal.finish();
    assert_eq!(result, json!({"cancelled":true}));
    let drawn = lines(&output);
    let pane = drawn
        .iter()
        .rposition(|line| *line == "─ preview ─")
        .unwrap();
    let preview: Vec<_> = drawn[pane + 1..]
        .iter()
        .take_while(|line| !line.starts_with("↑ up"))
        .filter(|line| !line.is_empty())
        .collect();
    assert_eq!(preview, Vec::<&&str>::new());
}

/// Answers each page request as the function says; `None` never answers.
struct Scripted(fn(&PageRequest) -> Option<Result<Page>>);
impl Source for Scripted {
    async fn page(&self, request: PageRequest) -> Result<Page> {
        match (self.0)(&request) {
            Some(reply) => reply,
            None => std::future::pending().await,
        }
    }
    async fn preview(&self, _: PreviewRequest) -> Result<String> {
        std::future::pending().await
    }
    fn act(&self, _: Action, _: &[String]) -> Result<()> {
        Ok(())
    }
    fn shown(&self, _: &str, _: &[Candidate]) -> Result<()> {
        Ok(())
    }
}

/// The page state in words, with the page number and the cursors known.
fn page_state(state: &State) -> String {
    let page = match &state.page {
        CurrentPage::Loading { highlight, .. } => format!("loading, to highlight {highlight:?}"),
        CurrentPage::Loaded {
            page, highlighted, ..
        } => format!(
            "loaded {}, highlighted {}",
            page.rows
                .iter()
                .map(|row| row.title.as_str())
                .collect::<Vec<_>>()
                .join(","),
            highlighted + 1
        ),
        CurrentPage::Failed(why) => format!("failed: {why}"),
    };
    format!(
        "page {} of {:?}: {page}",
        state.page_number + 1,
        state.cursors
    )
}

fn settled(state: &mut State) {
    while !state.settle() {
        std::thread::yield_now();
    }
}

#[test]
fn page_state_transitions() {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .build()
        .unwrap();
    fn page(names: &[&str], next: Option<&str>) -> Page {
        Page {
            rows: names.iter().map(|name| row(name)).collect(),
            count: Some(3),
            next: next.map(str::to_owned),
        }
    }
    let source = Arc::new(Scripted(|request| {
        match (request.fuzzy.as_deref().unwrap(), &request.cursor) {
            ("", Some(_)) => Some(Ok(page(&["Gamma"], None))),
            ("", None) => Some(Err(Failure::not_yet_available("unavailable"))),
            ("al", None) => Some(Ok(page(&["Alpha", "Beta"], Some("after Beta")))),
            _ => None,
        }
    }));
    let mut state = State::new(
        "sessions".into(),
        None,
        None,
        page(&["Alpha", "Beta"], Some("after Beta")),
    );
    assert_eq!(
        page_state(&state),
        "page 1 of [None]: loaded Alpha title,Beta title, highlighted 1"
    );
    assert!(!state.settle(), "a loaded page waits for nothing");

    // Moving inside a loaded page keeps it loaded.
    state.move_row(false, &source, &runtime);
    state.move_row(true, &source, &runtime);
    assert_eq!(
        page_state(&state),
        "page 1 of [None]: loaded Alpha title,Beta title, highlighted 2"
    );
    state.marked.push("session:codex:Beta".into());

    // Past the last row the next page is loading, and arrives.
    state.move_row(true, &source, &runtime);
    assert_eq!(
        page_state(&state),
        "page 2 of [None, Some(\"after Beta\")]: loading, to highlight First"
    );
    assert_eq!(state.highlighted(), None);
    settled(&mut state);
    assert_eq!(
        page_state(&state),
        "page 2 of [None, Some(\"after Beta\")]: loaded Gamma title, highlighted 1"
    );
    state.move_row(true, &source, &runtime);
    assert_eq!(
        page_state(&state),
        "page 2 of [None, Some(\"after Beta\")]: loaded Gamma title, highlighted 1",
        "the last page has no page after it"
    );

    // Before the first row the earlier page is loading, and fails.
    state.move_row(false, &source, &runtime);
    assert_eq!(
        page_state(&state),
        "page 1 of [None, Some(\"after Beta\")]: loading, to highlight Last"
    );
    settled(&mut state);
    assert_eq!(
        page_state(&state),
        "page 1 of [None, Some(\"after Beta\")]: failed: unavailable"
    );
    assert_eq!(state.highlighted(), None);
    assert_eq!(state.chosen(), ["session:codex:Beta"]);
    state.move_row(true, &source, &runtime);
    state.move_row(false, &source, &runtime);
    assert_eq!(
        page_state(&state),
        "page 1 of [None, Some(\"after Beta\")]: failed: unavailable",
        "a failed first page has no page on either side"
    );

    // A new filter starts again from the first page, from any state, and the
    // request it replaces takes its highlight with it.
    state.fuzzy.push('a');
    state.restart(&source, &runtime);
    assert_eq!(
        page_state(&state),
        "page 1 of [None]: loading, to highlight First"
    );
    assert_eq!(state.chosen(), Vec::<String>::new());
    assert!(!state.settle(), "the page has not arrived");
    state.move_row(true, &source, &runtime);
    state.move_row(false, &source, &runtime);
    assert_eq!(
        page_state(&state),
        "page 1 of [None]: loading, to highlight First"
    );
    state.fuzzy.push('l');
    state.restart(&source, &runtime);
    settled(&mut state);
    assert_eq!(
        page_state(&state),
        "page 1 of [None]: loaded Alpha title,Beta title, highlighted 1"
    );

    // The earlier page can be asked for while a later one is on its way.
    state.move_row(true, &source, &runtime);
    state.move_row(true, &source, &runtime);
    assert_eq!(
        page_state(&state),
        "page 2 of [None, Some(\"after Beta\")]: loading, to highlight First"
    );
    state.move_row(false, &source, &runtime);
    assert_eq!(
        page_state(&state),
        "page 1 of [None, Some(\"after Beta\")]: loading, to highlight Last"
    );
}
