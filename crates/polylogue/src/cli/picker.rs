//! Inline selection for the command line. The caller supplies row pages,
//! overview readings and actions; the picker owns keys and terminal drawing.

use crate::daemon::Failure;
use std::{future::Future, sync::Arc, time::Duration};

mod matching;
mod screen;
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
pub use matching::fuzzy_page;
use screen::Screen;
use std::sync::mpsc;

pub type Result<T> = std::result::Result<T, Failure>;

/// One selectable row, including the text used by fuzzy filtering.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Candidate {
    pub reference: String,
    pub title: String,
    pub first_prompt: String,
    pub repository: String,
    pub description: String,
}

/// A transfer page, not a limit on the selection's size.
#[derive(Clone, Debug, Default)]
pub struct Page {
    pub rows: Vec<Candidate>,
    pub count: Option<u64>,
    pub next: Option<String>,
}

#[derive(Clone, Debug)]
pub enum Selection {
    Query(String),
    Ambiguous {
        question: String,
        candidates: Vec<Candidate>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Fuzzy,
    Query,
}

/// Query mode replaces the selection; fuzzy mode pages the original selection.
#[derive(Clone, Debug)]
pub struct PageRequest {
    pub selection: String,
    pub cursor: Option<String>,
    /// Some text asks for catalog candidates through `fuzzy_page`; None asks
    /// the daemon to evaluate `selection` in query mode.
    pub fuzzy: Option<String>,
}

/// The source renders an overview at this width and honours this deadline.
#[derive(Clone, Debug)]
pub struct PreviewRequest {
    pub reference: String,
    pub width: u16,
    pub deadline: Duration,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    Enter,
    Read,
    Browser,
    Copy,
}

/// Pages and previews are cancellable futures. Actions run with the terminal
/// restored, so a pager can take input and return to this same selection.
pub trait Source: Send + Sync + 'static {
    fn page(&self, request: PageRequest) -> impl Future<Output = Result<Page>> + Send;
    fn preview(&self, request: PreviewRequest) -> impl Future<Output = Result<String>> + Send;
    fn act(&self, action: Action, references: &[String]) -> Result<()>;
    /// Replaces the terminal's row list, in displayed order, with its question.
    fn shown(&self, question: &str, rows: &[Candidate]) -> Result<()>;
}

#[derive(Clone, Debug)]
pub struct Options {
    pub selection: Selection,
    /// The words Enter will run, or empty to print references.
    pub enter: String,
    pub multiple: bool,
    pub json: bool,
    pub format: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Choice {
    Selected(Vec<String>),
    Cancelled,
}

/// Opens over query rows or an ambiguity's candidates and returns references.
/// A complete one-row selection runs the supplied Enter action directly.
pub fn open<S: Source>(source: Arc<S>, options: Options) -> Result<Choice> {
    let mut screen = Screen::stdout();
    run(source, options, &mut screen)
}

const PREVIEW_DEADLINE: Duration = Duration::from_millis(100);
const FRAME: Duration = Duration::from_millis(8);

#[derive(Clone, Copy)]
enum Command {
    Next,
    Previous,
    Mark,
    Enter,
    Read,
    Browser,
    Copy,
    Print,
    Mode,
    Preview,
    Cancel,
}

struct Key {
    code: KeyCode,
    modifiers: KeyModifiers,
    label: &'static str,
    command: Command,
}

fn keys() -> &'static [Key] {
    use Command::*;
    static KEYS: std::sync::OnceLock<Vec<Key>> = std::sync::OnceLock::new();
    KEYS.get_or_init(|| {
        let mut keys = vec![
            Key {
                code: KeyCode::Up,
                modifiers: KeyModifiers::NONE,
                label: "↑ up",
                command: Previous,
            },
            Key {
                code: KeyCode::Down,
                modifiers: KeyModifiers::NONE,
                label: "↓ down",
                command: Next,
            },
            Key {
                code: KeyCode::Tab,
                modifiers: KeyModifiers::NONE,
                label: "Tab mark",
                command: Mark,
            },
            Key {
                code: KeyCode::Enter,
                modifiers: KeyModifiers::NONE,
                label: "Enter",
                command: Enter,
            },
            Key {
                code: KeyCode::Esc,
                modifiers: KeyModifiers::NONE,
                label: "Esc cancel",
                command: Cancel,
            },
        ];
        // Shared web letters retain their meaning in the picker.
        for (action, label, command) in [
            (web_shell::keys::KeyAction::Next, "^j down", Next),
            (web_shell::keys::KeyAction::Previous, "^k up", Previous),
            (web_shell::keys::KeyAction::CopyMenu, "^y copy", Copy),
        ] {
            let key = web_shell::keys::KEYS
                .iter()
                .find(|key| key.action == action && key.key.len() == 1 && key.picker.is_some())
                .expect("shared picker letter");
            keys.push(Key {
                code: KeyCode::Char(key.key.chars().next().unwrap()),
                modifiers: KeyModifiers::CONTROL,
                label,
                command,
            });
        }
        for (letter, label, command) in [
            ('r', "^r read", Read),
            ('o', "^o browser", Browser),
            ('p', "^p print", Print),
            ('e', "^e query/fuzzy", Mode),
            ('/', "^/ preview", Preview),
        ] {
            keys.push(Key {
                code: KeyCode::Char(letter),
                modifiers: KeyModifiers::CONTROL,
                label,
                command,
            });
        }
        // Legacy terminals encode Ctrl-/ as the ASCII unit separator.
        keys.push(Key {
            code: KeyCode::Char('_'),
            modifiers: KeyModifiers::CONTROL,
            label: "",
            command: Preview,
        });
        keys
    })
}

fn footer(enter: &str) -> String {
    keys()
        .iter()
        .filter(|key| !key.label.is_empty())
        .map(|key| {
            if matches!(key.command, Command::Enter) {
                format!(
                    "Enter {}",
                    if enter.is_empty() {
                        "print references"
                    } else {
                        enter
                    }
                )
            } else {
                key.label.to_owned()
            }
        })
        .collect::<Vec<_>>()
        .join(" · ")
}

/// Dropping a request cancels its future and disconnects its reply. A later
/// result therefore has no path back into the displayed selection.
struct Work<T> {
    task: tokio::task::JoinHandle<()>,
    reply: mpsc::Receiver<Result<T>>,
}
impl<T: Send + 'static> Work<T> {
    fn start(
        runtime: &tokio::runtime::Runtime,
        work: impl Future<Output = Result<T>> + Send + 'static,
    ) -> Self {
        let (send, reply) = mpsc::channel();
        let task = runtime.spawn(async move {
            let _ = send.send(work.await);
        });
        Self { task, reply }
    }
    fn take(&self) -> Option<Result<T>> {
        match self.reply.try_recv() {
            Ok(reply) => Some(reply),
            Err(mpsc::TryRecvError::Empty) => None,
            Err(mpsc::TryRecvError::Disconnected) => Some(Err(Failure::internal(
                "picker request ended without a reply",
            ))),
        }
    }
}
impl<T> Drop for Work<T> {
    fn drop(&mut self) {
        self.task.abort();
    }
}

struct State {
    question: String,
    ambiguous: Option<Vec<Candidate>>,
    mode: Mode,
    fuzzy: String,
    query: String,
    page: Page,
    index: usize,
    top: usize,
    cursors: Vec<Option<String>>,
    page_number: usize,
    select_last: bool,
    marked: Vec<String>,
    preview: bool,
    overview: String,
    preview_key: Option<(String, u16)>,
    message: String,
    shown: (String, Vec<String>),
}
impl State {
    fn new(selection: Selection) -> Self {
        let (question, ambiguous) = match selection {
            Selection::Query(question) => (question, None),
            Selection::Ambiguous {
                question,
                candidates,
            } => (question, Some(candidates)),
        };
        Self {
            query: question.clone(),
            question,
            ambiguous,
            mode: Mode::Fuzzy,
            fuzzy: String::new(),
            page: Page::default(),
            index: 0,
            top: 0,
            cursors: vec![None],
            page_number: 0,
            select_last: false,
            marked: Vec::new(),
            preview: true,
            overview: String::new(),
            preview_key: None,
            message: String::new(),
            shown: (String::new(), vec![]),
        }
    }
    fn text(&self) -> &str {
        match self.mode {
            Mode::Fuzzy => &self.fuzzy,
            Mode::Query => &self.query,
        }
    }
    fn text_mut(&mut self) -> &mut String {
        match self.mode {
            Mode::Fuzzy => &mut self.fuzzy,
            Mode::Query => &mut self.query,
        }
    }
    fn request(&self) -> PageRequest {
        PageRequest {
            selection: match self.mode {
                Mode::Fuzzy => self.question.clone(),
                Mode::Query => self.query.clone(),
            },
            fuzzy: (self.mode == Mode::Fuzzy).then(|| self.fuzzy.clone()),
            cursor: self.cursors[self.page_number].clone(),
        }
    }
    fn highlighted(&self) -> Option<&Candidate> {
        self.page.rows.get(self.index)
    }
    fn chosen(&self) -> Vec<String> {
        if self.marked.is_empty() {
            self.highlighted()
                .map(|row| vec![row.reference.clone()])
                .unwrap_or_default()
        } else {
            self.marked.clone()
        }
    }
    fn reset(&mut self) {
        self.page = Page::default();
        self.index = 0;
        self.top = 0;
        self.cursors = vec![None];
        self.page_number = 0;
        self.marked.clear();
        self.message.clear();
        self.clear_preview();
    }
    fn clear_preview(&mut self) {
        self.preview_key = None;
        self.overview.clear();
    }
    fn accept(&mut self, page: Page) {
        self.page = page;
        self.index = if self.select_last {
            self.page.rows.len().saturating_sub(1)
        } else {
            0
        };
        self.top = 0;
        self.select_last = false;
        self.message.clear();
        self.clear_preview();
    }
    fn start_page<S: Source>(
        &self,
        source: Arc<S>,
        runtime: &tokio::runtime::Runtime,
    ) -> Work<Page> {
        let request = self.request();
        if self.mode == Mode::Fuzzy
            && let Some(rows) = &self.ambiguous
        {
            let rows = rows.clone();
            return Work::start(runtime, async move { fuzzy_page(rows, &request, 64) });
        }
        Work::start(runtime, async move { source.page(request).await })
    }
    fn move_row(&mut self, next: bool) -> bool {
        self.clear_preview();
        if next && self.index + 1 < self.page.rows.len() {
            self.index += 1;
        } else if !next && self.index > 0 {
            self.index -= 1;
        } else if next && let Some(cursor) = &self.page.next {
            self.cursors.truncate(self.page_number + 1);
            self.cursors.push(Some(cursor.clone()));
            self.page_number += 1;
            self.page = Page::default();
            return true;
        } else if !next && self.page_number > 0 {
            self.page_number -= 1;
            self.select_last = true;
            self.page = Page::default();
            return true;
        }
        false
    }
}

fn run<S: Source>(source: Arc<S>, options: Options, screen: &mut Screen) -> Result<Choice> {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .enable_all()
        .build()
        .map_err(Failure::io)?;
    let mut state = State::new(options.selection.clone());
    // No terminal mode is changed until the cardinality is known.
    let mut page = if let Some(rows) = &state.ambiguous {
        fuzzy_page(rows.clone(), &state.request(), 64)?
    } else {
        runtime.block_on(source.page(state.request()))?
    };
    // Empty transfer pages do not imply that a selection is empty.
    while page.rows.is_empty() && page.next.is_some() {
        state.accept(page);
        state.cursors[0] = state.page.next.clone();
        let request = state.request();
        page = runtime.block_on(source.page(request))?;
    }
    if page.rows.is_empty() {
        return Err(Failure::new(
            polylogue_model::Code::Empty,
            "no rows match the selection",
        ));
    }
    state.accept(page);
    if state.page.rows.len() == 1 && state.page.next.is_none() {
        return finish(&*source, &options, screen, state.chosen());
    }
    let machine = options.json || matches!(options.format.as_deref(), Some("json" | "jsonl"));
    if !screen.is_terminal() || machine {
        let candidates = state
            .page
            .rows
            .iter()
            .map(|row| {
                format!(
                    "{}  {}  {}\npolylogue {} {}",
                    row.reference,
                    row.description,
                    row.title,
                    polylogue_render::shell_quote(&row.reference),
                    if options.enter.is_empty() {
                        "read"
                    } else {
                        &options.enter
                    }
                )
            })
            .collect();
        let mut failure = Failure::new(
            polylogue_model::Code::Ambiguous,
            "choose a reference from the candidates; the picker needs terminal input and output without a machine format",
        );
        failure.error = failure.error.with_candidates(candidates);
        return Err(failure);
    }
    screen.enter()?;
    let result = interact(
        &*source,
        source.clone(),
        &options,
        screen,
        &runtime,
        &mut state,
    );
    let restored = screen.leave();
    let choice = result?;
    restored?;
    match choice {
        Choice::Selected(references) => finish(&*source, &options, screen, references),
        Choice::Cancelled => Ok(Choice::Cancelled),
    }
}

fn finish(
    source: &impl Source,
    options: &Options,
    screen: &mut Screen,
    references: Vec<String>,
) -> Result<Choice> {
    if options.enter.is_empty() {
        screen.print(&references)?;
    } else {
        source.act(Action::Enter, &references)?;
    }
    Ok(Choice::Selected(references))
}

fn interact<S: Source>(
    actions: &S,
    source: Arc<S>,
    options: &Options,
    screen: &mut Screen,
    runtime: &tokio::runtime::Runtime,
    state: &mut State,
) -> Result<Choice> {
    let mut page: Option<Work<Page>> = None;
    let mut preview: Option<Work<String>> = None;
    let mut dirty = true;
    loop {
        if let Some(reply) = page.as_ref().and_then(Work::take) {
            page = None;
            match reply {
                Ok(rows) => state.accept(rows),
                Err(error) => state.message = error.message.clone(),
            }
            dirty = true;
        }
        let (width, height) = screen.size()?;
        let preview_key = state
            .highlighted()
            .filter(|_| state.preview)
            .map(|row| (row.reference.clone(), width.saturating_sub(1)));
        if preview_key != state.preview_key {
            preview = None;
            state.overview.clear();
            state.preview_key = preview_key.clone();
            if let Some((reference, width)) = preview_key {
                let source = source.clone();
                preview = Some(Work::start(runtime, async move {
                    let request = PreviewRequest {
                        reference,
                        width,
                        deadline: PREVIEW_DEADLINE,
                    };
                    tokio::time::timeout(PREVIEW_DEADLINE, source.preview(request))
                        .await
                        .map_err(|_| {
                            Failure::deadline_exceeded("overview exceeded its 100 ms deadline")
                        })?
                }));
            }
            dirty = true;
        }
        if let Some(reply) = preview.as_ref().and_then(Work::take) {
            preview = None;
            state.overview = match reply {
                Ok(text) => text,
                Err(error) => error.message.clone(),
            };
            dirty = true;
        }
        if dirty {
            draw(actions, options, screen, state, width, height)?;
            dirty = false;
        }
        if !event::poll(FRAME).map_err(Failure::io)? {
            continue;
        }
        let event = event::read().map_err(Failure::io)?;
        let key = match event {
            Event::Resize(_, _) => {
                state.clear_preview();
                dirty = true;
                continue;
            }
            Event::Paste(text) => {
                state.text_mut().push_str(&text);
                state.reset();
                page = Some(state.start_page(source.clone(), runtime));
                preview = None;
                dirty = true;
                continue;
            }
            Event::Key(key) if key.kind != KeyEventKind::Release => key,
            _ => continue,
        };
        dirty = true;
        let command = keys()
            .iter()
            .find(|binding| binding.code == key.code && binding.modifiers == key.modifiers)
            .map(|binding| binding.command);
        match command {
            Some(Command::Cancel) => return Ok(Choice::Cancelled),
            Some(Command::Enter) => {
                let chosen = state.chosen();
                if !chosen.is_empty() && page.is_none() {
                    return Ok(Choice::Selected(chosen));
                }
            }
            Some(Command::Next | Command::Previous) => {
                if state.move_row(matches!(command, Some(Command::Next))) {
                    page = Some(state.start_page(source.clone(), runtime));
                }
            }
            Some(Command::Mark) => {
                if let Some(row) = state.highlighted().cloned() {
                    if let Some(index) = state
                        .marked
                        .iter()
                        .position(|reference| reference == &row.reference)
                    {
                        state.marked.remove(index);
                    } else {
                        if !options.multiple {
                            state.marked.clear();
                        }
                        state.marked.push(row.reference);
                    }
                    if state.move_row(true) {
                        page = Some(state.start_page(source.clone(), runtime));
                    }
                }
            }
            Some(Command::Mode) => {
                state.mode = match state.mode {
                    Mode::Fuzzy => Mode::Query,
                    Mode::Query => Mode::Fuzzy,
                };
                state.reset();
                page = Some(state.start_page(source.clone(), runtime));
                preview = None;
            }
            Some(Command::Preview) => {
                state.preview = !state.preview;
            }
            Some(Command::Read | Command::Browser | Command::Copy) => {
                if let Some(row) = state.highlighted() {
                    let reference = row.reference.clone();
                    let action = match command {
                        Some(Command::Read) => Action::Read,
                        Some(Command::Browser) => Action::Browser,
                        _ => Action::Copy,
                    };
                    preview = None;
                    state.clear_preview();
                    screen.leave()?;
                    let result = actions.act(action, &[reference]);
                    screen.enter()?;
                    if let Err(error) = result {
                        state.message = error.message.clone();
                    }
                }
            }
            Some(Command::Print) => {
                if let Some(row) = state.highlighted() {
                    screen.print(std::slice::from_ref(&row.reference))?;
                }
            }
            None if edit(state.text_mut(), key) => {
                state.reset();
                page = Some(state.start_page(source.clone(), runtime));
                preview = None;
            }
            None => {}
        }
    }
}

fn edit(text: &mut String, key: KeyEvent) -> bool {
    match key.code {
        KeyCode::Char(c)
            if !key
                .modifiers
                .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT) =>
        {
            text.push(c);
            true
        }
        KeyCode::Backspace => text.pop().is_some(),
        KeyCode::Char('u') if key.modifiers == KeyModifiers::CONTROL => {
            let changed = !text.is_empty();
            text.clear();
            changed
        }
        _ => false,
    }
}

fn wrapped(text: &str, width: usize) -> Vec<String> {
    let mut lines = vec![String::new()];
    for word in text.split_whitespace() {
        let mut word = word;
        while !word.is_empty() {
            let line = lines.last_mut().unwrap();
            if !line.is_empty() && line.len() + word.len() + 1 > width {
                lines.push(String::new());
                continue;
            }
            let chunk = screen::clipped(word, width.max(4));
            if !line.is_empty() {
                line.push(' ');
            }
            line.push_str(&chunk);
            word = &word[chunk.len()..];
            if !word.is_empty() {
                lines.push(String::new());
            }
        }
    }
    lines
}

fn draw(
    source: &impl Source,
    options: &Options,
    screen: &mut Screen,
    state: &mut State,
    width: u16,
    height: u16,
) -> Result<()> {
    let width = width.saturating_sub(1).max(4) as usize;
    let footer = wrapped(&screen::safe_line(&footer(&options.enter)), width);
    let available = (height as usize).saturating_sub(footer.len() + 3);
    let preview_lines = if state.preview {
        (available / 2).min(6)
    } else {
        0
    };
    let row_lines = available.saturating_sub(preview_lines).max(1);
    if state.index < state.top {
        state.top = state.index;
    } else if state.index >= state.top + row_lines {
        state.top = state.index + 1 - row_lines;
    }
    let shown = &state.page.rows
        [state.top.min(state.page.rows.len())..(state.top + row_lines).min(state.page.rows.len())];
    let question = match state.mode {
        Mode::Fuzzy => &state.question,
        Mode::Query => &state.query,
    };
    let remembered = (
        question.clone(),
        shown.iter().map(|row| row.reference.clone()).collect(),
    );
    if remembered != state.shown {
        source.shown(question, shown)?;
        state.shown = remembered;
    }
    let count = state
        .page
        .count
        .map_or_else(|| "?".into(), |count| count.to_string());
    let mut lines = vec![format!(
        "{} › {} [{}] → {} rows",
        screen::safe_line(&state.question),
        screen::safe_line(state.text()),
        if state.mode == Mode::Fuzzy {
            "fuzzy"
        } else {
            "query"
        },
        count
    )];
    for (index, row) in shown.iter().enumerate() {
        lines.push(format!(
            "{}{} {} {}  {}  {}",
            if state.top + index == state.index {
                "›"
            } else {
                " "
            },
            if state.marked.contains(&row.reference) {
                "*"
            } else {
                " "
            },
            index + 1,
            screen::safe_line(&row.title),
            screen::safe_line(&row.repository),
            screen::safe_line(&row.description)
        ));
    }
    while lines.len() < row_lines + 1 {
        lines.push(String::new());
    }
    if preview_lines > 0 {
        lines.push("─ preview ─".into());
        lines.extend(
            state
                .overview
                .lines()
                .take(preview_lines.saturating_sub(1))
                .map(screen::safe_line),
        );
        while lines.len() < row_lines + preview_lines + 1 {
            lines.push(String::new());
        }
    }
    lines.push(screen::safe_line(&state.message));
    lines.extend(footer);
    screen.draw(&lines)
}

#[cfg(test)]
mod tests;
