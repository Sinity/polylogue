//! Inline selection for the command line. The caller supplies row pages,
//! overview readings and actions; the picker owns keys and terminal drawing.

use crate::daemon::Failure;
use std::{future::Future, sync::Arc, time::Duration};

mod matching;
mod screen;
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
pub use matching::{CANDIDATE_PAGE, FuzzyRows, fuzzy_page};
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
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Choice {
    Selected(Vec<String>),
    Cancelled,
    Empty,
}

/// Opens over query rows or an ambiguity's candidates and returns references.
/// A complete one-row selection runs the supplied Enter action directly.
pub fn open<S: Source>(source: Arc<S>, options: Options) -> Result<Choice> {
    let mut screen = Screen::terminal();
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
        // Crossterm decodes the ASCII unit separator as Ctrl-7.
        keys.push(Key {
            code: KeyCode::Char('7'),
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

/// Which row a page highlights when it arrives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Highlight {
    First,
    Last,
}

/// The page of rows the picker is on. Only a loaded page has rows, a
/// highlighted row and a place in the terminal's row list.
enum CurrentPage {
    /// The page was asked for and has not arrived. Replacing this value drops
    /// the request, and with it the highlight chosen for that request.
    Loading {
        request: Work<Page>,
        highlight: Highlight,
    },
    Loaded {
        page: Page,
        highlighted: usize,
        /// The first row drawn.
        top: usize,
    },
    /// The request failed; the text says why.
    Failed(String),
}
impl CurrentPage {
    fn loaded(page: Page, highlight: Highlight) -> Self {
        let highlighted = match highlight {
            Highlight::First => 0,
            Highlight::Last => page.rows.len().saturating_sub(1),
        };
        Self::Loaded {
            page,
            highlighted,
            top: 0,
        }
    }
}

/// The preview pane's reading of one row at one width.
enum Overview {
    /// No row is highlighted, or the pane is closed.
    Absent,
    /// Replacing this value drops the request, so its reply is never drawn.
    Loading {
        row: (String, u16),
        request: Work<String>,
    },
    /// The text is the overview, or why there is none.
    Arrived { row: (String, u16), text: String },
}
impl Overview {
    fn row(&self) -> Option<&(String, u16)> {
        match self {
            Self::Absent => None,
            Self::Loading { row, .. } | Self::Arrived { row, .. } => Some(row),
        }
    }
    fn text(&self) -> &str {
        match self {
            Self::Arrived { text, .. } => text,
            Self::Absent | Self::Loading { .. } => "",
        }
    }
    /// Takes the reply once it is in; true when the pane changed.
    fn settle(&mut self) -> bool {
        let Self::Loading { row, request } = self else {
            return false;
        };
        let Some(reply) = request.take() else {
            return false;
        };
        *self = Self::Arrived {
            row: row.clone(),
            text: reply.unwrap_or_else(|error| error.message.clone()),
        };
        true
    }
}

struct State {
    question: String,
    ambiguous: Option<Vec<Candidate>>,
    mode: Mode,
    fuzzy: String,
    query: String,
    page: CurrentPage,
    cursors: Vec<Option<String>>,
    page_number: usize,
    marked: Vec<String>,
    preview: bool,
    overview: Overview,
    /// Why the last action failed.
    message: String,
    shown: (String, Vec<String>),
}
impl State {
    /// The picker opens on a loaded page, read at `cursor`.
    fn new(
        question: String,
        ambiguous: Option<Vec<Candidate>>,
        cursor: Option<String>,
        page: Page,
    ) -> Self {
        Self {
            query: question.clone(),
            question,
            ambiguous,
            mode: Mode::Fuzzy,
            fuzzy: String::new(),
            page: CurrentPage::loaded(page, Highlight::First),
            cursors: vec![cursor],
            page_number: 0,
            marked: Vec::new(),
            preview: true,
            overview: Overview::Absent,
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
    /// The selection text the current mode asks for.
    fn selection(&self) -> &str {
        match self.mode {
            Mode::Fuzzy => &self.question,
            Mode::Query => &self.query,
        }
    }

    fn request(&self) -> PageRequest {
        PageRequest {
            selection: self.selection().to_owned(),
            fuzzy: (self.mode == Mode::Fuzzy).then(|| self.fuzzy.clone()),
            cursor: self.cursors[self.page_number].clone(),
        }
    }
    fn highlighted(&self) -> Option<&Candidate> {
        match &self.page {
            CurrentPage::Loaded {
                page, highlighted, ..
            } => page.rows.get(*highlighted),
            CurrentPage::Loading { .. } | CurrentPage::Failed(_) => None,
        }
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
            return Work::start(runtime, async move {
                fuzzy_page(rows, &request, CANDIDATE_PAGE)
            });
        }
        Work::start(runtime, async move { source.page(request).await })
    }

    // The page's transitions: `load` from any state to loading, `settle` from
    // loading to loaded or failed. `restart` and `move_row` decide which page
    // to load and call `load`.

    /// Asks for the page `page_number` names, in place of whatever was shown
    /// or on its way.
    fn load<S: Source>(
        &mut self,
        source: &Arc<S>,
        runtime: &tokio::runtime::Runtime,
        highlight: Highlight,
    ) {
        self.overview = Overview::Absent;
        self.page = CurrentPage::Loading {
            request: self.start_page(source.clone(), runtime),
            highlight,
        };
    }
    /// Takes the reply once it is in; true when the page changed.
    fn settle(&mut self) -> bool {
        let CurrentPage::Loading { request, highlight } = &self.page else {
            return false;
        };
        let Some(reply) = request.take() else {
            return false;
        };
        self.page = match reply {
            Ok(page) => CurrentPage::loaded(page, *highlight),
            Err(error) => CurrentPage::Failed(error.message.clone()),
        };
        self.message.clear();
        self.overview = Overview::Absent;
        true
    }
    /// A changed filter, query or mode starts again from the first page.
    fn restart<S: Source>(&mut self, source: &Arc<S>, runtime: &tokio::runtime::Runtime) {
        self.cursors = vec![None];
        self.page_number = 0;
        self.marked.clear();
        self.message.clear();
        self.load(source, runtime, Highlight::First);
    }
    /// Moves the highlight within a loaded page, or loads the page past its
    /// edge. The earlier page can be asked for from any state.
    fn move_row<S: Source>(
        &mut self,
        next: bool,
        source: &Arc<S>,
        runtime: &tokio::runtime::Runtime,
    ) {
        self.overview = Overview::Absent;
        let later = match &mut self.page {
            CurrentPage::Loaded {
                page, highlighted, ..
            } => {
                if next && *highlighted + 1 < page.rows.len() {
                    *highlighted += 1;
                    return;
                }
                if !next && *highlighted > 0 {
                    *highlighted -= 1;
                    return;
                }
                page.next.clone()
            }
            CurrentPage::Loading { .. } | CurrentPage::Failed(_) => None,
        };
        if next {
            let Some(cursor) = later else { return };
            self.cursors.truncate(self.page_number + 1);
            self.cursors.push(Some(cursor));
            self.page_number += 1;
            self.load(source, runtime, Highlight::First);
        } else if self.page_number > 0 {
            self.page_number -= 1;
            self.load(source, runtime, Highlight::Last);
        }
    }
}

fn run<S: Source>(source: Arc<S>, options: Options, screen: &mut Screen) -> Result<Choice> {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .enable_all()
        .build()
        .map_err(Failure::io)?;
    let (question, ambiguous) = match options.selection.clone() {
        Selection::Query(question) => (question, None),
        Selection::Ambiguous {
            question,
            candidates,
        } => (question, Some(candidates)),
    };
    // No terminal mode is changed until the cardinality is known.
    let mut request = PageRequest {
        selection: question.clone(),
        cursor: None,
        fuzzy: Some(String::new()),
    };
    let mut page = if let Some(rows) = &ambiguous {
        fuzzy_page(rows.clone(), &request, CANDIDATE_PAGE)?
    } else {
        runtime.block_on(source.page(request.clone()))?
    };
    // Empty transfer pages do not imply that a selection is empty.
    while page.rows.is_empty() && page.next.is_some() {
        request.cursor = page.next;
        page = runtime.block_on(source.page(request.clone()))?;
    }
    if page.rows.is_empty() {
        return Ok(Choice::Empty);
    }
    if page.rows.len() == 1 && page.next.is_none() {
        return finish(&*source, &options, vec![page.rows[0].reference.clone()]);
    }
    if !screen.is_terminal() {
        let candidates = page
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
            "choose a reference from the candidates; the picker needs terminal input and output",
        );
        failure.error = failure.error.with_candidates(candidates);
        return Err(failure);
    }
    let mut state = State::new(question, ambiguous, request.cursor, page);
    screen.enter()?;
    let result = interact(source.clone(), &options, screen, &runtime, &mut state);
    let restored = screen.leave();
    let choice = result?;
    restored?;
    match choice {
        Choice::Selected(references) => finish(&*source, &options, references),
        Choice::Cancelled | Choice::Empty => Ok(choice),
    }
}

fn finish(source: &impl Source, options: &Options, references: Vec<String>) -> Result<Choice> {
    if !options.enter.is_empty() {
        source.act(Action::Enter, &references)?;
    }
    Ok(Choice::Selected(references))
}

fn interact<S: Source>(
    source: Arc<S>,
    options: &Options,
    screen: &mut Screen,
    runtime: &tokio::runtime::Runtime,
    state: &mut State,
) -> Result<Choice> {
    let mut dirty = true;
    loop {
        dirty |= state.settle();
        let (width, height) = screen.size()?;
        let wanted = state
            .highlighted()
            .filter(|_| state.preview)
            .map(|row| (row.reference.clone(), width.saturating_sub(1)));
        if wanted.as_ref() != state.overview.row() {
            state.overview = match wanted {
                Some(row) => Overview::Loading {
                    request: start_overview(source.clone(), runtime, row.clone()),
                    row,
                },
                None => Overview::Absent,
            };
            dirty = true;
        }
        dirty |= state.overview.settle();
        if dirty {
            draw(&*source, options, screen, state, width, height)?;
            dirty = false;
        }
        if !event::poll(FRAME).map_err(Failure::io)? {
            continue;
        }
        let event = event::read().map_err(Failure::io)?;
        let key = match event {
            Event::Resize(_, _) => {
                state.overview = Overview::Absent;
                dirty = true;
                continue;
            }
            Event::Paste(text) => {
                state.text_mut().push_str(&text);
                state.restart(&source, runtime);
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
                if !chosen.is_empty() && !matches!(state.page, CurrentPage::Loading { .. }) {
                    return Ok(Choice::Selected(chosen));
                }
            }
            Some(Command::Next | Command::Previous) => {
                state.move_row(matches!(command, Some(Command::Next)), &source, runtime);
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
                    state.move_row(true, &source, runtime);
                }
            }
            Some(Command::Mode) => {
                state.mode = match state.mode {
                    Mode::Fuzzy => Mode::Query,
                    Mode::Query => Mode::Fuzzy,
                };
                state.restart(&source, runtime);
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
                    state.overview = Overview::Absent;
                    screen.leave()?;
                    let result = source.act(action, &[reference]);
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
            None if edit(state.text_mut(), key) => state.restart(&source, runtime),
            None => {}
        }
    }
}

fn start_overview<S: Source>(
    source: Arc<S>,
    runtime: &tokio::runtime::Runtime,
    (reference, width): (String, u16),
) -> Work<String> {
    Work::start(runtime, async move {
        let request = PreviewRequest {
            reference,
            width,
            deadline: PREVIEW_DEADLINE,
        };
        tokio::time::timeout(PREVIEW_DEADLINE, source.preview(request))
            .await
            .map_err(|_| {
                Failure::deadline_exceeded(format!(
                    "overview exceeded its {} ms deadline",
                    PREVIEW_DEADLINE.as_millis()
                ))
            })?
    })
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
    if let CurrentPage::Loaded {
        highlighted, top, ..
    } = &mut state.page
    {
        if *highlighted < *top {
            *top = *highlighted;
        } else if *highlighted >= *top + row_lines {
            *top = *highlighted + 1 - row_lines;
        }
    }
    let question = state.selection();
    let (shown, highlighted, top, count, message): (&[Candidate], _, _, _, &str) = match &state.page
    {
        CurrentPage::Loaded {
            page,
            highlighted,
            top,
        } => {
            let shown =
                &page.rows[(*top).min(page.rows.len())..(*top + row_lines).min(page.rows.len())];
            // Only a loaded page replaces the terminal's row list.
            let remembered = (
                question.to_owned(),
                shown.iter().map(|row| row.reference.clone()).collect(),
            );
            if remembered != state.shown {
                source.shown(question, shown)?;
                state.shown = remembered;
            }
            let count = page
                .count
                .map_or_else(|| "?".into(), |count| count.to_string());
            (
                shown,
                Some(*highlighted),
                *top,
                format!("{count} rows"),
                &state.message,
            )
        }
        CurrentPage::Loading { .. } => (&[], None, 0, "loading".into(), &state.message),
        CurrentPage::Failed(why) => (&[], None, 0, "failed".into(), why),
    };
    let mut lines = vec![format!(
        "{} › {} [{}] → {}",
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
            if highlighted == Some(top + index) {
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
                .text()
                .split('\n')
                .take(preview_lines.saturating_sub(1))
                .map(screen::safe_line),
        );
        while lines.len() < row_lines + preview_lines + 1 {
            lines.push(String::new());
        }
    }
    lines.push(screen::safe_line(message));
    lines.extend(footer);
    screen.draw(&lines)
}

#[cfg(test)]
mod tests;
