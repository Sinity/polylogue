//! One selection followed by declared verbs. Shell arguments retain their boundaries.
use command_surface::{ArgumentForm, StepDeclaration};
use polylogue_model::Code;
use serde_json::Value;
use std::{ops::Range, path::PathBuf};

#[derive(Debug)]
pub(crate) struct ParsedLine {
    pub first: FirstWord,
    pub selection: Option<SelectionWord>,
    pub listing: ListingOptions,
    pub steps: Vec<ParsedStep>,
    pub line: LineOptions,
}
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum FirstWord {
    Nothing,
    Stdin,
    Command,
    Verb,
    SavedQuery { name: String },
    Unit,
    Reference,
    RowNumbers,
    QueryText,
    SearchWord,
}
impl FirstWord {
    pub fn rule(&self) -> u8 {
        match self {
            Self::Nothing => 1,
            Self::Stdin => 2,
            Self::Command => 3,
            Self::Verb => 4,
            Self::SavedQuery { .. } => 5,
            Self::Unit => 6,
            Self::Reference | Self::RowNumbers => 7,
            Self::QueryText => 8,
            Self::SearchWord => 9,
        }
    }
}
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum SelectionWord {
    Stdin,
    Text(String),
    References(Vec<String>),
}
#[derive(Debug)]
pub(crate) struct ParsedStep {
    pub verb: &'static str,
    pub step: Value,
    pub to: Option<Destination>,
    pub words: Range<usize>,
    pub stdin: Option<String>,
}
#[derive(Debug, Default)]
pub(crate) struct LineOptions {
    pub json: bool,
    pub fresh: bool,
    pub deadline: Option<String>,
    pub request_id: Option<String>,
    pub dry_run: bool,
    pub explain: bool,
    pub all: bool,
    pub wait: Option<String>,
}
#[derive(Debug, Default)]
pub(crate) struct ListingOptions {
    pub page_size: Option<std::num::NonZeroU64>,
    pub fields: Vec<String>,
    pub why: bool,
    pub to: Option<Destination>,
    pub cursor: Option<String>,
    pub budget: Option<String>,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Destination {
    Stdout,
    Pager,
    File(PathBuf),
    Dir(PathBuf),
    Clipboard,
    Browser,
}
impl Destination {
    fn parse(text: &str) -> Self {
        match text {
            "stdout" => Self::Stdout,
            "pager" => Self::Pager,
            "clipboard" => Self::Clipboard,
            "browser" => Self::Browser,
            _ if text.ends_with('/') => Self::Dir(text.into()),
            _ => Self::File(text.into()),
        }
    }
}
#[derive(Debug)]
pub(crate) struct Refusal {
    pub code: Code,
    pub message: String,
    pub position: Option<usize>,
    pub try_line: Option<String>,
    pub candidates: Vec<String>,
}
impl Refusal {
    pub(crate) fn new(code: Code, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            position: None,
            try_line: None,
            candidates: vec![],
        }
    }
    fn at(mut self, position: usize, words: &[String]) -> Self {
        self.position = Some(position);
        self.try_line = Some(shell_line(words));
        self
    }
}
#[derive(Default)]
pub(crate) struct FirstWordCatalog {
    pub saved: Vec<String>,
    pub shipped: Vec<String>,
}
impl FirstWordCatalog {
    pub fn load(paths: &crate::config::Paths, archive: &std::path::Path) -> Self {
        let saved = polylogue_store::completion::CatalogReader::open(
            &paths.completion_catalog(archive),
            &crate::runtime_dir::archive_key(archive),
        )
        .ok()
        .flatten()
        .and_then(|reader| reader.saved_names("").ok())
        .unwrap_or_default()
        .into_iter()
        .map(|item| item.name)
        .collect();
        Self {
            saved,
            shipped: vec![],
        }
    }
}
pub(crate) fn shell_line(words: &[String]) -> String {
    std::iter::once("polylogue".to_owned())
        .chain(words.iter().map(|word| polylogue_render::shell_quote(word)))
        .collect::<Vec<_>>()
        .join(" ")
}
fn first(word: Option<&str>, catalog: &FirstWordCatalog, verbs: &[StepDeclaration]) -> FirstWord {
    let Some(word) = word else {
        return FirstWord::Nothing;
    };
    if word == "-" {
        return FirstWord::Stdin;
    }
    if matches!(word, "--help" | "-h") || crate::commands::is_cli_command(word) {
        return FirstWord::Command;
    }
    if verbs.iter().any(|verb| verb.verb() == word) {
        return FirstWord::Verb;
    }
    if catalog
        .saved
        .iter()
        .chain(&catalog.shipped)
        .any(|name| name == word)
    {
        return FirstWord::SavedQuery { name: word.into() };
    }
    if polylogue_query::ast::Unit::all().any(|unit| unit.name() == word) {
        return FirstWord::Unit;
    }
    if crate::row_numbers::is_selection(word) {
        return FirstWord::RowNumbers;
    }
    if polylogue_model::Reference::parse(word).is_ok()
        || word.starts_with("https://")
        || word.starts_with("http://")
        || matches!(word, "latest" | "self" | "here")
        || (word.len() >= 8 && word.chars().all(|c| c.is_ascii_hexdigit() || c == '-'))
        || word
            .strip_prefix('@')
            .is_some_and(|name| name.starts_with(char::is_alphabetic))
    {
        return FirstWord::Reference;
    }
    if word
        .chars()
        .any(|c| c.is_whitespace() || " :|\"()~*,".contains(c))
    {
        FirstWord::QueryText
    } else {
        FirstWord::SearchWord
    }
}
fn takes_word(argument: &command_surface::ArgumentSpec) -> bool {
    argument.form != ArgumentForm::Flag
}
fn value(args: &[String], index: &mut usize) -> Result<String, Refusal> {
    *index += 1;
    args.get(*index)
        .cloned()
        .ok_or_else(|| Refusal::new(Code::BadRequest, "option needs a value"))
}
fn line_option(
    args: &[String],
    index: &mut usize,
    line: &mut LineOptions,
) -> Result<bool, Refusal> {
    let (word, assigned) = args[*index]
        .split_once('=')
        .map_or((args[*index].as_str(), None), |(name, value)| {
            (name, Some(value))
        });
    let flag = match word {
        "--json" => Some(&mut line.json),
        "--fresh" => Some(&mut line.fresh),
        "--dry-run" => Some(&mut line.dry_run),
        "--explain" => Some(&mut line.explain),
        "--all" => Some(&mut line.all),
        _ => None,
    };
    if let Some(flag) = flag {
        *flag = assigned
            .map(str::parse::<bool>)
            .transpose()
            .map_err(|_| Refusal::new(Code::BadValue, format!("{word} needs true or false")))?
            .unwrap_or(true);
        return Ok(true);
    }
    match args[*index].as_str() {
        "--deadline" => line.deadline = Some(value(args, index)?),
        "--request-id" => line.request_id = Some(value(args, index)?),
        "--wait" => line.wait = Some(value(args, index)?),
        _ => return Ok(false),
    }
    Ok(true)
}
fn listing_option(
    args: &[String],
    index: &mut usize,
    listing: &mut ListingOptions,
) -> Result<bool, Refusal> {
    match args[*index].as_str() {
        "--page-size" => {
            listing.page_size = Some(value(args, index)?.parse().map_err(|_| {
                Refusal::new(Code::BadValue, "--page-size needs a positive integer")
            })?)
        }
        "--fields" => listing.fields = value(args, index)?.split(',').map(str::to_owned).collect(),
        "--why" => listing.why = true,
        word if word.starts_with("--why=") => {
            listing.why = word[6..]
                .parse()
                .map_err(|_| Refusal::new(Code::BadValue, "--why needs true or false"))?
        }
        "--to" => listing.to = Some(Destination::parse(&value(args, index)?)),
        "--cursor" => listing.cursor = Some(value(args, index)?),
        "--budget" => listing.budget = Some(value(args, index)?),
        _ => return Ok(false),
    }
    Ok(true)
}
/// Parses without reading stdin or contacting the daemon.
pub(crate) fn parse(
    args: &[String],
    catalog: &FirstWordCatalog,
    terminal: bool,
) -> Result<ParsedLine, Refusal> {
    let declarations = crate::protocol::steps::declarations();
    let mut parsed = ParsedLine {
        first: FirstWord::Nothing,
        selection: None,
        listing: ListingOptions::default(),
        steps: vec![],
        line: LineOptions::default(),
    };
    let mut index = 0;
    while index < args.len()
        && (line_option(args, &mut index, &mut parsed.line)?
            || listing_option(args, &mut index, &mut parsed.listing)?)
    {
        index += 1;
    }
    let selection_index = index;
    let literal_selection = args.get(index).is_some_and(|word| word == "--");
    if literal_selection {
        index += 1;
    }
    parsed.first = first(
        args.get(index).map(String::as_str),
        catalog,
        declarations.verbs(),
    );
    if literal_selection {
        if index == args.len() {
            return Err(Refusal::new(Code::BadRequest, "-- needs a selection"));
        }
        parsed.first = FirstWord::QueryText;
    }
    if parsed.first == FirstWord::Command {
        return Ok(parsed);
    }
    if parsed.first == FirstWord::RowNumbers && !terminal {
        return Err(Refusal::new(
            Code::RowNumbersOffTerminal,
            "row numbers work on a terminal; use the references printed above",
        ));
    }
    match &parsed.first {
        FirstWord::Nothing | FirstWord::Verb => {}
        FirstWord::Stdin => {
            parsed.selection = Some(SelectionWord::Stdin);
            index += 1;
        }
        FirstWord::SavedQuery { name } => {
            parsed.selection = Some(SelectionWord::Text(format!("query:{name}")));
            index += 1;
        }
        _ => {
            parsed.selection = Some(SelectionWord::Text(args[index].clone()));
            index += 1;
        }
    }
    let mut stdin_count = usize::from(parsed.first == FirstWord::Stdin);
    while index < args.len() {
        if line_option(args, &mut index, &mut parsed.line)? {
            index += 1;
            continue;
        }
        if parsed.steps.is_empty() && args[index].starts_with('-') {
            if !listing_option(args, &mut index, &mut parsed.listing)? {
                return Err(Refusal::new(
                    Code::BadRequest,
                    format!("unknown listing option {}", args[index]),
                ));
            }
            index += 1;
            continue;
        }
        let Some(declaration) = declarations
            .verbs()
            .iter()
            .find(|verb| verb.verb() == args[index])
        else {
            let mut corrected = args[..index].to_vec();
            let mut message = format!("{:?} is not a verb", args[index]);
            if parsed.steps.is_empty() {
                let end = (index..args.len())
                    .find(|&i| {
                        declarations
                            .verbs()
                            .iter()
                            .any(|verb| verb.verb() == args[i])
                    })
                    .unwrap_or(args.len());
                corrected = args[..selection_index].to_vec();
                corrected.push(args[selection_index..end].join(" "));
                corrected.extend_from_slice(&args[end..]);
                message.push_str("; a query is one argument");
            } else if let Some(step) = parsed.steps.last() {
                let words = declarations
                    .verbs()
                    .iter()
                    .find(|verb| verb.verb() == step.verb)
                    .unwrap()
                    .arguments()
                    .iter()
                    .filter(|field| field.form == ArgumentForm::Positional)
                    .count();
                message.push_str(&format!(
                    "; {} takes {words} word{}",
                    step.verb,
                    if words == 1 { "" } else { "s" }
                ));
                if matches!(step.verb, "note" | "send" | "edit" | "answer") {
                    corrected = args[..step.words.start + 1].to_vec();
                    corrected.push(args[step.words.start + 1..].join(" "));
                }
            }
            return Err(Refusal::new(Code::NotAVerb, message).at(index, &corrected));
        };
        let start = index;
        index += 1;
        let positionals: Vec<_> = declaration
            .arguments()
            .iter()
            .filter(|field| field.form == ArgumentForm::Positional)
            .collect();
        let mut owed = positionals.len();
        let optional = matches!(declaration.verb(), "save" | "edit");
        let mut words = vec![];
        let mut to = None;
        let mut stdin = None;
        while index < args.len() {
            if line_option(args, &mut index, &mut parsed.line)? {
                index += 1;
                continue;
            }
            if args[index] == "--" && owed > 0 {
                words.push("--".into());
                let forced = value(args, &mut index)?;
                words.push(forced);
                owed -= 1;
                index += 1;
                continue;
            }
            let word = &args[index];
            if word.starts_with('-') && word != "-" {
                if word == "--to" && matches!(declaration.verb(), "read" | "brief") {
                    to = Some(Destination::parse(&value(args, &mut index)?));
                    index += 1;
                    continue;
                }
                let name = word.trim_start_matches('-').replace('-', "_");
                let Some(field) = declaration.arguments().iter().find(|field| {
                    field.name == name && field.form != ArgumentForm::Positional && field.listed
                }) else {
                    return Err(Refusal::new(
                        Code::BadRequest,
                        format!("{}: unknown option {word}", declaration.verb()),
                    ));
                };
                words.push(word.clone());
                if takes_word(field) {
                    let text = value(args, &mut index)?;
                    if field.name == "parts" && text == "-" {
                        stdin_count += 1;
                        stdin = Some(field.name.clone());
                    }
                    words.push(text);
                }
                index += 1;
                continue;
            }
            let verb = declarations.verbs().iter().any(|verb| verb.verb() == word);
            if owed > 0 && !(optional && verb) {
                if word == "-" {
                    stdin_count += 1;
                    stdin = Some(positionals[positionals.len() - owed].name.clone());
                }
                words.push(word.clone());
                owed -= 1;
                index += 1;
            } else {
                break;
            }
        }
        if owed > 0 && !optional {
            let name = positionals[positionals.len() - owed].name.to_uppercase();
            let mut corrected = args[..index].to_vec();
            corrected.push(name);
            return Err(Refusal::new(
                Code::BadRequest,
                format!(
                    "{} needs {}",
                    declaration.verb(),
                    if declaration.verb() == "tag" {
                        "a name"
                    } else {
                        "a word"
                    }
                ),
            )
            .at(index, &corrected));
        }
        let step = declarations
            .parse_step(declaration.verb(), &words)
            .map_err(|failure| Refusal::new(Code::BadRequest, failure.message))?;
        let typed: crate::protocol::line::Step = serde_json::from_value(step.clone())
            .map_err(|error| Refusal::new(Code::BadRequest, error.to_string()))?;
        parsed.steps.push(ParsedStep {
            verb: typed.verb(),
            step,
            to,
            words: start..index,
            stdin,
        });
    }
    if stdin_count > 1 {
        return Err(Refusal::new(
            Code::BadRequest,
            "standard input can be read once per line",
        ));
    }
    if parsed.first == FirstWord::Verb
        && !matches!(
            parsed.steps.first().map(|step| step.verb),
            Some("read" | "resume" | "brief")
        )
    {
        let verb = parsed.steps.first().unwrap().verb;
        let mut corrected = args.to_vec();
        corrected.insert(selection_index, "@1".into());
        let mut refusal = Refusal::new(Code::BadRequest, format!("{verb} needs rows first"))
            .at(selection_index, &corrected);
        corrected[selection_index] = "QUERY".into();
        corrected.insert(selection_index + 1, "pick".into());
        refusal.candidates.push(shell_line(&corrected));
        return Err(refusal);
    }
    for (index, step) in parsed.steps.iter().enumerate() {
        if step.step["follow"] == true && index + 1 != parsed.steps.len() {
            return Err(Refusal::new(
                Code::BadRequest,
                "read --follow must be the last step",
            ));
        }
    }
    Ok(parsed)
}
#[cfg(test)]
mod tests;
