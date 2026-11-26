//! One selection followed by declared verbs. Shell arguments retain their boundaries.
pub(crate) use command_surface::line::FirstWord;
use command_surface::{
    ArgumentForm,
    line::{SelectionRules, SplitError},
};
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
pub(crate) trait FirstWordRule {
    fn rule(&self) -> u8;
}
impl FirstWordRule for FirstWord {
    fn rule(&self) -> u8 {
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
            Self::Additional(never) => match *never {},
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
    /// A read that follows live events, which only the last step may do.
    pub follow: bool,
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
}
impl Destination {
    fn parse(text: &str) -> Result<Self, Refusal> {
        if let Some(unbuilt) = crate::unbuilt::listed("polylogue", &format!("--to {text}")) {
            return Err(Refusal::new(Code::NotBuilt, unbuilt.failure().message));
        }
        Ok(match text {
            "stdout" => Self::Stdout,
            "pager" => Self::Pager,
            _ if text.ends_with('/') => Self::Dir(text.into()),
            _ => Self::File(text.into()),
        })
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
/// The saved query names a first word can be: the authored ones the daemon
/// confirmed for this line, then the shipped ones.
#[derive(Default)]
pub(crate) struct SavedNames {
    pub authored: Vec<String>,
}
pub(crate) fn shell_line(words: &[String]) -> String {
    std::iter::once("polylogue".to_owned())
        .chain(words.iter().map(|word| polylogue_render::shell_quote(word)))
        .collect::<Vec<_>>()
        .join(" ")
}
impl SelectionRules for SavedNames {
    type Additional = std::convert::Infallible;
    fn is_command(&self, word: &str) -> bool {
        matches!(word, "--help" | "-h") || crate::commands::is_cli_command(word)
    }
    fn saved_name(&self, word: &str) -> Option<String> {
        self.authored
            .iter()
            .map(String::as_str)
            .chain(
                polylogue_model::saved_queries::SHIPPED_QUERIES
                    .iter()
                    .map(|query| query.name),
            )
            .any(|name| name == word)
            .then(|| word.into())
    }
    fn is_unit(&self, word: &str) -> bool {
        polylogue_query::ast::Unit::all().any(|unit| unit.name() == word)
    }
    fn is_row_numbers(&self, word: &str) -> bool {
        crate::row_numbers::is_selection(word)
    }
    fn is_reference(&self, word: &str) -> bool {
        polylogue_model::Reference::parse(word).is_ok()
            || word.starts_with("https://")
            || word.starts_with("http://")
            || polylogue_query::catalog::REFERENCE_WORDS.contains(&word)
            || polylogue_query::defaults::id_shaped(word)
            || word
                .strip_prefix('@')
                .is_some_and(|name| name.starts_with(char::is_alphabetic))
    }
    fn additional(&self, _: &str) -> Option<Self::Additional> {
        None
    }
    fn is_query(&self, word: &str) -> bool {
        word.chars()
            .any(|c| c.is_whitespace() || " :|\"()~*,".contains(c))
    }
}
/// An option word and its value when written `--name=value`.
fn split_option(word: &str) -> (&str, Option<&str>) {
    word.split_once('=')
        .map_or((word, None), |(name, value)| (name, Some(value)))
}
/// A flag's value: true when bare, else the text after `=` as true or false.
fn flag_value(word: &str, assigned: Option<&str>) -> Result<bool, Refusal> {
    assigned.map_or(Ok(true), |text| {
        text.parse()
            .map_err(|_| Refusal::new(Code::BadValue, format!("{word} needs true or false")))
    })
}
/// The value of an option: the text after `=`, else the next word.
fn value(args: &[String], index: &mut usize, assigned: Option<&str>) -> Result<String, Refusal> {
    if let Some(assigned) = assigned {
        return Ok(assigned.to_owned());
    }
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
    let (word, assigned) = split_option(&args[*index]);
    let flag = match word {
        "--json" => Some(&mut line.json),
        "--fresh" => Some(&mut line.fresh),
        "--dry-run" => Some(&mut line.dry_run),
        "--explain" => Some(&mut line.explain),
        "--all" => Some(&mut line.all),
        _ => None,
    };
    if let Some(flag) = flag {
        *flag = flag_value(word, assigned)?;
        return Ok(true);
    }
    match word {
        "--deadline" => line.deadline = Some(value(args, index, assigned)?),
        "--request-id" => line.request_id = Some(value(args, index, assigned)?),
        "--wait" => line.wait = Some(value(args, index, assigned)?),
        _ => return Ok(false),
    }
    Ok(true)
}
fn listing_option(
    args: &[String],
    index: &mut usize,
    listing: &mut ListingOptions,
) -> Result<bool, Refusal> {
    let (word, assigned) = split_option(&args[*index]);
    match word {
        "--page-size" => {
            listing.page_size = Some(value(args, index, assigned)?.parse().map_err(|_| {
                Refusal::new(Code::BadValue, "--page-size needs a positive integer")
            })?)
        }
        "--fields" => {
            listing.fields = value(args, index, assigned)?
                .split(',')
                .map(str::to_owned)
                .collect()
        }
        "--why" => listing.why = flag_value(word, assigned)?,
        "--to" => listing.to = Some(Destination::parse(&value(args, index, assigned)?)?),
        "--cursor" => listing.cursor = Some(value(args, index, assigned)?),
        "--budget" => listing.budget = Some(value(args, index, assigned)?),
        _ => return Ok(false),
    }
    Ok(true)
}
/// Skips the line and listing options before the selection.
fn leading_options(
    args: &[String],
    index: &mut usize,
    line: &mut LineOptions,
    listing: &mut ListingOptions,
) -> Result<(), Refusal> {
    while *index < args.len()
        && (line_option(args, index, line)? || listing_option(args, index, listing)?)
    {
        *index += 1;
    }
    Ok(())
}
/// Whether the words name a command, read as the line parser reads its
/// first word but without parsing the rest of the line.
pub(crate) fn names_command(args: &[String]) -> bool {
    let mut index = 0;
    if leading_options(
        args,
        &mut index,
        &mut LineOptions::default(),
        &mut ListingOptions::default(),
    )
    .is_err()
        || args.get(index).is_some_and(|word| word == "--")
    {
        return false;
    }
    command_surface::line::classify_first(
        args.get(index).map(String::as_str),
        &SavedNames::default(),
        &crate::protocol::steps::declarations(),
    ) == FirstWord::Command
}
impl ParsedLine {
    /// A first word read as a search word that is shaped like a saved query
    /// name: only the daemon knows whether one is saved under it.
    pub(crate) fn possible_saved_name(&self) -> Option<&str> {
        match (&self.first, &self.selection) {
            (FirstWord::SearchWord, Some(SelectionWord::Text(word)))
                if polylogue_model::annotation::policy::valid_saved_name(word) =>
            {
                Some(word)
            }
            _ => None,
        }
    }
}
/// Parses without reading stdin or contacting the daemon.
pub(crate) fn parse(
    args: &[String],
    saved: &SavedNames,
    terminal: bool,
) -> Result<ParsedLine, Refusal> {
    let declarations = crate::daemon::verbs::declarations();
    let mut parsed = ParsedLine {
        first: FirstWord::Nothing,
        selection: None,
        listing: ListingOptions::default(),
        steps: vec![],
        line: LineOptions::default(),
    };
    let mut index = 0;
    leading_options(args, &mut index, &mut parsed.line, &mut parsed.listing)?;
    let selection_index = index;
    let literal_selection = args.get(index).is_some_and(|word| word == "--");
    if literal_selection {
        index += 1;
    }
    parsed.first = if literal_selection {
        if index == args.len() {
            return Err(Refusal::new(Code::BadRequest, "-- needs a selection"));
        }
        FirstWord::QueryText
    } else {
        command_surface::line::classify_first(
            args.get(index).map(String::as_str),
            saved,
            &declarations,
        )
    };
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
                // The `--` marker stays its own word, so the suggested line
                // keeps taking the next word literally.
                let query_start = selection_index + usize::from(literal_selection);
                corrected = args[..query_start].to_vec();
                corrected.push(args[query_start..end].join(" "));
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
                let count = match words {
                    0 => "no words".to_owned(),
                    1 => "one word".to_owned(),
                    n => format!("{n} words"),
                };
                message.push_str(&format!("; {} takes {count}", step.verb));
                if matches!(step.verb, "note" | "send" | "edit" | "answer") {
                    corrected = args[..step.words.start + 1].to_vec();
                    corrected.push(args[step.words.start + 1..].join(" "));
                }
            }
            return Err(Refusal::new(Code::NotAVerb, message).at(index, &corrected));
        };
        let start = index;
        let mut to = None;
        let split = declarations
            .split_step(args, start, |verb, args, index| {
                if line_option(args, index, &mut parsed.line)? {
                    return Ok(true);
                }
                let (word, assigned) = split_option(&args[*index]);
                if word == "--to" && matches!(verb, "read" | "brief") {
                    to = Some(Destination::parse(&value(args, index, assigned)?)?);
                    return Ok(true);
                }
                Ok(false)
            })
            .map_err(|error| match error {
                SplitError::Local(error) => error,
                SplitError::UnknownOption { index } => {
                    match crate::unbuilt::option(declaration.verb(), &args[index]) {
                        Some(unbuilt) => Refusal::new(Code::NotBuilt, unbuilt.message),
                        None => Refusal::new(
                            Code::BadRequest,
                            format!("{}: unknown option {}", declaration.verb(), args[index]),
                        ),
                    }
                }
                SplitError::MissingValue { .. } => {
                    Refusal::new(Code::BadRequest, "option needs a value")
                }
                SplitError::MissingArgument { index, name } => {
                    let mut corrected = args[..index].to_vec();
                    corrected.push(name.to_uppercase());
                    Refusal::new(
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
                    .at(index, &corrected)
                }
                SplitError::Arguments(failure) => Refusal::new(Code::BadRequest, failure.message),
                SplitError::UnknownVerb { .. } => unreachable!("the selected verb is declared"),
            })?;
        index = split.end;
        let mut stdin = None;
        for word in &split.argument_words {
            if !word.literal
                && args[word.index] == "-"
                && (word.positional || word.field == "parts")
            {
                stdin_count += 1;
                stdin = Some(word.field.clone());
            }
        }
        let step = split.step;
        let typed: crate::protocol::line::Step = serde_json::from_value(step.clone())
            .map_err(|error| Refusal::new(Code::BadRequest, error.to_string()))?;
        let follow = matches!(&typed, crate::protocol::line::Step::Read(read) if read.follow);
        parsed.steps.push(ParsedStep {
            verb: typed.verb(),
            follow,
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
        if step.follow && index + 1 != parsed.steps.len() {
            return Err(Refusal::new(
                Code::BadRequest,
                "read --follow must be the last step",
            ));
        }
    }
    Ok(parsed)
}
#[cfg(test)]
pub(crate) mod tests;
