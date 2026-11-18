//! Reference links in authored Markdown. CommonMark owns the text grammar;
//! the reference parser owns which destinations name archive records.
use crate::{Reference, ReferenceError, assertion::LinkRole};
use pulldown_cmark::{Event, Options, Parser, Tag, TagEnd};
use std::collections::HashSet;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextLink {
    pub role: LinkRole,
    pub reference: Reference,
    pub quote: Option<String>,
    pub form: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum TextError {
    #[error("reference {written:?}: {error}")]
    Reference {
        written: String,
        #[source]
        error: ReferenceError,
    },
    #[error("unknown embed form {written:?}")]
    Form { written: String },
    #[error("{what} is not built")]
    NotBuilt { what: &'static str },
}

#[derive(Default)]
struct QuotedText {
    closing: Option<char>,
    text: String,
    complete: bool,
}
impl QuotedText {
    fn push(&mut self, text: &str) {
        if self.complete {
            return;
        }
        for ch in text.chars() {
            match self.closing {
                Some(end) if ch == end => {
                    self.complete = true;
                    break;
                }
                Some(_) => self.text.push(ch),
                None if ch == '"' => self.closing = Some('"'),
                None if ch == '“' => self.closing = Some('”'),
                None => {}
            }
        }
    }
    fn finish(self) -> Option<String> {
        self.complete.then_some(self.text)
    }
}

#[derive(Default)]
struct Footnote {
    quote: QuotedText,
    // A definition may put its quote after its references.
    citations: Vec<usize>,
}

/// References in text order, with each role and reference occurring once.
pub fn links(text: &str) -> Result<Vec<TextLink>, TextError> {
    let mut output: Vec<TextLink> = Vec::new();
    let mut seen = HashSet::new();
    let mut footnotes: Vec<Footnote> = Vec::new();
    let mut images = 0usize;
    let mut code_blocks = 0usize;
    for event in Parser::new_ext(text, Options::ENABLE_FOOTNOTES) {
        match event {
            Event::Start(Tag::CodeBlock(_)) => code_blocks += 1,
            Event::End(TagEnd::CodeBlock) => code_blocks -= 1,
            Event::Start(Tag::FootnoteDefinition(_)) => footnotes.push(Footnote::default()),
            Event::End(TagEnd::FootnoteDefinition) => {
                let note = footnotes.pop().expect("balanced CommonMark footnotes");
                let quote = note.quote.finish();
                for index in note.citations {
                    output[index].quote = quote.clone();
                }
            }
            Event::Start(Tag::Image {
                dest_url, title, ..
            }) => {
                if code_blocks == 0
                    && images == 0
                    && let Some(reference) = destination(&dest_url)?
                {
                    let form = embed_form(&title)?;
                    if matches!(reference, Reference::Query { .. })
                        && matches!(form.as_deref(), Some("value" | "live value"))
                    {
                        return Err(TextError::NotBuilt {
                            what: "values kept in sentences",
                        });
                    }
                    append(
                        &mut output,
                        &mut seen,
                        TextLink {
                            role: LinkRole::Embeds,
                            reference,
                            quote: None,
                            form,
                        },
                    );
                }
                images += 1;
            }
            Event::End(TagEnd::Image) => images -= 1,
            Event::Start(Tag::Link { dest_url, .. }) if code_blocks == 0 && images == 0 => {
                if let Some(reference) = destination(&dest_url)? {
                    let role = if footnotes.is_empty() {
                        LinkRole::Mentions
                    } else {
                        if matches!(reference, Reference::Session(_)) {
                            return Err(TextError::NotBuilt {
                                what: "citing a session by quote",
                            });
                        }
                        LinkRole::Cites
                    };
                    let index = append(
                        &mut output,
                        &mut seen,
                        TextLink {
                            role,
                            reference,
                            quote: None,
                            form: None,
                        },
                    );
                    if let Some(note) = footnotes.last_mut()
                        && let Some(index) = index
                    {
                        note.citations.push(index);
                    }
                }
            }
            Event::Text(text) | Event::Code(text) if code_blocks == 0 && images == 0 => {
                if let Some(note) = footnotes.last_mut() {
                    note.quote.push(&text);
                }
            }
            Event::SoftBreak | Event::HardBreak if code_blocks == 0 && images == 0 => {
                if let Some(note) = footnotes.last_mut() {
                    note.quote.push("\n");
                }
            }
            _ => {}
        }
    }
    Ok(output)
}

fn append(
    output: &mut Vec<TextLink>,
    seen: &mut HashSet<(LinkRole, Reference)>,
    link: TextLink,
) -> Option<usize> {
    if !seen.insert((link.role, link.reference.clone())) {
        return None;
    }
    let index = output.len();
    output.push(link);
    Some(index)
}

fn destination(written: &str) -> Result<Option<Reference>, TextError> {
    let url = written.split_once(':').is_some_and(|(scheme, _)| {
        scheme.eq_ignore_ascii_case("http") || scheme.eq_ignore_ascii_case("https")
    });
    let parsed = if url {
        Reference::parse(&format!("url:{written}"))
    } else {
        Reference::parse(written)
    };
    match parsed {
        Ok(reference) => Ok(Some(reference)),
        Err(ReferenceError::NoKind(_) | ReferenceError::UnknownKind(_)) => Ok(None),
        Err(error) => Err(TextError::Reference {
            written: written.into(),
            error,
        }),
    }
}

fn embed_form(written: &str) -> Result<Option<String>, TextError> {
    let form = written.trim();
    if form.is_empty() {
        return Ok(None);
    }
    if matches!(
        form,
        "card"
            | "excerpt"
            | "text"
            | "full"
            | "table"
            | "list"
            | "count"
            | "value"
            | "chart"
            | "compare"
            | "live value"
    ) {
        Ok(Some(form.into()))
    } else {
        Err(TextError::Form {
            written: written.into(),
        })
    }
}

#[cfg(test)]
mod tests;
