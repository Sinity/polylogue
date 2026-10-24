//! Role-free document output using the same masking and block projection as reads.
use crate::{Format, Rendered, View};
use polylogue_model::Document;
use serde_json::{Value, json};
use std::io::{self, Write};

pub fn document(mut document: Document, view: View, format: Format) -> io::Result<Value> {
    mask(&mut document);
    if format == Format::Json {
        document.blocks.retain(|b| view.keeps_block(false, b));
        return Ok(json!({"document":document}));
    }
    crate::formatted(&[Rendered::Document(&document)], format, view)
}

pub(crate) fn mask(document: &mut Document) {
    let mut masker = polylogue_model::masking::MessageMasker::default();
    for block in &mut document.blocks {
        masker.block(block);
    }
}

fn heading(doc: &Document) -> String {
    crate::mask(doc.title.as_deref().unwrap_or(doc.id.native().as_str()))
}
fn pieces(doc: &Document, view: View) -> impl Iterator<Item = crate::Piece> + '_ {
    doc.blocks
        .iter()
        .filter(move |b| view.keeps_block(false, b))
        .filter_map(crate::piece)
}
pub(crate) fn terminal(doc: &Document, view: View, out: &mut dyn Write) -> io::Result<()> {
    writeln!(
        out,
        "{}\ndocument:{} · {} · authorship {}\n",
        heading(doc),
        doc.id,
        doc.purpose.as_str(),
        doc.authorship.as_str()
    )?;
    for p in pieces(doc, view) {
        writeln!(out, "{}", p.text)?;
    }
    Ok(())
}
pub(crate) fn markdown(doc: &Document, view: View, out: &mut dyn Write) -> io::Result<()> {
    writeln!(
        out,
        "# {}\n\n`document:{}` · {} · authorship {}\n",
        heading(doc),
        doc.id,
        doc.purpose.as_str(),
        doc.authorship.as_str()
    )?;
    for p in pieces(doc, view) {
        writeln!(out, "{}\n", p.text)?;
    }
    Ok(())
}
pub(crate) fn html(doc: &Document, view: View, out: &mut dyn Write) -> io::Result<()> {
    use crate::html::escape;
    write!(
        out,
        "<section class=\"document\"><h1>{}</h1><p><code>document:{}</code> · {} · authorship {}</p>",
        escape(&heading(doc)),
        escape(&doc.id.to_string()),
        doc.purpose.as_str(),
        doc.authorship.as_str()
    )?;
    for p in pieces(doc, view) {
        write!(out, "<pre class=\"{}\">{}</pre>", p.kind, escape(&p.text))?;
    }
    write!(out, "</section>")
}

#[cfg(test)]
mod tests;
