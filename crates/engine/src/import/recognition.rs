//! Recognition of a file on its own: each origin's one rule
//! ([`OriginSpec::recognize`]), given the whole document's top-level
//! structure, so field order and a long first value do not matter.
use super::*;
use polylogue_sources::document::{Head, Shape, read_head, rewind_past_byte_order_mark};
use polylogue_sources::origin::Alone;

/// A refusal's structures tried that no single origin declares.
const BUNDLE: &str = "Polylogue bundle";
const LAYOUT: &str = "live tool layout";
const NAMED: &str = "explicit format";

/// Every structure import recognizes, as a refusal lists them.
pub(super) fn structures() -> Vec<String> {
    polylogue_sources::origin::Package::ALL
        .iter()
        .map(|package| package.structure())
        .chain(
            ORIGINS
                .iter()
                .filter_map(|origin| match origin.import.alone {
                    Alone::Structure(name) => Some(name),
                    _ => None,
                }),
        )
        .chain([BUNDLE, LAYOUT, NAMED])
        .map(str::to_owned)
        .collect()
}

pub(super) fn declared(
    path: &Path,
    file: &mut (impl std::io::Read + std::io::Seek),
    named_origin: Option<&str>,
) -> std::io::Result<Option<&'static OriginSpec>> {
    file.rewind()?;
    let bytes = read_head(&mut *file)?;
    rewind_past_byte_order_mark(file)?;
    if polylogue_sources::polylogue_bundle::manifest(&mut *file).is_ok() {
        file.rewind()?;
        return Ok(polylogue_sources::origin::by_name(
            polylogue_sources::polylogue_bundle::ORIGIN,
        ));
    }
    file.rewind()?;
    let shape = Shape::read(std::io::Read::by_ref(file));
    rewind_past_byte_order_mark(file)?;
    let head = Head {
        bytes: &bytes,
        shape: shape.as_ref(),
    };
    if let Some(name) = named_origin {
        return Ok(polylogue_sources::origin::by_name(name).filter(|o| {
            importable(o) && ((o.recognize)(path, &head) || o.arrival == Arrival::Import)
        }));
    }
    // A declared structure wherever it lies; then a tool's file outside its
    // root, by its preserved layout.
    let structure = ORIGINS.iter().find(|o| {
        matches!(o.import.alone, Alone::Structure(_)) && importable(o) && (o.recognize)(path, &head)
    });
    Ok(structure.or_else(|| {
        ORIGINS.iter().find(|o| {
            matches!(o.import.alone, Alone::Layout)
                && importable(o)
                && o.locations.iter().any(|p| collect::ends_in(path, p))
                && (o.recognize)(path, &head)
        })
    }))
}

/// An origin whose files an import stores: a parsed one, or one whose
/// files are kept as bytes until it has a parser, as the live path keeps
/// them. A row-dump origin is a database read row by row, never a file.
fn importable(origin: &OriginSpec) -> bool {
    origin.support == Support::Parsed
        || (origin.support == Support::NoParser
            && matches!(origin.capture, polylogue_sources::Capture::Bytes))
}

pub(super) fn refuse(report: &mut ImportReport, path: &Path, reason: impl Into<String>) {
    report.refused.push(ImportRefusal {
        path: path.to_path_buf(),
        reason: reason.into(),
        structures_tried: structures(),
    });
}
