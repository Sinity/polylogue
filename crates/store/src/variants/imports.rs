//! Import history never replaces the rendition's original authorship.
use super::*;
use polylogue_model::{Reference, RevisionReference, authored::ImportAttribution};

fn reference(conn: &Connection, id: &str, n: Option<u64>) -> Result<Reference> {
    let variant =
        rows::get(conn, id)?.ok_or_else(|| StoreError::NotFound(format!("variant {id}")))?;
    Ok(Reference::Variant {
        revision: RevisionReference::exact(variant.root, variant.id),
        part: n,
    })
}
fn identify(conn: &Connection, reference: &Reference) -> Result<Option<String>> {
    let Reference::Variant {
        revision: RevisionReference {
            root,
            revision: Some(id),
        },
        part,
    } = reference
    else {
        return Err(StoreError::Invalid(
            "variant import history needs an exact rendition or part".into(),
        ));
    };
    let Some(variant) = rows::get(conn, id)? else {
        return Ok(None);
    };
    if variant.root != *root {
        return Ok(None);
    }
    if let Some(n) = part
        && rows::get_part(conn, id, *n)?.is_none()
    {
        return Ok(None);
    }
    Ok(Some(id.clone()))
}
fn insert(
    conn: &Connection,
    reference: &Reference,
    variant: &str,
    attribution: &ImportAttribution,
    observed: bool,
    counter: i64,
) -> Result<()> {
    let source = Source::of(conn)?;
    conn.execute(&source.sql("INSERT OR IGNORE INTO @source.variant_import(ref,variant,observed,input,chain,machine,actor,at,commit_seq) VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9)"),params![reference.to_string(),variant,observed,attribution.input,attribution.chain,attribution.machine,attribution.actor.to_string(),attribution.at.to_string(),counter])?;
    Ok(())
}

pub(super) fn restore(
    conn: &Connection,
    effect: &VariantEffect,
    current: &ImportAttribution,
    counter: i64,
) -> Result<()> {
    super::apply(conn, effect, counter)?;
    for variant in &effect.variants {
        let reference = reference(conn, &variant.id, None)?;
        insert(conn, &reference, &variant.id, current, true, counter)?;
    }
    for part in &effect.variant_parts {
        let reference = reference(conn, &part.variant, Some(part.n))?;
        insert(conn, &reference, &part.variant, current, true, counter)?;
    }
    Ok(())
}

/// What a sender states stays declared; only `restore` can record how an import
/// was observed.
pub fn declare_imports(
    conn: &Connection,
    reference: &Reference,
    attributions: &[ImportAttribution],
    counter: i64,
) -> Result<()> {
    let id =
        identify(conn, reference)?.ok_or_else(|| StoreError::NotFound(reference.to_string()))?;
    for attribution in attributions {
        insert(conn, reference, &id, attribution, false, counter)?;
    }
    Ok(())
}

pub(super) fn copy(from: &Connection, to: &Connection) -> Result<()> {
    let source = Source::reading(from)?;
    let mut statement=from.prepare_cached(&source.sql("SELECT ref,variant,observed,input,chain,machine,actor,at FROM @source.variant_import ORDER BY ref,machine,input,chain,actor,at,observed"))?;
    let mut rows = statement.query([])?;
    while let Some(row) = rows.next()? {
        let reference: String = row.get(0)?;
        let variant: String = row.get(1)?;
        let at: String = row.get(7)?;
        let reference = Reference::parse(&reference)
            .map_err(|error| StoreError::stored(0, error.to_string()))?;
        let attribution = ImportAttribution {
            input: row.get(3)?,
            chain: row.get(4)?,
            machine: row.get(5)?,
            actor: crate::encoding::Columns::id(row, 6)?,
            at: Timestamp::parse_rfc3339(&at)
                .ok_or_else(|| StoreError::stored(7, "import time is not an RFC 3339 time"))?,
        };
        insert(to, &reference, &variant, &attribution, row.get(2)?, 0)?;
    }
    Ok(())
}
impl Reader<'_> {}
