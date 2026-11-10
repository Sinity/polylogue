use super::*;

const COLUMNS: &str = "id,root,base,source,source_hash,kind,from_lang,to_lang,author,via,method,machine,created_at,commit_seq";

fn variant(row: &rusqlite::Row<'_>) -> Result<Variant> {
    use crate::encoding::Columns;
    Ok(Variant {
        id: row.get(0)?,
        root: row.get(1)?,
        base: row.get(2)?,
        source: row.get(3)?,
        source_hash: BlobHash(row.hash(4)?),
        kind: row.name::<VariantKind>(5)?,
        from_lang: row.get(6)?,
        to_lang: row.get(7)?,
        author: row.id::<Author>(8)?,
        via: row.get(9)?,
        method: row.get(10)?,
        machine: row.get(11)?,
        created_at: Timestamp::parse_rfc3339(&row.get::<_, String>(12)?)
            .ok_or_else(|| StoreError::stored(12, "variant time is not an RFC 3339 time"))?,
        commit_seq: row.get(13)?,
    })
}
pub(crate) fn get(conn: &Connection, id: &str) -> Result<Option<Variant>> {
    let source = Source::reading(conn)?;
    let mut statement = conn.prepare(&source.sql(&format!(
        "SELECT {COLUMNS} FROM @source.variant WHERE id=?1"
    )))?;
    let mut rows = statement.query([id])?;
    rows.next()?.map(variant).transpose()
}
pub(super) fn visit(
    conn: &Connection,
    filter: &str,
    value: Option<i64>,
    mut f: impl FnMut(Variant) -> Result<()>,
) -> Result<()> {
    let source = Source::reading(conn)?;
    let mut statement = conn.prepare(&source.sql(&format!(
        "SELECT {COLUMNS} FROM @source.variant {filter} ORDER BY id"
    )))?;
    let mut rows = if let Some(value) = value {
        statement.query([value])?
    } else {
        statement.query([])?
    };
    while let Some(row) = rows.next()? {
        f(variant(row)?)?;
    }
    Ok(())
}
fn part(row: &rusqlite::Row<'_>) -> Result<VariantPart> {
    use crate::encoding::Columns;
    Ok(VariantPart {
        variant: row.get(0)?,
        n: row.get(1)?,
        of: row.get(2)?,
        of_hash: BlobHash(row.hash(3)?),
        relation: row.name::<VariantRelation>(4)?,
        body: crate::encoding::unpack_text(&row.get::<_, Vec<u8>>(5)?)?,
    })
}
pub(super) fn parts(
    conn: &Connection,
    id: &str,
    mut visit: impl FnMut(VariantPart) -> Result<()>,
) -> Result<()> {
    parts_while(conn, id, |part| {
        visit(part)?;
        Ok(true)
    })
}
pub(crate) fn parts_while(
    conn: &Connection,
    id: &str,
    mut visit: impl FnMut(VariantPart) -> Result<bool>,
) -> Result<()> {
    let source = Source::reading(conn)?;
    let mut statement=conn.prepare_cached(&source.sql("SELECT variant,n,of,of_hash,relation,body FROM @source.variant_part WHERE variant=?1 ORDER BY n"))?;
    let mut rows = statement.query([id])?;
    while let Some(row) = rows.next()? {
        if !visit(part(row)?)? {
            break;
        }
    }
    Ok(())
}
pub(super) fn parts_for(
    conn: &Connection,
    id: &str,
    source_reference: &str,
    mut visit: impl FnMut(VariantPart) -> Result<()>,
) -> Result<()> {
    parts_for_while(conn, id, source_reference, |part| {
        visit(part)?;
        Ok(true)
    })
}
pub(super) fn parts_for_while(
    conn: &Connection,
    id: &str,
    source_reference: &str,
    mut visit: impl FnMut(VariantPart) -> Result<bool>,
) -> Result<()> {
    let source = Source::reading(conn)?;
    let mut statement = conn.prepare_cached(&source.sql(
        "SELECT variant,n,of,of_hash,relation,body FROM @source.variant_part WHERE variant=?1 AND of=?2 ORDER BY n",
    ))?;
    let mut rows = statement.query(params![id, source_reference])?;
    while let Some(row) = rows.next()? {
        if !visit(part(row)?)? {
            break;
        }
    }
    Ok(())
}
/// Indexed record-local targets, grouped by exact original address then part n.
pub(super) fn parts_in(
    conn: &Connection,
    id: &str,
    record: &polylogue_model::Reference,
    mut visit: impl FnMut(VariantPart) -> Result<()>,
) -> Result<()> {
    let (root, intervals) = record_intervals(record)?;
    let source = Source::reading(conn)?;
    let mut statement=conn.prepare_cached(&source.sql("SELECT variant,n,of,of_hash,relation,body FROM @source.variant_part WHERE variant=?1 AND of=?2 UNION ALL SELECT variant,n,of,of_hash,relation,body FROM @source.variant_part WHERE variant=?1 AND of>=?3 AND of<?4 UNION ALL SELECT variant,n,of,of_hash,relation,body FROM @source.variant_part WHERE variant=?1 AND of>=?5 AND of<?6 UNION ALL SELECT variant,n,of,of_hash,relation,body FROM @source.variant_part WHERE variant=?1 AND of>=?7 AND of<?8 ORDER BY of,n"))?;
    let mut rows = statement.query(params![
        id,
        root,
        intervals[0].0,
        intervals[0].1,
        intervals[1].0,
        intervals[1].1,
        intervals[2].0,
        intervals[2].1
    ])?;
    while let Some(row) = rows.next()? {
        let original = polylogue_model::Reference::parse(&row.get::<_, String>(2)?)
            .map_err(|error| StoreError::Invalid(error.to_string()))?;
        if within(&original, record) {
            visit(part(row)?)?;
        }
    }
    Ok(())
}

fn within(original: &polylogue_model::Reference, record: &polylogue_model::Reference) -> bool {
    let mut current = Some(original.clone());
    while let Some(reference) = current {
        if &reference == record {
            return true;
        }
        current = reference.container();
    }
    false
}
fn record_intervals(
    record: &polylogue_model::Reference,
) -> Result<(String, Vec<(String, String)>)> {
    use polylogue_model::Reference;
    let root = record.to_string();
    let mut prefixes = vec![format!("span:{root}#")];
    match record {
        Reference::Message(message) => {
            prefixes.push(format!("block:{message}#"));
            prefixes.push(format!("span:block:{message}#"));
        }
        Reference::Document(_) => {}
        _ => {
            return Err(StoreError::Invalid(
                "rendition record scope must be a message or document".into(),
            ));
        }
    }
    let mut intervals = Vec::new();
    for prefix in prefixes {
        let mut end = prefix.clone();
        end.pop();
        end.push('$');
        intervals.push((prefix, end));
    }
    while intervals.len() < 3 {
        intervals.push((String::new(), String::new()));
    }
    Ok((root, intervals))
}
pub(super) fn sources_in(
    conn: &Connection,
    record: &polylogue_model::Reference,
    mut visit: impl FnMut(polylogue_model::Reference) -> Result<()>,
) -> Result<()> {
    let (root, intervals) = record_intervals(record)?;
    let source = Source::reading(conn)?;
    let mut statement=conn.prepare_cached(&source.sql("SELECT source FROM @source.variant WHERE source=?1 UNION SELECT source FROM @source.variant WHERE source>=?2 AND source<?3 UNION SELECT source FROM @source.variant WHERE source>=?4 AND source<?5 UNION SELECT source FROM @source.variant WHERE source>=?6 AND source<?7 ORDER BY source"))?;
    let mut rows = statement.query(params![
        root,
        intervals[0].0,
        intervals[0].1,
        intervals[1].0,
        intervals[1].1,
        intervals[2].0,
        intervals[2].1
    ])?;
    while let Some(row) = rows.next()? {
        let reference = polylogue_model::Reference::parse(&row.get::<_, String>(0)?)
            .map_err(|error| StoreError::Invalid(error.to_string()))?;
        if within(&reference, record) {
            visit(reference)?;
        }
    }
    Ok(())
}

pub(super) fn get_part(conn: &Connection, id: &str, n: u64) -> Result<Option<VariantPart>> {
    let source = Source::reading(conn)?;
    let mut statement=conn.prepare_cached(&source.sql("SELECT variant,n,of,of_hash,relation,body FROM @source.variant_part WHERE variant=?1 AND n=?2"))?;
    let mut rows = statement.query(params![
        id,
        i64::try_from(n)
            .map_err(|_| StoreError::Invalid("variant part number is out of range".into()))?
    ])?;
    rows.next()?.map(part).transpose()
}

pub(super) fn insert(conn: &Connection, v: &Variant, counter: i64) -> Result<()> {
    if let Some(existing) = get(conn, &v.id)? {
        let mut expected = v.clone();
        expected.commit_seq = existing.commit_seq;
        return if existing == expected {
            Ok(())
        } else {
            Err(StoreError::Conflict(format!("variant:{}@{}", v.root, v.id)))
        };
    }
    let source = Source::of(conn)?;
    conn.execute(&source.sql("INSERT INTO @source.variant(id,root,base,source,source_hash,kind,from_lang,to_lang,author,via,method,machine,created_at,commit_seq) VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14)"),params![v.id,v.root,v.base,v.source,&v.source_hash.0[..],v.kind.as_str(),v.from_lang,v.to_lang,v.author.to_string(),v.via,v.method,v.machine,v.created_at.to_string(),counter])?;
    Ok(())
}
pub(super) fn insert_part(conn: &Connection, p: &VariantPart) -> Result<()> {
    if let Some(existing) = get_part(conn, &p.variant, p.n)? {
        return if existing == *p {
            Ok(())
        } else {
            Err(StoreError::Conflict(format!(
                "variant:{}#{}",
                p.variant, p.n
            )))
        };
    }
    let source = Source::of(conn)?;
    conn.execute(&source.sql("INSERT INTO @source.variant_part(variant,n,of,of_hash,relation,body) VALUES(?1,?2,?3,?4,?5,?6)"),params![p.variant,i64::try_from(p.n).map_err(|_|StoreError::Invalid("variant part number is out of range".into()))?,p.of,&p.of_hash.0[..],p.relation.as_str(),crate::encoding::pack_text(&p.body)?])?;
    Ok(())
}
