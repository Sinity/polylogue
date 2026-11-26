use super::*;

const COLUMNS: &str = "id,root,base,source,source_hash,kind,from_lang,to_lang,author,via,method,machine,created_at,commit_seq";

fn alternate(row: &rusqlite::Row<'_>) -> Result<Alternate> {
    use crate::encoding::Columns;
    Ok(Alternate {
        id: row.get(0)?,
        root: row.get(1)?,
        base: row.get(2)?,
        source: row.get(3)?,
        source_hash: BlobHash(row.hash(4)?),
        kind: row.name::<AlternateKind>(5)?,
        from_lang: row.get(6)?,
        to_lang: row.get(7)?,
        author: row.id::<Author>(8)?,
        via: row.get(9)?,
        method: row.get(10)?,
        machine: row.get(11)?,
        created_at: Timestamp::parse_rfc3339(&row.get::<_, String>(12)?)
            .ok_or_else(|| StoreError::stored(12, "alternate time is not an RFC 3339 time"))?,
        commit_seq: row.get(13)?,
    })
}
pub(crate) fn get(conn: &Connection, id: &str) -> Result<Option<Alternate>> {
    let source = Source::reading(conn)?;
    let mut statement = conn.prepare(&source.sql(&format!(
        "SELECT {COLUMNS} FROM @source.alternate WHERE id=?1"
    )))?;
    let mut rows = statement.query([id])?;
    rows.next()?.map(alternate).transpose()
}
pub(super) fn visit(
    conn: &Connection,
    filter: &str,
    value: Option<i64>,
    mut f: impl FnMut(Alternate) -> Result<()>,
) -> Result<()> {
    let source = Source::reading(conn)?;
    let mut statement = conn.prepare(&source.sql(&format!(
        "SELECT {COLUMNS} FROM @source.alternate {filter} ORDER BY id"
    )))?;
    let mut rows = if let Some(value) = value {
        statement.query([value])?
    } else {
        statement.query([])?
    };
    while let Some(row) = rows.next()? {
        f(alternate(row)?)?;
    }
    Ok(())
}
fn part(row: &rusqlite::Row<'_>) -> Result<AlternatePart> {
    use crate::encoding::Columns;
    Ok(AlternatePart {
        alternate: row.get(0)?,
        n: row.get(1)?,
        of: row.get(2)?,
        of_hash: BlobHash(row.hash(3)?),
        relation: row.name::<AlternateRelation>(4)?,
        body: crate::encoding::unpack_text(&row.get::<_, Vec<u8>>(5)?)?,
    })
}
pub(super) fn parts(
    conn: &Connection,
    id: &str,
    mut visit: impl FnMut(AlternatePart) -> Result<()>,
) -> Result<()> {
    parts_while(conn, id, |part| {
        visit(part)?;
        Ok(true)
    })
}
pub(crate) fn parts_while(
    conn: &Connection,
    id: &str,
    mut visit: impl FnMut(AlternatePart) -> Result<bool>,
) -> Result<()> {
    let source = Source::reading(conn)?;
    let mut statement=conn.prepare_cached(&source.sql("SELECT alternate,position,source,source_hash,relation,body FROM @source.alternate_part WHERE alternate=?1 ORDER BY position"))?;
    let mut rows = statement.query([id])?;
    while let Some(row) = rows.next()? {
        if !visit(part(row)?)? {
            break;
        }
    }
    Ok(())
}

pub(super) fn get_part(conn: &Connection, id: &str, n: u64) -> Result<Option<AlternatePart>> {
    let source = Source::reading(conn)?;
    let mut statement=conn.prepare_cached(&source.sql("SELECT alternate,position,source,source_hash,relation,body FROM @source.alternate_part WHERE alternate=?1 AND position=?2"))?;
    let mut rows = statement.query(params![
        id,
        crate::to_i64(n, "alternate part number is out of range")?
    ])?;
    rows.next()?.map(part).transpose()
}

pub(super) fn insert(conn: &Connection, v: &Alternate, counter: i64) -> Result<()> {
    if let Some(existing) = get(conn, &v.id)? {
        let mut expected = v.clone();
        expected.commit_seq = existing.commit_seq;
        return if existing == expected {
            Ok(())
        } else {
            Err(StoreError::Conflict(super::reference_string(
                &v.root, &v.id,
            )))
        };
    }
    let source = Source::of(conn)?;
    conn.execute(&source.sql("INSERT INTO @source.alternate(id,root,base,source,source_hash,kind,from_lang,to_lang,author,via,method,machine,created_at,commit_seq) VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14)"),params![v.id,v.root,v.base,v.source,&v.source_hash.0[..],v.kind.as_str(),v.from_lang,v.to_lang,v.author.to_string(),v.via,v.method,v.machine,v.created_at.to_string(),counter])?;
    Ok(())
}
pub(super) fn insert_part(conn: &Connection, p: &AlternatePart) -> Result<()> {
    if let Some(existing) = get_part(conn, &p.alternate, p.n)? {
        return if existing == *p {
            Ok(())
        } else {
            Err(StoreError::Conflict(format!(
                "alternate:{}#{}",
                p.alternate, p.n
            )))
        };
    }
    let source = Source::of(conn)?;
    conn.execute(&source.sql("INSERT INTO @source.alternate_part(alternate,position,source,source_hash,relation,body) VALUES(?1,?2,?3,?4,?5,?6)"),params![p.alternate,crate::to_i64(p.n, "alternate part number is out of range")?,p.of,&p.of_hash.0[..],p.relation.as_str(),crate::encoding::pack_text(&p.body)?])?;
    Ok(())
}
