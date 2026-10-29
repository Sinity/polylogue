use super::*;

pub(super) fn reading(conn: &Connection) -> Result<Source> {
    if conn.query_row(
        "SELECT EXISTS(SELECT 1 FROM temp.sqlite_schema WHERE name='source_variant')",
        [],
        |row| row.get::<_, bool>(0),
    )? {
        Ok(Source::Cache)
    } else {
        Source::of(conn)
    }
}

const COLUMNS: &str = "id,root,base,source,source_hash,kind,from_lang,to_lang,author,via,method,machine,created_at,commit_seq";

fn hash(value: Vec<u8>) -> Result<BlobHash> {
    Ok(BlobHash(value.try_into().map_err(|_| {
        StoreError::Invalid("variant hash must contain 32 bytes".into())
    })?))
}
fn variant(row: &rusqlite::Row<'_>) -> Result<Variant> {
    let kind: String = row.get(5)?;
    let author: String = row.get(8)?;
    Ok(Variant {
        id: row.get(0)?,
        root: row.get(1)?,
        base: row.get(2)?,
        source: row.get(3)?,
        source_hash: hash(row.get(4)?)?,
        kind: VariantKind::from_name(&kind)
            .ok_or_else(|| StoreError::Invalid("unknown variant kind".into()))?,
        from_lang: row.get(6)?,
        to_lang: row.get(7)?,
        author: Author::parse(&author)
            .map_err(|_| StoreError::Invalid("unknown variant author".into()))?,
        via: row.get(9)?,
        method: row.get(10)?,
        machine: row.get(11)?,
        created_at: Timestamp::from_unix_ms(row.get(12)?)
            .ok_or_else(|| StoreError::Invalid("variant time is out of range".into()))?,
        commit_seq: row
            .get::<_, i64>(13)?
            .try_into()
            .map_err(|_| StoreError::Invalid("negative variant counter".into()))?,
    })
}
pub(super) fn get(conn: &Connection, id: &str) -> Result<Option<Variant>> {
    let source = reading(conn)?;
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
    let source = reading(conn)?;
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
pub(super) fn parts(
    conn: &Connection,
    id: &str,
    mut f: impl FnMut(VariantPart) -> Result<()>,
) -> Result<()> {
    let source = reading(conn)?;
    let mut statement=conn.prepare(&source.sql("SELECT variant,n,of,of_hash,relation,body FROM @source.variant_part WHERE variant=?1 ORDER BY n"))?;
    let mut rows = statement.query([id])?;
    while let Some(row) = rows.next()? {
        let relation: String = row.get(4)?;
        f(VariantPart {
            variant: row.get(0)?,
            n: row
                .get::<_, i64>(1)?
                .try_into()
                .map_err(|_| StoreError::Invalid("negative variant part number".into()))?,
            of: row.get(2)?,
            of_hash: hash(row.get(3)?)?,
            relation: VariantRelation::from_name(&relation)
                .ok_or_else(|| StoreError::Invalid("unknown variant relation".into()))?,
            body: crate::encoding::unpack_text(&row.get::<_, Vec<u8>>(5)?)?,
        })?;
    }
    Ok(())
}
pub(super) fn insert(conn: &Connection, v: &Variant, counter: i64) -> Result<()> {
    let source = Source::of(conn)?;
    conn.execute(&source.sql("INSERT INTO @source.variant(id,root,base,source,source_hash,kind,from_lang,to_lang,author,via,method,machine,created_at,commit_seq) VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14)"),params![v.id,v.root,v.base,v.source,&v.source_hash.0[..],v.kind.as_str(),v.from_lang,v.to_lang,v.author.to_string(),v.via,v.method,v.machine,v.created_at.unix_ms(),counter])?;
    Ok(())
}
pub(super) fn insert_part(conn: &Connection, p: &VariantPart) -> Result<()> {
    let source = Source::of(conn)?;
    conn.execute(&source.sql("INSERT INTO @source.variant_part(variant,n,of,of_hash,relation,body) VALUES(?1,?2,?3,?4,?5,?6)"),params![p.variant,i64::try_from(p.n).map_err(|_|StoreError::Invalid("variant part number is out of range".into()))?,p.of,&p.of_hash.0[..],p.relation.as_str(),crate::encoding::pack_text(&p.body)?])?;
    Ok(())
}
