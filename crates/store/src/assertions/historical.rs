//! Historical values share identity, graph and ingress storage with current
//! assertions. They never pass through current native value validation.
use super::*;
use polylogue_model::{
    authored::{ImportAttribution, ImportHistory},
    historical::{AssertionEvidence, HistoricalAssertionEvidence},
};

fn decode(conn: &Connection, row: &rusqlite::Row<'_>) -> Result<AssertionEvidence> {
    if let Some(body) = row.get::<_, Option<Vec<u8>>>(12)? {
        let fact: HistoricalAssertionEvidence = crate::encoding::unpack_json(&body)?;
        if fact.id != row.get::<_, String>(0)? || fact.root != row.get::<_, String>(1)? {
            return Err(invalid(
                "historical assertion identity disagrees with its row",
            ));
        }
        let stored_links = links(conn, &fact.id)?;
        let mut declared_links = fact.links.clone();
        declared_links.sort_by_cached_key(|link| (link.role.as_str(), link.reference.to_string()));
        if stored_links != declared_links {
            return Err(invalid(
                "historical assertion links disagree with their indexed facts",
            ));
        }
        let commit_seq = u64::try_from(row.get::<_, i64>(10)?)
            .map_err(|_| invalid("negative historical assertion counter"))?;
        return Ok(AssertionEvidence::Historical {
            fact,
            commit_seq,
            imports: serde_json::from_str(&row.get::<_, String>(11)?)?,
        });
    }
    let mut assertion = record(row)?;
    assertion.fact.links = links(conn, &assertion.fact.id)?;
    Ok(AssertionEvidence::Current(assertion))
}

pub fn get_evidence(conn: &Connection, id: &str) -> Result<Option<AssertionEvidence>> {
    let source = Source::reading(conn)?;
    let mut statement = conn.prepare_cached(&source.sql(&format!(
        "SELECT {COLUMNS} FROM @source.assertion WHERE id=?1"
    )))?;
    let mut rows = statement.query([id])?;
    rows.next()?.map(|row| decode(conn, row)).transpose()
}

pub fn visit_historical(
    conn: &Connection,
    after: Option<u64>,
    mut visit: impl FnMut(&HistoricalAssertionEvidence, u64, &ImportHistory) -> Result<()>,
) -> Result<()> {
    let after = after
        .map(i64::try_from)
        .transpose()
        .map_err(|_| invalid("historical assertion counter is out of range"))?;
    let source = Source::reading(conn)?;
    let mut statement=conn.prepare_cached(&source.sql(&format!("SELECT {COLUMNS} FROM @source.assertion WHERE historical IS NOT NULL AND (?1 IS NULL OR commit_seq>?1) ORDER BY commit_seq,id")))?;
    let mut rows = statement.query([after])?;
    while let Some(row) = rows.next()? {
        let AssertionEvidence::Historical {
            fact,
            commit_seq,
            imports,
        } = decode(conn, row)?
        else {
            return Err(invalid("historical assertion row has a current payload"));
        };
        visit(&fact, commit_seq, &imports)?;
    }
    Ok(())
}

pub fn apply_historical(
    conn: &Connection,
    fact: &HistoricalAssertionEvidence,
    counter: i64,
) -> Result<()> {
    if fact.id.is_empty() || fact.root.is_empty() || counter < 0 {
        return Err(invalid(
            "historical assertion identity or counter is invalid",
        ));
    }
    let mut keys = std::collections::BTreeSet::new();
    for link in &fact.links {
        if !keys.insert((link.role, &link.reference)) {
            return Err(invalid(
                "historical assertion contains duplicate indexed links",
            ));
        }
    }
    if let Some(known) = get_evidence(conn, &fact.id)? {
        return if matches!(known,AssertionEvidence::Historical {fact:known,..} if known==*fact) {
            Ok(())
        } else {
            Err(StoreError::Conflict(
                exact(&fact.root, &fact.id).to_string(),
            ))
        };
    }
    let source = Source::of(conn)?;
    let kind = fact.kind.name();
    let body = crate::encoding::pack_bytes(&serde_json::to_vec(fact)?)?;
    conn.execute(&source.sql("INSERT INTO @source.assertion(id,root,kind,key,value,author,via,machine,created_at,valid_until,commit_seq,historical) VALUES(?1,?2,?3,?4,?5,?6,NULL,?7,?8,?9,?10,?11)"),params![
        fact.id,fact.root,kind,fact.key,fact.value.as_ref().map(serde_json::to_string).transpose()?,
        fact.recorded_author,fact.machine,
        fact.created_at.map(|at|at.to_string()),fact.valid_until.map(|at|at.to_string()),counter,body,
    ])?;
    for link in &fact.links {
        conn.execute(&source.sql("INSERT INTO @source.assertion_link(assertion,role,ref,seen_hash) VALUES(?1,?2,?3,?4)"),params![
            fact.id,link.role.as_str(),link.reference.to_string(),link.seen_hash.map(|hash|hash.0.to_vec()),
        ])?;
    }
    Ok(())
}

pub fn restore_historical(
    conn: &Connection,
    fact: &HistoricalAssertionEvidence,
    claims: &[ImportAttribution],
    current: &ImportAttribution,
    counter: i64,
) -> Result<()> {
    if matches!(
        fact.source,
        polylogue_model::historical::SourceSchema::RetainedPublication { .. }
    ) {
        return Err(invalid(
            "queued publication is not canonical accepted assertion evidence",
        ));
    }
    apply_historical(conn, fact, counter)?;
    let Some(AssertionEvidence::Historical { mut imports, .. }) = get_evidence(conn, &fact.id)?
    else {
        return Err(invalid("restored historical assertion is absent"));
    };
    imports.merge(claims, current);
    super::imports::save(conn, &fact.id, &imports)
}
