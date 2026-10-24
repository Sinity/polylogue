//! Canonical container custody belongs to member observations, independently of the index.
use crate::{Batch, ImportContainer, Result, StoreError};
use polylogue_model::BlobHash;
use rusqlite::{Connection, OptionalExtension, params};

pub(crate) fn require_authority(conn: &Connection) -> Result<()> {
    if crate::schema::version(conn)? >= 10 {
        let present:bool=conn.query_row("SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='observation_container')",[],|r|r.get(0))?;
        if !present {
            return Err(StoreError::Invalid(
                "container custody authority is missing from archive10".into(),
            ));
        }
    }
    Ok(())
}

pub(crate) fn coordinate(batch: &Batch) -> Option<&ImportContainer> {
    batch
        .observation
        .as_ref()?
        .import
        .as_ref()?
        .container
        .as_ref()
}

pub(crate) fn linked(batch: &Batch) -> bool {
    coordinate(batch).is_some()
}

/// A new original must occur before its members in the same staged group.
/// Already retained originals still need their exact immutable byte basis.
pub(crate) fn original(conn: &Connection, input: &str, present: bool) -> Result<()> {
    let hash = BlobHash::parse(input)
        .map_err(|_| StoreError::Invalid("container input must name original bytes".into()))?;
    let valid: bool = conn.query_row("SELECT EXISTS(SELECT 1 FROM input i JOIN revision r ON r.input=i.id
        JOIN extent e ON e.input=r.input AND e.chain=r.chain JOIN blob b ON b.hash=e.blob
        WHERE i.id=?1 AND i.format='export-file' AND r.produced='imported'
          AND r.number=1 AND r.chain=1 AND e.offset=0 AND e.length=r.length
          AND e.blob_offset=0 AND e.blob=?2 AND b.size=r.length
          AND NOT EXISTS(SELECT 1 FROM extent other WHERE other.input=i.id AND other.chain=r.chain AND other.offset<>0)
          AND NOT EXISTS(SELECT 1 FROM revision other WHERE other.input=i.id AND other.number<>1))",
        params![input,&hash.0[..]],|r|r.get(0))?;
    if !valid {
        return Err(StoreError::Invalid(
            "container has no exact retained original byte basis".into(),
        ));
    }
    if present {
        crate::retention::require_present(conn, "", input, Some(1))?;
    }
    Ok(())
}

pub(crate) fn record(
    conn: &Connection,
    observation: &str,
    member: &crate::ImportMember,
) -> Result<()> {
    let Some(container) = &member.container else {
        return Ok(());
    };
    original(conn, &container.input, false)?;
    let own: String = conn.query_row(
        "SELECT input FROM observation WHERE id=?1",
        [observation],
        |r| r.get(0),
    )?;
    if own == container.input {
        return Err(StoreError::Invalid("a ZIP cannot be its own member".into()));
    }
    let acquired: bool = conn.query_row(
        "SELECT EXISTS(SELECT 1 FROM observation WHERE input=?1 AND class='imported'
        AND json_valid(details_json) AND json_extract(details_json,'$.import.acquisition')=?2)",
        params![container.input, member.acquisition],
        |r| r.get(0),
    )?;
    if !acquired {
        return Err(StoreError::Invalid(
            "member has no matching container acquisition".into(),
        ));
    }
    let index = i64::try_from(container.member_index).map_err(|_| {
        StoreError::Invalid("ZIP member index exceeds archive address space".into())
    })?;
    conn.execute(
        "INSERT INTO observation_container VALUES (?1,?2,?3)",
        params![observation, container.input, index],
    )?;
    Ok(())
}

pub(crate) fn refuse_excision(conn: &Connection, input: &str) -> Result<()> {
    let related: Option<String> = conn
        .query_row(
            "SELECT CASE WHEN o.input=?1 THEN c.container_input ELSE o.input END
        FROM observation_container c JOIN observation o ON o.id=c.observation_id
        WHERE (o.input=?1 AND EXISTS(SELECT 1 FROM extent WHERE input=c.container_input))
            OR (c.container_input=?1 AND EXISTS(SELECT 1 FROM extent WHERE input=o.input)) LIMIT 1",
            [input],
            |r| r.get(0),
        )
        .optional()?;
    if let Some(related) = related {
        return Err(StoreError::ExcisionFormat {
            input: input.into(),
            format: "zip".into(),
            detail: format!(
                "original container and member bytes are both retained ({related}); coordinated ZIP removal is unsupported"
            ),
        });
    }
    Ok(())
}

pub(crate) fn verify_all(conn: &Connection) -> Result<()> {
    let mut stmt=conn.prepare("SELECT c.observation_id,c.container_input,c.member_index,o.input,o.details_json FROM observation_container c JOIN observation o ON o.id=c.observation_id ORDER BY c.observation_id")?;
    let mut rows = stmt.query([])?;
    while let Some(row) = rows.next()? {
        let container: String = row.get(1)?;
        let input: String = row.get(3)?;
        original(conn, &container, true)?;
        if input == container {
            return Err(StoreError::Invalid("a ZIP cannot be its own member".into()));
        }
        let hash = BlobHash::parse(&input).map_err(|_| {
            StoreError::Invalid("container member has no immutable input identity".into())
        })?;
        let valid:bool=conn.query_row("SELECT EXISTS(SELECT 1 FROM revision r JOIN extent e ON e.input=r.input AND e.chain=r.chain JOIN blob b ON b.hash=e.blob WHERE r.input=?1 AND r.number=1 AND r.chain=1 AND r.produced='imported' AND e.offset=0 AND e.blob_offset=0 AND e.length=r.length AND b.size=r.length AND e.blob=?2)",params![input,&hash.0[..]],|r|r.get(0))?;
        if !valid {
            return Err(StoreError::Invalid(
                "container member has no exact retained byte basis".into(),
            ));
        }
        crate::retention::require_present(conn, "", &input, Some(1))?;
        let details: String = row.get(4)?;
        let details: serde_json::Value = serde_json::from_str(&details)?;
        let member: crate::ImportMember = serde_json::from_value(details["import"].clone())?;
        let acquired:bool=conn.query_row("SELECT EXISTS(SELECT 1 FROM observation WHERE input=?1 AND class='imported' AND json_valid(details_json) AND json_extract(details_json,'$.import.acquisition')=?2)",params![container,member.acquisition],|r|r.get(0))?;
        if member.version != 1
            || member.acquisition.is_empty()
            || member.member.is_empty()
            || !acquired
        {
            return Err(StoreError::Invalid(
                "container member has no matching acquisition".into(),
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
