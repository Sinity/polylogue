//! One publication pass's ordered units, current replacements and held positions.
use super::{Published, Replacement, Unit, excision};
use crate::{Result, StoreError};
use rusqlite::{Connection, OptionalExtension, params};

pub struct Inventory<'a> {
    db: Connection,
    _file: tempfile::NamedTempFile,
    // Keep the source read transaction alive throughout this inventory's use.
    _source: &'a Published,
    length: u64,
    held: u64,
    has_excision: bool,
}

impl<'a> Inventory<'a> {
    pub(super) fn new(source: &'a Published) -> Result<Self> {
        let legacy: bool = source.conn.query_row(
            "SELECT EXISTS(SELECT 1 FROM excision WHERE json_type(refused_json,'$.publication') IS NULL)",
            [], |r| r.get(0),
        )?;
        if legacy {
            return Err(StoreError::Invalid("primary publication cannot reconstruct unit identities for legacy excisions without replacement metadata".into()));
        }
        let file = tempfile::NamedTempFile::new()?;
        let db = Connection::open(file.path())?;
        db.execute_batch("PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF; PRAGMA cache_size=-1024;
            PRAGMA mmap_size=0; PRAGMA temp_store=FILE;
            CREATE TABLE unit(seq INTEGER NOT NULL,kind TEXT NOT NULL,key TEXT NOT NULL,
                PRIMARY KEY(seq,kind,key)) WITHOUT ROWID;
            CREATE TABLE replacement(seq INTEGER NOT NULL,kind TEXT NOT NULL,key TEXT NOT NULL,data TEXT NOT NULL,owner_seq INTEGER NOT NULL,owner_key TEXT NOT NULL,
                PRIMARY KEY(seq,kind,key)) WITHOUT ROWID;
            CREATE TABLE position(position INTEGER PRIMARY KEY,seq INTEGER NOT NULL,kind TEXT NOT NULL,key TEXT NOT NULL);
            CREATE TABLE held(position INTEGER PRIMARY KEY);
            BEGIN;")?;
        let mut has_excision = false;
        {
            let mut insert = db.prepare("INSERT OR IGNORE INTO unit VALUES (?1,?2,?3)")?;
            let mut current = db.prepare(
                "INSERT INTO replacement VALUES (?1,?2,?3,?4,?5,?6)
                ON CONFLICT(seq,kind,key) DO UPDATE SET data=excluded.data,
                    owner_seq=excluded.owner_seq,owner_key=excluded.owner_key
                WHERE (excluded.owner_seq,excluded.owner_key)>(replacement.owner_seq,replacement.owner_key)",
            )?;
            // Leave ordering to the on-disk primary key, not an in-memory source sort.
            let mut source_units = source.conn.prepare(
                "SELECT commit_seq,'revision',input||'/'||number FROM revision WHERE commit_seq>0
                UNION ALL SELECT commit_seq,'note',id FROM note WHERE commit_seq>0
                UNION ALL SELECT commit_seq,'excision',id FROM excision WHERE commit_seq>0
                UNION ALL SELECT commit_seq,'setting',id FROM setting WHERE commit_seq>0
                UNION ALL SELECT commit_seq,'delivery',id FROM delivery WHERE commit_seq>0",
            )?;
            let mut rows = source_units.query([])?;
            while let Some(row) = rows.next()? {
                let unit = row_unit(row)?;
                if unit.key.is_empty() {
                    return Err(StoreError::Invalid(
                        "invalid publication unit identity".into(),
                    ));
                }
                insert.execute(params![unit.commit_seq, unit.kind, unit.key])?;
            }
            excision::visit_publications(&source.conn, |owner, publication| {
                if owner.commit_seq < 1 || owner.key.is_empty() {
                    return Err(StoreError::Invalid("invalid excision unit identity".into()));
                }
                has_excision = true;
                for replacement in &publication.replacements {
                    let unit = &replacement.unit;
                    if unit >= owner {
                        return Err(StoreError::Invalid(
                            "replacement names no earlier unit identity".into(),
                        ));
                    }
                    insert.execute(params![unit.commit_seq, unit.kind, unit.key])?;
                    current.execute(params![
                        unit.commit_seq,
                        unit.kind,
                        unit.key,
                        serde_json::to_string(replacement)?,
                        owner.commit_seq,
                        owner.key
                    ])?;
                }
                Ok(())
            })?;
        }
        db.execute_batch("INSERT INTO position SELECT ROW_NUMBER() OVER (ORDER BY seq,kind,key),seq,kind,key FROM unit ORDER BY seq,kind,key;
            DROP TABLE unit; COMMIT;")?;
        let length =
            db.query_row("SELECT COUNT(*) FROM position", [], |r| r.get::<_, i64>(0))? as u64;
        Ok(Self {
            db,
            _file: file,
            _source: source,
            length,
            held: 0,
            has_excision,
        })
    }

    pub fn len(&self) -> u64 {
        self.length
    }
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }
    pub fn held(&self) -> u64 {
        self.held
    }
    pub fn has_excision(&self) -> bool {
        self.has_excision
    }

    pub fn unit(&self, position: u64) -> Result<Unit> {
        Ok(self.db.query_row(
            "SELECT seq,kind,key FROM position WHERE position=?1",
            [position_number(position)?],
            row_unit,
        )?)
    }

    /// Call only for a committed acknowledgment, including a successful Have reply.
    pub fn mark_held(&mut self, positions: &[u64]) -> Result<()> {
        if positions
            .iter()
            .any(|&position| position == 0 || position > self.length)
        {
            return Err(StoreError::Invalid(
                "held position is outside the publication inventory".into(),
            ));
        }
        let tx = self.db.transaction()?;
        let mut changed = 0;
        {
            let mut insert = tx.prepare("INSERT OR IGNORE INTO held VALUES (?1)")?;
            for &position in positions {
                changed += insert.execute([position_number(position)?])?;
            }
        }
        tx.commit()?;
        self.held += changed as u64;
        Ok(())
    }

    pub fn last_missing(&self) -> Result<Option<u64>> {
        let position: Option<i64> = self.db.query_row(
            "SELECT MAX(position) FROM position p
             WHERE NOT EXISTS(SELECT 1 FROM held h WHERE h.position=p.position)",
            [],
            |r| r.get(0),
        )?;
        Ok(position.map(|n| n as u64))
    }

    pub fn next_missing(&self, after: u64) -> Result<Option<(u64, Unit)>> {
        Ok(self
            .db
            .query_row(
                "SELECT position,seq,kind,key FROM position p WHERE position>?1
             AND NOT EXISTS(SELECT 1 FROM held h WHERE h.position=p.position)
             ORDER BY position LIMIT 1",
                [position_number(after)?],
                |r| {
                    Ok((
                        r.get::<_, i64>(0)? as u64,
                        Unit {
                            commit_seq: r.get(1)?,
                            kind: r.get(2)?,
                            key: r.get(3)?,
                        },
                    ))
                },
            )
            .optional()?)
    }

    pub fn next_replacement(&self, after: Option<&Unit>) -> Result<Option<Replacement>> {
        let (seq, kind, key) = after.map_or((0, "", ""), |u| {
            (u.commit_seq, u.kind.as_str(), u.key.as_str())
        });
        let data: Option<String> = self
            .db
            .query_row(
                "SELECT data FROM replacement WHERE (seq,kind,key)>(?1,?2,?3)
             ORDER BY seq,kind,key LIMIT 1",
                params![seq, kind, key],
                |r| r.get(0),
            )
            .optional()?;
        data.map(|data| Ok(serde_json::from_str(&data)?))
            .transpose()
    }
}

fn position_number(position: u64) -> Result<i64> {
    i64::try_from(position).map_err(|_| {
        StoreError::Invalid("publication position exceeds SQLite integer range".into())
    })
}

fn row_unit(row: &rusqlite::Row<'_>) -> rusqlite::Result<Unit> {
    Ok(Unit {
        commit_seq: row.get(0)?,
        kind: row.get(1)?,
        key: row.get(2)?,
    })
}

#[cfg(test)]
mod tests;
