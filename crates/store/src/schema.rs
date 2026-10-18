//! Table declarations. `archive.db` and `embeddings.db` evolve by numbered
//! additive migrations applied when the daemon opens the archive.
//! `index.db` is never migrated: a new version builds a new file and swaps
//! it in.

use crate::Result;
use rusqlite::Connection;

pub const ARCHIVE: &str = include_str!("schema/archive.sql");
pub const INDEX: &str = include_str!("schema/index.sql");
pub const EMBEDDINGS: &str = include_str!("schema/embeddings.sql");

/// Bumped whenever `INDEX` changes; a different stored version means the
/// index is rebuilt into a new file.
pub const INDEX_VERSION: u32 = 1;

/// Migrations for `archive.db`, applied in order after the base schema.
/// Each is additive; an archive at a higher number than this list is refused.
pub const ARCHIVE_MIGRATIONS: &[&str] = &[];

pub fn create_archive(conn: &Connection) -> Result<()> {
    conn.execute_batch(ARCHIVE)?;
    Ok(())
}

pub fn create_index(conn: &Connection) -> Result<()> {
    conn.execute_batch(INDEX)?;
    Ok(())
}

pub fn create_embeddings(conn: &Connection) -> Result<()> {
    conn.execute_batch(EMBEDDINGS)?;
    Ok(())
}

/// Applies pending archive migrations inside one transaction.
pub fn migrate_archive(_conn: &mut Connection) -> Result<()> {
    // Architecture 4.1: numbered additive migrations, refuse newer archives.
    todo!()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_schema_applies_to_an_empty_database() {
        for (name, create) in [
            ("archive", create_archive as fn(&Connection) -> Result<()>),
            ("index", create_index),
            ("embeddings", create_embeddings),
        ] {
            let conn = Connection::open_in_memory().unwrap();
            create(&conn).unwrap_or_else(|e| panic!("{name}: {e}"));
        }
    }
}
