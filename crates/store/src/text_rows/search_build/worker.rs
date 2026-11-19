//! One independent native database builds both declared search corpora.
use super::TextBatch;
use crate::{ConnExt as _, Result};
use rusqlite::{Connection, TransactionBehavior, params};
use std::sync::mpsc::Receiver;
use std::thread::JoinHandle;

const TABLES: [(i64, &str); 2] = [(1, "text_fts"), (2, "output_fts")];

pub(super) fn start(
    conn: &Connection,
    receiver: Receiver<TextBatch>,
    stop: crate::cancellation::Cancellation,
    cache_mib: i64,
) -> Result<JoinHandle<Result<tempfile::TempPath>>> {
    let index = conn.path().filter(|path| !path.is_empty()).ok_or_else(|| {
        crate::StoreError::Invalid("search construction requires an index file".into())
    })?;
    let file = super::SearchBuild::path(std::path::Path::new(index));
    // There is one archive writer. A previous process's disposable tree is
    // discarded, never resumed or used as evidence of completed model work.
    match std::fs::remove_file(&file) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.into()),
    }
    let path = tempfile::TempPath::try_from_path(file)?;
    let mut declarations = Vec::new();
    for (_, table) in TABLES {
        let sql: String = conn.scalar("SELECT sql FROM sqlite_schema WHERE name=?1", [table])?;
        let config = conn
            .prepare(&format!(
                "SELECT k,v FROM {table}_config WHERE k<>'version' ORDER BY k"
            ))?
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, rusqlite::types::Value>(1)?,
                ))
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        declarations.push((table, sql, config));
    }
    let subject = conn.path().unwrap_or_default().to_owned();
    Ok(polylogue_model::spawn_named(
        "polylogue-build-search",
        move || {
            crate::cancellation::under(&stop, || {
                let _worker = crate::timing::worker();
                let side = Connection::open(&path)?;
                // Disposable derived trees. Only their copied main-index
                // transaction is durable; no temporary checkpoint owns progress.
                side.execute_batch("PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF; PRAGMA mmap_size=0; PRAGMA temp_store=FILE;")?;
                crate::archive::writer_settings(&side, cache_mib)?;
                let interrupt = stop.clone();
                crate::cancellation::interrupt_on(&side, move || interrupt.check())?;
                for (table, sql, config) in declarations {
                    side.execute_batch(&sql)?;
                    for (key, value) in config {
                        side.execute(
                            &format!("INSERT INTO {table}({table},rank) VALUES(?1,?2)"),
                            params![key, value],
                        )?;
                    }
                }
                let tx =
                    rusqlite::Transaction::new_unchecked(&side, TransactionBehavior::Immediate)?;
                let mut highest = 0;
                for mut batch in receiver {
                    crate::timing::time(crate::timing::Stage::SearchIndex, || -> Result<()> {
                        stop.check()?;
                        for group in batch.groups.drain(..) {
                            let decoded = crate::text_rows::groups::decode(&group.body)?;
                            for index in group.members {
                                let text = &mut batch.texts[index];
                                text.body = crate::text_rows::groups::slice_decoded(
                                    &decoded,
                                    text.start,
                                    text.length,
                                )?;
                            }
                        }
                        // Snapshots retain ascending text IDs within each feed.
                        for text in batch.texts {
                            stop.check()?;
                            let (body, _) = crate::text_rows::folded_text(text.body)?;
                            for (bit, table) in TABLES {
                                // A higher ID cannot already have a native entry.
                                // Older IDs may gain or lose either membership.
                                let present = text.id <= highest && tx.prepare_cached(&format!("SELECT EXISTS(SELECT 1 FROM {table}_docsize WHERE id=?1)"))?
                                .query_row([text.id], |row| row.get::<_, bool>(0))?;
                                if text.bits & bit != 0 && !present {
                                    crate::text_rows::insert_entry(&tx, text.id, bit, &body)?;
                                } else if text.bits & bit == 0 && present {
                                    tx.execute(
                                        &format!("DELETE FROM {table} WHERE rowid=?1"),
                                        [text.id],
                                    )?;
                                }
                            }
                            highest = highest.max(text.id);
                        }
                        crate::fault::check("search.build_batch", &subject)?;
                        Ok(())
                    })?;
                }
                crate::timing::time(crate::timing::Stage::SearchIndex, || -> Result<()> {
                    stop.check()?;
                    tx.commit()?;
                    Ok(())
                })?;
                drop(side);
                Ok(path)
            })
        },
    )?)
}
