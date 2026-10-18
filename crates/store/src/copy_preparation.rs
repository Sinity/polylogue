//! Prepares a stopped local copy of an archive for use (local restore only).
//! Work in progress is stored as JSON in `meta.copy_preparation` and blocks
//! ordinary writers until the same request finishes; a `client_request` row
//! records completion.
use crate::{Error, Result, client_request, counter, db, open, recovery, wal};
use polylogue_model::{BlobHash, authored::uuid_v7};
use rusqlite::{Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use std::{
    fs::{self, File},
    io::Read,
    path::{Path, PathBuf},
    sync::atomic::AtomicBool,
    time::{SystemTime, UNIX_EPOCH},
};

fn refused(reason: impl Into<String>) -> Error {
    Error::CopyPreparationRefused(reason.into())
}
/// A new random UUIDv7 for `meta.acquisition_lineage`.
pub(crate) fn fresh_lineage() -> Result<String> {
    let mut bytes = [0u8; 16];
    File::open("/dev/urandom")?.read_exact(&mut bytes)?;
    let ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| Error::Range)?
        .as_millis();
    if ms >= 1u128 << 48 {
        return Err(Error::Range);
    }
    bytes[..6].copy_from_slice(&(ms as u64).to_be_bytes()[2..]);
    bytes[6] = (bytes[6] & 15) | 0x70;
    bytes[8] = (bytes[8] & 63) | 0x80;
    let hex = bytes.iter().map(|b| format!("{b:02x}")).collect::<String>();
    Ok(format!(
        "{}-{}-{}-{}-{}",
        &hex[..8],
        &hex[8..12],
        &hex[12..16],
        &hex[16..20],
        &hex[20..]
    ))
}
fn meta(conn: &Connection, path: &Path, key: &str) -> Result<Option<String>> {
    db(path, || {
        conn.query_row(
            "SELECT CASE WHEN length(CAST(value AS BLOB)) <= 4096 THEN value END \
             FROM meta WHERE key = ?1",
            [key],
            |r| r.get::<_, Option<String>>(0),
        )
        .optional()
    })?
    .map(|s| s.ok_or_else(|| refused(format!("{key} is too large"))))
    .transpose()
}
pub(crate) fn lineage(conn: &Connection, path: &Path) -> Result<Option<String>> {
    let value = meta(conn, path, "acquisition_lineage")?;
    if value.as_deref().is_some_and(|v| !uuid_v7(v)) {
        return Err(refused("invalid acquisition lineage"));
    }
    Ok(value)
}
/// An unfinished preparation, stored in `meta.copy_preparation`. The field
/// names are on disk in existing archives.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Pending {
    pub version: u32,
    pub request_id: String,
    pub intent_hash: BlobHash,
    pub mode: String,
    pub backing: String,
    pub retired_writers: bool,
    pub source_lineage: String,
    pub target_lineage: String,
    pub copy_cut: u64,
    pub batch_cursor: u64,
}
pub(crate) fn pending(conn: &Connection, path: &Path) -> Result<Option<Pending>> {
    meta(conn, path, "copy_preparation")?
        .map(|v| {
            let p: Pending =
                serde_json::from_str(&v).map_err(|_| refused("invalid copy_preparation"))?;
            let hash = intent_hash()?;
            if p.version != 1
                || !uuid_v7(&p.request_id)
                || p.intent_hash != hash
                || p.mode != "restore"
                || p.backing != "local"
                || !p.retired_writers
                || !uuid_v7(&p.source_lineage)
                || p.target_lineage != p.source_lineage
                || p.batch_cursor != 0
                || p.copy_cut > i64::MAX as u64 - 2
            {
                return Err(refused("unsupported copy_preparation value"));
            }
            Ok(p)
        })
        .transpose()
}
fn intent_hash() -> Result<BlobHash> {
    client_request::digest(
        &serde_json::json!({"command":"prepare-copy","mode":"restore","backing":"local","retired_writers":true}),
    )
}
/// Refuses a format migration while a preparation is pending: it must finish
/// with the binary that started it. A missing lineage does not block migration.
pub(crate) fn migration_guard(root: &Path) -> Result<()> {
    let path = root.join("archive.db");
    let conn = open(&path, true, true)?;
    if let Some(active) = pending(&conn, &path)? {
        return Err(refused(format!(
            "copy preparation {} is pending; finish it with the binary that started it before upgrading",
            active.request_id
        )));
    }
    Ok(())
}
pub(crate) fn writer_guard(root: &Path) -> Result<()> {
    let path = root.join("archive.db");
    if !path.exists() || path.metadata()?.len() == 0 {
        return Ok(());
    }
    let conn = open(&path, true, true)?;
    if let Some(active) = pending(&conn, &path)? {
        return Err(refused(format!(
            "copy preparation {} is pending; rerun the same preparation before writing",
            active.request_id
        )));
    }
    if lineage(&conn, &path)?.is_none() {
        return Err(refused(
            "archive has no acquisition lineage; it can be read but not written",
        ));
    }
    Ok(())
}
/// Counts present file blobs whose file is missing. Contents are not checked.
pub(crate) fn unavailable_blobs(conn: &Connection, path: &Path, root: &Path) -> Result<u64> {
    let mut statement = db(path, || {
        conn.prepare(
            "SELECT hash FROM blob WHERE place = 'file' AND state = 'present' ORDER BY hash",
        )
    })?;
    let mut rows = db(path, || statement.query([]))?;
    let mut missing = 0;
    while let Some(row) = db(path, || rows.next())? {
        let bytes: Vec<u8> = db(path, || row.get(0))?;
        let hash = BlobHash::from_bytes(bytes.try_into().map_err(|_| Error::Schema(path.into()))?)
            .to_string();
        let file = root
            .join("blobs")
            .join(&hash[..2])
            .join(format!("{hash}.zst"));
        if !file.metadata().is_ok_and(|m| m.is_file()) {
            missing += 1;
        }
    }
    Ok(missing)
}
#[derive(Debug, Serialize)]
pub struct Report {
    pub outcome: &'static str,
    pub action: &'static str,
    pub request_id: String,
    pub acquisition_lineage: String,
    pub latest_acknowledged_head: &'static str,
    pub recovery: Option<polylogue_model::RecoveryRecord>,
}
/// A preparation in progress. Holds the archive's writer lock from `begin` to
/// `finish`, so the caller can report `progress` before the cancellable checks.
pub struct Preparation {
    root: PathBuf,
    archive: Connection,
    _lock: File,
    plan: Pending,
    /// The request already completed earlier; `finish` only reports it.
    already_completed: bool,
}
impl Preparation {
    pub fn begin(
        root: &Path,
        mode: &str,
        backing: &str,
        request_id: &str,
        retired_writers: bool,
        cancel: &AtomicBool,
    ) -> Result<Self> {
        // Check the arguments before opening the copied database.
        if mode != "restore" || backing != "local" {
            return Err(refused("only local restore is supported"));
        }
        if !retired_writers {
            return Err(refused(
                "restore requires --retired-writers (the original writer is stopped)",
            ));
        }
        if !uuid_v7(request_id) {
            return Err(refused("preparation requires a UUIDv7 request ID"));
        }
        wal::check_cancel(cancel)?;
        crate::initialize_sqlite()?;
        let root = fs::canonicalize(root)?;
        let lock = crate::startup_lock(&root, false, true)?;
        wal::validate(&root, cancel)?;
        let path = root.join("archive.db");
        crate::preflight_inner(&root, true)?;
        let checked = open(&path, true, true)?;
        let source = lineage(&checked, &path)?.ok_or_else(|| {
            refused("archive has no acquisition lineage; restore cannot create one")
        })?;
        drop(checked);
        recovery::migrate(&root, cancel)?;
        let mut archive = open(&path, false, true)?;
        db(&path, || archive.pragma_update(None, "synchronous", "FULL"))?;
        let hash = intent_hash()?;
        let already_completed = client_request::replay(&archive, &path, request_id, hash)?;
        let existing = pending(&archive, &path)?;
        if let Some(p) = &existing
            && !already_completed
        {
            if p.request_id != request_id || p.intent_hash != hash {
                return Err(refused(format!(
                    "copy preparation {} is pending; a different request cannot continue it",
                    p.request_id
                )));
            }
            if p.source_lineage != source || counter(&archive, &path)? != p.copy_cut + 1 {
                return Err(refused(
                    "archive changed since the pending preparation started",
                ));
            }
        }
        let plan = existing.filter(|_| !already_completed).unwrap_or(Pending {
            version: 1,
            request_id: request_id.into(),
            intent_hash: hash,
            mode: "restore".into(),
            backing: "local".into(),
            retired_writers: true,
            source_lineage: source.clone(),
            target_lineage: source,
            copy_cut: counter(&archive, &path)?,
            batch_cursor: 0,
        });
        if !already_completed && meta(&archive, &path, "copy_preparation")?.is_none() {
            if plan.copy_cut > i64::MAX as u64 - 2 {
                return Err(Error::Range);
            }
            let json = serde_json::to_string(&plan).map_err(|e| refused(e.to_string()))?;
            db(&path, || {
                let tx = archive.transaction()?;
                tx.execute(
                    "INSERT INTO meta(key, value) VALUES('copy_preparation', ?1)",
                    [json],
                )?;
                tx.execute(
                    "UPDATE meta SET value=?1 WHERE key='write_counter'",
                    [(plan.copy_cut + 1).to_string()],
                )?;
                tx.commit()
            })?;
        }
        Ok(Self {
            root,
            archive,
            _lock: lock,
            plan,
            already_completed,
        })
    }
    pub fn progress(&self) -> serde_json::Value {
        let pending = (!self.already_completed).then_some(&self.plan);
        serde_json::json!({
            "outcome": if pending.is_some() { "pending" } else { "historical_completion" },
            "copy_preparation": pending,
            "request_id": self.plan.request_id,
            "activation_granted": false,
        })
    }
    pub fn finish(mut self, cancel: &AtomicBool) -> Result<Report> {
        let path = self.root.join("archive.db");
        if !self.already_completed {
            wal::check_cancel(cancel)?;
            recovery::integrity(&self.archive, &path, cancel)?;
            let embeddings = self.root.join("embeddings.db");
            recovery::integrity(&open(&embeddings, true, true)?, &embeddings, cancel)?;
            recovery::blobs(&self.root, &self.archive, cancel)?;
            wal::check_cancel(cancel)?;
            if pending(&self.archive, &path)?.as_ref() != Some(&self.plan)
                || counter(&self.archive, &path)? != self.plan.copy_cut + 1
                || lineage(&self.archive, &path)?.as_deref() != Some(&self.plan.source_lineage)
            {
                return Err(refused("archive changed during preparation"));
            }
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_err(|_| Error::Range)?
                .as_millis();
            let tx = db(&path, || self.archive.transaction())?;
            client_request::insert(
                &tx,
                &path,
                &self.plan.request_id,
                self.plan.intent_hash,
                i64::try_from(now).map_err(|_| Error::Range)?,
            )?;
            db(&path, || {
                tx.execute(
                    "UPDATE meta SET value=?1 WHERE key='acquisition_lineage'",
                    [&self.plan.target_lineage],
                )?;
                tx.execute(
                    "UPDATE meta SET value=?1 WHERE key='write_counter'",
                    [(self.plan.copy_cut + 2).to_string()],
                )?;
                tx.execute("DELETE FROM meta WHERE key='copy_preparation'", [])?;
                tx.commit()
            })?;
        }
        Ok(Report {
            outcome: "degraded",
            action: if self.already_completed {
                "historical_completion"
            } else {
                "prepared"
            },
            request_id: self.plan.request_id,
            acquisition_lineage: lineage(&self.archive, &path)?
                .ok_or_else(|| refused("missing lineage"))?,
            latest_acknowledged_head: "unknown",
            recovery: recovery::record(&self.archive, &path)?,
        })
    }
}
#[cfg(test)]
mod tests;
