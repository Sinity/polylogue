//! Immutable retained input registration. Archive commit precedes disposable
//! index registration, so retry repairs a crash between the two databases.
use crate::{BlobHash, Error, Result, Writer, counter, db};
use rusqlite::{OptionalExtension, params};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputFormat {
    ClaudeCodeJsonl,
    CodexRollout,
}
impl InputFormat {
    fn name(self) -> &'static str {
        match self {
            Self::ClaudeCodeJsonl => "claude-code-jsonl",
            Self::CodexRollout => "codex-rollout",
        }
    }
}

/// Caller-supplied acquisition evidence; retention never invents a host/path/time.
pub struct ImportObservation<'a> {
    pub id: &'a str,
    pub machine: &'a str,
    pub host: &'a str,
    pub path: &'a str,
    pub seen_at: i64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RetainedInput {
    pub input_pk: i64,
    pub id: String,
    pub chain: i64,
    pub length: u64,
}

impl Writer {
    /// Register an already retained whole blob as one immutable imported input.
    /// No source filesystem access or provider interpretation occurs here. This
    /// does not infer lineage with any live file that happens to share messages.
    /// Metadata allowance is caller admission, not a provider semantic size cap.
    pub fn retain_immutable_input(
        &mut self,
        hash: BlobHash,
        format: InputFormat,
        observation: &ImportObservation<'_>,
        metadata_allowance: usize,
    ) -> Result<RetainedInput> {
        let mut metadata = 64usize + format.name().len();
        for value in [
            observation.id,
            observation.machine,
            observation.host,
            observation.path,
        ] {
            if value.is_empty() {
                return Err(Error::Invariant("empty immutable-import provenance".into()));
            }
            metadata = metadata.checked_add(value.len()).ok_or(Error::Range)?;
        }
        if metadata > metadata_allowance {
            return Err(Error::ParserAdmission {
                resource: "immutable import metadata bytes",
                required: metadata as u64,
            });
        }
        let id = hash.to_string();
        let index_file = self.root.join("index.db");
        let excised: Option<bool> = db(&index_file, || {
            self._index
                .query_row("SELECT excised FROM input_ref WHERE id=?1", [&id], |r| {
                    r.get(0)
                })
                .optional()
        })?;
        if excised == Some(true) {
            return Err(Error::Invariant(
                "immutable input is excised in index".into(),
            ));
        }
        let file = self.root.join("archive.db");
        let tx = db(&file, || self.archive.transaction())?;
        let length: Option<i64> = db(&file, || {
            tx.query_row(
                "SELECT size FROM blob WHERE hash=?1 AND state='present'",
                [hash.bytes().as_slice()],
                |row| row.get(0),
            )
            .optional()
        })?;
        let length = length.ok_or(Error::UnknownBlob)?;
        let length_u64 = u64::try_from(length).map_err(|_| Error::Range)?;
        let next = counter(&tx, &file)?
            .checked_add(1)
            .filter(|value| *value <= i64::MAX as u64)
            .ok_or(Error::Range)?;
        let mut changed = db(&file, || {
            tx.execute(
                "INSERT OR IGNORE INTO input(id,format) VALUES(?1,?2)",
                params![id, format.name()],
            )
        })?;
        let existing_format: String = db(&file, || {
            tx.query_row("SELECT format FROM input WHERE id=?1", [&id], |r| r.get(0))
        })?;
        if existing_format != format.name() {
            return Err(Error::Invariant("immutable input format changed".into()));
        }
        changed += db(&file, || {
            tx.execute(
                "INSERT OR IGNORE INTO revision(input,number,produced,chain,length,blake3,local_seq,excised) VALUES(?1,1,'imported',1,?2,?3,?4,0)",
                params![id, length, hash.bytes().as_slice(), next as i64],
            )
        })?;
        let valid_revision: bool = db(&file, || {
            tx.query_row(
                "SELECT produced='imported' AND chain=1 AND length=?2 AND blake3=?3 AND excised=0 FROM revision WHERE input=?1 AND number=1",
                params![id, length, hash.bytes().as_slice()],
                |r| r.get(0),
            )
        })?;
        if !valid_revision {
            return Err(Error::Invariant(
                "immutable input revision changed or excised".into(),
            ));
        }
        changed += db(&file, || {
            tx.execute(
                "INSERT OR IGNORE INTO extent(input,chain,offset,length,blob,blob_offset) VALUES(?1,1,0,?2,?3,0)",
                params![id, length, hash.bytes().as_slice()],
            )
        })?;
        let valid_extent: bool = db(&file, || {
            tx.query_row(
                "SELECT length=?2 AND blob=?3 AND blob_offset=0 FROM extent WHERE input=?1 AND chain=1 AND offset=0",
                params![id, length, hash.bytes().as_slice()],
                |r| r.get(0),
            )
        })?;
        if !valid_extent {
            return Err(Error::Invariant("immutable input extent changed".into()));
        }
        changed += db(&file, || {
            tx.execute(
                "INSERT OR IGNORE INTO machine(id,host,first_seen) VALUES(?1,?2,?3)",
                params![observation.machine, observation.host, observation.seen_at],
            )
        })?;
        changed += db(&file, || {
            tx.execute(
                "UPDATE machine SET first_seen=?3 WHERE id=?1 AND host=?2 AND first_seen>?3",
                params![observation.machine, observation.host, observation.seen_at],
            )
        })?;
        changed += db(&file, || {
            tx.execute(
                "INSERT OR IGNORE INTO observation(id,input,machine,path,class,revision,length,seen_at,polylogue_version) VALUES(?1,?2,?3,?4,'imported',1,?5,?6,?7)",
                params![observation.id, id, observation.machine, observation.path, length, observation.seen_at, env!("CARGO_PKG_VERSION")],
            )
        })?;
        let valid_observation: bool = db(&file, || {
            tx.query_row(
                "SELECT input=?2 AND machine=?3 AND path=?4 AND class='imported' AND revision=1 AND length=?5 AND seen_at=?6 FROM observation WHERE id=?1",
                params![observation.id, id, observation.machine, observation.path, length, observation.seen_at],
                |r| r.get(0),
            )
        })?;
        if !valid_observation {
            return Err(Error::Invariant(
                "immutable import observation identity reused".into(),
            ));
        }
        if changed != 0 {
            db(&file, || {
                tx.execute(
                    "UPDATE meta SET value=?1 WHERE key='write_counter'",
                    [next.to_string()],
                )
            })?;
        }
        db(&file, || tx.commit())?;
        // An interrupted/disposable index can always reconstruct this reference
        // from archive input. No index reference certifies canonical convergence.
        let file = self.root.join("index.db");
        let tx = db(&file, || self._index.transaction())?;
        db(&file, || {
            tx.execute("INSERT OR IGNORE INTO input_ref(id) VALUES(?1)", [&id])
        })?;
        let (input_pk, excised): (i64, bool) = db(&file, || {
            tx.query_row("SELECT pk,excised FROM input_ref WHERE id=?1", [&id], |r| {
                Ok((r.get(0)?, r.get(1)?))
            })
        })?;
        if excised {
            return Err(Error::Invariant(
                "immutable input is excised in index".into(),
            ));
        }
        db(&file, || tx.commit())?;
        Ok(RetainedInput {
            input_pk,
            id,
            chain: 1,
            length: length_u64,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn observation(id: &str) -> ImportObservation<'_> {
        ImportObservation {
            id,
            machine: "synthetic-machine",
            host: "test-host",
            path: "copy:synthetic.jsonl",
            seen_at: 7,
        }
    }
    #[test]
    fn immutable_import_is_retained_and_retry_does_not_invent_another_revision() {
        let root = tempfile::tempdir().unwrap();
        let mut writer = Writer::open(root.path()).unwrap();
        let hash = writer.put_file_blob(b"synthetic\n").unwrap();
        let imported = writer
            .retain_immutable_input(
                hash,
                InputFormat::CodexRollout,
                &observation("import-1"),
                1024,
            )
            .unwrap();
        let counter = writer.status().unwrap().write_counter;
        assert_eq!(
            writer
                .retain_immutable_input(
                    hash,
                    InputFormat::CodexRollout,
                    &observation("import-1"),
                    1024
                )
                .unwrap(),
            imported
        );
        assert_eq!(writer.status().unwrap().write_counter, counter);
        let reader = writer.read_handle().unwrap();
        assert_eq!(
            reader.read_blob_range(hash, 0, imported.length).unwrap(),
            b"synthetic\n"
        );
        assert_eq!(
            writer
                .archive
                .query_row("SELECT count(*) FROM revision", [], |r| r.get::<_, u64>(0))
                .unwrap(),
            1
        );
        assert_eq!(
            writer
                .archive
                .query_row("SELECT count(*) FROM observation", [], |r| r
                    .get::<_, u64>(0))
                .unwrap(),
            1
        );
        assert!(
            writer
                .retain_immutable_input(
                    hash,
                    InputFormat::ClaudeCodeJsonl,
                    &observation("import-2"),
                    1024
                )
                .is_err()
        );
        assert_eq!(writer.status().unwrap().write_counter, counter);
    }
    #[test]
    fn disposable_registration_failure_keeps_archive_and_retry_repairs_index() {
        let root = tempfile::tempdir().unwrap();
        let mut writer = Writer::open(root.path()).unwrap();
        let hash = writer.put_file_blob(b"retained before index\n").unwrap();
        writer._index.execute_batch("CREATE TRIGGER reject_input BEFORE INSERT ON input_ref BEGIN SELECT RAISE(ABORT,'synthetic registration refusal'); END;").unwrap();
        assert!(matches!(
            writer.retain_immutable_input(
                hash,
                InputFormat::CodexRollout,
                &observation("import-1"),
                1024
            ),
            Err(Error::Sqlite { .. })
        ));
        let archive_counter = writer.status().unwrap().write_counter;
        assert_eq!(
            writer
                .archive
                .query_row("SELECT count(*) FROM extent", [], |r| r.get::<_, u64>(0))
                .unwrap(),
            1
        );
        writer
            ._index
            .execute_batch("DROP TRIGGER reject_input;")
            .unwrap();
        drop(writer);
        let mut writer = Writer::open(root.path()).unwrap();
        let imported = writer
            .retain_immutable_input(
                hash,
                InputFormat::CodexRollout,
                &observation("import-1"),
                1024,
            )
            .unwrap();
        assert_eq!(writer.status().unwrap().write_counter, archive_counter);
        assert_eq!(imported.chain, 1);
        assert_eq!(imported.length, 22);
    }
    #[test]
    fn provenance_conflict_admission_and_missing_bytes_do_not_mutate_archive() {
        let root = tempfile::tempdir().unwrap();
        let mut writer = Writer::open(root.path()).unwrap();
        let a = writer.put_file_blob(b"a").unwrap();
        let b = writer.put_file_blob(b"b").unwrap();
        let before = writer.status().unwrap().write_counter;
        assert!(matches!(
            writer.retain_immutable_input(a, InputFormat::CodexRollout, &observation("id"), 1),
            Err(Error::ParserAdmission { .. })
        ));
        assert!(matches!(
            writer.retain_immutable_input(
                BlobHash::of(b"missing"),
                InputFormat::CodexRollout,
                &observation("missing"),
                1024
            ),
            Err(Error::UnknownBlob)
        ));
        assert_eq!(writer.status().unwrap().write_counter, before);
        writer
            .retain_immutable_input(a, InputFormat::CodexRollout, &observation("id"), 1024)
            .unwrap();
        let before = writer.status().unwrap().write_counter;
        assert!(
            writer
                .retain_immutable_input(b, InputFormat::CodexRollout, &observation("id"), 1024)
                .is_err()
        );
        assert_eq!(writer.status().unwrap().write_counter, before);
        assert_eq!(
            writer
                .archive
                .query_row("SELECT count(*) FROM input", [], |r| r.get::<_, u64>(0))
                .unwrap(),
            1
        );
    }
}
