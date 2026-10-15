//! Immutable retained input registration. Archive commit precedes disposable
//! index registration, so retry repairs a crash between the two databases.
use crate::{BlobHash, Error, Result, Writer, counter, db};
use polylogue_model::InputId;
use rusqlite::{OptionalExtension, params};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputFormat {
    ClaudeCodeJsonl,
    CodexRollout,
}
impl InputFormat {
    pub fn name(self) -> &'static str {
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

#[derive(Clone, Debug)]
pub struct ImmutableImport {
    pub input: RetainedInput,
    pub hash: BlobHash,
    pub format: InputFormat,
    pub context: Vec<String>,
}

fn validate_context(format: InputFormat, context: &[&str]) -> Result<()> {
    const MAXIMUM: usize = 65536;
    let mut required = context
        .len()
        .checked_mul(8)
        .and_then(|n| n.checked_add(8))
        .ok_or(Error::Range)?;
    if required > MAXIMUM {
        return Err(Error::ImportContextTooLarge {
            required,
            maximum: MAXIMUM,
        });
    }
    for field in context {
        required = required.checked_add(field.len()).ok_or(Error::Range)?;
        if required > MAXIMUM {
            return Err(Error::ImportContextTooLarge {
                required,
                maximum: MAXIMUM,
            });
        }
    }
    match format {
        InputFormat::CodexRollout if context.is_empty() => {}
        InputFormat::ClaudeCodeJsonl if matches!(context,["claude-main",stem] if !stem.is_empty() && !stem.contains('/') && !stem.contains('\0')) =>
            {}
        InputFormat::ClaudeCodeJsonl if matches!(context,["claude-subagent",agent] | ["claude-subagent",agent,_] if !agent.is_empty() && !agent.contains('/') && !agent.contains('\0') && context.iter().all(|v|!v.is_empty())) =>
            {}
        _ => {
            return Err(Error::Invariant(
                "missing or invalid immutable interpretation context".into(),
            ));
        }
    }
    Ok(())
}

fn request_hash(
    format: InputFormat,
    context: &[&str],
    observation: &ImportObservation<'_>,
    allowance: usize,
) -> Result<BlobHash> {
    validate_context(format, context)?;
    let mut length = 64usize + format.name().len();
    for field in [
        observation.id,
        observation.machine,
        observation.host,
        observation.path,
    ] {
        if field.is_empty() {
            return Err(Error::Invariant("empty immutable-import provenance".into()));
        }
        length = length.checked_add(field.len()).ok_or(Error::Range)?;
    }
    if length > allowance {
        return Err(Error::ParserAdmission {
            resource: "immutable import metadata bytes",
            required: length as u64,
        });
    }
    for field in context {
        length = length
            .checked_add(field.len())
            .and_then(|n| n.checked_add(8))
            .ok_or(Error::Range)?;
    }
    if length > allowance {
        return Err(Error::ParserAdmission {
            resource: "immutable import metadata bytes",
            required: length as u64,
        });
    }
    let value = serde_json::json!({"context":context,"command":"import","format":format.name(),"machine":observation.machine,"host":observation.host,"path":observation.path,"seen_at":observation.seen_at});
    let bytes = serde_json::to_vec(&value).map_err(|e| Error::Invariant(e.to_string()))?;
    Ok(BlobHash::of(&bytes))
}

impl Writer {
    /// Provider acquisition must check this before copying any source bytes.
    /// Until the refusal-aware collector is wired, even an unrelated excision
    /// prevents source acquisition. Generic blob storage is not that collector.
    pub fn check_import_acquisition(&self) -> Result<()> {
        let file = self.root.join("archive.db");
        let exists: bool = db(&file, || {
            self.archive
                .query_row("SELECT EXISTS(SELECT 1 FROM excision LIMIT 1)", [], |r| {
                    r.get(0)
                })
        })?;
        if exists {
            return Err(Error::ExcisionCheckRequired);
        }
        Ok(())
    }

    /// Recover one immutable import in primary-key order. Archive metadata is
    /// authoritative; repair its disposable index reference after a crash gap.
    /// This does not enumerate growable/native inputs or choose Claude context.
    pub fn next_immutable_import(&mut self, after: &str) -> Result<Option<ImmutableImport>> {
        self.check_import_acquisition()?;
        let file = self.root.join("archive.db");
        let row: Option<(String, String, i64, Vec<u8>, Vec<u8>, String)> = db(&file, || {
            self.archive.query_row(
            "SELECT i.id,i.format,r.length,r.blake3,i.original_hash,CASE WHEN length(CAST(i.context_json AS BLOB))<=524288 THEN i.context_json ELSE NULL END FROM input i JOIN revision r ON r.input=i.id AND r.number=1 WHERE i.id>?1 AND r.produced='imported' AND r.chain=1 ORDER BY i.id LIMIT 1", [after],
            |r| Ok((r.get(0)?,r.get(1)?,r.get(2)?,r.get(3)?,r.get(4)?,r.get(5)?))).optional()
        })?;
        self.decode_immutable_import(row)
    }

    /// Exact retained-import lookup. Does not inspect provider records or paths.
    pub fn immutable_import(&mut self, id: &str) -> Result<Option<ImmutableImport>> {
        self.check_import_acquisition()?;
        let file = self.root.join("archive.db");
        let row = db(&file, || {
            self.archive.query_row(
            "SELECT i.id,i.format,r.length,r.blake3,i.original_hash,CASE WHEN length(CAST(i.context_json AS BLOB))<=524288 THEN i.context_json ELSE NULL END FROM input i JOIN revision r ON r.input=i.id AND r.number=1 WHERE i.id=?1 AND r.produced='imported' AND r.chain=1", [id],
            |r| Ok((r.get(0)?,r.get(1)?,r.get(2)?,r.get(3)?,r.get(4)?,r.get(5)?))).optional()
        })?;
        self.decode_immutable_import(row)
    }

    fn decode_immutable_import(
        &mut self,
        row: Option<(String, String, i64, Vec<u8>, Vec<u8>, String)>,
    ) -> Result<Option<ImmutableImport>> {
        let Some((id, format, length, digest, original, context_json)) = row else {
            return Ok(None);
        };
        let hash = BlobHash::from_bytes(
            digest
                .try_into()
                .map_err(|_| Error::Invariant("invalid immutable import hash".into()))?,
        );
        let original = BlobHash::from_bytes(
            original
                .try_into()
                .map_err(|_| Error::Invariant("invalid import original digest".into()))?,
        );
        let context: Vec<String> =
            serde_json::from_str(&context_json).map_err(|e| Error::Invariant(e.to_string()))?;
        let parts: Vec<&str> = context.iter().map(String::as_str).collect();
        if hash != original || id != InputId::imported(&original, &format, &parts).as_str() {
            return Err(Error::Invariant(
                "immutable import identity tuple mismatch".into(),
            ));
        }
        let format = match format.as_str() {
            "codex-rollout" => InputFormat::CodexRollout,
            "claude-code-jsonl" => InputFormat::ClaudeCodeJsonl,
            _ => {
                return Err(Error::Invariant(
                    "unsupported immutable import format".into(),
                ));
            }
        };
        // Validate provider semantics as well as the identity framing.
        validate_context(format, &parts)?;
        let length = u64::try_from(length).map_err(|_| Error::Range)?;
        let input = self.register_import_reference(id, length)?;
        Ok(Some(ImmutableImport {
            input,
            hash,
            format,
            context,
        }))
    }

    /// Retry the same write request without opening its source path again.
    /// An existing request ID with changed command metadata is refused.
    pub fn replay_immutable_import(
        &mut self,
        format: InputFormat,
        context: &[&str],
        observation: &ImportObservation<'_>,
        allowance: usize,
    ) -> Result<Option<RetainedInput>> {
        let expected = request_hash(format, context, observation, allowance)?;
        self.check_import_acquisition()?;
        let file = self.root.join("archive.db");
        let previous: Option<Vec<u8>> = db(&file, || {
            self.archive
                .query_row(
                    "SELECT content_hash FROM client_request WHERE id=?1",
                    [observation.id],
                    |r| r.get(0),
                )
                .optional()
        })?;
        let Some(previous) = previous else {
            return Ok(None);
        };
        if previous.as_slice() != expected.bytes() {
            return Err(Error::ImportRequestConflict {
                id: observation.id.to_owned(),
            });
        }
        let id: String = db(&file, || {
            self.archive.query_row(
                "SELECT input FROM observation WHERE id=?1 AND class='imported'",
                [observation.id],
                |r| r.get(0),
            )
        })?;
        let imported = self.immutable_import(&id)?.ok_or_else(|| {
            Error::Invariant("import receipt has no retained immutable input".into())
        })?;
        Ok(Some(imported.input))
    }

    fn register_import_reference(&mut self, id: String, length: u64) -> Result<RetainedInput> {
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
            length,
        })
    }

    /// Register an already retained whole blob as one immutable imported input.
    /// No source filesystem access or provider interpretation occurs here. This
    /// does not infer lineage with any live file that happens to share messages.
    /// Metadata allowance is caller admission, not a provider semantic size cap.
    /// Archives with any excision require the not-yet-wired refusal-aware
    /// collector and return ExcisionCheckRequired before changing metadata.
    pub fn retain_immutable_input(
        &mut self,
        hash: BlobHash,
        format: InputFormat,
        context: &[&str],
        observation: &ImportObservation<'_>,
        metadata_allowance: usize,
    ) -> Result<RetainedInput> {
        let request_hash = request_hash(format, context, observation, metadata_allowance)?;
        if let Some(previous) =
            self.replay_immutable_import(format, context, observation, metadata_allowance)?
        {
            if previous.id != InputId::imported(&hash, format.name(), context).as_str() {
                return Err(Error::ImportRequestConflict {
                    id: observation.id.to_owned(),
                });
            }
            return Ok(previous);
        }
        let id = InputId::imported(&hash, format.name(), context).to_string();
        let context_json =
            serde_json::to_string(context).map_err(|e| Error::Invariant(e.to_string()))?;
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
        let request_inserted = db(&file, || {
            tx.execute(
                "INSERT INTO client_request(id,content_hash,created_at) VALUES(?1,?2,?3)",
                params![
                    observation.id,
                    request_hash.bytes().as_slice(),
                    observation.seen_at
                ],
            )
        })?;
        // This foundation primitive does not interpret refused_json or guess
        // session/text removal closure. Until the collector owns that check,
        // refuse explicitly instead of reintroducing removed provider bytes.
        let has_excisions: bool = db(&file, || {
            tx.query_row("SELECT EXISTS(SELECT 1 FROM excision LIMIT 1)", [], |r| {
                r.get(0)
            })
        })?;
        if has_excisions {
            return Err(Error::ExcisionCheckRequired);
        }
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
        let mut changed = request_inserted
            + db(&file, || {
                tx.execute(
                    "INSERT OR IGNORE INTO input(id,format,original_hash,context_json) VALUES(?1,?2,?3,?4)",
                    params![id, format.name(),hash.bytes().as_slice(),context_json],
                )
            })?;
        let existing_format: String = db(&file, || {
            tx.query_row("SELECT format FROM input WHERE id=?1", [&id], |r| r.get(0))
        })?;
        let tuple_valid: bool = db(&file, || {
            tx.query_row(
                "SELECT original_hash=?2 AND context_json=?3 FROM input WHERE id=?1",
                params![id, hash.bytes().as_slice(), context_json],
                |r| r.get(0),
            )
        })?;
        if existing_format != format.name() || !tuple_valid {
            return Err(Error::Invariant("immutable input format changed".into()));
        }
        changed += db(&file, || {
            tx.execute(
                "INSERT OR IGNORE INTO revision(input,number,produced,chain,length,blake3,local_seq) VALUES(?1,1,'imported',1,?2,?3,?4)",
                params![id, length, hash.bytes().as_slice(), next as i64],
            )
        })?;
        let valid_revision: bool = db(&file, || {
            tx.query_row(
                "SELECT produced='imported' AND chain=1 AND length=?2 AND blake3=?3 FROM revision WHERE input=?1 AND number=1",
                params![id, length, hash.bytes().as_slice()],
                |r| r.get(0),
            )
        })?;
        if !valid_revision {
            return Err(Error::Invariant("immutable input revision changed".into()));
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
        // An interrupted disposable index reconstructs this reference from archive.
        self.register_import_reference(id, length_u64)
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
                &[],
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
                    &[],
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
                .query_row("SELECT count(*) FROM revision", [], |r| r.get::<_, i64>(0))
                .unwrap(),
            1
        );
        assert_eq!(
            writer
                .archive
                .query_row("SELECT count(*) FROM observation", [], |r| r
                    .get::<_, i64>(0))
                .unwrap(),
            1
        );
        assert!(
            writer
                .retain_immutable_input(
                    hash,
                    InputFormat::ClaudeCodeJsonl,
                    &[],
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
                &[],
                &observation("import-1"),
                1024
            ),
            Err(Error::Sqlite { .. })
        ));
        let archive_counter = writer.status().unwrap().write_counter;
        assert_eq!(
            writer
                .archive
                .query_row("SELECT count(*) FROM extent", [], |r| r.get::<_, i64>(0))
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
                &[],
                &observation("import-1"),
                1024,
            )
            .unwrap();
        assert_eq!(writer.status().unwrap().write_counter, archive_counter);
        assert_eq!(imported.chain, 1);
        assert_eq!(imported.length, 22);
    }
    #[test]
    fn immutable_import_recovery_pages_and_repairs_only_registered_archive_inputs() {
        let root = tempfile::tempdir().unwrap();
        let mut writer = Writer::open(root.path()).unwrap();
        for (bytes, id) in [(b"a".as_slice(), "a"), (b"b".as_slice(), "b")] {
            let hash = writer.put_file_blob(bytes).unwrap();
            writer
                .retain_immutable_input(
                    hash,
                    InputFormat::CodexRollout,
                    &[],
                    &observation(id),
                    1024,
                )
                .unwrap();
        }
        writer._index.execute("DELETE FROM input_ref", []).unwrap();
        drop(writer);
        let mut writer = Writer::open(root.path()).unwrap();
        let first = writer.next_immutable_import("").unwrap().unwrap();
        assert_eq!(
            first.input.id,
            InputId::imported(&first.hash, first.format.name(), &[]).to_string()
        );
        assert_eq!(first.input.length, 1);
        assert_eq!(
            writer
                ._index
                .query_row("SELECT count(*) FROM input_ref", [], |r| r.get::<_, i64>(0))
                .unwrap(),
            1
        );
        let second = writer
            .next_immutable_import(&first.input.id)
            .unwrap()
            .unwrap();
        assert!(second.input.id > first.input.id);
        assert!(
            writer
                .next_immutable_import(&second.input.id)
                .unwrap()
                .is_none()
        );
        assert_eq!(
            writer.next_immutable_import("").unwrap().unwrap().input,
            first.input
        );
        writer.archive.execute("INSERT INTO excision(id,scope,created_at,local_seq) VALUES('removal','session:unknown',1,1)",[]).unwrap();
        assert!(matches!(
            writer.check_import_acquisition(),
            Err(Error::ExcisionCheckRequired)
        ));
        assert!(matches!(
            writer.next_immutable_import(""),
            Err(Error::ExcisionCheckRequired)
        ));
    }

    #[test]
    fn context_declared_bound_precedes_scanning_encoding_or_sql_mutation() {
        let root = tempfile::tempdir().unwrap();
        let mut writer = Writer::open(root.path()).unwrap();
        let hash = writer.put_file_blob(b"synthetic").unwrap();
        let huge = "x".repeat(65536);
        let before = writer.status().unwrap().write_counter;
        assert!(matches!(
            writer.retain_immutable_input(
                hash,
                InputFormat::ClaudeCodeJsonl,
                &["claude-main", &huge],
                &observation("large"),
                usize::MAX
            ),
            Err(Error::ImportContextTooLarge { maximum: 65536, .. })
        ));
        assert!(matches!(
            validate_context(InputFormat::CodexRollout, &vec![""; 8192]),
            Err(Error::ImportContextTooLarge { .. })
        ));
        assert!(
            validate_context(InputFormat::ClaudeCodeJsonl, &["claude-main", "bad\0stem"]).is_err()
        );
        assert_eq!(writer.status().unwrap().write_counter, before);
        assert_eq!(
            writer
                .archive
                .query_row("SELECT count(*) FROM input", [], |r| r.get::<_, i64>(0))
                .unwrap(),
            0
        );
    }

    #[test]
    fn same_bytes_different_contexts_have_distinct_inputs_and_one_physical_blob() {
        let root = tempfile::tempdir().unwrap();
        let mut writer = Writer::open(root.path()).unwrap();
        let hash = writer
            .put_file_blob(b"same synthetic provider bytes")
            .unwrap();
        let a = writer
            .retain_immutable_input(
                hash,
                InputFormat::ClaudeCodeJsonl,
                &["claude-main", "a"],
                &observation("a"),
                4096,
            )
            .unwrap();
        let b = writer
            .retain_immutable_input(
                hash,
                InputFormat::ClaudeCodeJsonl,
                &["claude-main", "b"],
                &observation("b"),
                4096,
            )
            .unwrap();
        let native = writer
            .retain_immutable_input(
                hash,
                InputFormat::CodexRollout,
                &[],
                &observation("native"),
                4096,
            )
            .unwrap();
        assert_ne!(a.id, b.id);
        assert_ne!(a.id, native.id);
        assert_eq!(
            writer
                .archive
                .query_row("SELECT count(*) FROM blob", [], |r| r.get::<_, i64>(0))
                .unwrap(),
            1
        );
        assert_eq!(
            writer.immutable_import(&a.id).unwrap().unwrap().context,
            vec!["claude-main", "a"]
        );
        assert!(
            writer
                .replay_immutable_import(
                    InputFormat::ClaudeCodeJsonl,
                    &["claude-main", "b"],
                    &observation("a"),
                    4096
                )
                .is_err()
        );
        assert!(
            writer
                .retain_immutable_input(
                    hash,
                    InputFormat::ClaudeCodeJsonl,
                    &[],
                    &observation("bad"),
                    4096
                )
                .is_err()
        );
        assert!(
            writer
                .retain_immutable_input(
                    hash,
                    InputFormat::CodexRollout,
                    &["unexpected"],
                    &observation("bad"),
                    4096
                )
                .is_err()
        );
        writer
            .archive
            .execute(
                "UPDATE input SET context_json='[\"claude-main\",\"different\"]' WHERE id=?1",
                [&a.id],
            )
            .unwrap();
        assert!(writer.immutable_import(&a.id).is_err());
        assert_eq!(writer.immutable_import(&b.id).unwrap().unwrap().input, b);
        // Nullable paired schema fields cannot admit a torn tuple.
        assert!(writer.archive.execute("INSERT INTO input(id,format,original_hash) VALUES('invalid','codex-rollout',zeroblob(32))",[]).is_err());
    }

    #[test]
    fn request_retry_replays_retained_result_and_rejects_changed_arguments() {
        let root = tempfile::tempdir().unwrap();
        let mut writer = Writer::open(root.path()).unwrap();
        let hash = writer.put_file_blob(b"original").unwrap();
        let imported = writer
            .retain_immutable_input(
                hash,
                InputFormat::CodexRollout,
                &[],
                &observation("request"),
                4096,
            )
            .unwrap();
        let counter = writer.status().unwrap().write_counter;
        drop(writer);
        let mut writer = Writer::open(root.path()).unwrap();
        assert_eq!(
            writer
                .replay_immutable_import(
                    InputFormat::CodexRollout,
                    &[],
                    &observation("request"),
                    4096
                )
                .unwrap(),
            Some(imported.clone())
        );
        assert_eq!(
            writer
                .immutable_import(&imported.id)
                .unwrap()
                .unwrap()
                .input,
            imported
        );
        assert_eq!(writer.status().unwrap().write_counter, counter);
        let changed = ImportObservation {
            path: "/different-path",
            ..observation("request")
        };
        assert!(
            writer
                .replay_immutable_import(InputFormat::CodexRollout, &[], &changed, 4096)
                .is_err()
        );
        assert!(
            writer
                .replay_immutable_import(
                    InputFormat::ClaudeCodeJsonl,
                    &[],
                    &observation("request"),
                    4096
                )
                .is_err()
        );
        assert!(
            writer
                .replay_immutable_import(InputFormat::CodexRollout, &[], &observation("new"), 4096)
                .unwrap()
                .is_none()
        );
        assert_eq!(writer.status().unwrap().write_counter, counter);
    }

    #[test]
    fn provenance_conflict_admission_and_missing_bytes_do_not_mutate_archive() {
        let root = tempfile::tempdir().unwrap();
        let mut writer = Writer::open(root.path()).unwrap();
        let a = writer.put_file_blob(b"a").unwrap();
        let b = writer.put_file_blob(b"b").unwrap();
        let before = writer.status().unwrap().write_counter;
        assert!(matches!(
            writer.retain_immutable_input(a, InputFormat::CodexRollout, &[], &observation("id"), 1),
            Err(Error::ParserAdmission { .. })
        ));
        assert!(matches!(
            writer.retain_immutable_input(
                BlobHash::of(b"missing"),
                InputFormat::CodexRollout,
                &[],
                &observation("missing"),
                1024
            ),
            Err(Error::UnknownBlob)
        ));
        assert_eq!(writer.status().unwrap().write_counter, before);
        writer
            .retain_immutable_input(a, InputFormat::CodexRollout, &[], &observation("id"), 1024)
            .unwrap();
        let before = writer.status().unwrap().write_counter;
        assert!(
            writer
                .retain_immutable_input(b, InputFormat::CodexRollout, &[], &observation("id"), 1024)
                .is_err()
        );
        assert_eq!(writer.status().unwrap().write_counter, before);
        assert_eq!(
            writer
                .archive
                .query_row("SELECT count(*) FROM input", [], |r| r.get::<_, i64>(0))
                .unwrap(),
            1
        );
    }

    #[test]
    fn existing_excisions_require_the_collector_before_retaining_input_metadata() {
        let root = tempfile::tempdir().unwrap();
        let mut writer = Writer::open(root.path()).unwrap();
        let hash = writer.put_file_blob(b"synthetic retained bytes").unwrap();
        writer.archive.execute("INSERT INTO excision(id,scope,created_at,local_seq) VALUES('synthetic-removal','session:unknown',1,1)", []).unwrap();
        let before = writer.status().unwrap().write_counter;
        assert!(matches!(
            writer.retain_immutable_input(
                hash,
                InputFormat::CodexRollout,
                &[],
                &observation("new-observation"),
                1024
            ),
            Err(Error::ExcisionCheckRequired)
        ));
        assert_eq!(writer.status().unwrap().write_counter, before);
        assert_eq!(
            writer
                .archive
                .query_row("SELECT count(*) FROM input", [], |r| r.get::<_, i64>(0))
                .unwrap(),
            0
        );
        assert_eq!(
            writer
                ._index
                .query_row("SELECT count(*) FROM input_ref", [], |r| r.get::<_, i64>(0))
                .unwrap(),
            0
        );
    }
}
