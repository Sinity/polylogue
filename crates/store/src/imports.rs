//! Immutable imported inputs. The archive commit comes before the index
//! registration, so retrying an import repairs a crash between the two.
use crate::{BlobHash, Error, Result, Writer, counter, db};
use polylogue_model::InputId;
use rusqlite::{OptionalExtension, params};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputFormat {
    ClaudeCodeJsonl,
    ClaudeCodeToolResult,
    CodexRollout,
    CodexHistorySqlite,
    /// Prompt references only; never parsed as a conversation.
    CodexPromptHistory,
}
impl InputFormat {
    pub fn name(self) -> &'static str {
        match self {
            Self::ClaudeCodeJsonl => "claude-code-jsonl",
            Self::ClaudeCodeToolResult => "claude-code-tool-result",
            Self::CodexRollout => "codex-rollout",
            Self::CodexHistorySqlite => "codex-history-sqlite",
            Self::CodexPromptHistory => "codex-prompt-history",
        }
    }
    fn from_name(name: &str) -> Option<Self> {
        [
            Self::ClaudeCodeJsonl,
            Self::ClaudeCodeToolResult,
            Self::CodexRollout,
            Self::CodexHistorySqlite,
            Self::CodexPromptHistory,
        ]
        .into_iter()
        .find(|format| format.name() == name)
    }

    /// Check the context strings an input of this format must carry. Pure; no IO.
    pub fn validate_context(self, context: &[&str]) -> Result<()> {
        const MAXIMUM: usize = 65536;
        let too_large = |required| Error::ImportContextTooLarge {
            required,
            maximum: MAXIMUM,
        };
        // Each field costs 8 framing bytes plus its length. Check the count
        // before scanning so a huge context is refused cheaply.
        let mut required = context
            .len()
            .checked_mul(8)
            .and_then(|n| n.checked_add(8))
            .ok_or(Error::Range)?;
        if required > MAXIMUM {
            return Err(too_large(required));
        }
        for field in context {
            required = required.checked_add(field.len()).ok_or(Error::Range)?;
            if required > MAXIMUM {
                return Err(too_large(required));
            }
        }
        let plain_name = |s: &str| !s.is_empty() && !s.contains('/') && !s.contains('\0');
        let valid = match self {
            Self::CodexRollout | Self::CodexHistorySqlite | Self::CodexPromptHistory => {
                context.is_empty()
            }
            Self::ClaudeCodeToolResult => matches!(context, [main, path, kind]
                if !main.is_empty()
                    && std::path::Path::new(path).is_absolute()
                    && matches!(*kind, "required" | "inline-duplicate")),
            Self::ClaudeCodeJsonl => match context {
                ["claude-main", stem] => plain_name(stem),
                ["claude-subagent", agent] => plain_name(agent),
                ["claude-subagent", agent, owner] => plain_name(agent) && !owner.is_empty(),
                _ => false,
            },
        };
        if !valid {
            return Err(Error::Invariant(format!(
                "invalid {} import context",
                self.name()
            )));
        }
        Ok(())
    }
}

/// Where and when the caller found the input. The store never fills these in.
pub struct ImportObservation<'a> {
    pub id: &'a str,
    pub machine: &'a str,
    pub host: &'a str,
    pub path: &'a str,
    pub seen_at: i64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StoredInput {
    pub input_pk: i64,
    pub id: String,
    pub chain: i64,
    pub length: u64,
}

#[derive(Clone, Debug)]
pub struct ImmutableImport {
    pub input: StoredInput,
    pub hash: BlobHash,
    pub format: InputFormat,
    pub context: Vec<String>,
}

/// One piece of an input: `(offset, length, blob, blob_offset)`.
pub(crate) type Part = (u64, u64, BlobHash, u64);

const MAX_PARTS: usize = 8192;

fn request_hash(
    format: InputFormat,
    context: &[&str],
    observation: &ImportObservation<'_>,
    allowance: usize,
) -> Result<BlobHash> {
    format.validate_context(context)?;
    let mut length = 64usize + format.name().len();
    for field in [
        observation.id,
        observation.machine,
        observation.host,
        observation.path,
    ] {
        if field.is_empty() {
            return Err(Error::Invariant(
                "import observation has an empty field".into(),
            ));
        }
        length = length.checked_add(field.len()).ok_or(Error::Range)?;
    }
    for field in context {
        length = length
            .checked_add(field.len())
            .and_then(|n| n.checked_add(8))
            .ok_or(Error::Range)?;
    }
    if length > allowance {
        return Err(Error::ParserAdmission {
            resource: "import metadata bytes",
            required: length as u64,
        });
    }
    let value = serde_json::json!({"context":context,"command":"import","format":format.name(),"machine":observation.machine,"host":observation.host,"path":observation.path,"seen_at":observation.seen_at});
    crate::client_request::digest(&value)
}

impl Writer {
    /// Call before copying any source bytes. Any excision in the archive blocks
    /// imports, because nothing yet checks new bytes against excisions.
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

    /// Look up an imported input by ID.
    pub fn immutable_import(&mut self, id: &str) -> Result<Option<ImmutableImport>> {
        self.check_import_acquisition()?;
        let file = self.root.join("archive.db");
        let row: Option<(String, String, i64, Vec<u8>, Vec<u8>, String)> = db(&file, || {
            self.archive.query_row(
            "SELECT i.id,i.format,r.length,r.blake3,i.original_hash,CASE WHEN length(CAST(i.context_json AS BLOB))<=524288 THEN i.context_json ELSE NULL END FROM input i JOIN revision r ON r.input=i.id AND r.number=1 WHERE i.id=?1 AND r.produced='imported' AND r.chain=1", [id],
            |r| Ok((r.get(0)?,r.get(1)?,r.get(2)?,r.get(3)?,r.get(4)?,r.get(5)?))).optional()
        })?;
        let Some((id, format, length, digest, original, context_json)) = row else {
            return Ok(None);
        };
        let (id, hash, format, context, length) =
            decode_import_identity(id, format, length, digest, original, context_json)?;
        let input = self.register_input_ref(id, length)?;
        Ok(Some(ImmutableImport {
            input,
            hash,
            format,
            context,
        }))
    }

    /// Return the input an earlier request with this observation ID stored,
    /// without opening the source path again. The same ID with different
    /// arguments is refused.
    pub fn replay_immutable_import(
        &mut self,
        format: InputFormat,
        context: &[&str],
        observation: &ImportObservation<'_>,
        allowance: usize,
    ) -> Result<Option<StoredInput>> {
        let expected = request_hash(format, context, observation, allowance)?;
        self.check_import_acquisition()?;
        let file = self.root.join("archive.db");
        if !crate::client_request::replay(&self.archive, &file, observation.id, expected)? {
            return Ok(None);
        }
        let id: String = db(&file, || {
            self.archive.query_row(
                "SELECT input FROM observation WHERE id=?1 AND class='imported'",
                [observation.id],
                |r| r.get(0),
            )
        })?;
        let imported = self
            .immutable_import(&id)?
            .ok_or_else(|| Error::Invariant("import request has no stored input".into()))?;
        Ok(Some(imported.input))
    }

    /// Like `replay_immutable_import`, but also refuses a replay whose stored
    /// input has different bytes.
    fn replay_same_import(
        &mut self,
        hash: BlobHash,
        format: InputFormat,
        context: &[&str],
        observation: &ImportObservation<'_>,
        allowance: usize,
    ) -> Result<Option<StoredInput>> {
        let Some(previous) =
            self.replay_immutable_import(format, context, observation, allowance)?
        else {
            return Ok(None);
        };
        if previous.id != InputId::imported(&hash, format.name(), context).as_str() {
            return Err(Error::ClientRequestConflict {
                id: observation.id.into(),
            });
        }
        Ok(Some(previous))
    }

    pub(crate) fn register_input_ref(&mut self, id: String, length: u64) -> Result<StoredInput> {
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
            return Err(Error::Invariant("input is excised".into()));
        }
        db(&file, || tx.commit())?;
        Ok(StoredInput {
            input_pk,
            id,
            chain: 1,
            length,
        })
    }

    /// Record a blob already in the store as one immutable imported input.
    /// Never reads the source path or parses the bytes, and never links the
    /// input to a live file with the same content. `metadata_allowance` bounds
    /// the request metadata, not the input.
    pub fn store_immutable_input(
        &mut self,
        hash: BlobHash,
        format: InputFormat,
        context: &[&str],
        observation: &ImportObservation<'_>,
        metadata_allowance: usize,
    ) -> Result<StoredInput> {
        // Replay first: a retry must succeed even if the blob is gone by now.
        if let Some(previous) =
            self.replay_same_import(hash, format, context, observation, metadata_allowance)?
        {
            return Ok(previous);
        }
        let file = self.root.join("archive.db");
        let length: Option<i64> = db(&file, || {
            self.archive
                .query_row(
                    "SELECT size FROM blob WHERE hash=?1 AND state='present'",
                    [hash.bytes().as_slice()],
                    |r| r.get(0),
                )
                .optional()
        })?;
        let length = u64::try_from(length.ok_or(Error::UnknownBlob)?).map_err(|_| Error::Range)?;
        self.store_immutable_parts(
            hash,
            length,
            &[(0, length, hash, 0)],
            format,
            context,
            observation,
            metadata_allowance,
        )
    }

    /// Record an immutable input made of ranges of stored blobs. The caller
    /// has computed `hash` over the whole input.
    pub(crate) fn store_immutable_parts(
        &mut self,
        hash: BlobHash,
        length_u64: u64,
        parts: &[Part],
        format: InputFormat,
        context: &[&str],
        observation: &ImportObservation<'_>,
        metadata_allowance: usize,
    ) -> Result<StoredInput> {
        let request_hash = request_hash(format, context, observation, metadata_allowance)?;
        if let Some(previous) =
            self.replay_same_import(hash, format, context, observation, metadata_allowance)?
        {
            return Ok(previous);
        }
        if parts.len() > MAX_PARTS {
            return Err(Error::ParserAdmission {
                resource: "import extents",
                required: parts.len() as u64,
            });
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
            return Err(Error::Invariant("input is excised".into()));
        }
        let file = self.root.join("archive.db");
        let tx = db(&file, || self.archive.transaction())?;
        let request_inserted = crate::client_request::insert(
            &tx,
            &file,
            observation.id,
            request_hash,
            observation.seen_at,
        )?;
        // Re-check inside the transaction: an import must never bring back
        // bytes an excision removed.
        let has_excisions: bool = db(&file, || {
            tx.query_row("SELECT EXISTS(SELECT 1 FROM excision LIMIT 1)", [], |r| {
                r.get(0)
            })
        })?;
        if has_excisions {
            return Err(Error::ExcisionCheckRequired);
        }
        let length = i64::try_from(length_u64).map_err(|_| Error::Range)?;
        let mut end = 0u64;
        for &(offset, size, piece, start) in parts {
            let stored: Option<i64> = db(&file, || {
                tx.query_row(
                    "SELECT size FROM blob WHERE hash=?1 AND state='present'",
                    [piece.bytes().as_slice()],
                    |r| r.get(0),
                )
                .optional()
            })?;
            let stored = stored
                .and_then(|n| u64::try_from(n).ok())
                .ok_or(Error::UnknownBlob)?;
            if offset != end || start.checked_add(size).is_none_or(|n| n > stored) {
                return Err(Error::Invariant(
                    "import parts are not contiguous or exceed their blob".into(),
                ));
            }
            end = end.checked_add(size).ok_or(Error::Range)?;
        }
        if end != length_u64 {
            return Err(Error::Invariant(
                "import parts do not cover the whole input".into(),
            ));
        }
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
        let same_input: bool = db(&file, || {
            tx.query_row(
                "SELECT format=?2 AND original_hash=?3 AND context_json=?4 FROM input WHERE id=?1",
                params![id, format.name(), hash.bytes().as_slice(), context_json],
                |r| r.get(0),
            )
        })?;
        if !same_input {
            return Err(Error::Invariant(
                "stored input has a different format, hash or context".into(),
            ));
        }
        changed += db(&file, || {
            tx.execute(
                "INSERT OR IGNORE INTO revision(input,number,produced,chain,length,blake3,local_seq) VALUES(?1,1,'imported',1,?2,?3,?4)",
                params![id, length, hash.bytes().as_slice(), next as i64],
            )
        })?;
        let same_revision: bool = db(&file, || {
            tx.query_row(
                "SELECT produced='imported' AND chain=1 AND length=?2 AND blake3=?3 FROM revision WHERE input=?1 AND number=1",
                params![id, length, hash.bytes().as_slice()],
                |r| r.get(0),
            )
        })?;
        if !same_revision {
            return Err(Error::Invariant(
                "stored input has a different first revision".into(),
            ));
        }
        for &(offset, size, piece, start) in parts {
            changed += db(&file, || {
                tx.execute("INSERT OR IGNORE INTO extent(input,chain,offset,length,blob,blob_offset) VALUES(?1,1,?2,?3,?4,?5)", params![id,i64::try_from(offset).map_err(|_|rusqlite::Error::InvalidQuery)?,i64::try_from(size).map_err(|_|rusqlite::Error::InvalidQuery)?,piece.bytes().as_slice(),i64::try_from(start).map_err(|_|rusqlite::Error::InvalidQuery)?])
            })?;
        }
        let stored: Vec<(i64, i64, Vec<u8>, i64)> = db(&file, || {
            let mut q=tx.prepare("SELECT offset,length,blob,blob_offset FROM extent WHERE input=?1 AND chain=1 ORDER BY offset LIMIT 8194")?;
            q.query_map([&id], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)))?
                .collect()
        })?;
        if stored.len() != parts.len()
            || stored.iter().zip(parts).any(|(a, b)| {
                u64::try_from(a.0).ok() != Some(b.0)
                    || u64::try_from(a.1).ok() != Some(b.1)
                    || a.2 != b.2.bytes()
                    || u64::try_from(a.3).ok() != Some(b.3)
            })
        {
            return Err(Error::Invariant(
                "stored input has different extents".into(),
            ));
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
        let same_observation: bool = db(&file, || {
            tx.query_row(
                "SELECT input=?2 AND machine=?3 AND path=?4 AND class='imported' AND revision=1 AND length=?5 AND seen_at=?6 FROM observation WHERE id=?1",
                params![observation.id, id, observation.machine, observation.path, length, observation.seen_at],
                |r| r.get(0),
            )
        })?;
        if !same_observation {
            return Err(Error::Invariant(
                "observation ID is already used by a different import".into(),
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
        // If this fails, the next lookup re-registers the input from the archive.
        self.register_input_ref(id, length_u64)
    }
}

/// Decode and check an imported input's archive row. Shared by the writer
/// and pinned readers; only the writer registers index references.
pub(crate) fn decode_import_identity(
    id: String,
    format: String,
    length: i64,
    digest: Vec<u8>,
    original: Vec<u8>,
    context_json: String,
) -> Result<(String, BlobHash, InputFormat, Vec<String>, u64)> {
    let hash = BlobHash::from_bytes(
        digest
            .try_into()
            .map_err(|_| Error::Invariant("invalid import hash".into()))?,
    );
    let original = BlobHash::from_bytes(
        original
            .try_into()
            .map_err(|_| Error::Invariant("invalid import original hash".into()))?,
    );
    let context: Vec<String> =
        serde_json::from_str(&context_json).map_err(|e| Error::Invariant(e.to_string()))?;
    let parts: Vec<&str> = context.iter().map(String::as_str).collect();
    if hash != original || id != InputId::imported(&original, &format, &parts).as_str() {
        return Err(Error::Invariant(
            "import ID does not match its hash, format and context".into(),
        ));
    }
    let format = InputFormat::from_name(&format)
        .ok_or_else(|| Error::Invariant("unknown import format".into()))?;
    format.validate_context(&parts)?;
    let length = u64::try_from(length).map_err(|_| Error::Range)?;
    Ok((id, hash, format, context, length))
}

#[cfg(test)]
#[path = "imports/tests.rs"]
mod tests;
