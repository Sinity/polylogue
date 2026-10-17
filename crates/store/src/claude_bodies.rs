//! Explicit Claude tool-result snapshots. Immutable input context owns the main
//! association and exact provider pathname; no disposable parser cell owns bytes.
use crate::{BlobHash, Error, ImportObservation, InputFormat, ReadHandle, Result, Writer, db};
use serde::{Deserialize, Serialize};
use std::{
    fs::File,
    os::unix::fs::MetadataExt,
    path::{Component, Path},
    sync::atomic::AtomicBool,
};

pub const MAX_BODY: u64 = 1024 * 1024;
pub const MAX_COMPANIONS: usize = 32;

/// The configured base may resolve normally. Each provider-owned appended
/// component is opened relative to its held parent descriptor with NOFOLLOW;
/// renaming or replacing its pathname cannot redirect the next open.
fn open_body(main: &Path, path: &Path) -> Result<Option<File>> {
    use rustix::fs::{Mode, OFlags, openat};
    let parent = main.parent().ok_or(Error::Range)?;
    let stem = main.file_stem().ok_or(Error::Range)?;
    let directory = parent.join(stem).join("tool-results");
    if !main.is_absolute()
        || path.parent() != Some(directory.as_path())
        || path.extension().and_then(|v| v.to_str()) != Some("txt")
        || path
            .components()
            .any(|c| matches!(c, Component::ParentDir | Component::CurDir))
    {
        return Err(Error::ObservedSourceRefused(
            "Claude body reference is outside the supported main/tool-results/*.txt location"
                .into(),
        ));
    }
    let base = File::open(parent)?;
    let flags = OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC;
    let open = || -> std::io::Result<File> {
        let session = openat(&base, stem, flags, Mode::empty()).map_err(std::io::Error::from)?;
        let results =
            openat(&session, "tool-results", flags, Mode::empty()).map_err(std::io::Error::from)?;
        let leaf = openat(
            &results,
            path.file_name()
                .ok_or_else(|| std::io::Error::other("missing Claude body filename"))?,
            OFlags::RDONLY | OFlags::NONBLOCK | OFlags::NOFOLLOW | OFlags::CLOEXEC,
            Mode::empty(),
        )
        .map_err(std::io::Error::from)?;
        Ok(File::from(leaf))
    };
    match open() {
        Ok(file) => Ok(Some(file)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e.into()),
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Companion {
    pub path: String,
    pub input: String,
    pub chain: i64,
    pub revision: u64,
    pub blob: BlobHash,
    pub length: u64,
    pub inline_duplicate: bool,
}
/// Verify an immutable snapshot using the caller's pinned archive transaction.
/// A normalized copy cannot substitute for a lost or changed retained witness.
pub(crate) fn read_pinned(
    connection: &rusqlite::Connection,
    root: &Path,
    body: &Companion,
) -> Result<Vec<u8>> {
    if body.length > MAX_BODY || body.chain != 1 || body.revision != 1 {
        return Err(Error::Range);
    }
    let file = root.join("archive.db");
    let valid: bool = db(&file, || {
        connection.query_row("SELECT EXISTS(SELECT 1 FROM input i JOIN revision r ON r.input=i.id AND r.number=1 JOIN extent e ON e.input=i.id AND e.chain=1 AND e.offset=0 JOIN blob b ON b.hash=e.blob WHERE i.id=?1 AND i.format='claude-code-tool-result' AND i.original_hash=?2 AND json_extract(i.context_json,'$[1]')=?3 AND r.chain=1 AND r.length=?4 AND r.blake3=?2 AND e.blob=?2 AND e.length=?4 AND e.blob_offset=0 AND b.size=?4 AND b.state='present') AND (SELECT count(*) FROM extent WHERE input=?1 AND chain=1)=1",rusqlite::params![body.input,body.blob.bytes().as_slice(),body.path,body.length as i64],|r|r.get(0))
    })?;
    if !valid {
        return Err(Error::Invariant(
            "required Claude retained body metadata unavailable".into(),
        ));
    }
    let reader = hashed_blobs::Reader::new(root, vec![]);
    let bytes = reader.read_range(&hashed_blobs::Hash(*body.blob.bytes()), 0, body.length)?;
    if BlobHash::of(&bytes) != body.blob {
        return Err(Error::Invariant(
            "required Claude retained body hash differs".into(),
        ));
    }
    Ok(bytes)
}
/// A sealed source descriptor pins the owning Message's exact witness. Other
/// origin formats retain their existing reader semantics.
pub(crate) fn verify_origin(
    connection: &rusqlite::Connection,
    root: &Path,
    bytes: &[u8],
) -> Result<()> {
    let Ok(value) = serde_json::from_slice::<serde_json::Value>(bytes) else {
        return Ok(());
    };
    if value.get("format").and_then(serde_json::Value::as_str)
        != Some("claude-code-tool-result-body")
    {
        return Ok(());
    }
    #[derive(Deserialize)]
    struct Binding {
        input: String,
        chain: i64,
    }
    #[derive(Deserialize)]
    struct Origin {
        binding: Binding,
        revision: u64,
        path: String,
        blob: BlobHash,
        start: u64,
        end: u64,
        codec_version: u64,
        transform: String,
    }
    let origin: Origin =
        serde_json::from_value(value).map_err(|e| Error::Invariant(e.to_string()))?;
    if origin.start != 0 || origin.codec_version != 1 || origin.transform != "utf8-tool-output-v1" {
        return Err(Error::Invariant("unsupported Claude body origin".into()));
    }
    read_pinned(
        connection,
        root,
        &Companion {
            path: origin.path,
            input: origin.binding.input,
            chain: origin.binding.chain,
            revision: origin.revision,
            blob: origin.blob,
            length: origin.end,
            inline_duplicate: false,
        },
    )?;
    Ok(())
}
pub(crate) fn manifest(
    connection: &rusqlite::Connection,
    file: &Path,
    main: &str,
) -> Result<Vec<Companion>> {
    let mut statement = db(file, || {
        connection.prepare("SELECT i.id,i.original_hash,r.length,CASE WHEN length(CAST(i.context_json AS BLOB))<=65536 THEN i.context_json END FROM input i JOIN revision r ON r.input=i.id AND r.number=1 WHERE i.format='claude-code-tool-result' AND json_extract(i.context_json,'$[0]')=?1 ORDER BY (SELECT max(o.rowid) FROM observation o WHERE o.input=i.id),i.id")
    })?;
    let mut rows = db(file, || statement.query([main]))?;
    let mut result = std::collections::BTreeMap::new();
    while let Some(row) = db(file, || rows.next())? {
        let id: String = db(file, || row.get(0))?;
        let hash: Vec<u8> = db(file, || row.get(1))?;
        let length: i64 = db(file, || row.get(2))?;
        let context: Option<String> = db(file, || row.get(3))?;
        let context = context.ok_or(Error::Range)?;
        if context.len() > 65536 {
            return Err(Error::Range);
        }
        let context: Vec<String> =
            serde_json::from_str(&context).map_err(|e| Error::Invariant(e.to_string()))?;
        InputFormat::ClaudeCodeToolResult
            .validate_context(&context.iter().map(String::as_str).collect::<Vec<_>>())?;
        let path = context[1].clone();
        result.insert(
            path.clone(),
            Companion {
                path,
                input: id,
                chain: 1,
                revision: 1,
                blob: BlobHash::from_bytes(hash.try_into().map_err(|_| Error::Range)?),
                length: u64::try_from(length).map_err(|_| Error::Range)?,
                inline_duplicate: context[2] == "inline-duplicate",
            },
        );
        if result.len() > MAX_COMPANIONS {
            return Err(Error::ObservedSourceRefused(
                "Claude companion count exceeds 32".into(),
            ));
        }
    }
    Ok(result.into_values().collect())
}
impl Writer {
    /// Called only for an explicitly named reference within this main's own
    /// tool-results directory. Missing provider copies preserve acquired bytes.
    /// All other open/read failures remain failures, including denied access.
    pub fn acquire_claude_body(
        &mut self,
        main: &str,
        main_path: &Path,
        path: &Path,
        observation: &ImportObservation<'_>,
        cancelled: &AtomicBool,
        inline: Option<(BlobHash, u64)>,
    ) -> Result<()> {
        self.check_import_acquisition()?;
        let Some(prepared) =
            self.blob_capture()?
                .prepare_claude_body(main_path, path, cancelled, inline, |_| Ok(()))?
        else {
            return Ok(());
        };
        self.retain_claude_body(main, prepared, observation)
    }
    pub fn retain_claude_body(
        &mut self,
        main: &str,
        prepared: PreparedClaudeBody,
        observation: &ImportObservation<'_>,
    ) -> Result<()> {
        self.check_import_acquisition()?;
        let PreparedClaudeBody {
            path,
            proof,
            inline,
        } = prepared;
        let duplicate = inline == Some((proof.hash(), proof.length()));
        let length = proof.length();
        let hash = self.retain_captured_blob(proof)?;
        let path = path.to_str().ok_or(Error::Range)?;
        let context = [
            main,
            path,
            if duplicate {
                "inline-duplicate"
            } else {
                "required"
            },
        ];
        if let Some(previous) = self.replay_immutable_import(
            InputFormat::ClaudeCodeToolResult,
            &context,
            observation,
            65536,
        )? {
            if previous.id
                != polylogue_model::InputId::imported(
                    &hash,
                    InputFormat::ClaudeCodeToolResult.name(),
                    &context,
                )
                .as_str()
            {
                return Err(Error::ClientRequestConflict {
                    id: observation.id.into(),
                });
            }
            return Ok(());
        }
        let previous = manifest(&self.archive, &self.root.join("archive.db"), main)?;
        if previous.iter().any(|body| {
            body.path == path
                && body.blob == hash
                && body.length == length
                && body.inline_duplicate == duplicate
        }) {
            return Ok(());
        }
        self.retain_immutable_input(
            hash,
            InputFormat::ClaudeCodeToolResult,
            &[
                main,
                path,
                if duplicate {
                    "inline-duplicate"
                } else {
                    "required"
                },
            ],
            observation,
            65536,
        )?;
        Ok(())
    }
}

pub struct PreparedClaudeBody {
    path: std::path::PathBuf,
    proof: crate::PreparedBlob,
    inline: Option<(BlobHash, u64)>,
}
impl crate::BlobCapture {
    pub(crate) fn prepare_claude_body(
        &self,
        main_path: &Path,
        path: &Path,
        cancelled: &AtomicBool,
        inline: Option<(BlobHash, u64)>,
        mut work: impl FnMut(u64) -> Result<()>,
    ) -> Result<Option<PreparedClaudeBody>> {
        let Some(mut file) = open_body(main_path, path)? else {
            return Ok(None);
        };
        let before = file.metadata()?;
        if !before.is_file() || before.len() > MAX_BODY {
            return Err(Error::ObservedSourceRefused(
                "Claude body requires a regular file of at most 1048576 bytes".into(),
            ));
        }
        let proof =
            self.capture_with_check((&mut file).take(before.len()), cancelled, &mut work)?;
        use std::io::Read;
        let after = file.metadata()?;
        let identity = |m: &std::fs::Metadata| {
            (
                m.dev(),
                m.ino(),
                m.len(),
                m.mtime(),
                m.mtime_nsec(),
                m.ctime(),
                m.ctime_nsec(),
            )
        };
        if identity(&before) != identity(&after) || proof.length() != before.len() {
            return Err(Error::ObservedSourceRefused(
                "Claude body changed during capture".into(),
            ));
        }
        Ok(Some(PreparedClaudeBody {
            path: path.to_path_buf(),
            proof,
            inline,
        }))
    }
}

impl ReadHandle {
    pub fn claude_body(&self, body: &Companion) -> Result<Vec<u8>> {
        if body.length > MAX_BODY {
            return Err(Error::Range);
        }
        let source = self.source(&body.input)?.ok_or(Error::UnknownBlob)?;
        if source.format != InputFormat::ClaudeCodeToolResult
            || source.binding != body.blob
            || source.input.chain != body.chain
            || source.revision != body.revision
            || source.input.length != body.length
            || source.context.get(1) != Some(&body.path)
        {
            return Err(Error::Invariant(
                "Claude retained companion basis differs".into(),
            ));
        }
        read_pinned(&self.archive, &self.root, body)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;
    #[test]
    fn body_opens_refuse_symlinks_in_each_appended_component() {
        use std::os::unix::fs::symlink;
        for component in ["session", "tool-results", "result.txt"] {
            let root = tempfile::tempdir().unwrap();
            let main = root.path().join("session.jsonl");
            let directory = root.path().join("session/tool-results");
            std::fs::create_dir_all(&directory).unwrap();
            let body = directory.join("result.txt");
            std::fs::write(&body, b"outside synthetic body").unwrap();
            let target = match component {
                "session" => root.path().join("session"),
                "tool-results" => directory.clone(),
                _ => body.clone(),
            };
            let outside = root.path().join("outside");
            std::fs::rename(&target, &outside).unwrap();
            symlink(&outside, &target).unwrap();
            assert!(open_body(&main, &body).is_err(), "{component}");
        }
    }
    #[test]
    fn configured_base_resolution_and_missing_appended_components_remain_supported() {
        use std::io::Read;
        use std::os::unix::fs::symlink;
        let root = tempfile::tempdir().unwrap();
        let base = root.path().join("base");
        std::fs::create_dir(&base).unwrap();
        let configured = root.path().join("configured");
        symlink(&base, &configured).unwrap();
        let main = configured.join("session.jsonl");
        let body = configured.join("session/tool-results/result.txt");
        assert!(open_body(&main, &body).unwrap().is_none());
        std::fs::create_dir(base.join("session")).unwrap();
        assert!(open_body(&main, &body).unwrap().is_none());
        std::fs::create_dir(base.join("session/tool-results")).unwrap();
        assert!(open_body(&main, &body).unwrap().is_none());
        std::fs::write(&body, b"inside configured authority").unwrap();
        let mut bytes = Vec::new();
        open_body(&main, &body)
            .unwrap()
            .unwrap()
            .read_to_end(&mut bytes)
            .unwrap();
        assert_eq!(bytes, b"inside configured authority");
    }
    #[test]
    fn snapshots_pin_latest_acquisition_and_retained_bytes_survive_provider_loss() {
        let root = tempfile::tempdir().unwrap();
        let archive = root.path().join("archive");
        let main_path = root.path().join("session.jsonl");
        let directory = root.path().join("session/tool-results");
        std::fs::create_dir_all(&directory).unwrap();
        let provider = directory.join("result.txt");
        let mut writer = Writer::open(&archive).unwrap();
        let stop = AtomicBool::new(false);
        for (request, bytes) in [
            ("a", b"first".as_slice()),
            ("b", b"second"),
            ("c", b"first"),
        ] {
            std::fs::write(&provider, bytes).unwrap();
            writer
                .acquire_claude_body(
                    "main",
                    &main_path,
                    &provider,
                    &ImportObservation {
                        id: request,
                        machine: "test",
                        host: "test",
                        path: "copy:synthetic",
                        seen_at: 1,
                    },
                    &stop,
                    None,
                )
                .unwrap();
            let pinned = manifest(&writer.archive, &archive.join("archive.db"), "main").unwrap();
            assert_eq!(pinned.len(), 1);
            assert_eq!(
                writer
                    .read_handle()
                    .unwrap()
                    .claude_body(&pinned[0])
                    .unwrap(),
                bytes
            );
        }
        let pinned = manifest(&writer.archive, &archive.join("archive.db"), "main")
            .unwrap()
            .pop()
            .unwrap();
        std::fs::remove_file(&provider).unwrap();
        writer
            .acquire_claude_body(
                "main",
                &main_path,
                &provider,
                &ImportObservation {
                    id: "missing",
                    machine: "test",
                    host: "test",
                    path: "copy:synthetic",
                    seen_at: 1,
                },
                &stop,
                None,
            )
            .unwrap();
        assert_eq!(
            writer.read_handle().unwrap().claude_body(&pinned).unwrap(),
            b"first"
        );
        writer
            .archive
            .execute(
                "UPDATE blob SET state='deleting' WHERE hash=?1",
                [pinned.blob.bytes().as_slice()],
            )
            .unwrap();
        assert!(writer.read_handle().unwrap().claude_body(&pinned).is_err());
    }
    #[test]
    fn failed_provider_reads_and_excisions_never_become_missing_or_empty() {
        let root = tempfile::tempdir().unwrap();
        let main_path = root.path().join("session.jsonl");
        let directory = root.path().join("session/tool-results");
        std::fs::create_dir_all(&directory).unwrap();
        let provider = directory.join("result.txt");
        let mut writer = Writer::open(&root.path().join("archive")).unwrap();
        let observation = ImportObservation {
            id: "a",
            machine: "test",
            host: "test",
            path: "copy:synthetic",
            seen_at: 1,
        };
        std::fs::write(&provider, b"private synthetic body").unwrap();
        std::fs::set_permissions(&provider, std::fs::Permissions::from_mode(0)).unwrap();
        assert!(
            writer
                .acquire_claude_body(
                    "main",
                    &main_path,
                    &provider,
                    &observation,
                    &AtomicBool::new(false),
                    None
                )
                .is_err()
        );
        std::fs::set_permissions(&provider, std::fs::Permissions::from_mode(0o600)).unwrap();
        writer.archive.execute("INSERT INTO excision(id,scope,created_at,local_seq) VALUES('x','session:synthetic',1,1)",[]).unwrap();
        assert!(matches!(
            writer.acquire_claude_body(
                "main",
                &main_path,
                &provider,
                &observation,
                &AtomicBool::new(false),
                None
            ),
            Err(Error::ExcisionCheckRequired)
        ));
    }
}
