//! Python Polylogue's source catalog (`--format python-archive`): a
//! folder with `source.db` and `blob/`. Each raw session becomes an input
//! read by its origin's parser, with its source facts and append lineage.
//! The exact catalog files and any blob no raw session cites are also kept.
use super::*;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::io::Read;

enum Pending {
    Staged(Box<package::Member>),
    Checked(CheckedMember),
}

impl Pending {
    fn checked(&self) -> bool {
        matches!(self, Self::Checked(_))
    }

    fn bytes(&self) -> (polylogue_model::BlobHash, u64, bool) {
        match self {
            Self::Staged(member) => (
                member.bytes.hash(),
                member.bytes.len(),
                member.origin.is_some(),
            ),
            Self::Checked(member) => (member.hash, member.length, true),
        }
    }
}

impl Engine {
    /// Only a complete native message stream from a receiver-verified catalog
    /// member can be read independently of its wrapping filename.
    pub(crate) fn stored_import_origin(
        &self,
        reader: Option<&polylogue_store::Reader<'_>>,
        path: &Path,
        origin: &'static OriginSpec,
        input: &str,
        chain: i64,
        length: u64,
    ) -> Result<&'static OriginSpec, EngineError> {
        if origin.name != "claude-code" {
            return Ok(origin);
        }
        let Some((root, path_text)) = wrapped_transcript(path) else {
            return Ok(origin);
        };
        let Ok(hash) = polylogue_model::BlobHash::parse(input) else {
            return Ok(origin);
        };
        let acquired;
        let reader = match reader {
            Some(reader) => reader,
            None => {
                acquired = self.reads.reader()?;
                &acquired
            }
        };
        let receipts = reader.import_receipts(input, path_text, origin.name, path_text)?;
        let verified = receipts.into_iter().any(|(machine, receipt)| {
            let identity =
                |author| import_identity(&machine, root, author, std::iter::once((path, hash)));
            receipt.version == 1
                && (receipt.import_id == identity(receipt.authorship.as_ref())
                    || receipt.authorship == Some(default_import_authorship())
                        && receipt.import_id == identity(None))
        });
        if verified {
            let stored = reader.stored_chain(input, chain, length)?;
            let body = stored.reader(Arc::clone(&self.blobs));
            if polylogue_sources::claude_code::wire_messages(std::io::BufReader::new(body))? {
                return Ok(&polylogue_sources::origin::CLAUDE_CODE_WIRE_MESSAGES);
            }
        }
        Ok(origin)
    }

    pub(super) fn import_python_archive(
        &mut self,
        root: &Path,
        options: &ImportOptions,
        report: &mut ImportReport,
    ) -> Result<(), EngineError> {
        let archive = self.archive.clone();
        let import_locations = (!options.explain && !archive.is_primary())
            .then(|| polylogue_store::python_archive::import_locations(&archive))
            .transpose()?;
        let mut ids = Vec::new();
        let mut package = ImportPackage {
            path: root.to_path_buf(),
            ..Default::default()
        };
        let mut pending: Vec<Pending> = Vec::new();
        let mut pending_bytes = 0_u64;
        const MEMBERS: usize = 64;
        const BYTES: u64 = 64 << 20;
        let found = visit(root, &archive, |item| {
            let item = match item {
                Member::Prepared(member) => Pending::Staged(member),
                Member::Raw(raw) => {
                    // A path match saves no verification and acknowledges no bytes.
                    let predicted = if let Some(imported_at) = &import_locations
                        && raw
                            .origin
                            .is_some_and(|origin| origin.name != "polylogue-bundle")
                    {
                        imported_at(
                            &self.config.machine,
                            raw.path.as_os_str().as_encoded_bytes(),
                        )
                        .map_err(std::io::Error::other)?
                    } else {
                        false
                    };
                    if predicted {
                        Pending::Checked(raw.check(root)?)
                    } else {
                        Pending::Staged(Box::new(raw.prepare(root, &archive)?))
                    }
                }
            };
            let (hash, length, accepted) = item.bytes();
            // Fresh bytes and predicted repeats use separate bounded groups.
            // On a miss, the entire predicted group takes ordinary admission.
            if pending.len() == MEMBERS
                || pending_bytes.saturating_add(length) > BYTES
                || pending
                    .first()
                    .is_some_and(|previous| previous.checked() != item.checked())
            {
                self.flush_python_members(root, &mut pending, options, report, &mut package)?;
                pending_bytes = 0;
            }
            if accepted {
                ids.push(hash.to_string());
            }
            pending.push(item);
            pending_bytes += length;
            if pending_bytes >= BYTES || archive.is_primary() {
                self.flush_python_members(root, &mut pending, options, report, &mut package)?;
                pending_bytes = 0;
            }
            Ok(())
        });
        // A later damaged source must not discard the verified prefix.
        self.flush_python_members(root, &mut pending, options, report, &mut package)?;
        if !found? {
            recognition::refuse(
                report,
                root,
                "not Python Polylogue's archive: it needs source.db and blob/",
            );
            return Ok(());
        }
        if !options.explain {
            let counts = self.reads.reader()?.import_counts(&ids)?;
            package.sessions = counts.sessions;
            package.documents = counts.documents;
        }
        report.packages.push(package);
        Ok(())
    }

    fn flush_python_members(
        &mut self,
        root: &Path,
        pending: &mut Vec<Pending>,
        options: &ImportOptions,
        report: &mut ImportReport,
        package: &mut ImportPackage,
    ) -> std::io::Result<()> {
        if pending.is_empty() {
            return Ok(());
        }
        let members = std::mem::take(pending);
        let before = report.pass.errors.len();
        let confirmed = if members[0].checked() {
            self.confirm_python_members(root, &members, options, report, package)
                .map_err(std::io::Error::other)?
        } else {
            false
        };
        if !confirmed {
            let mut staged = Vec::with_capacity(members.len());
            let prepared = members.into_iter().try_for_each(|member| {
                staged.push(match member {
                    Pending::Staged(member) => *member,
                    Pending::Checked(member) => member.raw.prepare(root, &self.archive)?,
                });
                Ok::<(), std::io::Error>(())
            });
            if !staged.is_empty() {
                self.import_members(root, staged, options, report)
                    .map_err(std::io::Error::other)?;
                package.assets += report
                    .packages
                    .pop()
                    .expect("imported member package")
                    .assets;
            }
            prepared?;
        }
        if report.pass.errors.len() > before {
            return Err(std::io::Error::other(
                report.pass.errors[before..].join("; "),
            ));
        }
        Ok(())
    }

    fn confirm_python_members(
        &mut self,
        root: &Path,
        group: &[Pending],
        options: &ImportOptions,
        report: &mut ImportReport,
        package: &mut ImportPackage,
    ) -> Result<bool, EngineError> {
        let mut metadata = Vec::with_capacity(group.len());
        let mut batches = Vec::with_capacity(group.len());
        for member in group {
            let Pending::Checked(member) = member else {
                return Ok(false);
            };
            let path = &member.raw.path;
            let id = import_identity(
                &self.config.machine,
                root,
                options.authorship.as_ref(),
                std::iter::once((path.as_path(), member.hash)),
            );
            let receipt = import_member(root, path, false, id, options);
            let mut batch = self.import_batch(
                path,
                member.origin,
                Some(receipt),
                None,
                member.hash,
                member.length,
            );
            batch.new_blob = Some(polylogue_store::NewBlob::whole(member.hash, member.length));
            metadata.push((path.clone(), member.origin, None, batch.input.clone()));
            batches.push(batch);
        }
        let Some(stored) = self.writer.confirm_imports(batches)? else {
            return Ok(false);
        };
        for (path, origin, _, _) in &metadata {
            classify_accepted(report, package, path, origin, false);
        }
        self.finish_import_group(metadata, stored.into_iter().map(Ok).collect(), report)?;
        Ok(true)
    }
}

fn wrapped_transcript(path: &Path) -> Option<(&Path, &str)> {
    let at = path.to_str()?;
    let prefix = at.strip_suffix(".jsonl.txt.json")?;
    let (root, member) = prefix.rsplit_once("!/raw_sessions/")?;
    let (raw_id, name) = member.split_once('/')?;
    if root.is_empty() || matches!(raw_id, "" | "." | "..") || name.is_empty() || name.contains('/')
    {
        return None;
    }
    Some((Path::new(root), at))
}

/// Visit the native catalog, each fragment and its declared joined reading in
/// order. A rejected append still has its independent original fragment.
fn visit(
    root: &Path,
    archive: &Archive,
    mut accept: impl FnMut(Member) -> std::io::Result<()>,
) -> std::io::Result<bool> {
    let Some(polylogue_store::python_archive::Catalog {
        database,
        wal,
        inputs,
    }) = polylogue_store::python_archive::Catalog::capture(root, archive)
        .map_err(std::io::Error::other)?
    else {
        return Ok(false);
    };
    let mut declared = BTreeMap::new();
    for input in &inputs {
        if declared.insert(input.raw_id.clone(), input).is_some() {
            return Err(std::io::Error::other(
                "Python Polylogue catalog repeats a raw_id",
            ));
        }
    }
    // Revision names given to exactly one raw session.
    let mut by_revision: BTreeMap<&str, Option<&Input>> = BTreeMap::new();
    for input in &inputs {
        if let Some(revision) = input.source_revision.as_deref() {
            by_revision
                .entry(revision)
                .and_modify(|held| *held = None)
                .or_insert(Some(input));
        }
    }
    let mut catalog = blake3::Hasher::new_derive_key("polylogue Python catalog snapshot v1");
    catalog.update(&database.hash().0);
    if let Some(wal) = &wal {
        catalog.update(&wal.hash().0);
    }
    let catalog = catalog.finalize();
    for (name, bytes) in
        std::iter::once(("source.db", database)).chain(wal.map(|bytes| ("source.db-wal", bytes)))
    {
        accept(Member::Prepared(Box::new(package::Member {
            path: PathBuf::from(format!("{}!/catalog/{catalog}/{name}", root.display())),
            bytes,
            container: false,
            origin: polylogue_sources::origin::by_name(KEPT),
            parent: None,
            container_kind: None,
            member_path_bytes: None,
            refusal: None,
        })))?;
    }
    for input in &inputs {
        let name = Path::new(&input.source_path)
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("input");
        if input.revision_kind == "append" {
            accept(Member::Raw(RawMember {
                path: PathBuf::from(format!(
                    "{}!/raw_sessions/{}/fragment",
                    root.display(),
                    input.raw_id
                )),
                parts: vec![Blob::from(input)],
                origin: polylogue_sources::origin::by_name(KEPT),
            }))?;
        }
        let at = PathBuf::from(format!(
            "{}!/raw_sessions/{}/{name}",
            root.display(),
            input.raw_id
        ));
        let parts = match lineage(root, input, &declared, &by_revision) {
            Ok(chain) => chain,
            Err(reason) => {
                // The fragment above keeps the bytes; only the joined
                // reading is refused.
                accept(Member::Prepared(Box::new(package::Member {
                    path: at,
                    bytes: archive
                        .prepare_bytes(std::io::empty())
                        .map_err(std::io::Error::other)?,
                    container: false,
                    origin: None,
                    parent: None,
                    container_kind: None,
                    member_path_bytes: None,
                    refusal: Some(reason),
                })))?;
                continue;
            }
        };
        accept(Member::Raw(RawMember {
            path: at,
            parts,
            origin: polylogue_store::python_polylogue::origin(&input.origin)
                .and_then(polylogue_sources::origin::by_name),
        }))?;
    }
    // Every blob no raw session cites, checked against its name.
    let cited: BTreeSet<String> = inputs
        .iter()
        .map(|input| polylogue_model::hex(&input.blob_hash))
        .collect();
    let mut prefixes: Vec<_> =
        std::fs::read_dir(root.join("blob"))?.collect::<std::io::Result<Vec<_>>>()?;
    prefixes.sort_by_key(|entry| entry.file_name());
    for prefix in prefixes {
        if !prefix.file_type()?.is_dir() {
            continue;
        }
        let mut files: Vec<_> =
            std::fs::read_dir(prefix.path())?.collect::<std::io::Result<Vec<_>>>()?;
        files.sort_by_key(|entry| entry.file_name());
        for file in files {
            let name = format!(
                "{}{}",
                prefix.file_name().to_string_lossy(),
                file.file_name().to_string_lossy()
            );
            if cited.contains(&name) || !file.file_type()?.is_file() {
                continue;
            }
            let mut source = std::fs::File::open(file.path())?;
            let (bytes, ()) = archive
                .prepare_write(|output| {
                    let mut actual = Sha256::new();
                    let mut buffer = [0; 64 * 1024];
                    loop {
                        let n = source.read(&mut buffer)?;
                        if n == 0 {
                            break;
                        }
                        actual.update(&buffer[..n]);
                        output.write_all(&buffer[..n])?;
                    }
                    let actual: [u8; 32] = actual.finalize().into();
                    let actual: String = polylogue_model::hex(&actual);
                    if actual != name {
                        return Err(polylogue_store::StoreError::invalid(format!(
                            "Python Polylogue blob {name} differs from its name"
                        )));
                    }
                    Ok(())
                })
                .map_err(std::io::Error::other)?;
            accept(Member::Prepared(Box::new(package::Member {
                path: PathBuf::from(format!("{}!/blob/{name}", root.display())),
                bytes,
                container: false,
                origin: polylogue_sources::origin::by_name(KEPT),
                parent: None,
                container_kind: None,
                member_path_bytes: None,
                refusal: None,
            })))?;
        }
    }
    Ok(true)
}
enum Member {
    Prepared(Box<package::Member>),
    Raw(RawMember),
}

struct Blob {
    hash: [u8; 32],
    size: u64,
    header: Vec<u8>,
}

impl From<&Input> for Blob {
    fn from(input: &Input) -> Self {
        Self {
            hash: input.blob_hash,
            size: input.blob_size,
            header: Vec::new(),
        }
    }
}

struct RawMember {
    path: PathBuf,
    parts: Vec<Blob>,
    origin: Option<&'static OriginSpec>,
}

struct CheckedMember {
    raw: RawMember,
    hash: polylogue_model::BlobHash,
    length: u64,
    origin: &'static OriginSpec,
}

impl RawMember {
    fn read(
        &self,
        root: &Path,
        mut write: impl FnMut(&[u8]) -> std::io::Result<()>,
    ) -> polylogue_store::Result<Vec<u8>> {
        let mut head = Vec::with_capacity(64);
        for part in &self.parts {
            copy_blob(root, part, |bytes| {
                let take = bytes.len().min(64 - head.len());
                head.extend_from_slice(&bytes[..take]);
                write(bytes)
            })?;
        }
        Ok(head)
    }

    fn named_origin(&self, head: &[u8]) -> Option<&'static OriginSpec> {
        if polylogue_sources::browser_capture::recognize(&self.path, head) {
            polylogue_sources::origin::by_name("browser-capture")
        } else {
            self.origin
        }
    }

    fn check(self, root: &Path) -> std::io::Result<CheckedMember> {
        let mut hash = blake3::Hasher::new();
        let mut length = 0;
        let head = self
            .read(root, |bytes| {
                hash.update(bytes);
                length += bytes.len() as u64;
                Ok(())
            })
            .map_err(std::io::Error::other)?;
        let origin = self
            .named_origin(&head)
            .expect("only named origins are checked for a repeat");
        Ok(CheckedMember {
            raw: self,
            hash: polylogue_model::BlobHash(*hash.finalize().as_bytes()),
            length,
            origin,
        })
    }

    fn prepare(self, root: &Path, archive: &Archive) -> std::io::Result<package::Member> {
        #[cfg(test)]
        polylogue_store::fault::check("python_archive.prepare_raw", &self.path.to_string_lossy())
            .map_err(std::io::Error::other)?;
        let (bytes, head) = archive
            .prepare_write(|output| self.read(root, |part| output.write_all(part)))
            .map_err(std::io::Error::other)?;
        let origin = match self.named_origin(&head) {
            Some(origin) => Some(origin),
            None => {
                let mut reader = bytes.reader().map_err(std::io::Error::other)?;
                super::recognition::declared(&self.path, &mut reader, None)?
                    .or_else(|| polylogue_sources::origin::by_name(KEPT))
            }
        };
        Ok(package::Member {
            path: self.path,
            bytes,
            container: false,
            origin,
            parent: None,
            container_kind: None,
            member_path_bytes: None,
            refusal: None,
        })
    }
}

/// Validate the source bytes while producing this member. No staged
/// copy of every source blob is kept alongside the joined readings.
fn copy_blob(
    root: &Path,
    blob: &Blob,
    mut write: impl FnMut(&[u8]) -> std::io::Result<()>,
) -> polylogue_store::Result<()> {
    let hash: String = polylogue_model::hex(&blob.hash);
    let path = blob_path(root, &hash);
    if !std::fs::symlink_metadata(&path)?.is_file() {
        return Err(polylogue_store::StoreError::invalid(format!(
            "non-regular Python Polylogue blob {hash}"
        )));
    }
    let mut source = std::fs::File::open(path)?;
    let mut actual = Sha256::new();
    let mut length = blob.header.len() as u64;
    if !blob.header.is_empty() {
        let mut header = vec![0; blob.header.len()];
        source.read_exact(&mut header)?;
        if header != blob.header {
            return Err(polylogue_store::StoreError::invalid(
                "the Codex append header changed during import",
            ));
        }
        actual.update(&header);
    }
    let mut buffer = [0; 64 * 1024];
    loop {
        let n = source.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        actual.update(&buffer[..n]);
        write(&buffer[..n])?;
        length += n as u64;
    }
    let actual: [u8; 32] = actual.finalize().into();
    if length != blob.size || actual != blob.hash {
        return Err(polylogue_store::StoreError::invalid(format!(
            "Python Polylogue blob {hash} differs from its catalog hash or size"
        )));
    }
    Ok(())
}

fn blob_path(root: &Path, hash: &str) -> PathBuf {
    root.join("blob").join(&hash[..2]).join(&hash[2..])
}

#[cfg(test)]
fn stage(root: &Path, archive: &Archive) -> std::io::Result<Option<Vec<package::Member>>> {
    let mut members = Vec::new();
    let found = visit(root, archive, |member| {
        members.push(match member {
            Member::Prepared(member) => *member,
            Member::Raw(raw) => raw.prepare(root, archive)?,
        });
        Ok(())
    })?;
    Ok(found.then_some(members))
}

type Input = polylogue_store::python_archive::Input;

/// The raw sessions whose bytes, in order, make `input`'s source revision:
/// its full baseline, then each append. An append names the revision it
/// continues by row ID, or else by revision name; its recorded offsets must
/// join the bytes before it. Otherwise the reason it cannot be joined.
fn lineage<'a>(
    root: &Path,
    input: &'a Input,
    declared: &BTreeMap<String, &'a Input>,
    by_revision: &BTreeMap<&str, Option<&'a Input>>,
) -> Result<Vec<Blob>, String> {
    let mut chain = Vec::new();
    let mut seen = BTreeSet::new();
    let mut current = input;
    while current.revision_kind == "append" {
        if !seen.insert(current.raw_id.as_str()) {
            return Err("its append lineage has a cycle".into());
        }
        chain.push(current);
        current = match (&current.previous, &current.previous_revision) {
            (Some(id), _) => declared
                .get(id)
                .copied()
                .ok_or_else(|| format!("it continues raw session {id}, which the catalog does not hold"))?,
            (None, Some(revision)) => by_revision
                .get(revision.as_str())
                .copied()
                .flatten()
                .ok_or_else(|| format!("it continues revision {revision}, which no one raw session of the catalog is"))?,
            (None, None) => return Err("it is an append with no recorded previous revision".into()),
        };
        if current.origin != input.origin {
            return Err("its append lineage changes origin".into());
        }
    }
    chain.push(current);
    chain.reverse();
    let baseline = chain[0];
    let mut length = baseline.blob_size;
    let mut parts = vec![Blob::from(baseline)];
    for append in &chain[1..] {
        let offset_error = || {
            format!(
                "the recorded offsets of append {} do not join the bytes before it",
                append.raw_id
            )
        };
        let end = append
            .append_end
            .filter(|end| append.append_start == Some(length) && *end >= length)
            .ok_or_else(offset_error)?;
        let mut part = Blob::from(*append);
        if end - length != part.size {
            part.header = codex_append_header(root, baseline, append, end - length)
                .ok_or_else(offset_error)?;
        }
        length = end;
        parts.push(part);
    }
    Ok(parts)
}

/// The Python capture prepended an identity-only line to some Codex tails.
/// Only the exact envelope for the declared rollout is outside its byte range.
fn codex_append_header(
    root: &Path,
    baseline: &Input,
    append: &Input,
    length: u64,
) -> Option<Vec<u8>> {
    if append.origin != "codex-session" {
        return None;
    }
    let rollout = |input: &Input| match polylogue_sources::codex::File::from_path(Path::new(
        &input.source_path,
    )) {
        Some(polylogue_sources::codex::File::Rollout { thread }) => thread,
        _ => None,
    };
    let thread = rollout(append)?;
    if rollout(baseline).as_ref() != Some(&thread)
        || !thread.bytes().enumerate().all(|(index, byte)| {
            if matches!(index, 8 | 13 | 18 | 23) {
                byte == b'-'
            } else {
                byte.is_ascii_hexdigit()
            }
        })
    {
        return None;
    }
    let expected =
        format!("{{\"type\":\"session_meta\",\"payload\":{{\"id\":\"{thread}\"}}}}\n").into_bytes();
    if append.blob_size.checked_sub(length)? != expected.len() as u64 {
        return None;
    }
    let path = blob_path(root, &polylogue_model::hex(&append.blob_hash));
    if !std::fs::symlink_metadata(&path).ok()?.is_file() {
        return None;
    }
    let mut actual = vec![0; expected.len()];
    std::fs::File::open(path)
        .ok()?
        .read_exact(&mut actual)
        .ok()?;
    (actual == expected).then_some(expected)
}

#[cfg(test)]
mod tests;
