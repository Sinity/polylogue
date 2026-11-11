//! The predecessor's source catalog (`--format predecessor-archive`): a
//! folder with `source.db` and `blob/`. Each raw session becomes an input
//! read by its origin's parser, with its source facts and append lineage;
//! every row of every table of `source.db` is kept verbatim, and every blob
//! no raw session cites (attachments, sidecars, hook payloads) is kept with
//! its bytes checked against its name.
use super::*;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::io::Read;

/// Stages and delivers one member at a time. The consumer commits and parses
/// it before another member is prepared, so a stopped import keeps its progress.
pub(super) fn visit(
    root: &Path,
    archive: &Archive,
    mut accept: impl FnMut(package::Member) -> std::io::Result<()>,
) -> std::io::Result<bool> {
    let Some(catalog) = polylogue_store::legacy_import::Catalog::open(root, &archive.staging_dir())
        .map_err(std::io::Error::other)?
    else {
        return Ok(false);
    };
    let (rows, _) = archive
        .prepare_write(|out| catalog.write_rows(out))
        .map_err(std::io::Error::other)?;
    let inputs = catalog.inputs().map_err(std::io::Error::other)?;
    let mut declared = BTreeMap::new();
    for input in &inputs {
        if declared.insert(input.raw_id.clone(), input).is_some() {
            return Err(std::io::Error::other(
                "predecessor catalog repeats a raw_id",
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
    accept(package::Member {
        path: PathBuf::from(format!("{}!/source.db/rows.jsonl", root.display())),
        bytes: rows,
        container: false,
        origin: polylogue_sources::origin::by_name(KEPT),
        parent: None,
        container_kind: None,
        member_path_bytes: None,
        refusal: None,
    })?;
    for input in &inputs {
        let name = Path::new(&input.source_path)
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("input");
        if input.revision_kind == "append" {
            accept(package::Member {
                path: PathBuf::from(format!(
                    "{}!/raw_sessions/{}/fragment",
                    root.display(),
                    input.raw_id
                )),
                bytes: archive
                    .prepare_write(|out| copy_blob(root, input, out))
                    .map(|(bytes, ())| bytes)
                    .map_err(std::io::Error::other)?,
                container: false,
                origin: polylogue_sources::origin::by_name(KEPT),
                parent: None,
                container_kind: None,
                member_path_bytes: None,
                refusal: None,
            })?;
        }
        let at = PathBuf::from(format!(
            "{}!/raw_sessions/{}/{name}",
            root.display(),
            input.raw_id
        ));
        let chain = match lineage(input, &declared, &by_revision) {
            Ok(chain) => chain,
            Err(reason) => {
                // The fragment above keeps the bytes; only the joined
                // reading is refused.
                accept(package::Member {
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
                })?;
                continue;
            }
        };
        let (combined, ()) = archive
            .prepare_write(|output| {
                for part in &chain {
                    copy_blob(root, part, output)?;
                }
                Ok(())
            })
            .map_err(std::io::Error::other)?;
        let mut reader = combined.reader().map_err(std::io::Error::other)?;
        let mut head = [0; 64];
        let n = reader.read(&mut head)?;
        // An origin the predecessor could not name ("unknown-export") is
        // read by the declared structure its bytes have, or kept unparsed.
        let spec = if polylogue_sources::browser_capture::recognize(&at, &head[..n]) {
            polylogue_sources::origin::by_name("browser-capture")
        } else if let Some(name) = polylogue_store::predecessor::origin(&input.origin) {
            polylogue_sources::origin::by_name(name)
        } else {
            let mut bytes = combined.reader().map_err(std::io::Error::other)?;
            super::recognition::declared(&at, &mut bytes, None)?
        }
        .or_else(|| polylogue_sources::origin::by_name(KEPT));
        accept(package::Member {
            path: at,
            bytes: combined,
            container: false,
            origin: spec,
            parent: None,
            container_kind: None,
            member_path_bytes: None,
            refusal: None,
        })?;
    }
    // Every blob no raw session cites, checked against its name.
    let cited: BTreeSet<String> = inputs
        .iter()
        .map(|input| input.blob_hash.iter().map(|b| format!("{b:02x}")).collect())
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
                    let actual: String = actual.iter().map(|b| format!("{b:02x}")).collect();
                    if actual != name {
                        return Err(polylogue_store::StoreError::Invalid(format!(
                            "predecessor blob {name} differs from its name"
                        )));
                    }
                    Ok(())
                })
                .map_err(std::io::Error::other)?;
            accept(package::Member {
                path: PathBuf::from(format!("{}!/blob/{name}", root.display())),
                bytes,
                container: false,
                origin: polylogue_sources::origin::by_name(KEPT),
                parent: None,
                container_kind: None,
                member_path_bytes: None,
                refusal: None,
            })?;
        }
    }
    Ok(true)
}
/// Validate the exact source bytes while producing this member. No staged
/// copy of every source blob is retained alongside the joined readings.
fn copy_blob(
    root: &Path,
    input: &Input,
    output: &mut dyn std::io::Write,
) -> polylogue_store::Result<()> {
    let hash: String = input.blob_hash.iter().map(|b| format!("{b:02x}")).collect();
    let path = root.join("blob").join(&hash[..2]).join(&hash[2..]);
    if !std::fs::symlink_metadata(&path)?.is_file() {
        return Err(polylogue_store::StoreError::Invalid(format!(
            "non-regular predecessor blob {hash}"
        )));
    }
    let mut source = std::fs::File::open(path)?;
    let mut actual = Sha256::new();
    let mut length = 0;
    let mut buffer = [0; 64 * 1024];
    loop {
        let n = source.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        actual.update(&buffer[..n]);
        output.write_all(&buffer[..n])?;
        length += n as u64;
    }
    let actual: [u8; 32] = actual.finalize().into();
    if length != input.blob_size || actual != input.blob_hash {
        return Err(polylogue_store::StoreError::Invalid(format!(
            "predecessor blob {hash} differs from its catalog hash or size"
        )));
    }
    Ok(())
}

#[cfg(test)]
fn stage(root: &Path, archive: &Archive) -> std::io::Result<Option<Vec<package::Member>>> {
    let mut members = Vec::new();
    let found = visit(root, archive, |member| {
        members.push(member);
        Ok(())
    })?;
    Ok(found.then_some(members))
}

type Input = polylogue_store::legacy_import::Input;

/// The raw sessions whose bytes, in order, make `input`'s source revision:
/// its full baseline, then each append. An append names the revision it
/// continues by row ID, or else by revision name; its recorded offsets must
/// join the bytes before it. Otherwise the reason it cannot be joined.
fn lineage<'a>(
    input: &'a Input,
    declared: &BTreeMap<String, &'a Input>,
    by_revision: &BTreeMap<&str, Option<&'a Input>>,
) -> Result<Vec<&'a Input>, String> {
    let mut chain = Vec::new();
    let mut seen = BTreeSet::new();
    let mut current = input;
    while current.revision_kind == "append" {
        if !seen.insert(current.raw_id.as_str()) {
            return Err("its append lineage has a cycle".into());
        }
        chain.push(current);
        current = match (&current.predecessor, &current.predecessor_revision) {
            (Some(id), _) => declared
                .get(id)
                .copied()
                .ok_or_else(|| format!("it continues raw session {id}, which the catalog does not hold"))?,
            (None, Some(revision)) => by_revision
                .get(revision.as_str())
                .copied()
                .flatten()
                .ok_or_else(|| format!("it continues revision {revision}, which no one raw session of the catalog is"))?,
            (None, None) => return Err("it is an append with no recorded predecessor".into()),
        };
        if current.origin != input.origin {
            return Err("its append lineage changes origin".into());
        }
    }
    chain.push(current);
    chain.reverse();
    let mut length = chain[0].blob_size;
    for append in &chain[1..] {
        if append.append_start != Some(length)
            || append.append_end != length.checked_add(append.blob_size)
        {
            return Err(format!(
                "the recorded offsets of append {} do not join the bytes before it",
                append.raw_id
            ));
        }
        length += append.blob_size;
    }
    Ok(chain)
}

#[cfg(test)]
mod tests;
