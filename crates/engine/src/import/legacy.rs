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

pub(super) fn stage(
    root: &Path,
    archive: &Archive,
) -> std::io::Result<Option<Vec<package::Member>>> {
    let Some(catalog) = polylogue_store::legacy_import::Catalog::open(root, &archive.staging_dir())
        .map_err(std::io::Error::other)?
    else {
        return Ok(None);
    };
    let (rows, _) = archive
        .prepare_write(|out| catalog.write_rows(out))
        .map_err(std::io::Error::other)?;
    let inputs = catalog.inputs().map_err(std::io::Error::other)?;
    let mut declared = BTreeMap::new();
    let mut blobs = BTreeMap::new();
    for input in &inputs {
        if declared.insert(input.raw_id.clone(), input).is_some() {
            return Err(std::io::Error::other(
                "predecessor catalog repeats a raw_id",
            ));
        }
        let hash = input
            .blob_hash
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>();
        let path = root.join("blob").join(&hash[..2]).join(&hash[2..]);
        if !std::fs::symlink_metadata(&path).is_ok_and(|metadata| metadata.is_file()) {
            return Err(std::io::Error::other(format!(
                "missing or non-regular predecessor blob {hash}"
            )));
        }
        let mut source = std::fs::File::open(&path).map_err(|error| {
            std::io::Error::other(format!("missing predecessor blob {hash}: {error}"))
        })?;
        let (copy, ()) = archive
            .prepare_write(|output| {
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
            })
            .map_err(std::io::Error::other)?;
        blobs.insert(input.raw_id.clone(), copy);
    }
    let mut members = Vec::new();
    members.push(package::Member {
        path: PathBuf::from(format!("{}!/source.db/rows.jsonl", root.display())),
        bytes: rows,
        container: false,
        origin: polylogue_sources::origin::by_name(KEPT),
        parent: None,
        container_kind: None,
        member_path_bytes: None,
    });
    for input in &inputs {
        let name = Path::new(&input.source_path)
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("input");
        if input.revision_kind == "append" {
            members.push(package::Member {
                path: PathBuf::from(format!(
                    "{}!/raw_sessions/{}/fragment",
                    root.display(),
                    input.raw_id
                )),
                bytes: archive
                    .prepare_bytes(
                        blobs[&input.raw_id]
                            .reader()
                            .map_err(std::io::Error::other)?,
                    )
                    .map_err(std::io::Error::other)?,
                container: false,
                origin: polylogue_sources::origin::by_name(KEPT),
                parent: None,
                container_kind: None,
                member_path_bytes: None,
            });
        }
        let mut chain = Vec::new();
        let mut seen = BTreeSet::new();
        let mut current = input;
        while current.revision_kind == "append" {
            if !seen.insert(current.raw_id.clone()) {
                return Err(std::io::Error::other(
                    "predecessor append lineage has a cycle",
                ));
            }
            chain.push(current);
            let predecessor = current.predecessor.as_ref().ok_or_else(|| {
                std::io::Error::other("predecessor append has no recorded predecessor")
            })?;
            current = declared.get(predecessor).copied().ok_or_else(|| {
                std::io::Error::other("predecessor append names a missing revision")
            })?;
            if current.origin != input.origin {
                return Err(std::io::Error::other("predecessor append changes origin"));
            }
        }
        let (combined, ()) = archive
            .prepare_write(|output| {
                let mut length = std::io::copy(&mut blobs[&current.raw_id].reader()?, output)?;
                for append in chain.into_iter().rev() {
                    if append.append_start != Some(length)
                        || append.append_end != length.checked_add(append.blob_size)
                    {
                        return Err(polylogue_store::StoreError::Invalid(
                            "predecessor append offsets do not join its recorded bytes".into(),
                        ));
                    }
                    length += std::io::copy(&mut blobs[&append.raw_id].reader()?, output)?;
                }
                Ok(())
            })
            .map_err(std::io::Error::other)?;
        let mut reader = combined.reader().map_err(std::io::Error::other)?;
        let mut head = [0; 64];
        let n = reader.read(&mut head)?;
        let at = PathBuf::from(format!(
            "{}!/raw_sessions/{}/{name}",
            root.display(),
            input.raw_id
        ));
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
        members.push(package::Member {
            path: at,
            bytes: combined,
            container: false,
            origin: spec,
            parent: None,
            container_kind: None,
            member_path_bytes: None,
        });
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
            members.push(package::Member {
                path: PathBuf::from(format!("{}!/blob/{name}", root.display())),
                bytes,
                container: false,
                origin: polylogue_sources::origin::by_name(KEPT),
                parent: None,
                container_kind: None,
                member_path_bytes: None,
            });
        }
    }
    Ok(Some(members))
}
#[cfg(test)]
mod tests;
