//! The predecessor's catalog supplies origin and append lineage. Only its
//! declared byte blobs and source facts cross this boundary.
use super::*;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::io::Read;

pub(super) fn stage(
    root: &Path,
    archive: &Archive,
) -> std::io::Result<Option<Vec<package::Member>>> {
    let (catalog, inputs) = archive
        .prepare_write(|mut catalog| {
            polylogue_store::legacy_import::inputs(root, |row| {
                serde_json::to_writer(&mut catalog, row)?;
                std::io::Write::write_all(&mut catalog, b"\n")?;
                Ok(())
            })
        })
        .map_err(std::io::Error::other)?;
    let Some(inputs) = inputs else {
        return Ok(None);
    };
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
        path: PathBuf::from(format!("{}!/raw_sessions/catalog.jsonl", root.display())),
        bytes: catalog,
        container: false,
        origin: polylogue_sources::origin::by_name(KEPT),
        parent: None,
        container_kind: None,
        member_path_bytes: None,
    });
    for input in &inputs {
        let format = origin(&input.origin).ok_or_else(|| {
            std::io::Error::other(format!("unsupported predecessor origin {}", input.origin))
        })?;
        let spec = polylogue_sources::origin::by_name(format).ok_or_else(|| {
            std::io::Error::other(format!("no parser for predecessor origin {format}"))
        })?;
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
        let spec = if polylogue_sources::browser_capture::recognize(&at, &head[..n]) {
            polylogue_sources::origin::by_name("browser-capture").expect("capture parser")
        } else {
            spec
        };
        members.push(package::Member {
            path: at,
            bytes: combined,
            container: false,
            origin: Some(spec),
            parent: None,
            container_kind: None,
            member_path_bytes: None,
        });
    }
    Ok(Some(members))
}
fn origin(origin: &str) -> Option<&'static str> {
    Some(match origin {
        "claude-code-session" => "claude-code",
        "codex-session" => "codex",
        "gemini-cli-session" => "gemini-cli",
        "hermes-session" => "hermes",
        "antigravity-session" => "antigravity",
        "grok-export" => "grok",
        "chatgpt-export" => "chatgpt",
        "claude-ai-export" => "claude-ai",
        "claude-design-session" => "claude-design",
        "aistudio-drive" => "ai-studio",
        "otel-genai" => "otel-genai",
        _ => return None,
    })
}

#[cfg(test)]
mod tests;
