//! The predecessor's catalog supplies origin and append lineage. Only its
//! declared byte blobs and source facts cross this boundary.
use super::*;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::io::{Read, Seek};

pub(super) fn stage(root: &Path, scratch: &Path) -> std::io::Result<Option<Vec<package::Member>>> {
    let Some(inputs) =
        polylogue_store::legacy_import::inputs(root).map_err(std::io::Error::other)?
    else {
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
        let mut source = std::fs::File::open(&path).map_err(|error| {
            std::io::Error::other(format!("missing predecessor blob {hash}: {error}"))
        })?;
        let mut copy = tempfile::NamedTempFile::new_in(scratch)?;
        let mut actual = Sha256::new();
        let mut length = 0;
        let mut buffer = [0; 64 * 1024];
        loop {
            let n = source.read(&mut buffer)?;
            if n == 0 {
                break;
            }
            actual.update(&buffer[..n]);
            std::io::Write::write_all(&mut copy, &buffer[..n])?;
            length += n as u64;
        }
        if length != input.blob_size || actual.finalize().as_slice() != input.blob_hash {
            return Err(std::io::Error::other(format!(
                "predecessor blob {hash} differs from its catalog hash or size"
            )));
        }
        blobs.insert(input.raw_id.clone(), copy.into_temp_path());
    }
    let mut members = Vec::new();
    let mut catalog = tempfile::NamedTempFile::new_in(scratch)?;
    for input in &inputs {
        serde_json::to_writer(&mut catalog, input).map_err(std::io::Error::other)?;
        std::io::Write::write_all(&mut catalog, b"\n")?;
    }
    members.push(package::Member {
        path: PathBuf::from(format!("{}!/raw_sessions/catalog.jsonl", root.display())),
        bytes: catalog.into_temp_path(),
        container: false,
        origin: polylogue_sources::origin::by_name(KEPT),
        parent: None,
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
                bytes: copy(&blobs[&input.raw_id], scratch)?,
                container: false,
                origin: polylogue_sources::origin::by_name(KEPT),
                parent: None,
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
        let mut combined = tempfile::NamedTempFile::new_in(scratch)?;
        let mut length = std::io::copy(
            &mut std::fs::File::open(&blobs[&current.raw_id])?,
            &mut combined,
        )?;
        for append in chain.into_iter().rev() {
            if append.append_start != Some(length)
                || append.append_end != length.checked_add(append.blob_size)
            {
                return Err(std::io::Error::other(
                    "predecessor append offsets do not join its recorded bytes",
                ));
            }
            length += std::io::copy(
                &mut std::fs::File::open(&blobs[&append.raw_id])?,
                &mut combined,
            )?;
        }
        combined.rewind()?;
        let mut head = [0; 64];
        let n = combined.read(&mut head)?;
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
            bytes: combined.into_temp_path(),
            container: false,
            origin: Some(spec),
            parent: None,
        });
    }
    Ok(Some(members))
}
fn copy(path: &Path, scratch: &Path) -> std::io::Result<tempfile::TempPath> {
    let mut file = tempfile::NamedTempFile::new_in(scratch)?;
    std::io::copy(&mut std::fs::File::open(path)?, &mut file)?;
    Ok(file.into_temp_path())
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
