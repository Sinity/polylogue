//! Containers are completely decoded and checked into private scratch files
//! before the first archive write. Payload memory is bounded by stream buffers.
use super::*;
use std::io::{Read, Seek};

pub(super) struct Member {
    pub path: PathBuf,
    pub bytes: tempfile::TempPath,
    pub container: bool,
    pub origin: Option<&'static OriginSpec>,
    pub parent: Option<(usize, u64)>,
    pub container_kind: Option<String>,
    pub member_path_bytes: Option<Vec<u8>>,
}

pub(super) fn stage(
    path: &Path,
    scratch: &Path,
    format: Option<&str>,
) -> std::io::Result<Vec<Member>> {
    if path.is_dir()
        && let Some(members) = super::legacy::stage(path, scratch)?
    {
        return Ok(members);
    }
    let mut members = Vec::new();
    if path.is_dir() {
        let mut paths = Vec::new();
        files_under(path, &mut paths)?;
        for child in paths {
            expand(
                child.clone(),
                copy_source(&child, scratch)?,
                None,
                scratch,
                &mut members,
            )?;
        }
    } else {
        expand(
            path.to_path_buf(),
            copy_source(path, scratch)?,
            None,
            scratch,
            &mut members,
        )?;
    }
    classify(&mut members, format)?;
    Ok(members)
}

fn copy_source(path: &Path, scratch: &Path) -> std::io::Result<tempfile::TempPath> {
    let mut source = std::fs::File::open(path)?;
    let before = FileFacts::of(&source.metadata()?);
    let mut copied = tempfile::NamedTempFile::new_in(scratch)?;
    let len = std::io::copy(&mut source, &mut copied)?;
    unchanged(&source, before, len)?;
    Ok(copied.into_temp_path())
}

fn expand(
    path: PathBuf,
    bytes: tempfile::TempPath,
    parent: Option<(usize, u64)>,
    scratch: &Path,
    out: &mut Vec<Member>,
) -> std::io::Result<()> {
    let mut file = std::fs::File::open(&bytes)?;
    let mut magic = [0; 512];
    let n = file.read(&mut magic)?;
    file.rewind()?;
    let container = magic[..n].starts_with(ZIP_MAGIC)
        || path
            .extension()
            .is_some_and(|e| e == "zip" || e == "tar" || e == "tgz")
        || path.to_string_lossy().ends_with(".tar.gz")
        || path.to_string_lossy().ends_with(".tar.zst");
    let index = out.len();
    if !container {
        out.push(Member {
            path,
            bytes,
            container: false,
            origin: None,
            parent,
            container_kind: None,
            member_path_bytes: None,
        });
        return Ok(());
    }
    let mut children = Vec::new();
    if magic[..n].starts_with(ZIP_MAGIC) || path.extension().is_some_and(|e| e == "zip") {
        let mut zip = zip::ZipArchive::new(&mut file).map_err(std::io::Error::other)?;
        for i in 0..zip.len() {
            let mut member = zip.by_index(i).map_err(std::io::Error::other)?;
            if !member.is_file() {
                continue;
            }
            let name = member.name_raw().to_vec();
            let expected = member.size();
            let mut decoded = tempfile::NamedTempFile::new_in(scratch)?;
            let written = std::io::copy(&mut member, &mut decoded)?;
            if written != expected {
                return Err(std::io::Error::other(
                    "container member length differs from declaration",
                ));
            }
            decoded.rewind()?;
            children.push((name, i as u64, decoded.into_temp_path()));
        }
    } else {
        let mut decoded = tempfile::tempfile_in(scratch)?;
        if magic[..n].starts_with(&[0x1f, 0x8b]) {
            std::io::copy(
                &mut flate2::read::MultiGzDecoder::new(&mut file),
                &mut decoded,
            )?;
        } else if magic[..n].starts_with(&[0x28, 0xb5, 0x2f, 0xfd]) {
            std::io::copy(
                &mut zstd::stream::read::Decoder::new(&mut file)?,
                &mut decoded,
            )?;
        } else {
            std::io::copy(&mut file, &mut decoded)?;
        }
        decoded.rewind()?;
        let mut tar = tar::Archive::new(decoded);
        for (i, entry) in tar.entries()?.enumerate() {
            let mut entry = entry?;
            if !entry.header().entry_type().is_file() {
                continue;
            }
            let name = entry.path()?.into_owned();
            let mut member = tempfile::NamedTempFile::new_in(scratch)?;
            std::io::copy(&mut entry, &mut member)?;
            member.rewind()?;
            children.push((
                name.as_os_str().as_encoded_bytes().to_vec(),
                i as u64,
                member.into_temp_path(),
            ));
        }
    }
    drop(file);
    out.push(Member {
        path: path.clone(),
        bytes,
        container: true,
        origin: Some(polylogue_sources::origin::by_name(KEPT).expect("container origin")),
        parent,
        container_kind: None,
        member_path_bytes: None,
    });
    let kind = if magic[..n].starts_with(ZIP_MAGIC) {
        "zip"
    } else if magic[..n].starts_with(&[0x1f, 0x8b]) {
        "tar+gzip"
    } else if magic[..n].starts_with(&[0x28, 0xb5, 0x2f, 0xfd]) {
        "tar+zstd"
    } else {
        "tar"
    };
    for (name, i, member) in children {
        let child = out.len();
        let display = String::from_utf8(name.clone())
            .unwrap_or_else(|_| name.iter().map(|byte| format!("%{byte:02X}")).collect());
        expand(
            PathBuf::from(format!("{}!/{display}", path.display())),
            member,
            Some((index, i)),
            scratch,
            out,
        )?;
        out[child].container_kind = Some(kind.into());
        out[child].member_path_bytes = Some(name);
    }
    Ok(())
}

fn classify(members: &mut [Member], format: Option<&str>) -> std::io::Result<()> {
    #[derive(Default)]
    struct Package {
        chatgpt: bool,
        claude: bool,
        recognized: bool,
        conversations: bool,
        user: bool,
        users: bool,
        aider_llm: bool,
        aider_md: bool,
        takeout: bool,
    }
    let mut anchors = Vec::new();
    for member in members.iter_mut().filter(|member| !member.container) {
        member.origin = super::recognition::declared(
            &member.path,
            &mut std::fs::File::open(&member.bytes)?,
            format,
        )?;
        let name = member
            .path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("");
        if (name.starts_with("conversations") && name.ends_with(".json"))
            || matches!(
                name,
                "prod-grok-backend.json" | "codex.json" | polylogue_sources::aider::HISTORY
            )
        {
            anchors.push((
                member.parent.map(|parent| parent.0),
                member.path.parent().unwrap_or(Path::new("")).to_path_buf(),
            ));
        }
        let mut prefix = PathBuf::new();
        for part in member.path.components() {
            prefix.push(part);
            if part.as_os_str() == "Takeout" {
                anchors.push((member.parent.map(|parent| parent.0), prefix));
                break;
            }
        }
    }
    let keys: Vec<_> = members
        .iter()
        .map(|member| {
            let parent = member.parent.map(|parent| parent.0);
            let root = anchors
                .iter()
                .filter(|(container, root)| *container == parent && member.path.starts_with(root))
                .max_by_key(|(_, root)| root.components().count())
                .map(|(_, root)| root.clone())
                .unwrap_or_else(|| member.path.parent().unwrap_or(Path::new("")).to_path_buf());
            (parent, root)
        })
        .collect();
    let mut packages = std::collections::BTreeMap::<(Option<usize>, PathBuf), Package>::new();
    for (member, key) in members
        .iter()
        .zip(&keys)
        .filter(|(member, _)| !member.container)
    {
        let package = packages.entry(key.clone()).or_default();
        package.recognized |= member.origin.is_some();
        package.takeout |= key.1.file_name().is_some_and(|name| name == "Takeout");
        let name = member
            .path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("");
        package.user |= name == "user.json";
        package.users |= name == "users.json";
        package.aider_llm |= name == polylogue_sources::aider::HISTORY;
        package.aider_md |= name == polylogue_sources::aider::COMPANION;
        if name.starts_with("conversations") && name.ends_with(".json") {
            package.conversations = true;
            package.chatgpt |= member.origin.is_some_and(|origin| origin.name == "chatgpt");
            package.claude |= member
                .origin
                .is_some_and(|origin| origin.name == "claude-ai");
        }
    }
    for package in packages.values_mut() {
        package.chatgpt |= package.conversations && package.user;
        package.claude |= package.conversations && package.users;
        if package.chatgpt && package.claude {
            return Err(std::io::Error::other(
                "export package declares conflicting ChatGPT and claude.ai structures",
            ));
        }
    }
    for (member, key) in members.iter_mut().zip(keys) {
        if member.container {
            member.origin = None;
            continue;
        }
        let package = packages.get(&key).expect("classified package");
        let path = member.path.to_string_lossy();
        let name = member
            .path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("");
        let route = if package.takeout {
            if path.contains("My Activity/Gemini Apps/") {
                Some(if name == "MyActivity.html" {
                    "gemini-app"
                } else {
                    KEPT
                })
            } else if path.contains("Drive/Google AI Studio/") {
                Some(
                    if member
                        .origin
                        .is_some_and(|origin| origin.name == "ai-studio")
                    {
                        "ai-studio"
                    } else {
                        KEPT
                    },
                )
            } else if path.contains("Gemini/") || path.contains("NotebookLM/") {
                Some(KEPT)
            } else {
                None
            }
        } else if package.claude {
            Some(
                if name == "conversations.json"
                    || matches!(name, "memories.json" | "projects.json")
                    || path.contains("/projects/")
                {
                    "claude-ai"
                } else if path.contains("/design_chats/")
                    && member
                        .origin
                        .is_some_and(|origin| origin.name == "claude-design")
                {
                    "claude-design"
                } else {
                    KEPT
                },
            )
        } else if package.chatgpt {
            Some(
                if name.starts_with("conversations") && name.ends_with(".json") {
                    "chatgpt"
                } else if name == "codex.json" {
                    "codex-cloud"
                } else {
                    KEPT
                },
            )
        } else if package.aider_llm
            && package.aider_md
            && matches!(
                name,
                polylogue_sources::aider::HISTORY | polylogue_sources::aider::COMPANION
            )
        {
            Some("aider")
        } else if member.origin.is_none() && package.recognized {
            Some(KEPT)
        } else {
            member.origin.map(|origin| origin.name)
        };
        member.origin = route.and_then(polylogue_sources::origin::by_name);
    }
    for index in (0..members.len())
        .rev()
        .filter(|index| members[*index].container)
        .collect::<Vec<_>>()
    {
        if members.iter().any(|member| {
            member.parent.is_some_and(|parent| parent.0 == index) && member.origin.is_some()
        }) {
            members[index].origin = polylogue_sources::origin::by_name(KEPT);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
