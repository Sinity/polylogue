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
}

pub(super) fn stage(
    path: &Path,
    scratch: &Path,
    format: Option<&str>,
) -> std::io::Result<Vec<Member>> {
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
            if member.enclosed_name().is_none() {
                return Err(std::io::Error::other("container member has an unsafe path"));
            }
            let name = member.name().to_owned();
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
            if name.is_absolute()
                || name
                    .components()
                    .any(|c| matches!(c, std::path::Component::ParentDir))
            {
                return Err(std::io::Error::other("container member has an unsafe path"));
            }
            let mut member = tempfile::NamedTempFile::new_in(scratch)?;
            std::io::copy(&mut entry, &mut member)?;
            member.rewind()?;
            children.push((
                name.to_string_lossy().into_owned(),
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
    });
    for (name, i, member) in children {
        expand(
            PathBuf::from(format!("{}!/{name}", path.display())),
            member,
            Some((index, i)),
            scratch,
            out,
        )?;
    }
    Ok(())
}

fn classify(members: &mut [Member], format: Option<&str>) -> std::io::Result<()> {
    #[derive(Default)]
    struct Package {
        chatgpt: bool,
        claude: bool,
        takeout: bool,
        conversations: bool,
        user: bool,
        users: bool,
    }
    let containers: std::collections::BTreeSet<_> = members
        .iter()
        .enumerate()
        .filter_map(|(i, m)| m.origin.is_some().then_some(i))
        .collect();
    let mut packages = std::collections::BTreeMap::<Option<usize>, Package>::new();
    for (index, member) in members.iter_mut().enumerate() {
        if containers.contains(&index) {
            continue;
        }
        member.origin = super::recognition::declared(
            &member.path,
            &mut std::fs::File::open(&member.bytes)?,
            format,
        )?;
        let package = packages.entry(member.parent.map(|p| p.0)).or_default();
        let path = member.path.to_string_lossy();
        package.takeout |= path.contains("Takeout/");
        let name = member
            .path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("");
        package.user |= name == "user.json";
        package.users |= name == "users.json";
        package.conversations |= name == "conversations.json";
        if member
            .path
            .file_name()
            .is_some_and(|n| n.to_string_lossy().starts_with("conversations"))
        {
            package.chatgpt |= member.origin.is_some_and(|o| o.name == "chatgpt");
            package.claude |= member.origin.is_some_and(|o| o.name == "claude-ai");
        }
    }
    for package in packages.values_mut() {
        package.chatgpt |= package.conversations && package.user;
        package.claude |= package.conversations && package.users;
    }
    for (index, member) in members.iter_mut().enumerate() {
        if containers.contains(&index) {
            member.origin = None;
            continue;
        }
        let package = packages
            .get(&member.parent.map(|p| p.0))
            .expect("classified package");
        let path = member.path.to_string_lossy();
        if package.takeout {
            let route = if path.contains("My Activity/Gemini Apps/") {
                Some("gemini-app")
            } else if path.contains("Drive/Google AI Studio/") {
                Some("ai-studio")
            } else if path.contains("Gemini/") || path.contains("NotebookLM/") {
                Some(KEPT)
            } else {
                None
            };
            member.origin = route.and_then(polylogue_sources::origin::by_name);
        } else if package.claude {
            let name = member
                .path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("");
            if [
                "conversations.json",
                "memories.json",
                "users.json",
                "projects.json",
            ]
            .contains(&name)
                || path.contains("/projects/")
            {
                member.origin = polylogue_sources::origin::by_name("claude-ai");
            } else if member.origin.is_none() {
                member.origin = polylogue_sources::origin::by_name(KEPT);
            }
        } else if member.origin.is_none() && package.chatgpt {
            let name = if member
                .path
                .file_name()
                .is_some_and(|n| n == "conversations.json")
            {
                "chatgpt"
            } else {
                KEPT
            };
            member.origin = polylogue_sources::origin::by_name(name);
        }
    }
    // Select a container only when its own children are selected. This is
    // evaluated from the leaves so nested containers retain their originals.
    for index in containers.into_iter().rev() {
        if members
            .iter()
            .any(|m| m.parent.is_some_and(|p| p.0 == index) && m.origin.is_some())
        {
            members[index].origin = polylogue_sources::origin::by_name(KEPT);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
