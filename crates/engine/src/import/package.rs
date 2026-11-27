//! Containers are completely decoded and checked into private encoded files
//! before the first archive write. Payload memory is bounded by stream buffers.
use super::*;
use std::io::{Read, Seek};

pub(super) struct Member {
    pub path: PathBuf,
    pub bytes: polylogue_store::staging::PreparedBytes,
    pub container: bool,
    pub origin: Option<&'static OriginSpec>,
    pub parent: Option<(usize, u64)>,
    pub container_kind: Option<String>,
    pub member_path_bytes: Option<Vec<u8>>,
    /// Why a member with no origin is refused, when it is not that no
    /// declared structure matched.
    pub refusal: Option<String>,
}

pub(super) fn stage(
    path: &Path,
    archive: &Archive,
    origin: Option<&str>,
) -> std::io::Result<Vec<Member>> {
    let mut members = Vec::new();
    let mut seen = Expansion::default();
    if path.is_dir() {
        let mut paths = Vec::new();
        files_under(path, &mut paths)?;
        for child in paths {
            expand(
                child.clone(),
                copy_source(&child, archive)?,
                None,
                archive,
                &mut seen,
                &mut members,
            )?;
        }
    } else {
        expand(
            path.to_path_buf(),
            copy_source(path, archive)?,
            None,
            archive,
            &mut seen,
            &mut members,
        )?;
    }
    classify(&mut members, origin)?;
    super::bundle::preflight(&mut members)?;
    Ok(members)
}

fn copy_source(
    path: &Path,
    archive: &Archive,
) -> std::io::Result<polylogue_store::staging::PreparedBytes> {
    let mut source = std::fs::File::open(path)?;
    let before = FileFacts::of(&source.metadata()?);
    let copied = archive
        .prepare_bytes(&mut source)
        .map_err(std::io::Error::other)?;
    unchanged(&source, before, copied.len())?;
    Ok(copied)
}

/// What container expansion has seen so far. Work follows distinct content:
/// a container holding its own bytes (directly or through a chain of
/// containers) is refused, and a container already expanded at another path
/// in the same import is kept as an original without expanding it again.
#[derive(Default)]
struct Expansion {
    ancestors: Vec<polylogue_model::BlobHash>,
    expanded: std::collections::HashSet<polylogue_model::BlobHash>,
}

/// Refuses a member whose declared decoded size cannot fit in the free
/// staging space, before any of it is decoded.
fn fits_in_staging(archive: &Archive, declared: u64) -> std::io::Result<()> {
    let directory = archive.staging_dir();
    std::fs::create_dir_all(&directory)?;
    let stats = rustix::fs::statvfs(&directory).map_err(std::io::Error::from)?;
    let free = stats.f_bavail.saturating_mul(stats.f_frsize);
    if declared > free {
        return Err(std::io::Error::new(
            std::io::ErrorKind::StorageFull,
            format!("container declares {declared} decoded bytes; staging has {free} free"),
        ));
    }
    Ok(())
}

fn expand(
    path: PathBuf,
    bytes: polylogue_store::staging::PreparedBytes,
    parent: Option<(usize, u64)>,
    archive: &Archive,
    seen: &mut Expansion,
    out: &mut Vec<Member>,
) -> std::io::Result<()> {
    let mut file = bytes.reader().map_err(std::io::Error::other)?;
    let mut magic = [0; 512];
    let n = file.read(&mut magic)?;
    file.rewind()?;
    let container = Container::of(&path, &magic[..n])?;
    let index = out.len();
    let hash = bytes.hash();
    if container.is_some() && seen.ancestors.contains(&hash) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("container {} contains its own bytes", path.display()),
        ));
    }
    if container.is_some() && !seen.expanded.insert(hash) {
        // The same container at another path: its members are already staged.
        out.push(Member {
            path,
            bytes,
            container: true,
            origin: None,
            parent,
            container_kind: None,
            member_path_bytes: None,
            refusal: None,
        });
        return Ok(());
    }
    let Some(container) = container else {
        out.push(Member {
            path,
            bytes,
            container: false,
            origin: None,
            parent,
            container_kind: None,
            member_path_bytes: None,
            refusal: None,
        });
        return Ok(());
    };
    let mut children = Vec::new();
    if let Container::Zip = container {
        let mut zip = zip::ZipArchive::new(&mut file).map_err(std::io::Error::other)?;
        let mut declared = 0u64;
        for i in 0..zip.len() {
            let member = zip.by_index_raw(i).map_err(std::io::Error::other)?;
            if member.is_file() {
                declared = declared.saturating_add(member.size());
            }
        }
        fits_in_staging(archive, declared)?;
        for i in 0..zip.len() {
            let mut member = zip.by_index(i).map_err(std::io::Error::other)?;
            if !member.is_file() {
                continue;
            }
            let name = member.name_raw().to_vec();
            let display = member.name().to_owned();
            let expected = member.size();
            // One byte past the declaration is enough to know it lied.
            let decoded = archive
                .prepare_bytes((&mut member).take(expected.saturating_add(1)))
                .map_err(std::io::Error::other)?;
            if decoded.len() != expected {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "member {display} of {} decodes to a length different from its declared {expected}",
                        path.display()
                    ),
                ));
            }
            children.push((name, i as u64, decoded, Some(display)));
        }
    } else {
        let decoded: Box<dyn Read + '_> = match container {
            Container::Gzip => Box::new(flate2::read::MultiGzDecoder::new(&mut file)),
            Container::Zstd => Box::new(zstd::stream::read::Decoder::new(&mut file)?),
            Container::Zip | Container::Tar => Box::new(&mut file),
        };
        let mut tar = tar::Archive::new(decoded);
        for (i, entry) in tar.entries()?.enumerate() {
            let mut entry = entry?;
            if !entry.header().entry_type().is_file() {
                continue;
            }
            let name = entry.path()?.into_owned();
            fits_in_staging(archive, entry.header().size()?)?;
            let member = archive
                .prepare_bytes(&mut entry)
                .map_err(std::io::Error::other)?;
            children.push((
                name.as_os_str().as_encoded_bytes().to_vec(),
                i as u64,
                member,
                None,
            ));
        }
        // A tar iterator stops at its end marker before the compression
        // footer. Drain the decoder so checksum/truncation failures still
        // refuse the whole package before it is imported.
        std::io::copy(&mut tar.into_inner(), &mut std::io::sink())?;
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
        refusal: None,
    });
    let kind = container.kind();
    seen.ancestors.push(hash);
    for (name, i, member, display) in children {
        let child = out.len();
        let display = display.unwrap_or_else(|| {
            String::from_utf8(name.clone())
                .unwrap_or_else(|_| name.iter().map(|byte| format!("%{byte:02X}")).collect())
        });
        expand(
            PathBuf::from(format!("{}!/{display}", path.display())),
            member,
            Some((index, i)),
            archive,
            seen,
            out,
        )?;
        out[child].container_kind = Some(kind.into());
        out[child].member_path_bytes = Some(name);
    }
    seen.ancestors.pop();
    Ok(())
}

/// What a container is, from its first bytes; its name only says that it
/// is meant to be one.
#[derive(Clone, Copy)]
enum Container {
    Zip,
    Tar,
    Gzip,
    Zstd,
}

impl Container {
    /// `None` for a file that is no container. A file named as a ZIP
    /// archive whose bytes are not one is refused rather than read as
    /// something else.
    fn of(path: &Path, magic: &[u8]) -> std::io::Result<Option<Container>> {
        if magic.starts_with(ZIP_MAGIC) {
            return Ok(Some(Container::Zip));
        }
        let name = path.to_string_lossy();
        if path.extension().is_some_and(|e| e == "zip") {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("{name} is named as a ZIP archive and does not start like one"),
            ));
        }
        let tar = path.extension().is_some_and(|e| e == "tar" || e == "tgz")
            || name.ends_with(".tar.gz")
            || name.ends_with(".tar.zst");
        Ok(tar.then(|| {
            if magic.starts_with(&[0x1f, 0x8b]) {
                Container::Gzip
            } else if magic.starts_with(&[0x28, 0xb5, 0x2f, 0xfd]) {
                Container::Zstd
            } else {
                Container::Tar
            }
        }))
    }

    /// The kind recorded on each member.
    fn kind(self) -> &'static str {
        match self {
            Container::Zip => "zip",
            Container::Tar => "tar",
            Container::Gzip => "tar+gzip",
            Container::Zstd => "tar+zstd",
        }
    }
}

fn classify(members: &mut [Member], origin: Option<&str>) -> std::io::Result<()> {
    use polylogue_sources::origin::{Others, Package};
    let mut package_roots = Vec::new();
    for member in members.iter_mut().filter(|member| !member.container) {
        member.origin = super::recognition::declared(
            &member.path,
            &mut member.bytes.reader().map_err(std::io::Error::other)?,
            origin,
        )?;
        if member.origin.is_none()
            && let Some(name) = origin
        {
            member.refusal = Some(format!(
                "origin {name} is not an importable origin or does not recognize this file"
            ));
        }
        if Package::root_name(file_name(&member.path)) {
            package_roots.push((
                member.parent.map(|parent| parent.0),
                member.path.parent().unwrap_or(Path::new("")).to_path_buf(),
            ));
        }
        let mut prefix = PathBuf::new();
        for part in member.path.components() {
            prefix.push(part);
            if part.as_os_str() == Package::TAKEOUT {
                package_roots.push((member.parent.map(|parent| parent.0), prefix));
                break;
            }
        }
    }
    let keys: Vec<_> = members
        .iter()
        .map(|member| {
            let parent = member.parent.map(|parent| parent.0);
            let root = package_roots
                .iter()
                .filter(|(container, root)| *container == parent && member.path.starts_with(root))
                .max_by_key(|(_, root)| root.components().count())
                .map(|(_, root)| root.clone())
                .unwrap_or_else(|| member.path.parent().unwrap_or(Path::new("")).to_path_buf());
            (parent, root)
        })
        .collect();
    // Each group of members under one root: what each is on its own, and
    // whether one of them makes the others members kept with it.
    #[derive(Default)]
    struct Group<'a> {
        members: Vec<(&'a str, Option<&'static OriginSpec>)>,
        recognized: bool,
    }
    let mut groups = std::collections::BTreeMap::<&(Option<usize>, PathBuf), Group>::new();
    for (member, key) in members
        .iter()
        .zip(&keys)
        .filter(|(member, _)| !member.container)
    {
        let group = groups.entry(key).or_default();
        group.members.push((file_name(&member.path), member.origin));
        group.recognized |= member
            .origin
            .is_some_and(|origin| (origin.import.holds_package)(&member.path));
    }
    let mut packages = std::collections::BTreeMap::new();
    for (key, group) in &groups {
        let package = Package::of(&key.1, &group.members)
            .map_err(|reason| std::io::Error::new(std::io::ErrorKind::InvalidData, reason))?;
        packages.insert(*key, (package, group.recognized));
    }
    let kept = polylogue_sources::origin::by_name(KEPT);
    let mut routes = Vec::with_capacity(members.len());
    for (member, key) in members.iter().zip(&keys) {
        if member.container {
            routes.push(None);
            continue;
        }
        let (package, recognized) = packages[key];
        let alone = member.origin;
        let path = member.path.to_string_lossy();
        let name = file_name(&member.path);
        let generic = if alone.is_none() && recognized {
            kept
        } else {
            alone
        };
        let route = match package {
            Some(package) => ORIGINS
                .iter()
                .find(|origin| {
                    origin.import.member.as_ref().is_some_and(|rule| {
                        rule.package == package
                            && (rule.is)(&path, name, alone.is_some_and(|a| a.name == origin.name))
                    })
                })
                .or(match package.others(&path) {
                    Others::Kept => kept,
                    Others::Refused => None,
                    Others::Alone => generic,
                }),
            None => generic,
        };
        routes.push(route);
    }
    for (member, route) in members.iter_mut().zip(routes) {
        member.origin = route;
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
    // A repeated container shares the route of its first, expanded copy.
    let kept: std::collections::HashSet<_> = members
        .iter()
        .filter(|member| member.container && member.origin.is_some())
        .map(|member| member.bytes.hash())
        .collect();
    for member in members.iter_mut().filter(|member| member.container) {
        if member.origin.is_none() && kept.contains(&member.bytes.hash()) {
            member.origin = polylogue_sources::origin::by_name(KEPT);
        }
    }
    Ok(())
}

fn file_name(path: &Path) -> &str {
    path.file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("")
}

#[cfg(test)]
mod tests;
