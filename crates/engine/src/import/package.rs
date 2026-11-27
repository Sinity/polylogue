//! Containers are completely decoded and checked into private encoded files
//! before the first archive write. Payload memory is bounded by stream buffers.
use super::*;
use std::io::Read;

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

/// A staged file that may still need expanding. Container errors are deferred
/// until its package has had a chance to identify it as an attachment.
struct PendingMember {
    member: Member,
    format: std::io::Result<Option<Container>>,
}

impl PendingMember {
    fn prepare(path: PathBuf, archive: &Archive, mut source: impl Read) -> std::io::Result<Self> {
        let (bytes, format) = archive
            .prepare_write(|output| {
                // The longest container signature is four bytes. Feed them
                // back into the encoder so these are exactly the staged bytes.
                let mut magic = Vec::with_capacity(4);
                source.by_ref().take(4).read_to_end(&mut magic)?;
                std::io::copy(&mut magic.as_slice().chain(source), output)?;
                Ok(Container::of(&path, &magic))
            })
            .map_err(std::io::Error::other)?;
        Ok(Self {
            member: Member {
                path,
                bytes,
                container: false,
                origin: None,
                parent: None,
                container_kind: None,
                member_path_bytes: None,
                refusal: None,
            },
            format,
        })
    }

    fn is_container(&self) -> bool {
        !matches!(self.format, Ok(None))
    }
}

pub(super) fn stage(
    path: &Path,
    archive: &Archive,
    origin: Option<&str>,
) -> std::io::Result<Vec<Member>> {
    let mut paths = Vec::new();
    let root = if path.is_dir() {
        files_under(path, &mut paths)?;
        path
    } else {
        paths.push(path.to_path_buf());
        path.parent().unwrap_or(Path::new(""))
    };
    let pending = paths
        .into_iter()
        .map(|path| prepare_source(path, archive))
        .collect::<std::io::Result<_>>()?;
    let mut members = Vec::new();
    expand_members(
        pending,
        root,
        archive,
        origin,
        &mut Expansion::default(),
        &mut members,
    )?;
    let kept = polylogue_sources::origin::by_name(KEPT);
    // Keep the original of any container with an accepted descendant.
    for index in (0..members.len()).rev() {
        if members[index].origin.is_some()
            && let Some((parent, _)) = members[index].parent
        {
            members[parent].origin = kept;
        }
    }
    // A repeated container shares the route of its first, expanded copy.
    let kept_hashes: std::collections::HashSet<_> = members
        .iter()
        .filter(|member| member.container && member.origin.is_some())
        .map(|member| member.bytes.hash())
        .collect();
    for member in members.iter_mut().filter(|member| member.container) {
        if member.origin.is_none() && kept_hashes.contains(&member.bytes.hash()) {
            member.origin = kept;
        }
    }
    super::bundle::preflight(&mut members)?;
    Ok(members)
}

fn expand_members(
    mut pending: Vec<PendingMember>,
    root: &Path,
    archive: &Archive,
    origin: Option<&str>,
    seen: &mut Expansion,
    out: &mut Vec<Member>,
) -> std::io::Result<()> {
    // Files at one container level share a package. Settle those roles before
    // expanding containers so uploaded archives are kept as attachment bytes.
    classify(&mut pending, root, origin)?;
    for pending in pending {
        expand(pending, archive, origin, seen, out)?;
    }
    Ok(())
}

fn prepare_source(path: PathBuf, archive: &Archive) -> std::io::Result<PendingMember> {
    let mut source = std::fs::File::open(&path)?;
    let before = FileFacts::of(&source.metadata()?);
    let pending = PendingMember::prepare(path, archive, &mut source)?;
    unchanged(&source, before, pending.member.bytes.len())?;
    Ok(pending)
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
    pending: PendingMember,
    archive: &Archive,
    origin: Option<&str>,
    seen: &mut Expansion,
    out: &mut Vec<Member>,
) -> std::io::Result<()> {
    let PendingMember { mut member, format } = pending;
    let Some(container) = format? else {
        out.push(member);
        return Ok(());
    };
    member.container = true;
    let path = &member.path;
    let hash = member.bytes.hash();
    if seen.ancestors.contains(&hash) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("container {} contains its own bytes", path.display()),
        ));
    }
    if !seen.expanded.insert(hash) {
        // The same container at another path: its members are already staged.
        out.push(member);
        return Ok(());
    }
    let mut file = member.bytes.reader().map_err(std::io::Error::other)?;
    let root = PathBuf::from(format!("{}!", path.display()));
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
            // A member name is provenance, even when it starts with a slash.
            let child = PendingMember::prepare(
                PathBuf::from(format!("{}/{display}", root.display())),
                archive,
                (&mut member).take(expected.saturating_add(1)),
            )?;
            if child.member.bytes.len() != expected {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "member {display} of {} decodes to a length different from its declared {expected}",
                        path.display()
                    ),
                ));
            }
            children.push((name, i as u64, child));
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
            let name = entry.path()?.as_os_str().as_encoded_bytes().to_vec();
            let display = String::from_utf8(name.clone())
                .unwrap_or_else(|_| name.iter().map(|byte| format!("%{byte:02X}")).collect());
            fits_in_staging(archive, entry.header().size()?)?;
            let child = PendingMember::prepare(
                PathBuf::from(format!("{}/{display}", root.display())),
                archive,
                &mut entry,
            )?;
            children.push((name, i as u64, child));
        }
        // A tar iterator stops at its end marker before the compression
        // footer. Drain the decoder so checksum/truncation failures still
        // refuse the whole package before it is imported.
        std::io::copy(&mut tar.into_inner(), &mut std::io::sink())?;
    }
    drop(file);
    let index = out.len();
    out.push(member);
    let kind = container.kind();
    let mut pending = Vec::new();
    for (name, i, mut child) in children {
        child.member.parent = Some((index, i));
        child.member.container_kind = Some(kind.into());
        child.member.member_path_bytes = Some(name);
        pending.push(child);
    }
    seen.ancestors.push(hash);
    expand_members(pending, &root, archive, origin, seen, out)?;
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

fn classify(
    pending: &mut [PendingMember],
    root: &Path,
    origin: Option<&str>,
) -> std::io::Result<()> {
    use polylogue_sources::origin::{Others, Package};
    let mut package_roots = Vec::new();
    for member in pending
        .iter_mut()
        .filter(|item| !item.is_container())
        .map(|item| &mut item.member)
    {
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
            package_roots.push(member.path.parent().unwrap_or(Path::new("")).to_path_buf());
        }
        let mut prefix = PathBuf::new();
        for part in member.path.components() {
            prefix.push(part);
            if part.as_os_str() == Package::TAKEOUT {
                package_roots.push(prefix);
                break;
            }
        }
    }
    let keys: Vec<_> = pending
        .iter()
        .map(|item| {
            let path = &item.member.path;
            package_roots
                .iter()
                .filter(|root| path.starts_with(root))
                .max_by_key(|root| root.components().count())
                .cloned()
                .unwrap_or_else(|| path.parent().unwrap_or(Path::new("")).to_path_buf())
        })
        .collect();
    // Each group of members under one root: what each is on its own, and
    // whether one of them makes the others members kept with it.
    #[derive(Default)]
    struct Group<'a> {
        members: Vec<(&'a str, Option<&'static OriginSpec>)>,
        recognized: bool,
    }
    let mut groups = std::collections::BTreeMap::<&PathBuf, Group>::new();
    for (item, key) in pending
        .iter()
        .zip(&keys)
        .filter(|(item, _)| !item.is_container())
    {
        let member = &item.member;
        let group = groups.entry(key).or_default();
        group.members.push((file_name(&member.path), member.origin));
        group.recognized |= member
            .origin
            .is_some_and(|origin| (origin.import.holds_package)(&member.path));
    }
    let mut packages = std::collections::BTreeMap::new();
    for (key, group) in &groups {
        let package = Package::of(key, &group.members)
            .map_err(|reason| std::io::Error::new(std::io::ErrorKind::InvalidData, reason))?;
        packages.insert(*key, (package, group.recognized));
    }
    let kept = polylogue_sources::origin::by_name(KEPT);
    for (item, key) in pending.iter_mut().zip(&keys) {
        let container = item.is_container();
        let member = &mut item.member;
        let (package, recognized) = packages.get(key).copied().unwrap_or_default();
        if container {
            let attachment = package.is_some_and(|package| {
                let relative = member
                    .path
                    .strip_prefix(root)
                    .unwrap_or(&member.path)
                    .to_string_lossy();
                ORIGINS.iter().any(|origin| {
                    origin
                        .import
                        .member
                        .as_ref()
                        .is_some_and(|rule| rule.package == package)
                        && origin
                            .assets
                            .as_ref()
                            .is_some_and(|assets| (assets.member_id)(&relative).is_some())
                })
            });
            if attachment {
                item.format = Ok(None);
                member.origin = kept;
            }
            continue;
        }
        let alone = member.origin;
        let path = member.path.to_string_lossy();
        let name = file_name(&member.path);
        let generic = if alone.is_none() && recognized {
            kept
        } else {
            alone
        };
        member.origin = match package {
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
