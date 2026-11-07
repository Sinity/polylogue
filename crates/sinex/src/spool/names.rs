//! A bounded-depth, rebuildable index of logical operation names. Full names
//! remain in immutable queue metadata; the directory prefix never limits them.
use super::*;

const PREFIX_BYTES: usize = 32;
const COMPONENT_BYTES: usize = 4;

fn hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    const DIGITS: &[u8] = b"0123456789abcdef";
    for byte in bytes {
        out.push(char::from(DIGITS[usize::from(byte >> 4)]));
        out.push(char::from(DIGITS[usize::from(byte & 15)]));
    }
    out
}

pub(super) fn insert(root: &Path, name: &str, identity: &str) -> Result<()> {
    let mut leaf = root.to_path_buf();
    for part in name.as_bytes()[..name.len().min(PREFIX_BYTES)].chunks(COMPONENT_BYTES) {
        leaf.push(hex(part));
    }
    leaf.push("entries");
    fs::create_dir_all(&leaf)?;
    write_index(&leaf.join(hash(identity.as_bytes())), identity)
}

pub(super) fn visit(
    root: &Path,
    prefix: &str,
    visit: &mut dyn FnMut(&str) -> Result<()>,
) -> Result<()> {
    let indexed = &prefix.as_bytes()[..prefix.len().min(PREFIX_BYTES)];
    walk(root, indexed, 0, visit)
}

fn walk(
    path: &Path,
    prefix: &[u8],
    depth: usize,
    visit: &mut dyn FnMut(&str) -> Result<()>,
) -> Result<()> {
    let required = hex(&prefix[..prefix.len().min(COMPONENT_BYTES)]);
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            return invalid("named index contains an unexpected file");
        }
        let name = entry.file_name();
        let name = name
            .to_str()
            .ok_or_else(|| Error::Invalid("invalid named index component".into()))?;
        if name == "entries" {
            if !prefix.is_empty() {
                continue;
            }
            for record in fs::read_dir(entry.path())? {
                let record = record?;
                let identity = read_json::<String>(&record.path())?
                    .ok_or_else(|| Error::Invalid("named index record disappeared".into()))?;
                if record.file_name().to_str() != Some(hash(identity.as_bytes()).as_str()) {
                    return invalid("named index identity differs from its record");
                }
                visit(&identity)?;
            }
        } else {
            if depth >= PREFIX_BYTES / COMPONENT_BYTES {
                return invalid("named index exceeds its fixed prefix depth");
            }
            if name.is_empty()
                || name.len() > COMPONENT_BYTES * 2
                || name.len() % 2 != 0
                || !name.bytes().all(|byte| byte.is_ascii_hexdigit())
            {
                return invalid("invalid named index component");
            }
            if name.starts_with(&required) {
                walk(
                    &entry.path(),
                    &prefix[prefix.len().min(COMPONENT_BYTES)..],
                    depth + 1,
                    visit,
                )?;
            }
        }
    }
    Ok(())
}
