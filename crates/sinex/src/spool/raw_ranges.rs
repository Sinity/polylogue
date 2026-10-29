//! Rebuildable references to committed provider bodies, including unresolved
//! custody. This index grants no source-head or parser progress authority.
use super::*;
use polylogue_store::primary::{RawBodyRange, SourceGeneration};

pub(crate) struct PinnedRange {
    pub order: u64,
    range: RawBodyRange,
    file: File,
    descriptor: EncodedBlob,
}

impl PinnedRange {
    /// Payload verification runs after releasing the producer custody lock.
    pub fn verify(self) -> Result<RawBodyRange> {
        let _verified = hashed_blobs::Prepared::from_encoded(
            self.file,
            self.descriptor,
            None,
            ReadLimits::default(),
        )?;
        Ok(self.range)
    }
}

fn generation_key(generation: &SourceGeneration) -> Result<String> {
    Ok(hash(&serde_json::to_vec(generation)?))
}

fn disposition(
    spool: &Spool,
    identity: &str,
) -> Result<Option<polylogue_store::primary::CaptureDisposition>> {
    let operation = spool.operation(identity)?;
    let parent = operation.parent().expect("operation parent");
    if parent == spool.root.join("operations") {
        return Ok(None);
    }
    let group = parent.parent().expect("acquisition parent");
    let receipt = read_checked::<Acquisition>(&group.join("acquisition.json"))?
        .ok_or_else(|| Error::Invalid("raw body has no published acquisition receipt".into()))?;
    if group.file_name().and_then(|name| name.to_str())
        != Some(hash(receipt.identity.as_bytes()).as_str())
    {
        return invalid("raw acquisition receipt has another identity");
    }
    Ok(Some(receipt.disposition))
}

pub(super) fn insert(spool: &Spool, prepared: &Prepared) -> Result<()> {
    let Some(Capture::Input(input)) = &prepared.capture else {
        return Ok(());
    };
    let (Some(generation), Some(_)) = (&input.generation, input.raw_range) else {
        return Ok(());
    };
    let mut path = spool.index("raw-generations", &generation_key(generation)?);
    for byte in &prepared.order.to_be_bytes()[..7] {
        path.push(format!("{byte:02x}"));
    }
    fs::create_dir_all(&path)?;
    path.push(format!("{:02x}", prepared.order.to_be_bytes()[7]));
    write_index(&path, &prepared.identity)
}

impl Spool {
    pub fn raw_cut(&self) -> Result<u64> {
        self.ensure_indexes()?;
        self.next
            .get()
            .checked_sub(1)
            .ok_or_else(|| Error::Invalid("invalid custody queue cut".into()))
    }

    pub fn raw_ranges(
        &self,
        generation: &SourceGeneration,
        after: Option<u64>,
        through: u64,
        limit: usize,
    ) -> Result<Vec<PinnedRange>> {
        generation.validate()?;
        if limit == 0 {
            return invalid("raw custody page size must be positive");
        }
        self.ensure_indexes()?;
        let root = self.index("raw-generations", &generation_key(generation)?);
        if !root.try_exists()? {
            return Ok(Vec::new());
        }
        let mut entries = Vec::new();
        walk(
            &root,
            0,
            0,
            after,
            through,
            limit,
            &mut |order, identity| {
                let prepared = self.required(identity)?;
                let Some(Capture::Input(input)) = &prepared.capture else {
                    return invalid("raw custody index points to a non-input operation");
                };
                if prepared.order != order || input.generation.as_ref() != Some(generation) {
                    return invalid("raw custody index differs from its immutable generation");
                }
                let range = input
                    .raw_range
                    .ok_or_else(|| Error::Invalid("raw custody has no provider interval".into()))?;
                let (path, descriptor) = self.encoded_body(identity)?;
                if descriptor.hash.0 != input.hash.0
                    || descriptor.decoded_len != input.length
                    || range.length()? != input.length
                {
                    return invalid("raw custody body differs from its original interval");
                }
                entries.push(PinnedRange {
                    order,
                    range: RawBodyRange {
                        generation: generation.clone(),
                        body: identity.into(),
                        range,
                        hash: input.hash,
                        decoded_length: input.length,
                        known_total: input.raw_total,
                        predecessor: input.raw_predecessor.clone(),
                        disposition: disposition(self, identity)?,
                    },
                    file: regular(&path)?,
                    descriptor,
                });
                Ok(())
            },
        )?;
        Ok(entries)
    }
}

/// Eight radix levels bound each directory read to 256 references. The page
/// cursor is queue order; unrelated parser payloads are never visited.
fn walk(
    path: &Path,
    depth: u32,
    prefix: u64,
    after: Option<u64>,
    through: u64,
    mut remaining: usize,
    visit: &mut dyn FnMut(u64, &str) -> Result<()>,
) -> Result<usize> {
    let mut children = Vec::new();
    for child in fs::read_dir(path)? {
        let child = child?;
        let name = child.file_name();
        let name = name
            .to_str()
            .ok_or_else(|| Error::Invalid("raw custody index has a non-UTF8 entry".into()))?;
        if name.len() != 2
            || !name
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return invalid("raw custody index has an invalid radix entry");
        }
        let byte =
            u8::from_str_radix(name, 16).map_err(|error| Error::Invalid(error.to_string()))?;
        children.push((byte, child.path(), child.file_type()?));
    }
    children.sort_by_key(|child| child.0);
    for (byte, child, kind) in children {
        if remaining == 0 {
            break;
        }
        let prefix = (prefix << 8) | u64::from(byte);
        let bits = 8 * (7 - depth);
        let maximum = (prefix << bits) | ((1u64 << bits) - 1);
        if after.is_some_and(|after| maximum <= after) || (prefix << bits) > through {
            continue;
        }
        if depth == 7 {
            if !kind.is_file() {
                return invalid("raw custody radix leaf is not a file");
            }
            let identity = read_json::<String>(&child)?
                .ok_or_else(|| Error::Invalid("raw custody index disappeared".into()))?;
            visit(prefix, &identity)?;
            remaining -= 1;
        } else {
            if !kind.is_dir() {
                return invalid("raw custody radix branch is not a directory");
            }
            remaining = walk(&child, depth + 1, prefix, after, through, remaining, visit)?;
        }
    }
    Ok(remaining)
}
