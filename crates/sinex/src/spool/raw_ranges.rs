//! Rebuildable references to committed provider bodies, including unresolved
//! custody. This index grants no source-head or parser progress authority.
use super::*;
use polylogue_store::primary::{RawBodyRange, SourceGeneration};

pub(crate) struct PinnedRange {
    pub order: u64,
    range: RawBodyRange,
    parts: Vec<(File, EncodedBlob)>,
}

impl PinnedRange {
    /// Payload verification runs after releasing the producer custody lock.
    /// One encoding names the body; chunks are read through and hashed as
    /// the body.
    pub fn verify(self) -> Result<RawBodyRange> {
        let several = self.parts.len() > 1;
        let mut whole = blake3::Hasher::new();
        let mut length = 0u64;
        for (file, descriptor) in self.parts {
            let verified = hashed_blobs::Prepared::from_encoded(
                file,
                descriptor,
                None,
                ReadLimits::default(),
            )?;
            if several {
                length += std::io::copy(
                    &mut verified.reader(None, ReadLimits::default())?,
                    &mut whole,
                )?;
            }
        }
        if several
            && (whole.finalize().as_bytes() != &self.range.hash.0
                || length != self.range.decoded_length)
        {
            return invalid("raw custody chunks differ from their original body");
        }
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
    let Some(Capture::Input(input)) = prepared.capture() else {
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
    write_index(&path, &prepared.identity)?;
    if let Some(previous) = &input.raw_predecessor {
        let root = spool.index("raw-children", &hash(previous.as_bytes()));
        fs::create_dir_all(&root)?;
        write_index(
            &root.join(hash(prepared.identity.as_bytes())),
            &prepared.identity,
        )?;
    } else if input
        .raw_range
        .is_some_and(|range| range.start == generation.window().start)
    {
        let root = spool.index("raw-roots", &generation_key(generation)?);
        fs::create_dir_all(&root)?;
        write_index(
            &root.join(hash(prepared.identity.as_bytes())),
            &prepared.identity,
        )?;
    }
    Ok(())
}

pub(crate) struct RawMetadata {
    pub order: u64,
    pub generation: SourceGeneration,
    pub range: polylogue_store::primary::RawRange,
    pub predecessor: Option<String>,
}

impl Spool {
    pub fn raw_cut(&self) -> Result<u64> {
        self.ensure_indexes()?;
        self.next
            .get()
            .checked_sub(1)
            .ok_or_else(|| Error::Invalid("invalid custody queue cut".into()))
    }

    pub fn raw_metadata(&self, identity: &str) -> Result<Option<RawMetadata>> {
        let Some(prepared) = self.get(identity)? else {
            return Ok(None);
        };
        let Some(Capture::Input(input)) = prepared.capture().cloned() else {
            return Ok(None);
        };
        let (Some(generation), Some(range)) = (input.generation, input.raw_range) else {
            return Ok(None);
        };
        Ok(Some(RawMetadata {
            order: prepared.order,
            generation,
            range,
            predecessor: input.raw_predecessor,
        }))
    }

    pub fn pin_raw(&self, identity: &str) -> Result<PinnedRange> {
        let prepared = self.required(identity)?;
        let Some(Capture::Input(input)) = prepared.capture() else {
            return invalid("raw custody reference is not an input");
        };
        let generation = input
            .generation
            .clone()
            .ok_or_else(|| Error::Invalid("raw custody generation is unknown".into()))?;
        let range = input
            .raw_range
            .ok_or_else(|| Error::Invalid("raw custody has no provider interval".into()))?;
        let parts = self.encoded_body(identity)?;
        let named = match parts.as_slice() {
            [(_, one)] => one.hash.0 == input.hash.0 && one.decoded_len == input.length,
            several => several.iter().map(|(_, d)| d.decoded_len).sum::<u64>() == input.length,
        };
        if !named || range.length()? != input.length {
            return invalid("raw custody body differs from its original interval");
        }
        Ok(PinnedRange {
            order: prepared.order,
            range: RawBodyRange {
                generation,
                body: identity.into(),
                range,
                hash: input.hash,
                decoded_length: input.length,
                known_total: input.raw_total,
                predecessor: input.raw_predecessor.clone(),
                disposition: disposition(self, identity)?,
            },
            parts: parts
                .into_iter()
                .map(|(path, descriptor)| Ok((regular(&path)?, descriptor)))
                .collect::<Result<_>>()?,
        })
    }

    pub fn raw_roots(&self, generation: &SourceGeneration, through: u64) -> Result<Vec<String>> {
        self.ensure_indexes()?;
        let root = self.index("raw-roots", &generation_key(generation)?);
        self.raw_candidates(&root, generation, through, &mut |_, entry| {
            Ok(entry.predecessor.is_none() && entry.range.start == generation.window().start)
        })
    }

    pub fn raw_children(
        &self,
        identity: &str,
        generation: &SourceGeneration,
        through: u64,
    ) -> Result<Vec<String>> {
        self.ensure_indexes()?;
        let root = self.index("raw-children", &hash(identity.as_bytes()));
        self.raw_candidates(&root, generation, through, &mut |_, entry| {
            if entry.predecessor.as_deref() != Some(identity) {
                return invalid("raw successor index has another predecessor");
            }
            Ok(true)
        })
    }

    fn raw_candidates(
        &self,
        root: &Path,
        generation: &SourceGeneration,
        through: u64,
        accept: &mut dyn FnMut(&str, &RawMetadata) -> Result<bool>,
    ) -> Result<Vec<String>> {
        if !root.try_exists()? {
            return Ok(vec![]);
        }
        let mut result = Vec::new();
        for record in fs::read_dir(root)? {
            let record = record?;
            let identity = read_json::<String>(&record.path())?
                .ok_or_else(|| Error::Invalid("raw custody index disappeared".into()))?;
            if record.file_name().to_str() != Some(hash(identity.as_bytes()).as_str()) {
                return invalid("raw custody index has another body identity");
            }
            let entry = self
                .raw_metadata(&identity)?
                .ok_or_else(|| Error::Invalid("raw custody index body disappeared".into()))?;
            if entry.order <= through
                && &entry.generation == generation
                && accept(&identity, &entry)?
            {
                result.push(identity);
                if result.len() == 2 {
                    break;
                }
            }
        }
        Ok(result)
    }

    pub fn raw_count(&self, generation: &SourceGeneration, through: u64) -> Result<u64> {
        self.ensure_indexes()?;
        let root = self.index("raw-generations", &generation_key(generation)?);
        if !root.try_exists()? {
            return Ok(0);
        }
        let mut count = 0u64;
        walk(
            &root,
            0,
            0,
            None,
            through,
            usize::MAX,
            &mut |order, identity| {
                let entry = self
                    .raw_metadata(identity)?
                    .ok_or_else(|| Error::Invalid("raw custody index body disappeared".into()))?;
                if entry.order != order || &entry.generation != generation {
                    return invalid("raw custody index differs from its immutable generation");
                }
                count = count
                    .checked_add(1)
                    .ok_or_else(|| Error::Invalid("raw custody count overflow".into()))?;
                Ok(())
            },
        )?;
        Ok(count)
    }
}

/// Eight radix levels bound each directory read to 256 references. The page
/// cursor is queue order; unrelated parser payloads are never visited.
pub(super) fn walk(
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
