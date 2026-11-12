//! Provider body capture, including unresolved acquisitions. Queue metadata
//! selects bodies without decoding their contents.
use super::*;
use polylogue_store::primary::{
    RawBodyRange, SourceGeneration,
    queue::{Lookup, Order},
};

pub(crate) struct PinnedRange {
    pub order: u64,
    range: RawBodyRange,
    parts: Vec<(File, EncodedBlob)>,
}

impl PinnedRange {
    /// Payload verification runs after releasing the producer capture lock.
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
            return invalid("raw capture chunks differ from their original body");
        }
        Ok(self.range)
    }
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
    let acknowledgment =
        read_checked::<Acquisition>(&group.join("acquisition.json"))?.ok_or_else(|| {
            Error::Invalid("raw body has no published acquisition acknowledgment".into())
        })?;
    if group.file_name().and_then(|name| name.to_str())
        != Some(hash(acknowledgment.identity.as_bytes()).as_str())
    {
        return invalid("raw acquisition acknowledgment has another identity");
    }
    Ok(Some(acknowledgment.disposition))
}

pub(crate) struct RawMetadata {
    pub order: u64,
    pub generation: SourceGeneration,
    pub range: polylogue_store::primary::RawRange,
    pub predecessor: Option<String>,
}

impl Spool {
    pub fn raw_cut(&self) -> Result<u64> {
        Ok(self.queue.last_order()?)
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
            return invalid("raw capture reference is not an input");
        };
        let generation = input
            .generation
            .clone()
            .ok_or_else(|| Error::Invalid("raw capture generation is unknown".into()))?;
        let range = input
            .raw_range
            .ok_or_else(|| Error::Invalid("raw capture has no provider interval".into()))?;
        let parts = self.encoded_body(identity)?;
        let named = match parts.as_slice() {
            [(_, one)] => one.hash.0 == input.hash.0 && one.decoded_len == input.length,
            several => several.iter().map(|(_, d)| d.decoded_len).sum::<u64>() == input.length,
        };
        if !named || range.length()? != input.length {
            return invalid("raw capture body differs from its original interval");
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
        self.raw_candidates(None, generation, through)
    }

    pub fn raw_children(
        &self,
        identity: &str,
        generation: &SourceGeneration,
        through: u64,
    ) -> Result<Vec<String>> {
        self.raw_candidates(Some(identity), generation, through)
    }

    fn raw_candidates(
        &self,
        predecessor: Option<&str>,
        generation: &SourceGeneration,
        through: u64,
    ) -> Result<Vec<String>> {
        // A second matching branch is enough for the caller to reject an
        // ambiguous continuation. Other generations do not make a branch.
        let generation_key = serde_json::to_vec(generation)?;
        let mut after = None;
        let mut result = Vec::new();
        loop {
            let entries = self.queue.lookup(
                match predecessor {
                    Some(identity) => Lookup::RawChildren(identity),
                    None => Lookup::RawRoots(&generation_key),
                },
                after,
                Some(through),
                Order::OldestFirst,
                64,
            )?;
            if entries.is_empty() {
                return Ok(result);
            }
            for entry in entries {
                after = Some(entry.order);
                let prepared = self.prepared(entry)?;
                if matches!(prepared.capture(), Some(Capture::Input(input)) if input.generation.as_ref() == Some(generation))
                {
                    result.push(prepared.identity.clone());
                    if result.len() == 2 {
                        return Ok(result);
                    }
                }
            }
        }
    }

    pub fn raw_count(&self, generation: &SourceGeneration, through: u64) -> Result<u64> {
        let generation = serde_json::to_vec(generation)?;
        let mut after = None;
        let mut count = 0u64;
        loop {
            let entries = self.queue.lookup(
                Lookup::RawGeneration(&generation),
                after,
                Some(through),
                Order::OldestFirst,
                64,
            )?;
            if entries.is_empty() {
                return Ok(count);
            }
            after = entries.last().map(|entry| entry.order);
            count = count
                .checked_add(entries.len() as u64)
                .ok_or_else(|| Error::Invalid("raw capture count overflow".into()))?;
        }
    }
}
