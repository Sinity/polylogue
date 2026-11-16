//! Provider byte-range bodies, including groups accepted with an unresolved
//! base conflict. Queue metadata
//! selects bodies without decoding their contents.
use super::*;
use polylogue_store::primary::{
    BodyRange, SourceGeneration,
    queue::{Lookup, Order},
};

pub(crate) struct PinnedRange {
    pub order: u64,
    range: BodyRange,
    parts: Vec<(File, EncodedBlob)>,
}

impl PinnedRange {
    /// Payload verification runs after releasing the producer capture lock.
    /// One encoding names the body; chunks are read through and hashed as
    /// the body.
    pub fn verify(self) -> Result<BodyRange> {
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
            return invalid("captured chunks differ from their original body");
        }
        Ok(self.range)
    }
}

fn group_outcome(
    spool: &Spool,
    identity: &str,
) -> Result<Option<polylogue_store::primary::CaptureOutcome>> {
    let operation = spool.operation(identity)?;
    let parent = operation.parent().expect("operation parent");
    if parent == spool.root.join("operations") {
        return Ok(None);
    }
    let group = parent.parent().expect("capture group directory");
    let header = read_checked::<CaptureGroupHeader>(&group.join("acquisition.json"))?
        .ok_or_else(|| Error::invalid("body has no published capture group header"))?;
    if group.file_name().and_then(|name| name.to_str())
        != Some(hash(header.identity.as_bytes()).as_str())
    {
        return invalid("capture group header has another identity");
    }
    Ok(Some(header.disposition))
}

pub(crate) struct BodyMetadata {
    pub order: u64,
    pub generation: SourceGeneration,
    pub range: polylogue_store::primary::RawRange,
    pub previous: Option<String>,
}

impl Spool {
    pub fn latest_order(&self) -> Result<u64> {
        Ok(self.queue.last_order()?)
    }

    pub fn body_metadata(&self, identity: &str) -> Result<Option<BodyMetadata>> {
        let Some(prepared) = self.get(identity)? else {
            return Ok(None);
        };
        let Some(Capture::Input(input)) = prepared.capture().cloned() else {
            return Ok(None);
        };
        let (Some(generation), Some(range)) = (input.generation, input.raw_range) else {
            return Ok(None);
        };
        Ok(Some(BodyMetadata {
            order: prepared.order,
            generation,
            range,
            previous: input.raw_predecessor,
        }))
    }

    pub fn pin_body_range(&self, identity: &str) -> Result<PinnedRange> {
        let prepared = self.required(identity)?;
        let Some(Capture::Input(input)) = prepared.capture() else {
            return invalid("body range reference is not an input");
        };
        let generation = input
            .generation
            .clone()
            .ok_or_else(|| Error::invalid("captured body's observed source state is unknown"))?;
        let range = input
            .raw_range
            .ok_or_else(|| Error::invalid("captured body has no provider byte range"))?;
        let parts = self.encoded_body(identity)?;
        let named = match parts.as_slice() {
            [(_, one)] => one.hash.0 == input.hash.0 && one.decoded_len == input.length,
            several => several.iter().map(|(_, d)| d.decoded_len).sum::<u64>() == input.length,
        };
        if !named || range.length()? != input.length {
            return invalid("captured body differs from its original byte range");
        }
        Ok(PinnedRange {
            order: prepared.order,
            range: BodyRange {
                generation,
                body: identity.into(),
                range,
                hash: input.hash,
                decoded_length: input.length,
                known_total: input.raw_total,
                previous: input.raw_predecessor.clone(),
                outcome: group_outcome(self, identity)?,
            },
            parts: parts
                .into_iter()
                .map(|(path, descriptor)| Ok((regular(&path)?, descriptor)))
                .collect::<Result<_>>()?,
        })
    }

    pub fn body_roots(&self, generation: &SourceGeneration, through: u64) -> Result<Vec<String>> {
        self.body_branches(None, generation, through)
    }

    pub fn body_children(
        &self,
        identity: &str,
        generation: &SourceGeneration,
        through: u64,
    ) -> Result<Vec<String>> {
        self.body_branches(Some(identity), generation, through)
    }

    fn body_branches(
        &self,
        parent: Option<&str>,
        generation: &SourceGeneration,
        through: u64,
    ) -> Result<Vec<String>> {
        // A second matching branch is enough for the caller to reject an
        // ambiguous continuation. Other generations do not make a branch.
        let observed_key = serde_json::to_vec(generation)?;
        let mut after = None;
        let mut result = Vec::new();
        loop {
            let entries = self.queue.lookup(
                match parent {
                    Some(identity) => Lookup::RawChildren(identity),
                    None => Lookup::RawRoots(&observed_key),
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

    pub fn body_count(&self, generation: &SourceGeneration, through: u64) -> Result<u64> {
        let generation = serde_json::to_vec(generation)?;
        let mut after = None;
        let mut count = 0u64;
        loop {
            let entries = self.queue.lookup(
                Lookup::ObservedSourceKey(&generation),
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
                .ok_or_else(|| Error::invalid("captured body count overflow"))?;
        }
    }
}
