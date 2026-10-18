//! Choosing one version of a message when several inputs or revisions carry it.
//! Callers supply ranked candidates and per-field facts; nothing here reads a
//! message body. Reference integers are opaque handles and never break ties.
use crate::{BlobHash, BlockSlot, Timestamp};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RouteKind {
    PageDom,
    AccountExport,
    Native,
}

/// Unknown ranks between in-progress and finished; a missing lifecycle field is
/// Unknown, never Finished.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishState {
    InProgress,
    Finished,
    Unknown,
}

/// Fields a provider supplies. Identity, derived values and acquisition state
/// are not slots.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MessageField {
    PromptKind,
    Model,
    At,
    CompletedAt,
    Parent,
    FromSession,
    FromName,
    StopReason,
    DurationMs,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UsageField {
    Input,
    Output,
    CacheRead,
    CacheWrite,
    CacheWrite5m,
    CacheWrite1h,
    Reasoning,
    ServiceTier,
    ProviderCostNusd,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BlockField {
    Text,
    ThinkingText,
    Encrypted,
    ThinkingData,
    /// Stored with the body but never counted for completeness or used to fill.
    ThinkingSignature,
    Tool,
    CallId,
    Input,
    Command,
    Outcome,
    OutcomeReason,
    ExitCode,
    DurationMs,
    Output,
    Name,
    MediaType,
    Size,
    Url,
    Direction,
    ProducedBy,
    OtherData,
}
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Slot {
    Message(MessageField),
    Usage(UsageField),
    /// The caller must match blocks by meaning before filling from another
    /// version; equal positions in differently shaped messages are not enough.
    Block {
        block: BlockSlot,
        field: BlockField,
    },
    /// One slot per top-level extra key that is not volatile; the value is opaque.
    Extra {
        block: Option<BlockSlot>,
        key: String,
    },
}
impl Slot {
    fn is_parent(&self) -> bool {
        *self == Self::Message(MessageField::Parent)
    }
    fn is_time(&self) -> bool {
        matches!(
            self,
            Self::Message(MessageField::At | MessageField::CompletedAt)
        )
    }
}

/// A handle the writer resolves to a stored value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct FactRef {
    pub candidate: u64,
    pub value: u64,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SuppliedValue {
    pub reference: u64,
    /// Set only for source times, for Codex's damaged-time rule.
    pub source_time: Option<Timestamp>,
}

/// Counts of what a version supplies. Empty text blocks and their fields,
/// decoder defaults, derived values and volatile keys do not count; false, zero
/// and known-empty values do. Each usage field and each extra key is its own slot.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PresenceFacts {
    pub nonempty_blocks: u64,
    pub blocks_with_nonempty_text: u64,
    #[serde(with = "slot_map")]
    pub slots: BTreeMap<Slot, SuppliedValue>,
}
impl PresenceFacts {
    pub fn completeness(&self) -> Result<Completeness, MergeError> {
        if self.blocks_with_nonempty_text > self.nonempty_blocks
            || self.slots.keys().any(|slot| {
                matches!(
                    slot,
                    Slot::Block {
                        field: BlockField::ThinkingSignature,
                        ..
                    }
                )
            })
        {
            return Err(MergeError::InvalidPresence);
        }
        Ok(Completeness {
            blocks: self.nonempty_blocks,
            text_blocks: self.blocks_with_nonempty_text,
            supplied_fields: u64::try_from(self.slots.len())
                .map_err(|_| MergeError::InvalidPresence)?,
        })
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Completeness {
    pub blocks: u64,
    pub text_blocks: u64,
    pub supplied_fields: u64,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Candidate {
    pub reference: u64,
    pub finish: FinishState,
    pub route: RouteKind,
    /// None until the version hash can be computed.
    pub version_hash: Option<BlobHash>,
    pub presence: PresenceFacts,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RankSummary {
    pub finish: FinishState,
    pub route: RouteKind,
    pub completeness: Completeness,
    pub version_hash: BlobHash,
}
impl Candidate {
    pub fn rank(&self) -> Result<RankSummary, MergeError> {
        let rank = RankSummary {
            finish: self.finish,
            route: self.route,
            completeness: self.presence.completeness()?,
            version_hash: self.version_hash.ok_or(MergeError::UnresolvedEvidence)?,
        };
        rank.prefix()?;
        Ok(rank)
    }
}
impl RankSummary {
    /// For comparing inputs, or chains already reduced to one version. Revisions
    /// within a chain are compared in `fold_chain`.
    pub fn compare(&self, other: &Self) -> Result<std::cmp::Ordering, MergeError> {
        Ok(self.key()?.cmp(&other.key()?))
    }
    fn prefix(&self) -> Result<(u8, RouteKind, Completeness), MergeError> {
        if self.completeness.text_blocks > self.completeness.blocks {
            return Err(MergeError::InvalidPresence);
        }
        let completion = match self.finish {
            FinishState::InProgress => 0,
            FinishState::Unknown => 1,
            FinishState::Finished => 2,
        };
        Ok((completion, self.route, self.completeness))
    }
    fn key(&self) -> Result<((u8, RouteKind, Completeness), [u8; 32]), MergeError> {
        Ok((self.prefix()?, *self.version_hash.bytes()))
    }
}

/// `candidate: None` means an edit snapshot deleted this message.
#[derive(Clone, Copy, Debug)]
pub struct Revision<'a> {
    pub length: u64,
    pub candidate: Option<&'a Candidate>,
}
#[derive(Clone, Copy, Debug)]
pub struct Chain<'a> {
    /// Whether this chain is the input's current one. The caller decides; it is
    /// never inferred from counts, IDs, times or arrival order.
    pub current: bool,
    pub revisions: &'a [Revision<'a>],
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RewritePolicy {
    Lossy { preserve_older_times: bool },
    Edit,
}

/// The chosen body. Versions with equal rank and hash are one choice that keeps
/// every candidate.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BodyChoice {
    pub rank: RankSummary,
    pub candidates: BTreeSet<u64>,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SlotChoice {
    pub sources: BTreeSet<FactRef>,
}

/// Whether the versions that supply a field agree. Conflict is not the same as
/// missing, and no reference wins over another with a different value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FieldResolution {
    Missing,
    Agreed(FactRef),
    Conflict,
}
impl FieldResolution {
    /// Add one supplying version. `equal` compares typed values, including time
    /// precision, never references. An error leaves the state unchanged.
    pub fn observe<E>(
        &mut self,
        value: FactRef,
        equal: impl FnOnce(FactRef, FactRef) -> Result<bool, E>,
    ) -> Result<(), E> {
        match *self {
            Self::Missing => *self = Self::Agreed(value),
            Self::Agreed(first) => {
                if !equal(first, value)? {
                    *self = Self::Conflict;
                }
            }
            Self::Conflict => {}
        }
        Ok(())
    }
}
impl SlotChoice {
    /// `observe` every source in order.
    pub fn resolve<E>(
        &self,
        mut equal: impl FnMut(FactRef, FactRef) -> Result<bool, E>,
    ) -> Result<FieldResolution, E> {
        let mut result = FieldResolution::Missing;
        for &value in &self.sources {
            result.observe(value, &mut equal)?;
        }
        Ok(result)
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EffectiveInput {
    pub body: BodyChoice,
    /// Field sources per slot; they never change `body.rank`.
    #[serde(with = "slot_map")]
    pub slots: BTreeMap<Slot, SlotChoice>,
}
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum MergeError {
    #[error("source presence facts are inconsistent")]
    InvalidPresence,
    #[error("required version evidence remains unresolved")]
    UnresolvedEvidence,
    #[error("an input has more than one current chain")]
    MultipleCurrentChains,
    /// Temporary: a later snapshot that names the current chain resolves it.
    #[error("edit snapshots have no current chain yet")]
    UnresolvedEditHead,
    #[error("duplicate chain revision length")]
    DuplicateRevision,
    #[error("candidate reference is repeated")]
    DuplicateCandidate,
    #[error("declared damaged time lacks source-time evidence")]
    MissingDamagedTime,
}

struct FoldedChain {
    current: bool,
    body: BodyChoice,
    slots: BTreeMap<Slot, SlotChoice>,
}
fn source(candidate: &Candidate, value: &SuppliedValue) -> FactRef {
    FactRef {
        candidate: candidate.reference,
        value: value.reference,
    }
}
fn one_slot(candidate: &Candidate, value: &SuppliedValue) -> SlotChoice {
    SlotChoice {
        sources: BTreeSet::from([source(candidate, value)]),
    }
}
fn body_from(candidate: &Candidate) -> Result<BodyChoice, MergeError> {
    Ok(BodyChoice {
        rank: candidate.rank()?,
        candidates: BTreeSet::from([candidate.reference]),
    })
}
fn fold_chain(chain: &Chain<'_>, edit: bool) -> Result<Option<FoldedChain>, MergeError> {
    let mut lengths = BTreeSet::new();
    for rev in chain.revisions {
        if !lengths.insert(rev.length) {
            return Err(MergeError::DuplicateRevision);
        }
    }
    let considered: Vec<_> = if edit {
        chain
            .revisions
            .iter()
            .max_by_key(|r| r.length)
            .into_iter()
            .collect()
    } else {
        chain.revisions.iter().collect()
    };
    let mut best = None;
    for revision in &considered {
        if let Some(candidate) = revision.candidate {
            let rank = candidate.rank()?;
            // On equal rank prefix, the later revision wins before the hash decides.
            let key = (rank.prefix()?, revision.length, *rank.version_hash.bytes());
            if best.as_ref().is_none_or(|(old, _)| key > *old) {
                best = Some((key, candidate));
            }
        }
    }
    let Some((_, selected)) = best else {
        return Ok(None);
    };
    let body = body_from(selected)?;
    let mut slots = BTreeMap::new();
    for revision in &considered {
        let Some(candidate) = revision.candidate else {
            continue;
        };
        let candidate_rank = candidate.rank()?;
        let key = (
            candidate_rank.prefix()?,
            revision.length,
            *candidate_rank.version_hash.bytes(),
        );
        for (slot, value) in &candidate.presence.slots {
            if slot.is_parent() && candidate.reference != selected.reference {
                continue;
            }
            let replace = match slots.get(slot) {
                None => true,
                Some((old, _)) => key > *old,
            };
            if replace {
                slots.insert(slot.clone(), (key, one_slot(candidate, value)));
            }
        }
    }
    Ok(Some(FoldedChain {
        current: chain.current,
        body,
        slots: slots.into_iter().map(|(s, (_, v))| (s, v)).collect(),
    }))
}

/// Reduce one input: first within each chain, then across chains. Under the edit
/// policy only the latest snapshot counts, including a deletion; older edits
/// cannot refill a deleted message.
pub fn fold_input(
    chains: &[Chain<'_>],
    policy: RewritePolicy,
) -> Result<Option<EffectiveInput>, MergeError> {
    if chains.iter().filter(|c| c.current).count() > 1 {
        return Err(MergeError::MultipleCurrentChains);
    }
    let mut refs = BTreeMap::new();
    for candidate in chains
        .iter()
        .flat_map(|c| c.revisions)
        .filter_map(|r| r.candidate)
    {
        if let Some(previous) = refs.insert(candidate.reference, candidate)
            && previous != candidate
        {
            return Err(MergeError::DuplicateCandidate);
        }
    }
    if policy == RewritePolicy::Edit {
        let current = match chains.iter().find(|c| c.current) {
            Some(current) => current,
            None => return Err(MergeError::UnresolvedEditHead),
        };
        if current.revisions.is_empty() {
            return Err(MergeError::UnresolvedEditHead);
        }
        return Ok(fold_chain(current, true)?.map(|chain| EffectiveInput {
            body: chain.body,
            slots: chain.slots,
        }));
    }
    let folded: Vec<_> = chains
        .iter()
        .map(|c| fold_chain(c, false))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .flatten()
        .collect();
    let mut best_key = None;
    let mut best_indices = Vec::new();
    for (index, chain) in folded.iter().enumerate() {
        let rank = &chain.body.rank;
        let key = (rank.prefix()?, chain.current, *rank.version_hash.bytes());
        match best_key.as_ref().map(|old| key.cmp(old)) {
            None | Some(std::cmp::Ordering::Greater) => {
                best_key = Some(key);
                best_indices = vec![index];
            }
            Some(std::cmp::Ordering::Equal) => best_indices.push(index),
            _ => {}
        }
    }
    let Some(&first) = best_indices.first() else {
        return Ok(None);
    };
    let mut body = folded[first].body.clone();
    for &index in best_indices.iter().skip(1) {
        body.candidates.extend(&folded[index].body.candidates);
    }
    let mut chosen_slots = BTreeMap::new();
    for (index, chain) in folded.iter().enumerate() {
        let rank = &chain.body.rank;
        let key = (rank.prefix()?, chain.current, *rank.version_hash.bytes());
        for (slot, choice) in &chain.slots {
            if slot.is_parent() && !best_indices.contains(&index) {
                continue;
            }
            match chosen_slots.get_mut(slot) {
                None => {
                    chosen_slots.insert(slot.clone(), (key, choice.clone()));
                }
                Some((old, existing)) if key > *old => {
                    *old = key;
                    *existing = choice.clone();
                }
                Some((old, existing)) if key == *old => existing.sources.extend(&choice.sources),
                _ => {}
            }
        }
    }
    let mut slots: BTreeMap<_, _> = chosen_slots.into_iter().map(|(s, (_, v))| (s, v)).collect();
    if matches!(
        policy,
        RewritePolicy::Lossy {
            preserve_older_times: true
        }
    ) {
        let mut times: BTreeMap<Slot, (Timestamp, SlotChoice)> = BTreeMap::new();
        for candidate in chains
            .iter()
            .filter(|c| !c.current)
            .flat_map(|c| c.revisions)
            .filter_map(|r| r.candidate)
        {
            for (slot, value) in candidate
                .presence
                .slots
                .iter()
                .filter(|(slot, _)| slot.is_time())
            {
                let time = value.source_time.ok_or(MergeError::MissingDamagedTime)?;
                match times.get_mut(slot) {
                    None => {
                        times.insert(slot.clone(), (time, one_slot(candidate, value)));
                    }
                    Some((old, choice)) if time < *old => {
                        *old = time;
                        *choice = one_slot(candidate, value);
                    }
                    Some((old, choice)) if time == *old => {
                        choice.sources.insert(source(candidate, value));
                    }
                    _ => {}
                }
            }
        }
        for (slot, (_, choice)) in times {
            slots.insert(slot, choice);
        }
    }
    Ok(Some(EffectiveInput { body, slots }))
}

/// Choose between inputs, each already reduced by `fold_input`. When an input
/// is removed or replaced, call this again on the new set; the previous winner
/// has no special standing.
pub fn select_effective(inputs: &[EffectiveInput]) -> Result<Option<EffectiveInput>, MergeError> {
    let mut best = None;
    let mut indices = Vec::new();
    for (index, input) in inputs.iter().enumerate() {
        let key = input.body.rank.key()?;
        match best.as_ref().map(|old| key.cmp(old)) {
            None | Some(std::cmp::Ordering::Greater) => {
                best = Some(key);
                indices = vec![index];
            }
            Some(std::cmp::Ordering::Equal) => indices.push(index),
            _ => {}
        }
    }
    let Some(&first) = indices.first() else {
        return Ok(None);
    };
    let mut body = inputs[first].body.clone();
    for &index in indices.iter().skip(1) {
        body.candidates.extend(&inputs[index].body.candidates);
    }
    let mut selected = BTreeMap::new();
    for (index, input) in inputs.iter().enumerate() {
        let key = input.body.rank.key()?;
        for (slot, choice) in &input.slots {
            if slot.is_parent() && !indices.contains(&index) {
                continue;
            }
            match selected.get_mut(slot) {
                None => {
                    selected.insert(slot.clone(), (key, choice.clone()));
                }
                Some((old, existing)) if key > *old => {
                    *old = key;
                    *existing = choice.clone();
                }
                Some((old, existing)) if key == *old => existing.sources.extend(&choice.sources),
                _ => {}
            }
        }
    }
    Ok(Some(EffectiveInput {
        body,
        slots: selected.into_iter().map(|(s, (_, v))| (s, v)).collect(),
    }))
}

// JSON object keys cannot represent a typed Slot. Encode ordered pairs and reject
// repeated slots on read rather than silently dropping a supplied fact.
mod slot_map {
    use super::Slot;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::collections::BTreeMap;
    pub fn serialize<S: Serializer, V: Serialize>(
        value: &BTreeMap<Slot, V>,
        ser: S,
    ) -> Result<S::Ok, S::Error> {
        ser.collect_seq(value.iter())
    }
    pub fn deserialize<'de, D: Deserializer<'de>, V: Deserialize<'de>>(
        de: D,
    ) -> Result<BTreeMap<Slot, V>, D::Error> {
        let pairs = Vec::<(Slot, V)>::deserialize(de)?;
        let mut result = BTreeMap::new();
        for (slot, value) in pairs {
            if result.insert(slot, value).is_some() {
                return Err(serde::de::Error::custom("duplicate semantic slot"));
            }
        }
        Ok(result)
    }
}
