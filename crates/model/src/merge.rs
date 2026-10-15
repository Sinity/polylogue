//! Pure per-message selection. Source mappers supply facts before fill/enrichment.
//! The caller retains revisions and calls this again when their eligibility changes.
//! No function reads a normalized Message or hashes a body. Reference integers are
//! opaque handles; their values never break content/rank ties.
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

/// Unknown remains unresolved until a provider rule supplies a classification.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishState {
    InProgress,
    Finished,
    Unknown,
}

/// Only source-supplied semantic fields may name slots. Own identity, derived
/// values, structural tags and acquisition state have no slots here.
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
    ProviderCostNusd,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BlockField {
    Text,
    ThinkingText,
    Encrypted,
    ThinkingData,
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
    /// The caller establishes semantic correspondence before using a donor:
    /// coincident ordinal positions in incompatible block shapes are insufficient.
    Block {
        block: BlockSlot,
        field: BlockField,
    },
    /// Each immediate nonvolatile extra key counts once; its payload is opaque.
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

/// The canonical writer resolves these handles; the model never loads payloads.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct FactRef {
    pub candidate: u64,
    pub value: u64,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SuppliedValue {
    pub reference: u64,
    /// Only source times use this, for the declared Codex damaged-time exception.
    pub source_time: Option<Timestamp>,
}

/// The mapper excludes empty text blocks AND their fields, semantic unknowns,
/// decoder defaults, derived facts, acquisition facts and declared volatile keys.
/// False, zero and known-empty payloads remain supplied facts. Arbitrary payloads
/// occupy one slot; typed usage leaves and immediate extras occupy separate slots.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PresenceFacts {
    pub nonempty_blocks: u64,
    pub blocks_with_nonempty_text: u64,
    #[serde(with = "slot_map")]
    pub slots: BTreeMap<Slot, SuppliedValue>,
}
impl PresenceFacts {
    pub fn completeness(&self) -> Result<Completeness, MergeError> {
        if self.blocks_with_nonempty_text > self.nonempty_blocks {
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
    /// None means required source evidence is unresolved, never a synthetic hash.
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
    /// Only cross-input or already-reduced rank comparison, never historical
    /// same-input revision selection.
    pub fn compare(&self, other: &Self) -> Result<std::cmp::Ordering, MergeError> {
        Ok(self.key()?.cmp(&other.key()?))
    }
    fn prefix(&self) -> Result<(bool, RouteKind, Completeness), MergeError> {
        if self.completeness.text_blocks > self.completeness.blocks {
            return Err(MergeError::InvalidPresence);
        }
        let finished = match self.finish {
            FinishState::InProgress => false,
            FinishState::Finished => true,
            FinishState::Unknown => return Err(MergeError::UnknownCompletion),
        };
        Ok((finished, self.route, self.completeness))
    }
    fn key(&self) -> Result<((bool, RouteKind, Completeness), [u8; 32]), MergeError> {
        Ok((self.prefix()?, *self.version_hash.bytes()))
    }
}

/// An absent candidate is an explicit edit-snapshot omission of this MessageID.
#[derive(Clone, Copy, Debug)]
pub struct Revision<'a> {
    pub length: u64,
    pub candidate: Option<&'a Candidate>,
}
#[derive(Clone, Copy, Debug)]
pub struct Chain<'a> {
    /// Caller-supplied head authority, never inferred from snapshot count, IDs,
    /// times or arrival. The sole chain of a proved complete immutable import
    /// may be current for its own input without asserting provider freshness.
    pub current: bool,
    pub revisions: &'a [Revision<'a>],
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RewritePolicy {
    Lossy { preserve_older_times: bool },
    Edit,
}

/// Equal-rank/equal-digest versions are one content choice with all provenance.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BodyChoice {
    pub rank: RankSummary,
    pub candidates: BTreeSet<u64>,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SlotChoice {
    pub sources: BTreeSet<FactRef>,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EffectiveInput {
    pub body: BodyChoice,
    /// Selected/fill facts retained per input; they never rescore `body.rank`.
    #[serde(with = "slot_map")]
    pub slots: BTreeMap<Slot, SlotChoice>,
}
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum MergeError {
    #[error("source presence facts are inconsistent")]
    InvalidPresence,
    #[error("source completion is unknown; ranking remains unresolved")]
    UnknownCompletion,
    #[error("required version evidence remains unresolved")]
    UnresolvedEvidence,
    #[error("an input has more than one current chain")]
    MultipleCurrentChains,
    /// History remains retained and another input may still contribute. A later
    /// authoritative head resolves this; it is not permanent source refusal.
    #[error("edit snapshots have no authoritative current head yet")]
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
    let eligible: Vec<_> = if edit {
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
    for revision in &eligible {
        if let Some(candidate) = revision.candidate {
            let rank = candidate.rank()?;
            // Same-chain later revision wins a rank-prefix tie BEFORE the digest.
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
    for revision in &eligible {
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

/// First reduce within each chain, then between chain representatives. For edits,
/// only the latest authoritative snapshot is eligible, including its absent-message
/// tombstone; older edit fields cannot fill a deliberate deletion.
/// This batch helper allocates in proportion to supplied candidate/slot metadata.
/// It is not a bounded durable scheduler or a whole-archive retained state.
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
        if let Some(previous) = refs.insert(candidate.reference, candidate) {
            if previous != candidate {
                return Err(MergeError::DuplicateCandidate);
            }
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

/// Compare ONLY one already-folded representative per input. Historical versions
/// never compete here. Removing/replacing an effective input requires re-election,
/// not a monotonic merge against the previous global winner.
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
