//! One-way conservation at numbered archive and exact native-version boundaries.
//! Retired actions are private decoding types and have no accepting dispatch.
mod legacy;
mod local;
use super::*;
#[cfg(test)]
pub(crate) use local::migrate_query_runs;
pub(crate) use local::migrate_source;
use polylogue_model::{
    authored::{AuthoredEffect, PreparedRequest},
    historical::{
        AcceptedEncoding, AcceptedHashEvidence, ConservedAccepted, HistoricalAcceptedRequest,
        NativeProjectionHistory, NativeProjectionObservation, NativeProjectionProvenance,
        OriginalBytes, SourceSchema,
    },
    user::{UserOutcome, UserRequest},
};

/// Callers establish the exact source schema before entry. A local migration
/// supplies its recorded hash; native projection supplies only actually fetched
/// envelope metadata and derives semantic identity by the declared old rule.
pub fn conserve_request(
    source: SourceSchema,
    request_json: &[u8],
    outcome_json: Option<&[u8]>,
    effect_json: Option<&[u8]>,
    expected_hash: Option<BlobHash>,
    native: Option<NativeProjectionProvenance>,
) -> Result<ConservedAccepted> {
    let old: legacy::Request = crate::encoding::read_json(request_json)?;
    let failure = |message: &str| {
        StoreError::Invalid(format!("request {} from {source:?}: {message}", old.id))
    };
    if !old.write.is_predecessor() {
        return Err(failure(
            "the action is not in its declared predecessor schema",
        ));
    }
    if old.id.is_empty() {
        return Err(failure("empty original identity"));
    }
    match (&source, &native) {
        (SourceSchema::PolylogueArchive { version: 4 }, None) if expected_hash.is_some() => {}
        (
            SourceSchema::LocalProjection {
                index_version: 14,
                source_schema_version: 4,
            },
            None,
        ) if expected_hash.is_some() => {}
        (
            SourceSchema::LocalProjection {
                index_version: 14,
                source_schema_version: 4,
            },
            None,
        ) if expected_hash.is_none()
            && old.authorship.is_some()
            && old.at.is_some()
            && outcome_json.is_some()
            && effect_json.is_some() => {}
        (
            SourceSchema::NativeProjection {
                source_version: 3,
                kind_version: 4,
                ..
            },
            Some(_),
        ) if expected_hash.is_none()
            && old.authorship.is_some()
            && old.at.is_some()
            && outcome_json.is_some()
            && effect_json.is_some() => {}
        (
            SourceSchema::RetainedPublication {
                source_version: 3,
                parser,
                parser_version: 3,
                ..
            },
            None,
        ) if matches!(
            parser.as_str(),
            "polylogue-request" | "polylogue-request-restore"
        ) && expected_hash.is_none()
            && old.authorship.is_some()
            && old.at.is_some()
            && outcome_json.is_some()
            && effect_json.is_some() => {}
        _ => return Err(failure("missing or unsupported original source encoding")),
    }
    if let SourceSchema::NativeProjection { kind, .. } = &source {
        let expected = old
            .write
            .source3_kind()
            .ok_or_else(|| failure("the action is not in the frozen Source3 model"))?;
        if kind != expected && kind != "polylogue.request_restore" {
            return Err(failure(
                "the decoded action disagrees with its actual frozen native kind",
            ));
        }
    }
    if matches!(source, SourceSchema::RetainedPublication { .. })
        && old.write.source3_kind().is_none()
    {
        return Err(failure(
            "the queued action is not in the frozen Source3 model",
        ));
    }
    let mut native_history = NativeProjectionHistory::default();
    if let Some(provenance) = native {
        native_history
            .observe(NativeProjectionObservation {
                source: source.clone(),
                provenance,
            })
            .map_err(|error| failure(&error.to_string()))?;
    }
    let (intent_bytes, rule) = old.intent_bytes()?;
    let hash = BlobHash::of(&intent_bytes);
    if expected_hash.is_some_and(|expected| expected != hash) {
        return Err(failure(
            "original semantic bytes do not match the recorded content hash",
        ));
    }
    let hash_evidence = if expected_hash.is_some() {
        AcceptedHashEvidence::Observed
    } else {
        AcceptedHashEvidence::Derived { rule }
    };
    let outcome = outcome_json
        .map(crate::encoding::read_json::<UserOutcome>)
        .transpose()?;
    if outcome
        .as_ref()
        .is_some_and(|outcome| outcome.pending || outcome.id.is_empty())
    {
        return Err(failure(
            "the original outcome is not a complete accepted outcome",
        ));
    }
    let effects = effect_json
        .map(|bytes| legacy::effects(&source, Some(&old), bytes))
        .transpose()?;
    let original = OriginalBytes {
        cached: vec![],
        request: Some(request_json.to_vec()),
        intent: Some(intent_bytes),
        outcome: outcome_json.map(<[u8]>::to_vec),
        effect: effect_json.map(<[u8]>::to_vec),
        records: Vec::new(),
        imports: None,
    };
    if let (Some(write), Some(authorship), Some(at), Some(outcome), Some(effects)) = (
        old.write.current(),
        old.authorship.clone(),
        old.at,
        outcome.as_ref(),
        effects.as_ref(),
    ) && effects.historical_assertions.is_empty()
    {
        let request = UserRequest {
            id: old.id.clone(),
            machine: old.machine.clone(),
            at,
            authorship,
            write,
        };
        if crate::primary::authored::request_hash(&request)? == hash.0 {
            let prepared = PreparedRequest {
                request,
                outcome: outcome.clone(),
                effect: effects.current.clone(),
                original: Some(AcceptedEncoding {
                    source: source.clone(),
                    content_hash: hash,
                    hash_evidence,
                    original,
                    native: native_history,
                }),
            };
            crate::primary::authored::validate(&prepared)?;
            return Ok(ConservedAccepted::Compatible(Box::new(prepared)));
        }
    }
    let facts = effects
        .as_ref()
        .map(legacy::fact_references)
        .unwrap_or_default();
    let record = HistoricalAcceptedRequest {
        id: old.id.clone(),
        source,
        content_hash: hash,
        hash_evidence,
        original: Some(original),
        intent: Some(old.write.historical()),
        machine: Some(old.machine),
        at: old.at,
        authorship: old.authorship,
        outcome,
        conserved_effects: effects,
        facts,
        imports: Default::default(),
        native: native_history,
    };
    record
        .validate()
        .map_err(|error| StoreError::Invalid(error.to_string()))?;
    Ok(ConservedAccepted::Historical(Box::new(record)))
}

/// Numbered local upgrades may find an outcome/effect whose original request
/// was never retained. Known bytes remain evidence; no intent/hash verification
/// or accepting machine is reconstructed from a fact or import observation.
pub(crate) fn conserve_without_request(
    source: SourceSchema,
    id: String,
    content_hash: BlobHash,
    at: Option<polylogue_model::Timestamp>,
    outcome_json: Option<&[u8]>,
    effect_json: Option<&[u8]>,
) -> Result<ConservedAccepted> {
    if !matches!(
        source,
        SourceSchema::PolylogueArchive { version: 4 }
            | SourceSchema::LocalProjection {
                index_version: 14,
                source_schema_version: 4
            }
    ) {
        return Err(invalid(
            "missing original request bodies are only conserved by the numbered local upgrade",
        ));
    }
    let outcome = outcome_json
        .map(crate::encoding::read_json::<UserOutcome>)
        .transpose()?;
    if outcome
        .as_ref()
        .is_some_and(|outcome| outcome.pending || outcome.id.is_empty())
    {
        return Err(invalid(
            "the retained outcome is not a complete accepted outcome",
        ));
    }
    let conserved_effects = effect_json
        .map(|bytes| legacy::effects(&source, None, bytes))
        .transpose()?;
    let facts = conserved_effects
        .as_ref()
        .map(legacy::fact_references)
        .unwrap_or_default();
    let record = HistoricalAcceptedRequest {
        id,
        source,
        content_hash,
        hash_evidence: AcceptedHashEvidence::Observed,
        original: Some(OriginalBytes {
            cached: vec![],
            request: None,
            intent: None,
            outcome: outcome_json.map(<[u8]>::to_vec),
            effect: effect_json.map(<[u8]>::to_vec),
            records: Vec::new(),
            imports: None,
        }),
        intent: None,
        machine: None,
        at,
        authorship: None,
        outcome,
        conserved_effects,
        facts,
        imports: Default::default(),
        native: Default::default(),
    };
    record
        .validate()
        .map_err(|error| invalid(&error.to_string()))?;
    Ok(ConservedAccepted::Historical(Box::new(record)))
}

#[cfg(test)]
mod tests;

/// The SDK uses the original retained action's frozen kind when continuing its
/// exact Source3 publication. This does not construct a current Product or intent.
fn retained_source3_request(accepted: &ConservedAccepted) -> Result<legacy::Request> {
    let (source, original) = match accepted {
        ConservedAccepted::Compatible(prepared) => {
            let original = prepared
                .original
                .as_ref()
                .ok_or_else(|| invalid("retained request has no original encoding"))?;
            (&original.source, Some(&original.original))
        }
        ConservedAccepted::Historical(record) => (&record.source, record.original.as_ref()),
    };
    if !matches!(
        source,
        SourceSchema::RetainedPublication {
            source_version: 3,
            parser_version: 3,
            ..
        }
    ) {
        return Err(invalid(
            "frozen Source3 kind lookup requires retained publication custody",
        ));
    }
    let bytes = original
        .and_then(|original| original.request.as_deref())
        .ok_or_else(|| StoreError::ReferenceUnavailable {
            reference: format!("request:{}", accepted.id()),
            reason: "the original retained request body is unavailable".into(),
        })?;
    let request: legacy::Request = crate::encoding::read_json(bytes)?;
    if request.id != accepted.id() {
        return Err(StoreError::Conflict(accepted.id().into()));
    }
    if request.write.source3_kind().is_none() {
        return Err(invalid(
            "the retained action is not in the frozen Source3 model",
        ));
    }
    Ok(request)
}

pub fn retained_source3_kind(accepted: &ConservedAccepted) -> Result<&'static str> {
    retained_source3_request(accepted)?
        .write
        .source3_kind()
        .ok_or_else(|| invalid("the retained action is not in the frozen Source3 model"))
}

/// Frozen Source3 summary only. The SDK retains the original envelope, key and
/// record bytes and may use this only under verified retained publication custody.
pub fn retained_source3_payload(accepted: &ConservedAccepted) -> Result<serde_json::Value> {
    retained_source3_request(accepted)?.source3_payload()
}

/// Portable typed projections must agree with the retained original effects.
/// This decodes evidence only; it never accepts its action or authenticates
/// sender-carried origin metadata.
pub(crate) fn validate_historical_effects(record: &HistoricalAcceptedRequest) -> Result<()> {
    let original = record.original.as_ref();
    let bytes = original.and_then(|original| original.effect.as_deref());
    let request = original
        .and_then(|original| original.request.as_deref())
        .and_then(|bytes| crate::encoding::read_json::<legacy::Request>(bytes).ok());
    match (bytes, &record.conserved_effects) {
        (Some(bytes), Some(expected)) => {
            let actual = legacy::effects(&record.source, request.as_ref(), bytes)?;
            if actual != *expected || legacy::fact_references(&actual) != record.facts {
                return Err(invalid(
                    "historical effects or fact membership disagree with the retained original encoding",
                ));
            }
        }
        (None, None) if record.facts.is_empty() => {}
        _ => {
            return Err(invalid(
                "historical effect availability disagrees with its retained original encoding",
            ));
        }
    }
    Ok(())
}
