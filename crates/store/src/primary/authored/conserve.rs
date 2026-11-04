//! Versioned conservation restores frozen data. It never accepts or executes
//! the retired intent, or turns a native projection hash into a client receipt.
use super::*;
use polylogue_model::historical::{
    ConservedAccepted, HistoricalAcceptedRequest, NativeProjectionHistory, SourceSchema,
};

fn conflict(id: &str) -> StoreError {
    StoreError::Conflict(id.into())
}

fn same_contract(a: &SourceSchema, b: &SourceSchema) -> bool {
    match (a, b) {
        (
            SourceSchema::NativeProjection {
                source: a,
                source_version: 3,
                source_fingerprint: af,
                kind_version: 4,
                ..
            },
            SourceSchema::NativeProjection {
                source: b,
                source_version: 3,
                source_fingerprint: bf,
                kind_version: 4,
                ..
            },
        ) => a == b && af == bf,
        (
            SourceSchema::NativeProjection {
                source: a,
                source_version: 3,
                source_fingerprint: af,
                kind_version: 4,
                ..
            },
            SourceSchema::RetainedPublication {
                source: b,
                source_version: 3,
                source_fingerprint: bf,
                parser_version: 3,
                ..
            },
        )
        | (
            SourceSchema::RetainedPublication {
                source: a,
                source_version: 3,
                source_fingerprint: af,
                parser_version: 3,
                ..
            },
            SourceSchema::NativeProjection {
                source: b,
                source_version: 3,
                source_fingerprint: bf,
                kind_version: 4,
                ..
            },
        ) => a == b && af == bf,
        (
            SourceSchema::LocalProjection {
                index_version: 14,
                source_schema_version: 4,
            },
            SourceSchema::RetainedPublication {
                source_version: 3,
                parser_version: 3,
                ..
            },
        )
        | (
            SourceSchema::RetainedPublication {
                source_version: 3,
                parser_version: 3,
                ..
            },
            SourceSchema::LocalProjection {
                index_version: 14,
                source_schema_version: 4,
            },
        ) => true,
        (
            SourceSchema::LocalProjection {
                index_version: 14,
                source_schema_version: 4,
            },
            SourceSchema::NativeProjection {
                source_version: 3,
                kind_version: 4,
                ..
            },
        )
        | (
            SourceSchema::NativeProjection {
                source_version: 3,
                kind_version: 4,
                ..
            },
            SourceSchema::LocalProjection {
                index_version: 14,
                source_schema_version: 4,
            },
        ) => true,
        _ => a == b,
    }
}

fn same_json(a: &[u8], b: &[u8]) -> bool {
    // Arbitrary-precision JSON numbers retain their representation. This only
    // ignores object key order/encoding, never array order or body field values.
    match (
        crate::encoding::read_json::<serde_json::Value>(a),
        crate::encoding::read_json::<serde_json::Value>(b),
    ) {
        (Ok(a), Ok(b)) => a == b,
        _ => false,
    }
}
fn same_optional_json(a: &Option<Vec<u8>>, b: &Option<Vec<u8>>) -> bool {
    match (a, b) {
        (Some(a), Some(b)) => same_json(a, b),
        (None, None) => true,
        _ => false,
    }
}
fn same_original(
    a: &polylogue_model::historical::OriginalBytes,
    b: &polylogue_model::historical::OriginalBytes,
) -> bool {
    a.intent == b.intent
        && same_optional_json(&a.request, &b.request)
        && same_optional_json(&a.outcome, &b.outcome)
        && same_optional_json(&a.effect, &b.effect)
}

/// Preserve original-source classifications as recorded metadata, independently
/// of the receiving row's actual import observations and declared claims.
pub(crate) fn merge_original(
    known: &mut polylogue_model::historical::OriginalBytes,
    incoming: &polylogue_model::historical::OriginalBytes,
) {
    for context in &incoming.cached {
        if !known.cached.contains(context) {
            known.cached.push(context.clone());
        }
    }
    for bytes in &incoming.records {
        if !known.records.contains(bytes) {
            known.records.push(bytes.clone());
        }
    }
    known.records.sort();
    if let Some(incoming) = &incoming.imports {
        let known = known.imports.get_or_insert_with(Default::default);
        for entry in &incoming.observed {
            if !known.observed.contains(entry) {
                known.observed.push(entry.clone());
            }
        }
        for entry in &incoming.declared {
            if !known.declared.contains(entry) {
                known.declared.push(entry.clone());
            }
        }
    }
}

pub(crate) fn same_prepared(a: &PreparedRequest, b: &PreparedRequest) -> bool {
    let mut a = a.clone();
    let mut b = b.clone();
    if let (Some(a), Some(b)) = (&mut a.original, &mut b.original) {
        if !same_contract(&a.source, &b.source) || !same_original(&a.original, &b.original) {
            return false;
        }
        b.source = a.source.clone();
        b.hash_evidence = a.hash_evidence.clone();
        b.original = a.original.clone();
        a.native = Default::default();
        b.native = Default::default();
    }
    a == b
}

pub(crate) fn declared_prepared(prepared: &PreparedRequest) -> PreparedRequest {
    let mut prepared = prepared.clone();
    if let Some(original) = &mut prepared.original {
        let observed = std::mem::take(&mut original.native.observed);
        original.native.declare(observed);
    }
    prepared
}

fn original_contains(
    known: &polylogue_model::historical::OriginalBytes,
    incoming: &polylogue_model::historical::OriginalBytes,
) -> bool {
    incoming
        .cached
        .iter()
        .all(|context| known.cached.contains(context))
        && incoming
            .records
            .iter()
            .all(|record| known.records.contains(record))
        && match (&known.imports, &incoming.imports) {
            (_, None) => true,
            (Some(known), Some(incoming)) => {
                incoming
                    .observed
                    .iter()
                    .all(|entry| known.observed.contains(entry))
                    && incoming
                        .declared
                        .iter()
                        .all(|entry| known.declared.contains(entry))
            }
            (None, Some(_)) => false,
        }
}

pub(crate) fn restored_contains(known: &PreparedRequest, incoming: &PreparedRequest) -> bool {
    same_prepared(known, incoming)
        && match (&known.original, &incoming.original) {
            (Some(known), Some(incoming)) => {
                original_contains(&known.original, &incoming.original)
                    && incoming
                        .native
                        .observed
                        .iter()
                        .chain(&incoming.native.declared)
                        .all(|claim| known.native.declared.contains(claim))
            }
            (None, None) => true,
            _ => false,
        }
}

pub(crate) fn merge_declared_prepared(
    conn: &Connection,
    known: &mut PreparedRequest,
    incoming: &PreparedRequest,
) -> Result<()> {
    if let (Some(known_original), Some(incoming_original)) =
        (&mut known.original, &incoming.original)
    {
        merge_original(&mut known_original.original, &incoming_original.original);
        known_original.native.declare(
            incoming_original
                .native
                .observed
                .iter()
                .chain(&incoming_original.native.declared)
                .cloned(),
        );
        let source = Source::of(conn)?;
        conn.execute(
            &source.sql("UPDATE @source.client_request SET original=?2 WHERE id=?1"),
            params![known.request.id, pack(known_original)?],
        )?;
    }
    Ok(())
}

pub(crate) fn same_historical(
    a: &HistoricalAcceptedRequest,
    b: &HistoricalAcceptedRequest,
) -> bool {
    if !same_contract(&a.source, &b.source) {
        return false;
    }
    let mut a = a.clone();
    let mut b = b.clone();
    match (&a.original, &b.original) {
        (Some(a), Some(b)) if same_original(a, b) => {}
        (None, None) => {}
        _ => return false,
    }
    b.source = a.source.clone();
    b.hash_evidence = a.hash_evidence.clone();
    b.original = a.original.clone();
    a.imports = Default::default();
    b.imports = Default::default();
    a.native = Default::default();
    b.native = Default::default();
    if let (Some(a), Some(b)) = (&a.conserved_effects, &mut b.conserved_effects) {
        if a.historical_assertions.len() != b.historical_assertions.len() {
            return false;
        }
        for (a, b) in a
            .historical_assertions
            .iter()
            .zip(&mut b.historical_assertions)
        {
            if !same_contract(&a.source, &b.source) {
                return false;
            }
            b.source = a.source.clone();
        }
    }
    a == b
}

fn merge_native(
    known: &mut NativeProjectionHistory,
    incoming: &NativeProjectionHistory,
    original: &mut polylogue_model::historical::OriginalBytes,
) -> Result<()> {
    for observed in &incoming.observed {
        if known.observed.iter().any(|previous| {
            previous.provenance.event == observed.provenance.event
                && previous.provenance.record_hash != observed.provenance.record_hash
        }) {
            return Err(StoreError::Invalid(
                "one native event changed its immutable record hash".into(),
            ));
        }
        if let Some(previous) = known
            .observed
            .iter()
            .find(|previous| previous.same_observation(observed))
        {
            if previous != observed {
                let mut previous_header = previous.clone();
                let mut current_header = observed.clone();
                previous_header.provenance.projected_payload.clear();
                current_header.provenance.projected_payload.clear();
                if previous_header != current_header
                    || !same_json(
                        &previous.provenance.projected_payload,
                        &observed.provenance.projected_payload,
                    )
                {
                    return Err(StoreError::Invalid("an actual native observation changed under its event, hashes and fetched epoch".into()));
                }
                if !original
                    .records
                    .contains(&observed.provenance.projected_payload)
                {
                    original
                        .records
                        .push(observed.provenance.projected_payload.clone());
                    original.records.sort();
                }
            }
        } else {
            known
                .observe(observed.clone())
                .map_err(|error| StoreError::Invalid(error.to_string()))?;
        }
    }
    known.declare(incoming.declared.clone());
    Ok(())
}

fn declared_imports(conn: &Connection, id: &str, claims: &[ImportAttribution]) -> Result<()> {
    let source = Source::of(conn)?;
    let prior: String = conn.query_row(
        &source.sql("SELECT imports FROM @source.client_request WHERE id=?1"),
        [id],
        |row| row.get(0),
    )?;
    let mut history: ImportHistory = serde_json::from_str(&prior)?;
    for claim in claims {
        if !history.declared.contains(claim) {
            history.declared.push(claim.clone());
        }
    }
    let merged = serde_json::to_string(&history)?;
    if merged != prior {
        conn.execute(
            &source.sql("UPDATE @source.client_request SET imports=?2 WHERE id=?1"),
            params![id, merged],
        )?;
    }
    Ok(())
}

/// Only the numbered source/projection upgrade calls this local counterpart.
pub(crate) fn conserve_local(
    tx: &Transaction<'_>,
    accepted: &ConservedAccepted,
    claims: &[ImportAttribution],
    counter: i64,
) -> Result<()> {
    let source = match accepted {
        ConservedAccepted::Compatible(prepared) => {
            &prepared
                .original
                .as_ref()
                .ok_or_else(|| conflict(&prepared.request.id))?
                .source
        }
        ConservedAccepted::Historical(record) => &record.source,
    };
    if !matches!(
        source,
        SourceSchema::PolylogueArchive { version: 4 }
            | SourceSchema::LocalProjection {
                index_version: 14,
                source_schema_version: 4
            }
    ) {
        return Err(StoreError::Invalid(
            "local conservation requires the numbered predecessor schema".into(),
        ));
    }
    conserve_on(tx, accepted, claims, counter)
}

fn apply_historical_fact(
    tx: &Transaction<'_>,
    incoming: &polylogue_model::historical::HistoricalAssertionEvidence,
    counter: i64,
) -> Result<()> {
    if let Some(known) = crate::assertions::get_evidence(tx, &incoming.id)? {
        let polylogue_model::historical::AssertionEvidence::Historical { fact: known, .. } = known
        else {
            return Err(conflict(&incoming.id));
        };
        let mut compared = incoming.clone();
        if same_contract(&known.source, &compared.source) {
            compared.source = known.source.clone();
        }
        if known != compared {
            return Err(conflict(&incoming.id));
        }
        return Ok(());
    }
    crate::assertions::apply_historical(tx, incoming, counter)
}

fn conserve_on(
    tx: &Transaction<'_>,
    accepted: &ConservedAccepted,
    claims: &[ImportAttribution],
    counter: i64,
) -> Result<()> {
    if counter < 0 {
        return Err(StoreError::Invalid("negative conservation counter".into()));
    }
    let source = Source::of(tx)?;
    match accepted {
        ConservedAccepted::Compatible(incoming) => {
            validate(incoming)?;
            let original = incoming
                .original
                .as_ref()
                .ok_or_else(|| conflict(&incoming.request.id))?;
            let mut merged = incoming.as_ref().clone();
            match frozen(tx, &incoming.request.id)? {
                Some(FrozenRequest::Historical) => return Err(conflict(&incoming.request.id)),
                Some(FrozenRequest::Prepared(mut known)) => {
                    if !same_prepared(&known, incoming) {
                        return Err(conflict(&incoming.request.id));
                    }
                    merge_original(
                        &mut known
                            .original
                            .as_mut()
                            .expect("checked original encoding")
                            .original,
                        &original.original,
                    );
                    let known_original =
                        known.original.as_mut().expect("checked original encoding");
                    merge_native(
                        &mut known_original.native,
                        &original.native,
                        &mut known_original.original,
                    )?;
                    merged = *known;
                }
                None => {
                    records::apply(tx, &incoming.effect, counter, true)?;
                    insert_request(tx, incoming, counter, &ImportHistory::default())?;
                }
            }
            tx.execute(
                &source.sql("UPDATE @source.client_request SET original=?2 WHERE id=?1"),
                params![
                    incoming.request.id,
                    merged.original.as_ref().map(pack).transpose()?
                ],
            )?;
            declared_imports(tx, &incoming.request.id, claims)
        }
        ConservedAccepted::Historical(incoming) => {
            incoming
                .validate()
                .map_err(|error| StoreError::Invalid(error.to_string()))?;
            let mut merged = incoming.as_ref().clone();
            if let Some(mut known) = super::historical::get(tx, &incoming.id)? {
                if !same_historical(&known, incoming) {
                    return Err(conflict(&incoming.id));
                }
                if let (Some(known), Some(incoming)) = (&mut known.original, &incoming.original) {
                    merge_original(known, incoming);
                }
                let original = known
                    .original
                    .as_mut()
                    .ok_or_else(|| conflict(&incoming.id))?;
                merge_native(&mut known.native, &incoming.native, original)?;
                merged = known;
            } else {
                if frozen(tx, &incoming.id)?.is_some() {
                    return Err(conflict(&incoming.id));
                }
                if let Some(effects) = &incoming.conserved_effects {
                    records::apply(tx, &effects.current, counter, true)?;
                    for fact in &effects.historical_assertions {
                        apply_historical_fact(tx, fact, counter)?;
                    }
                }
                merged.imports = ImportHistory::default();
                tx.execute(&source.sql("INSERT INTO @source.client_request(id,content_hash,created_at,historical,commit_seq) VALUES(?1,?2,?3,?4,?5)"),params![incoming.id,incoming.content_hash.0.as_slice(),incoming.at.map(|at|at.unix_ms()),pack(&merged)?,counter])?;
            }
            tx.execute(
                &source.sql("UPDATE @source.client_request SET historical=?2 WHERE id=?1"),
                params![incoming.id, pack(&merged)?],
            )?;
            declared_imports(tx, &incoming.id, claims)
        }
    }
}
