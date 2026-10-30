//! Immutable request-reading contract. This artifact is read-only and never
//! participates in current producer registration or publication.
use crate::{Error, Result};
use sinex_model::SourceDeclaration;
use sinex_protocol::ConfirmedEvent;
use std::{collections::BTreeMap, sync::OnceLock};

pub(super) const SOURCE_FINGERPRINT: &str =
    "99135262379a31eb3d06d6412d46673e6b15dcbc2dc8c8c9362b2056f6924737";
const REQUEST_KINDS: &[&str] = &[
    "polylogue.note",
    "polylogue.setting",
    "polylogue.kept_result",
    "polylogue.delivery",
    "polylogue.delivery_outcome",
    "polylogue.request",
    "polylogue.request_restore",
];

struct Contract {
    declaration: SourceDeclaration,
    hashes: BTreeMap<(String, u64), String>,
}

fn contract() -> Result<&'static Contract> {
    static CONTRACT: OnceLock<std::result::Result<Contract, String>> = OnceLock::new();
    CONTRACT
        .get_or_init(|| {
            let declaration: SourceDeclaration =
                serde_json::from_str(include_str!("declaration.json"))
                    .map_err(|error| error.to_string())?;
            if declaration.source.name != "polylogue"
                || declaration.source.version != 3
                || declaration.family != "polylogue"
                || declaration.fingerprint() != SOURCE_FINGERPRINT
            {
                return Err("frozen request source declaration changed identity".into());
            }
            let hashes = declaration
                .kinds
                .iter()
                .map(|kind| {
                    (
                        (kind.kind.name.clone(), kind.kind.version),
                        kind.definition_hash(),
                    )
                })
                .collect();
            Ok(Contract {
                declaration,
                hashes,
            })
        })
        .as_ref()
        .map_err(|error| Error::Invalid(error.clone()))
}

pub(super) fn declaration() -> Result<&'static SourceDeclaration> {
    Ok(&contract()?.declaration)
}

/// Called only after the current typed decoder matched this same immutable
/// kind hash. A prior source version does not upgrade native route evidence.
pub(super) fn origin_supported(event: &ConfirmedEvent) -> Result<bool> {
    let contract = contract()?;
    Ok(
        event.source_ref.as_ref() == Some(&contract.declaration.source)
            && event.source_fingerprint.as_deref() == Some(SOURCE_FINGERPRINT)
            && event.producer_family.as_deref() == Some("polylogue")
            && event
                .producer_identity
                .as_ref()
                .is_some_and(|value| !value.is_empty())
            && contract
                .hashes
                .get(&(event.event_type.clone(), event.kind_version))
                == Some(&event.kind_definition_hash),
    )
}

pub(crate) fn request_kind(name: &str) -> bool {
    REQUEST_KINDS.contains(&name)
}

pub(super) fn matches(event: &ConfirmedEvent) -> Result<bool> {
    if event.kind_version != 4 || !request_kind(&event.event_type) {
        return Ok(false);
    }
    if !origin_supported(event)? {
        return Err(Error::Refused {
            name: format!("{}@{}", event.event_type, event.kind_version),
            reason: "historical request does not match its frozen native source contract".into(),
        });
    }
    if !event.projection.verifies(&event.payload) {
        return Err(Error::Invalid(
            "historical request projection changed its payload identity".into(),
        ));
    }
    Ok(true)
}

/// Run2 is a separate exact historical read contract, never producer registration.
pub(super) fn matches_run(event: &ConfirmedEvent) -> Result<bool> {
    if event.event_type != "polylogue.query_run" || event.kind_version != 2 {
        return Ok(false);
    }
    if !origin_supported(event)? {
        return Err(Error::Refused {
            name: format!("{}@{}", event.event_type, event.kind_version),
            reason: "historical run does not match its frozen native source contract".into(),
        });
    }
    if !event.projection.verifies(&event.payload) {
        return Err(Error::Invalid(
            "historical run projection changed its payload identity".into(),
        ));
    }
    Ok(true)
}

/// Query1 is unchanged data across the two namespaces. An attributed current
/// source uses the ordinary decoder; every historical query needs this exact
/// retained Source3 contract before its old definition can be read.
pub(super) fn matches_query(event: &ConfirmedEvent) -> Result<bool> {
    if event.event_type != "polylogue.query" || event.kind_version != 1 {
        return Ok(false);
    }
    if event.source_ref.as_ref() == Some(&crate::source_reference()) {
        return Ok(false);
    }
    if !origin_supported(event)? {
        return Err(Error::Refused {
            name: format!("{}@{}", event.event_type, event.kind_version),
            reason: "historical query does not match its frozen native source contract".into(),
        });
    }
    if !event.projection.verifies(&event.payload) {
        return Err(Error::Invalid(
            "historical query projection changed its payload identity".into(),
        ));
    }
    Ok(true)
}
