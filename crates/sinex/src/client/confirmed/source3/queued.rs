//! Pending publication is a private scratch projection of retained custody.
//! It has no fetched event identity and cannot enter confirmed conservation.
use super::{contract, envelope};
use crate::{Error, Result};
use polylogue_model::{authored::PendingAuthored, historical::QueuedPublicationCustody};
use std::io::Read;

pub(crate) fn validate_source(custody: &QueuedPublicationCustody) -> Result<()> {
    let source = &contract::declaration()?.source;
    if custody.source != source.name
        || custody.source_version != source.version
        || custody.source_fingerprint != contract::SOURCE_FINGERPRINT
        || custody.producer_family != "polylogue"
        || custody.producer_identity.is_empty()
        || custody.receipt_number <= 0
        || custody.parser_version != 3
        || !matches!(
            custody.parser.as_str(),
            "polylogue-request" | "polylogue-request-restore"
        )
    {
        return Err(Error::Refused {
            name: custody.frame.clone(),
            reason: "retained request has no verified supported publication binding".into(),
        });
    }
    Ok(())
}

/// The producer has verified the immutable spool frame, complete body and
/// authenticated own-Recorded binding before invoking this scratch visitor.
pub(crate) fn visit_body(
    custody: &QueuedPublicationCustody,
    body: impl Read,
    visit: &mut dyn FnMut(&PendingAuthored) -> polylogue_store::Result<()>,
) -> Result<()> {
    validate_source(custody)?;
    let mut conserve = |fields: envelope::Fields<'_>| {
        let accepted = polylogue_store::assertions::conversion::conserve_request(
            custody.source_schema(),
            fields.request,
            Some(fields.outcome),
            Some(fields.effect),
            None,
            None,
        )?;
        visit(&PendingAuthored::ConservedQueued {
            accepted,
            claims: fields.claims,
            custody: custody.clone(),
        })
        .map_err(Error::Store)
    };
    if custody.parser == "polylogue-request" {
        let encoded: Box<serde_json::value::RawValue> = crate::spool::decode_json(body)?;
        conserve(envelope::prepared(encoded.get().as_bytes())?)
    } else {
        envelope::visit_restored(body, conserve)
    }
}
