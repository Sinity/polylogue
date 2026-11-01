//! Versioned request custody is interpreted only after matching its retained
//! frame and body. Unknown predecessor authority keeps the original spool.
use crate::spool::{Prepared, Spool};
use crate::{Error, Result};
use polylogue_model::{
    authored::{PendingAuthored, PreparedRequest},
    historical::QueuedPublicationCustody,
};
use polylogue_store::primary::{Product, ProductWrite};
use serde::{
    Deserialize,
    de::{SeqAccess, Visitor},
};
use sinex_protocol::{Ingress, MaterialFrame};
use std::fmt;
mod wire;

pub(super) fn unavailable(identity: &str) -> Error {
    Error::Refused {
        name: identity.into(),
        reason: "retained predecessor request needs its authenticated own-Recorded source binding"
            .into(),
    }
}

/// `custody` is an authenticated own-Recorded observation, not payload metadata
/// or a cached permission to publish. The current header permission is separate.
pub(super) fn visit(
    queue: &Spool,
    entry: &Prepared,
    producer: &str,
    custody: Option<&QueuedPublicationCustody>,
    visit: &mut dyn FnMut(&PendingAuthored) -> polylogue_store::Result<()>,
) -> Result<()> {
    let frame = queue.decode_frame(&entry.identity)?;
    let Ingress::Material { material, .. } = &frame else {
        return Err(Error::Invalid("authored intent has no material".into()));
    };
    if material.source.name == "polylogue" && material.source.version == 3 {
        let custody = custody.ok_or_else(|| unavailable(&entry.identity))?;
        validate(entry, producer, &frame, material, custody)?;
        // Validate the whole immutable body before any scratch callback. The
        // queue guard prevents replacement between verification and decoding.
        crate::client::confirmed::source3::visit_body(
            custody,
            queue.body(&entry.identity)?,
            &mut |_| Ok(()),
        )?;
        return crate::client::confirmed::source3::visit_body(
            custody,
            queue.body(&entry.identity)?,
            visit,
        );
    }
    if material.source != crate::source_reference() {
        return Err(unavailable(&entry.identity));
    }
    match material.parser.as_str() {
        "polylogue-request" => {
            let prepared: PreparedRequest =
                crate::spool::decode_json(queue.body(&entry.identity)?)?;
            visit(&PendingAuthored::Live { prepared }).map_err(Error::Store)
        }
        "polylogue-request-restore" => {
            let writes: Vec<ProductWrite> =
                crate::spool::decode_json(queue.body(&entry.identity)?)?;
            for write in writes {
                let pending = match write.product {
                    Product::RestoredRequest {
                        prepared,
                        claims,
                        current,
                    } => PendingAuthored::Restore {
                        prepared,
                        claims,
                        current,
                    },
                    Product::RestoredHistory {
                        evidence,
                        claims,
                        current,
                    } => PendingAuthored::RestoreHistory {
                        evidence,
                        claims,
                        current,
                    },
                    Product::RestoredAssertion {
                        fact,
                        claims,
                        current,
                    } => PendingAuthored::RestoreAssertion {
                        fact,
                        claims,
                        current,
                    },
                    _ => {
                        return Err(Error::Invalid(
                            "non-restoration product in authored restore frame".into(),
                        ));
                    }
                };
                visit(&pending).map_err(Error::Store)?;
            }
            Ok(())
        }
        _ => Err(Error::Invalid("unknown authored material parser".into())),
    }
}

fn validate(
    entry: &Prepared,
    producer: &str,
    frame: &Ingress,
    material: &MaterialFrame,
    custody: &QueuedPublicationCustody,
) -> Result<()> {
    crate::client::confirmed::source3::validate_source(custody)?;
    let body = entry
        .body
        .as_ref()
        .ok_or_else(|| unavailable(&entry.identity))?;
    let expected_frame = format!("blake3:{}", custody.frame_content_hash);
    let expected_body = format!("blake3:{}", custody.blob);
    if custody.producer_identity != producer
        || custody.frame != entry.identity
        || custody.frame != frame.frame()
        || expected_frame != entry.frame.blob
        || expected_frame
            != sinex_transport::producer::content_hash(frame)
                .map_err(crate::client::submit_error)?
        || custody.material != material.id.to_string()
        || expected_body != material.blob
        || expected_body != body.blob
        || custody.length != material.length
        || custody.length != body.length
        || custody.source != material.source.name
        || custody.source_version != material.source.version
        || custody.parser != material.parser
        || custody.parser_version != material.parser_version
        || entry.phase.is_refused()
        || entry.unresolved
        || !entry.is_request()
    {
        return Err(Error::Refused {
            name: entry.identity.clone(),
            reason: "owner retained binding differs from the original request custody".into(),
        });
    }
    Ok(())
}

/// Classify only the retired request publication path. Ignored event payloads
/// stream through the existing verified frame reader rather than new custody.
pub(crate) fn requires_retained(queue: &Spool, entry: &Prepared) -> Result<bool> {
    #[derive(Deserialize)]
    struct MaterialSource {
        source: sinex_protocol::SourceRef,
        parser: String,
    }
    #[derive(Deserialize)]
    struct Header {
        kind: String,
        source: Option<sinex_protocol::SourceRef>,
        material: Option<MaterialSource>,
        #[serde(default)]
        events: RequestEvents,
    }
    let header: Header = crate::spool::decode_json(queue.frame(&entry.identity)?)?;
    let old = |source: &sinex_protocol::SourceRef| {
        source.name == "polylogue" && source.version != crate::source_reference().version
    };
    Ok(match header.kind.as_str() {
        "material" => header.material.as_ref().is_some_and(|material| {
            old(&material.source)
                && matches!(
                    material.parser.as_str(),
                    "polylogue-request" | "polylogue-request-restore"
                )
        }),
        "events" => header.source.as_ref().is_some_and(old) && header.events.0,
        _ => false,
    })
}

#[derive(Default)]
struct RequestEvents(bool);
impl<'de> Deserialize<'de> for RequestEvents {
    fn deserialize<D: serde::Deserializer<'de>>(decoder: D) -> std::result::Result<Self, D::Error> {
        struct Events;
        impl<'de> Visitor<'de> for Events {
            type Value = RequestEvents;
            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("producer event array")
            }
            fn visit_seq<A: SeqAccess<'de>>(
                self,
                mut sequence: A,
            ) -> std::result::Result<Self::Value, A::Error> {
                #[derive(Deserialize)]
                struct EventKind {
                    kind: sinex_model::KindRef,
                }
                let mut found = false;
                while let Some(event) = sequence.next_element::<EventKind>()? {
                    found |= crate::client::confirmed::source3::request_kind(&event.kind.name);
                }
                Ok(RequestEvents(found))
            }
        }
        decoder.deserialize_seq(Events)
    }
}

#[cfg(test)]
mod tests;

impl crate::Sinex {
    pub(super) fn drain_retained_request(&self, identity: &str) -> Result<()> {
        let observed = self.observe_retained(identity, Vec::new())?;
        let frame = {
            let queue = self.queue()?;
            let Ingress::Material { material, .. } = queue.decode_frame(identity)? else {
                return Err(unavailable(identity));
            };
            if queue.capture_group(identity)?.is_some() {
                return Ok(());
            }
            wire::build(&observed.custody, &material, queue.body(identity)?)?
        };
        let Ingress::Events { events, .. } = &frame else {
            unreachable!()
        };
        let mut uses = Vec::new();
        for event in events {
            let header = crate::client::confirmed::source3::header(&event.kind)?;
            if !uses.contains(&header) {
                uses.push(header);
            }
        }
        let permitted = self.observe_retained(identity, uses)?;
        match permitted.headers {
            sinex_protocol::RecordedHeaders::PermittedHeaders => {}
            sinex_protocol::RecordedHeaders::NotPermitted { reason } => {
                return Err(Error::Refused {
                    name: identity.into(),
                    reason,
                });
            }
            _ => return Err(unavailable(identity)),
        }
        // This freezes only the original Source3 envelope, material reference
        // and operation identity. Publication rechecks permission online.
        self.queue()?.prepare_group(frame, &[identity.to_owned()])?;
        Ok(())
    }
}

pub(crate) fn validate_publication(
    queue: &Spool,
    entry: &Prepared,
    producer: &str,
    custody: &QueuedPublicationCustody,
) -> Result<()> {
    visit(queue, entry, producer, Some(custody), &mut |pending| {
        let PendingAuthored::ConservedQueued { accepted, .. } = pending else {
            return Err(polylogue_store::StoreError::Invalid(
                "retained publication is not queued history".into(),
            ));
        };
        wire::require_source(accepted).map_err(Into::into)
    })
}
