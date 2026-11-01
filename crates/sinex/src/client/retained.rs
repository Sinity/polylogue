//! Own-Recorded observations neither register old kinds nor grant admission.
use super::{ACKNOWLEDGMENT_WAIT, Client};
use crate::{Authentication, Config, Error, Result};
use polylogue_model::{BlobHash, historical::QueuedPublicationCustody};
use sinex_protocol::{
    Ingress, MaterialFrame, RecordedContinuation, RecordedContinuationRequest, RecordedHeaderUse,
    RecordedHeaders, RecordedMaterialBinding, RecordedPage, RecordedQuery,
};

pub(crate) struct Observation {
    pub custody: QueuedPublicationCustody,
    pub headers: RecordedHeaders,
}

/// This private read path is available even when the current write grant was
/// removed. Hello still validates the authenticated peer and protocol; its
/// fallback family is not retained source evidence.
pub(crate) async fn observe(
    config: &Config,
    authentication: &Authentication,
    request: &RecordedContinuationRequest,
) -> Result<RecordedContinuation> {
    let (client, subjects, _welcome) = super::transport(config, authentication).await?;
    query(&client, &subjects, &config.producer, request).await
}

impl Client {
    pub(crate) async fn retained(
        &self,
        request: &RecordedContinuationRequest,
    ) -> Result<RecordedContinuation> {
        query(&self.client, &self.subjects, &self.producer, request).await
    }
}

async fn query(
    client: &async_nats::Client,
    subjects: &sinex_protocol::Subjects,
    producer: &str,
    request: &RecordedContinuationRequest,
) -> Result<RecordedContinuation> {
    let query = RecordedQuery {
        after: i64::MAX,
        continuations: vec![request.clone()],
        ..Default::default()
    };
    if serde_json::to_vec(&query)?.len() > client.server_info().max_payload {
        return Err(Error::Invalid(
            "retained selector exceeds the RPC envelope".into(),
        ));
    }
    let page: RecordedPage = sinex_transport::replies::request_json(
        client,
        subjects.recorded(producer),
        producer,
        &query,
        ACKNOWLEDGMENT_WAIT,
    )
    .await
    .map_err(|error| Error::Unreachable(error.to_string()))?;
    let mut replies = page.continuations.into_iter();
    let reply = replies.next().ok_or_else(|| Error::Refused {
        name: request.frame.clone(),
        reason: "owner does not support a retained source comparison".into(),
    })?;
    let echoed = match &reply {
        RecordedContinuation::Unknown { request, .. }
        | RecordedContinuation::Retained { request, .. } => request,
    };
    if echoed != request || replies.next().is_some() {
        return Err(Error::Invalid(
            "retained comparison changed its selector".into(),
        ));
    }
    Ok(reply)
}

/// Compare actual encoded custody with a positively retained material. An
/// echo or a matching hash alone cannot establish the binding.
pub(crate) fn selector(
    queue: &crate::spool::Spool,
    entry: &crate::spool::Prepared,
    uses: Vec<RecordedHeaderUse>,
) -> Result<RecordedContinuationRequest> {
    let frame = queue.decode_frame(&entry.identity)?;
    let Ingress::Material { material, .. } = &frame else {
        return Err(Error::Invalid(
            "retained request has no original material".into(),
        ));
    };
    let body = entry.body().ok_or_else(|| unavailable(&entry.identity))?;
    if material.source.name != "polylogue"
        || material.source.version != 3
        || material.parser_version != 3
        || material.blob != body.blob
        || material.length != body.length
        || frame.frame() != entry.identity
        || sinex_transport::producer::content_hash(&frame).map_err(super::submit_error)?
            != entry.frame.blob
        || entry.phase.is_refused()
        || entry.unresolved
        || entry.revoked
        || entry.is_private()
    {
        return Err(unavailable(&entry.identity));
    }
    std::io::copy(&mut queue.body(&entry.identity)?, &mut std::io::sink())?;
    Ok(RecordedContinuationRequest {
        frame: entry.identity.clone(),
        frame_content_hash: entry.frame.blob.clone(),
        material: material.id,
        blob: material.blob.clone(),
        length: material.length,
        uses,
    })
}

pub(crate) fn permitted(
    entry: &crate::spool::Prepared,
    material: &MaterialFrame,
    request: &RecordedContinuationRequest,
    reply: RecordedContinuation,
) -> Result<()> {
    let RecordedContinuation::Retained {
        request: echoed,
        binding,
        headers,
    } = reply
    else {
        return Err(unavailable(&entry.identity));
    };
    if &echoed != request {
        return Err(unavailable(&entry.identity));
    }
    validate_binding(entry, material, request, &binding)?;
    match headers {
        RecordedHeaders::PermittedHeaders if !request.uses.is_empty() => Ok(()),
        RecordedHeaders::NotPermitted { reason } => Err(Error::Refused {
            name: entry.identity.clone(),
            reason,
        }),
        _ => Err(unavailable(&entry.identity)),
    }
}

pub(crate) fn observation(
    producer: &str,
    entry: &crate::spool::Prepared,
    material: &MaterialFrame,
    request: &RecordedContinuationRequest,
    reply: RecordedContinuation,
) -> Result<Observation> {
    let RecordedContinuation::Retained {
        request: echoed,
        binding,
        headers,
    } = reply
    else {
        return Err(unavailable(&entry.identity));
    };
    if &echoed != request {
        return Err(unavailable(&entry.identity));
    }
    validate_binding(entry, material, request, &binding)?;
    let custody = QueuedPublicationCustody {
        producer_family: binding.family,
        producer_identity: producer.into(),
        receipt_number: binding
            .receipt
            .number
            .ok_or_else(|| unavailable(&entry.identity))?,
        frame: entry.identity.clone(),
        frame_content_hash: hash(&entry.frame.blob)?,
        material: binding.material.to_string(),
        blob: hash(&binding.blob)?,
        length: binding.length,
        source: binding.source.name,
        source_version: binding.source.version,
        source_fingerprint: binding.source_fingerprint,
        parser: material.parser.clone(),
        parser_version: material.parser_version,
    };
    crate::client::confirmed::source3::validate_source(&custody)?;
    Ok(Observation { custody, headers })
}

fn validate_binding(
    entry: &crate::spool::Prepared,
    material: &MaterialFrame,
    request: &RecordedContinuationRequest,
    binding: &RecordedMaterialBinding,
) -> Result<()> {
    let receipt = super::verified_receipt(entry, binding.receipt.clone())?;
    if receipt.refusal.is_some()
        || receipt.number.is_none_or(|number| number <= 0)
        || receipt.material != Some(material.id)
        || receipt.content_hash() != Some(request.frame_content_hash.as_str())
        || binding.material != material.id
        || binding.material != request.material
        || binding.blob != material.blob
        || binding.blob != request.blob
        || binding.length != material.length
        || binding.length != request.length
        || binding.source != material.source
        || binding.family != "polylogue"
        || binding.source_fingerprint != super::confirmed::source3::source_fingerprint()
    {
        return Err(unavailable(&entry.identity));
    }
    Ok(())
}

fn hash(value: &str) -> Result<BlobHash> {
    BlobHash::parse(
        value
            .strip_prefix("blake3:")
            .ok_or_else(|| Error::Invalid("retained material omitted its hash algorithm".into()))?,
    )
    .map_err(|error| Error::Invalid(error.to_string()))
}

fn unavailable(identity: &str) -> Error {
    Error::Refused {
        name: identity.into(),
        reason: "the owner has no verified original Source3 publication binding".into(),
    }
}

impl Client {
    /// Only an already staged original Events group can take this route. Each
    /// dependency is the exact retained material from that immutable group.
    pub(crate) async fn permit_retained_events(
        &self,
        prepared: &crate::spool::Prepared,
    ) -> Result<()> {
        let materials = {
            let queue = self
                .spool
                .lock()
                .map_err(|_| Error::Invalid("producer spool panicked".into()))?;
            let frame = queue.decode_frame(&prepared.identity)?;
            let Ingress::Events {
                source: Some(source),
                coverage: None,
                events,
                failures,
                ..
            } = frame
            else {
                return Err(unavailable(&prepared.identity));
            };
            if source.name != "polylogue"
                || source.version != 3
                || !failures.is_empty()
                || events.is_empty()
            {
                return Err(unavailable(&prepared.identity));
            }
            let mut uses = Vec::with_capacity(events.len());
            for event in &events {
                if !super::confirmed::source3::request_kind(&event.kind.name)
                    && event.kind.name != "polylogue.blob"
                {
                    return Err(unavailable(&prepared.identity));
                }
                let mut header = super::confirmed::source3::header(&event.kind)?;
                header.class = event.class;
                if !uses.contains(&header) {
                    uses.push(header);
                }
            }
            let mut members = Vec::new();
            queue.visit_captures(&prepared.identity, |identity| {
                let entry = queue.get(identity)?.ok_or_else(|| unavailable(identity))?;
                let request = selector(&queue, &entry, uses.clone())?;
                let Ingress::Material { material, .. } = queue.decode_frame(identity)? else {
                    return Err(unavailable(identity));
                };
                members.push((entry, material, request));
                Ok(())
            })?;
            if members.is_empty() {
                return Err(unavailable(&prepared.identity));
            }
            // The exact group owns every located output material. A payload
            // hash match cannot import a different observation as a dependency.
            if events
                .iter()
                .filter_map(|event| event.location.as_ref())
                .any(|location| {
                    !members
                        .iter()
                        .any(|(_, material, _)| material.id == location.material)
                })
            {
                return Err(unavailable(&prepared.identity));
            }
            members
        };
        for (entry, material, request) in materials {
            let reply = self.retained(&request).await?;
            if entry.is_request() {
                let observed =
                    observation(&self.producer, &entry, &material, &request, reply.clone())?;
                let queue = self
                    .spool
                    .lock()
                    .map_err(|_| Error::Invalid("producer spool panicked".into()))?;
                crate::producer::validate_retained_publication(
                    &queue,
                    &entry,
                    &self.producer,
                    &observed.custody,
                )?;
            }
            permitted(&entry, &material, &request, reply)?;
        }
        Ok(())
    }

    /// Reconciliation reads the original receipt and actual admitted headers.
    /// It does not submit, manufacture a current event or apply a new effect.
    pub(crate) async fn verify_retained_receipt(
        &self,
        prepared: &crate::spool::Prepared,
        receipt: &sinex_protocol::Acknowledgment,
    ) -> Result<()> {
        let expected = {
            let queue = self
                .spool
                .lock()
                .map_err(|_| Error::Invalid("producer spool panicked".into()))?;
            match queue.decode_frame(&prepared.identity)? {
                Ingress::Material { .. } => return Ok(()),
                Ingress::Events {
                    source: Some(source),
                    events,
                    ..
                } if source.name == "polylogue" && source.version == 3 => events,
                _ => return Err(unavailable(&prepared.identity)),
            }
        };
        super::verified_receipt(prepared, receipt.clone())?;
        for id in &receipt.admitted {
            let reply: sinex_protocol::FetchEventReply = self
                .request(
                    self.subjects.fetch_event(&self.producer),
                    sinex_protocol::FetchEvent { event: *id },
                )
                .await?;
            if reply.refusal.is_some() || reply.erased.is_some() {
                return Err(unavailable(&prepared.identity));
            }
            let actual = reply.event.ok_or_else(|| unavailable(&prepared.identity))?;
            if actual.id != *id
                || !actual.projection.verifies(&actual.payload)
                || actual.producer_identity.as_deref() != Some(self.producer.as_str())
                || !super::confirmed::source3::origin_supported(&actual)?
                || !expected.iter().any(|event| {
                    event.kind.name == actual.event_type
                        && event.kind.version == actual.kind_version
                        && event.output_index == actual.output_index
                        && event.location == actual.location
                        && event.parents == actual.parents
                        && event.payload.get("record_key") == actual.payload.get("record_key")
                })
            {
                return Err(unavailable(&prepared.identity));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
