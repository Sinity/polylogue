//! Sending bytes through Sinex's producer protocol on the same host. The
//! message and its bytes are first written to this producer's spool under
//! Sinex's store root; Sinex then takes ownership of them, copies them and
//! acknowledges the message. A retry after an interruption reuses the
//! spooled request, so it never reads the source again or names a new
//! material.
use crate::{Error, Result, Sinex, checked};
use sinex_model::{Class, InputRef, MaterialId};
use sinex_protocol::producer_bytes::{
    Bytes, Material, Publication as SpoolRequest, Role, encode_frame,
};
use sinex_protocol::{Acknowledgment, Ingress, MaterialFrame};
use sinex_transport::spool::{self, Frozen as Spooled};
use std::io::Read;

impl Sinex {
    pub(crate) fn check_message(&self, message: &Ingress) -> Result<()> {
        if encode_frame(message).len().saturating_add(4096) > self.max_message {
            return Err(Error::Invalid(
                "a producer message exceeds the NATS server's message size".into(),
            ));
        }
        Ok(())
    }

    /// Writes the message, and the material's bytes when it has one, to the
    /// spool. A message too large for one NATS message travels as an
    /// envelope.
    fn spool(&self, message: Ingress, material: Option<&mut dyn Read>) -> Result<Spooled> {
        let body = encode_frame(&message);
        let envelope = (body.len().saturating_add(4096) > self.max_message).then(|| Bytes {
            blob: format!("blake3:{}", blake3::hash(&body)),
            length: body.len() as u64,
        });
        let declaration = match &message {
            Ingress::Material { material, .. } => Some(Material::from_frame(material)),
            _ => None,
        };
        let request = SpoolRequest {
            version: 1,
            frame: message.frame().to_owned(),
            class: Class::Open,
            material: declaration,
            envelope,
        };
        request.validate().map_err(Error::Invalid)?;
        let spooled = spool::freeze(
            &self.store_root,
            &self.producer,
            &Spooled {
                publication: request,
                frame: message,
            },
        )
        .map_err(Error::Invalid)?;
        let request = &spooled.publication;
        if let Some(bytes) = material {
            spool::stage(
                &self.store_root,
                &self.producer,
                request,
                Role::Material,
                bytes,
            )
            .map_err(Error::Invalid)?;
        }
        if request.envelope.is_some() {
            spool::stage(
                &self.store_root,
                &self.producer,
                request,
                Role::Envelope,
                body.as_slice(),
            )
            .map_err(Error::Invalid)?;
        }
        Ok(spooled)
    }

    fn final_acknowledgment(
        request: &SpoolRequest,
        acknowledgment: Acknowledgment,
    ) -> Result<Acknowledgment> {
        let acknowledgment = checked(acknowledgment)?;
        if acknowledgment.frame != request.frame
            || acknowledgment.material != request.material.as_ref().map(|m| m.id)
        {
            return Err(Error::Invalid(
                "the final acknowledgment differs from the spooled request".into(),
            ));
        }
        Ok(acknowledgment)
    }

    /// Removes spooled requests Sinex has acknowledged, for example when the
    /// final acknowledgment arrived after a stop. Reads only this producer's
    /// spool, whatever the archive's size.
    pub(crate) async fn finish_staged(&mut self) -> Result<()> {
        let mut spooled =
            spool::pending(&self.store_root, &self.producer).map_err(Error::Invalid)?;
        loop {
            let mut requests = Vec::with_capacity(500);
            for entry in spooled.by_ref().take(500) {
                // Drop the possibly large message before loading the next one.
                requests.push(entry.map_err(Error::Invalid)?.publication);
            }
            if requests.is_empty() {
                return Ok(());
            }
            let names = requests
                .iter()
                .map(|request| request.frame.clone())
                .collect::<Vec<_>>();
            let held = self.have(&names).await?;
            for request in &requests {
                if let Some(acknowledgment) = held.get(&request.frame) {
                    Self::final_acknowledgment(request, acknowledgment.clone())?;
                }
            }
            for request in requests {
                if held.contains_key(&request.frame) {
                    spool::discard(&self.store_root, &self.producer, &request)
                        .map_err(Error::Invalid)?;
                }
            }
        }
    }

    async fn deliver(&mut self, spooled: Spooled) -> Result<Acknowledgment> {
        let announce = Ingress::Reserve {
            frame: format!(
                "{}/reserve/{}",
                spooled.publication.frame,
                uuid::Uuid::now_v7()
            ),
            publication: spooled.publication.clone(),
        };
        self.check_message(&announce)?;
        let answer = self.transmit(&announce).await?;
        if answer.outcome["producer_bytes"] != 1
            || !answer
                .outcome
                .as_object()
                .is_some_and(|o| o.contains_key("complete"))
        {
            return Err(Error::Invalid(
                "this Sinex does not accept spooled producer bytes".into(),
            ));
        }
        let done = if answer.outcome["complete"].is_null() {
            for (role, _) in spooled.publication.roles() {
                // Sinex may already have removed these after a lost final
                // acknowledgment; a completed request needs no bytes.
                spool::snapshot(&self.store_root, &self.producer, &spooled.publication, role)
                    .map_err(Error::Invalid)?;
            }
            let wire = spooled.wire_frame();
            self.check_message(&wire)?;
            self.transmit(&wire).await?
        } else {
            checked(serde_json::from_value(answer.outcome["complete"].clone())?)?
        };
        let done = Self::final_acknowledgment(&spooled.publication, done)?;
        spool::discard(&self.store_root, &self.producer, &spooled.publication)
            .map_err(Error::Invalid)?;
        Ok(done)
    }

    /// Sends a message of events: directly when it fits one NATS message,
    /// otherwise as a spooled envelope.
    pub(crate) async fn send(&mut self, message: &Ingress) -> Result<Acknowledgment> {
        if !matches!(message, Ingress::Events { .. }) {
            return Err(Error::Invalid(
                "only messages of events are sent directly".into(),
            ));
        }
        if let Some(spooled) = spool::load(&self.store_root, &self.producer, message.frame())
            .map_err(Error::Invalid)?
        {
            return self.deliver(spooled).await;
        }
        if encode_frame(message).len().saturating_add(4096) <= self.max_message {
            return self.transmit(message).await;
        }
        let spooled = self.spool(message.clone(), None)?;
        self.deliver(spooled).await
    }

    /// Places `bytes` as the material sent under `name`. A request already
    /// in the spool is resent as it is; `bytes` is then not read.
    pub(crate) async fn material<R: Read>(
        &mut self,
        name: String,
        hash: &hashed_blobs::Hash,
        length: u64,
        bytes: &mut R,
        manifest: serde_json::Value,
    ) -> Result<MaterialId> {
        let content = format!("blake3:{}", hash.hex());
        let spooled =
            match spool::load(&self.store_root, &self.producer, &name).map_err(Error::Invalid)? {
                Some(spooled) => {
                    if spooled
                        .publication
                        .material
                        .as_ref()
                        .is_none_or(|m| m.bytes.blob != content)
                    {
                        return Err(Error::Invalid(format!(
                            "the spooled request for {name} names other bytes"
                        )));
                    }
                    spooled
                }
                None => {
                    let material = MaterialFrame {
                        id: MaterialId::new(),
                        input: InputRef {
                            kind: "polylogue".into(),
                            locator: name.clone(),
                            host: String::new(),
                        },
                        blob: content,
                        length,
                        parser: "polylogue".into(),
                        parser_version: 1,
                        part_index: 0,
                        previous_part: None,
                        manifest,
                    };
                    self.spool(
                        Ingress::Material {
                            frame: name,
                            material: Box::new(material),
                        },
                        Some(bytes),
                    )?
                }
            };
        self.deliver(spooled)
            .await?
            .material
            .ok_or_else(|| Error::Invalid("a placed material has no identity".into()))
    }
}

#[cfg(test)]
mod tests;
