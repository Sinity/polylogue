//! Reserve the receiver's ownership of exact private bytes before installation.
use super::*;
use sinex_protocol::producer_bytes::{
    Bytes, Material, Publication, encode_frame,
    spool::{self, Frozen},
};

impl Sinex {
    pub(super) fn check_message(&self, frame: &Ingress) -> Result<()> {
        if encode_frame(frame).len().saturating_add(4096) > self.max_message {
            return Err(Error::Invalid(
                "producer control frame exceeds negotiated NATS payload".into(),
            ));
        }
        Ok(())
    }

    fn freeze_bytes(
        &self,
        frame: Ingress,
        class: Class,
        material: Option<&mut std::fs::File>,
    ) -> Result<Frozen> {
        if class == Class::Sealed {
            return Err(Error::Invalid(
                "sealed producer byte publication is unsupported".into(),
            ));
        }
        let body = encode_frame(&frame);
        let envelope = (body.len().saturating_add(4096) > self.max_message).then(|| Bytes {
            blob: format!("blake3:{}", blake3::hash(&body)),
            length: body.len() as u64,
        });
        let declaration = match &frame {
            Ingress::Material { material, .. } => Some(Material::from_frame(material)),
            _ => None,
        };
        let publication = Publication {
            version: 1,
            frame: frame.frame().to_owned(),
            class,
            material: declaration,
            envelope,
        };
        publication.validate().map_err(Error::Invalid)?;
        let frozen = spool::freeze(
            &self.store_root,
            &self.producer,
            &Frozen { publication, frame },
        )
        .map_err(Error::Invalid)?;
        let publication = &frozen.publication;
        if let Some(bytes) = material {
            spool::stage(
                &self.store_root,
                &self.producer,
                publication,
                sinex_protocol::producer_bytes::Role::Material,
                bytes,
            )
            .map_err(Error::Invalid)?;
        }
        if publication.envelope.is_some() {
            spool::stage(
                &self.store_root,
                &self.producer,
                publication,
                sinex_protocol::producer_bytes::Role::Envelope,
                body.as_slice(),
            )
            .map_err(Error::Invalid)?;
        }
        Ok(frozen)
    }

    async fn deliver_frozen(&mut self, frozen: Frozen) -> Result<Acknowledgment> {
        let reservation = Ingress::Reserve {
            frame: format!(
                "{}/reserve/{}",
                frozen.publication.frame,
                uuid::Uuid::now_v7()
            ),
            publication: frozen.publication.clone(),
        };
        self.check_message(&reservation)?;
        let reserved = self.transmit(&reservation).await?;
        if reserved.outcome["producer_bytes"] != 1
            || !reserved
                .outcome
                .as_object()
                .is_some_and(|o| o.contains_key("complete"))
        {
            return Err(Error::Invalid(
                "receiver does not support durable producer byte custody".into(),
            ));
        }
        let done = if reserved.outcome["complete"].is_null() {
            for (role, _) in frozen.publication.roles() {
                // Receiver cleanup may already have removed these after a lost
                // final ack; the completed reservation above needs no bytes.
                spool::snapshot(&self.store_root, &self.producer, &frozen.publication, role)
                    .map_err(Error::Invalid)?;
            }
            let wire = frozen.wire_frame();
            self.check_message(&wire)?;
            self.transmit(&wire).await?
        } else {
            checked(serde_json::from_value(
                reserved.outcome["complete"].clone(),
            )?)?
        };
        if done.frame != frozen.publication.frame
            || done.material != frozen.publication.material.as_ref().map(|m| m.id)
        {
            return Err(Error::Invalid(
                "final acknowledgment differs from frozen producer ownership".into(),
            ));
        }
        spool::discard(&self.store_root, &self.producer, &frozen.publication)
            .map_err(Error::Invalid)?;
        Ok(done)
    }

    /// Inline events retain their existing path; byte-bearing frames have one
    /// canonical frozen owner, even when a material itself needs indirection.
    pub(super) async fn send(&mut self, frame: &Ingress) -> Result<Acknowledgment> {
        let class = match frame {
            Ingress::Events { events, .. } => {
                events.iter().map(|e| e.class).max().unwrap_or(Class::Open)
            }
            _ => {
                return Err(Error::Invalid(
                    "byte-bearing frames require their frozen source snapshot".into(),
                ));
            }
        };
        if let Some(frozen) =
            spool::load(&self.store_root, &self.producer, frame.frame()).map_err(Error::Invalid)?
        {
            if frozen.publication.class != class {
                return Err(Error::Invalid("frozen frame classification differs".into()));
            }
            return self.deliver_frozen(frozen).await;
        }
        if encode_frame(frame).len().saturating_add(4096) <= self.max_message {
            return self.transmit(frame).await;
        }
        let frozen = self.freeze_bytes(frame.clone(), class, None)?;
        self.deliver_frozen(frozen).await
    }

    pub(super) async fn material(
        &mut self,
        frame: String,
        known: Option<&Acknowledgment>,
        hash: &hashed_blobs::Hash,
        source: impl FnOnce() -> Result<sinex_rows::BlobSnapshot>,
        manifest: serde_json::Value,
        class: Class,
    ) -> Result<MaterialId> {
        if class == Class::Sealed {
            return Err(Error::Invalid(
                "sealed producer byte publication is unsupported".into(),
            ));
        }
        let frozen =
            spool::load(&self.store_root, &self.producer, &frame).map_err(Error::Invalid)?;
        if let Some(known) = known {
            let done = checked(known.clone())?;
            if done.frame != frame {
                return Err(Error::Invalid("held acknowledgment frame differs".into()));
            }
            if let Some(frozen) = frozen {
                if done.material != frozen.publication.material.as_ref().map(|m| m.id) {
                    return Err(Error::Invalid(
                        "held material differs from frozen identity".into(),
                    ));
                }
                spool::discard(&self.store_root, &self.producer, &frozen.publication)
                    .map_err(Error::Invalid)?;
            }
            return done
                .material
                .ok_or_else(|| Error::Invalid(format!("{} registered no material", done.frame)));
        }
        let frozen = if let Some(frozen) = frozen {
            if frozen.publication.class != class
                || frozen
                    .publication
                    .material
                    .as_ref()
                    .is_none_or(|m| m.bytes.blob != format!("blake3:{}", hash.hex()))
            {
                return Err(Error::Invalid(
                    "frozen material differs from requested bytes or class".into(),
                ));
            }
            frozen
        } else {
            let mut source = source()?;
            let material = MaterialFrame {
                id: MaterialId::new(),
                input: InputRef {
                    kind: "polylogue".into(),
                    locator: frame.clone(),
                    host: String::new(),
                },
                blob: format!("blake3:{}", hash.hex()),
                length: source.length,
                parser: "polylogue".into(),
                parser_version: 1,
                part_index: 0,
                previous_part: None,
                manifest,
            };
            self.freeze_bytes(
                Ingress::Material {
                    frame,
                    material: Box::new(material),
                },
                class,
                Some(&mut source.file),
            )?
        };
        self.deliver_frozen(frozen)
            .await?
            .material
            .ok_or_else(|| Error::Invalid("material publication registered no identity".into()))
    }
}

#[cfg(test)]
mod tests;
