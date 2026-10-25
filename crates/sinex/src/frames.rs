//! Sending frames to Sinex on the same host. A material's bytes are written
//! to the frame's spool directory under Sinex's store root; Sinex stores them
//! before it records the frame and removes the directory afterwards. A frame
//! too large for one NATS message is written there as its JSON and sent as an
//! indirect frame under the same name.
//!
//! Every frame name is fixed by what the frame carries, so nothing about a
//! send is kept here. A send is done when Sinex acknowledges it; until then
//! the next pass finds it missing through `have` and sends it again from the
//! archive's own bytes, and Sinex answers a frame it already recorded from
//! its record.
use crate::{Error, Result, Sinex, content_name};
use sinex_model::{InputRef, MaterialId};
use sinex_protocol::{Acknowledgment, Ingress, MaterialFrame};
use sinex_transport::spool;
use std::io::Read;

/// Room left in a NATS message for its subject and headers.
const MESSAGE_OVERHEAD: usize = 4096;

impl Sinex {
    /// Sends `frame` and waits for its acknowledgment. A frame too large for
    /// one message travels indirectly.
    pub(crate) async fn send(&mut self, frame: Ingress) -> Result<Acknowledgment> {
        let body = serde_json::to_vec(&frame)?;
        if body.len().saturating_add(MESSAGE_OVERHEAD) <= self.max_message {
            return self.transmit(frame.frame(), body).await;
        }
        let name = frame.frame().to_owned();
        let blob = format!("blake3:{}", blake3::hash(&body).to_hex());
        let length = body.len() as u64;
        spool::stage(
            &self.store_root,
            &self.producer,
            &name,
            &blob,
            length,
            body.as_slice(),
        )
        .map_err(Error::Invalid)?;
        let indirect = serde_json::to_vec(&Ingress::Indirect {
            frame: name.clone(),
            blob,
            length,
        })?;
        self.transmit(&name, indirect).await
    }

    /// Registers `bytes` as the material named `name` and returns its
    /// identity. When Sinex recorded `name` already, its record answers with
    /// the material it registered then.
    pub(crate) async fn material(
        &mut self,
        name: String,
        hash: &hashed_blobs::Hash,
        length: u64,
        bytes: impl Read,
        manifest: serde_json::Value,
    ) -> Result<MaterialId> {
        let blob = content_name(hash);
        spool::stage(
            &self.store_root,
            &self.producer,
            &name,
            &blob,
            length,
            bytes,
        )
        .map_err(Error::Invalid)?;
        let frame = Ingress::Material {
            frame: name.clone(),
            material: Box::new(MaterialFrame {
                id: MaterialId::new(),
                input: InputRef {
                    kind: "polylogue".into(),
                    locator: name,
                    host: String::new(),
                },
                blob,
                length,
                parser: "polylogue".into(),
                parser_version: 1,
                part_index: 0,
                previous_part: None,
                manifest,
            }),
        };
        self.send(frame)
            .await?
            .material
            .ok_or_else(|| Error::Invalid("a registered material has no identity".into()))
    }
}
