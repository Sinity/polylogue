//! Private canonical custody precedes publication and survives cache replacement.
use super::*;
use polylogue_store::primary::{CanonicalCustody, RawInputProof};

impl Sinex {
    pub(crate) fn stage_canonical_products(
        &self,
        identity: &str,
        writes: &[ProductWrite],
        historical_proof: Option<&RawInputProof>,
    ) -> Result<PreparedWrite> {
        if identity.is_empty()
            || writes.is_empty()
            || writes.iter().any(|write| {
                write.location.is_some()
                    || !matches!(write.product, Product::AttachmentBinding { .. })
            })
        {
            return Err(Error::Invalid(
                "private canonical custody requires an identified supported body".into(),
            ));
        }
        let name = format!("canonical-custody:{identity}:observation");
        {
            let queue = self.queue()?;
            if existing(&queue, &name, identity, writes)? {
                return Ok(PreparedWrite {
                    identity: name,
                    pending: true,
                });
            }
        }
        let preparation = hashed_blobs::Preparation::new(&self.config.store)?;
        let mut encoder =
            preparation.plain_writer(tempfile::tempfile_in(&self.config.store)?, 3)?;
        serde_json::to_writer(
            &mut encoder,
            &BorrowedCustody {
                identity,
                products: writes,
                historical_proof,
                revoked: false,
            },
        )?;
        let encoded = encoder.finish()?;
        let descriptor = encoded.descriptor();
        let queue = self.queue()?;
        if existing(&queue, &name, identity, writes)? {
            return Ok(PreparedWrite {
                identity: name,
                pending: true,
            });
        }
        queue.prepare_encoded_material(
            Ingress::Material {
                frame: name.clone(),
                material: Box::new(MaterialFrame {
                    source: crate::source_reference(),
                    id: crate::material_id(&self.config.producer, &name),
                    input: InputRef {
                        kind: "polylogue-private-canonical-custody".into(),
                        locator: identity.into(),
                        host: self.host.clone(),
                    },
                    blob: descriptor.hash.to_string(),
                    length: descriptor.decoded_len,
                    parser: "polylogue-canonical-custody".into(),
                    parser_version: 1,
                    part_index: 0,
                    previous_part: None,
                    manifest: Manifest {
                        role: MaterialRole::Records,
                        media_type: Some("application/json".into()),
                        details: serde_json::Map::from_iter([(
                            "polylogue_private_canonical".into(),
                            serde_json::json!(true),
                        )]),
                        ..Default::default()
                    },
                }),
            },
            PreparedBody {
                file: encoded.into_file(),
                descriptor,
                path: None,
            },
        )?;
        Ok(PreparedWrite {
            identity: name,
            pending: true,
        })
    }

    pub(crate) fn visit_canonical_custody(
        &self,
        visit: &mut dyn FnMut(&CanonicalCustody) -> polylogue_store::Result<()>,
    ) -> Result<()> {
        let queue = self.queue()?;
        queue.visit_names("canonical-custody:", &mut |identity| {
            let entry = queue
                .get(identity)?
                .ok_or_else(|| Error::Invalid("private canonical custody disappeared".into()))?;
            if !entry.is_private() {
                return Err(Error::Invalid(
                    "private custody index names a public operation".into(),
                ));
            }
            let mut body: CanonicalCustody = crate::spool::decode_json(queue.body(identity)?)?;
            if identity != format!("canonical-custody:{}:observation", body.identity) {
                return Err(Error::Invalid(
                    "private canonical body changed its original identity".into(),
                ));
            }
            body.revoked = entry.revoked;
            visit(&body).map_err(Error::Store)
        })
    }

    pub(crate) fn revoke_canonical_custody(&self) -> Result<()> {
        // Match connection operations' lock order and finish any in-flight
        // publication before making the durable revocation visible.
        let mut client = self
            .client
            .lock()
            .map_err(|_| Error::Invalid("producer client panicked".into()))?;
        self.queue()?.revoke_canonical()?;
        if let Some(client) = client.as_mut() {
            client.fence_publication();
        }
        Ok(())
    }
}

#[derive(serde::Serialize)]
struct BorrowedCustody<'a> {
    identity: &'a str,
    products: &'a [ProductWrite],
    historical_proof: Option<&'a RawInputProof>,
    revoked: bool,
}
fn existing(
    queue: &crate::spool::Spool,
    name: &str,
    identity: &str,
    writes: &[ProductWrite],
) -> Result<bool> {
    let Some(entry) = queue.get(name)? else {
        return Ok(false);
    };
    if !entry.is_private() {
        return Err(Error::Invalid(
            "canonical custody name belongs to another operation".into(),
        ));
    }
    let original: CanonicalCustody = crate::spool::decode_json(queue.body(name)?)?;
    if original.identity != identity || original.products != writes {
        return Err(Error::Invalid(
            "canonical custody changed the original frozen intent".into(),
        ));
    }
    Ok(true)
}
