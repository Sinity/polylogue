//! Durable acquisition and product preparation for the primary writer.
use crate::{Error, Result, Sinex};
use polylogue_model::BlobHash;
use polylogue_store::primary::{
    Capture, CapturedInput, Material, PendingCapture, PreparedInput, PreparedWrite, Product,
    ProductWrite, SourceLocation,
};
use polylogue_store::user_writes::UserRequest;
use sinex_model::{InputRef, Manifest, MaterialId};
use sinex_protocol::{Ingress, MaterialFrame};
use std::io::Read;

impl Sinex {
    fn prepared_capture(&self, captured: Capture, bytes: &mut dyn Read) -> Result<PreparedWrite> {
        let mut intent = serde_json::to_value(&captured)?;
        if let Some(observation) = intent
            .pointer_mut("/rows/observation")
            .and_then(serde_json::Value::as_object_mut)
        {
            observation.remove("id");
            observation.remove("seen_at");
        }
        let identity = format!(
            "capture:{}",
            blake3::hash(&serde_json::to_vec(&intent)?).to_hex()
        );
        if self
            .spool
            .lock()
            .map_err(|_| Error::Invalid("producer spool panicked".into()))?
            .get(&identity)?
            .is_some()
        {
            return Ok(PreparedWrite {
                identity,
                pending: true,
            });
        }
        let (input, hash, length, parser, part_index, previous_part) = match &captured {
            Capture::Input(input) => (
                InputRef {
                    kind: input.input.origin.clone(),
                    locator: input.input.path.clone(),
                    host: input.input.host.clone(),
                },
                input.hash,
                input.length,
                format!("polylogue-{}", input.input.origin),
                input.index,
                input
                    .previous
                    .as_ref()
                    .map(|id| {
                        id.parse()
                            .map(MaterialId)
                            .map_err(|error| Error::Invalid(format!("previous material: {error}")))
                    })
                    .transpose()?,
            ),
            Capture::Blob { hash, length } => (
                InputRef {
                    kind: "polylogue-attachment".into(),
                    locator: hash.to_string(),
                    host: self.host.clone(),
                },
                *hash,
                *length,
                "polylogue-attachment".into(),
                0,
                None,
            ),
        };
        let frame = Ingress::Material {
            frame: identity.clone(),
            material: Box::new(MaterialFrame {
                id: crate::material_id(&identity),
                input,
                blob: format!("blake3:{hash}"),
                length,
                parser,
                parser_version: 3,
                part_index,
                previous_part,
                manifest: Manifest {
                    details: [("polylogue_capture".into(), serde_json::to_value(&captured)?)]
                        .into_iter()
                        .collect(),
                    ..Default::default()
                },
            }),
        };
        self.spool
            .lock()
            .map_err(|_| Error::Invalid("producer spool panicked".into()))?
            .prepare_material(frame, bytes)?;
        Ok(PreparedWrite {
            identity,
            pending: true,
        })
    }

    pub(crate) fn prepare_input(
        &self,
        captured: &CapturedInput,
        bytes: &mut dyn Read,
    ) -> Result<PreparedInput> {
        let prepared = self.prepared_capture(Capture::Input(captured.clone()), bytes)?;
        Ok(PreparedInput::Pending {
            identity: prepared.identity,
        })
    }

    pub(crate) fn prepare_blob(
        &self,
        hash: BlobHash,
        length: u64,
        bytes: &mut dyn Read,
    ) -> Result<PreparedWrite> {
        self.prepared_capture(Capture::Blob { hash, length }, bytes)
    }

    pub(crate) fn pending_captures(&self) -> Result<Vec<PendingCapture>> {
        let entries = self
            .spool
            .lock()
            .map_err(|_| Error::Invalid("producer spool panicked".into()))?
            .prepared()?;
        entries
            .into_iter()
            .filter_map(|prepared| {
                let Ingress::Material { material, .. } = &prepared.frame else {
                    return None;
                };
                let captured = material.manifest.details.get("polylogue_capture")?;
                Some((|| {
                    let captured = serde_json::from_value(captured.clone())?;
                    let material = prepared
                        .acknowledgment
                        .as_ref()
                        .map(|ack| -> Result<_> {
                            if let Some(reason) = &ack.refusal {
                                return Err(Error::Refused {
                                    name: prepared.frame.frame().into(),
                                    reason: reason.clone(),
                                });
                            }
                            let id = ack.material.ok_or_else(|| {
                                Error::Invalid("material acknowledgment has no material".into())
                            })?;
                            let hash = material
                                .blob
                                .strip_prefix("blake3:")
                                .and_then(|hash| BlobHash::parse(hash).ok())
                                .ok_or_else(|| Error::Invalid("invalid material digest".into()))?;
                            let material = Material {
                                id: id.to_string(),
                                hash,
                                length: material.length,
                                extents: ack.material_extents.clone(),
                            };
                            crate::material::validate(&material)?;
                            Ok(material)
                        })
                        .transpose()?;
                    Ok(PendingCapture {
                        identity: prepared.frame.frame().into(),
                        captured,
                        material,
                    })
                })())
            })
            .collect()
    }

    pub(crate) fn prepare_products(&self, writes: &[ProductWrite]) -> Result<PreparedWrite> {
        let events = crate::products::events(writes)?;
        let desired = serde_json::to_value(&events)?;
        let scope = |events: &[sinex_protocol::ProducerEvent]| -> Result<String> {
            let keys: Vec<_> = events
                .iter()
                .map(|event| (&event.kind, &event.payload["record_key"]))
                .collect();
            Ok(blake3::hash(&serde_json::to_vec(&keys)?)
                .to_hex()
                .to_string())
        };
        let desired_scope = scope(&events)?;
        let spool = self
            .spool
            .lock()
            .map_err(|_| Error::Invalid("producer spool panicked".into()))?;
        let previous = spool.prepared()?.into_iter().rev().find(|prepared| {
            matches!(&prepared.frame,Ingress::Events {events,..} if scope(events).ok().as_deref()==Some(desired_scope.as_str()))
        });
        let prepared = if let Some(previous) = previous.filter(|prepared| {
            matches!(&prepared.frame,Ingress::Events {events,..} if serde_json::to_value(events).ok().as_ref()==Some(&desired))
        }) {previous} else {
            let identity = format!("products:{}",uuid::Uuid::now_v7());
            spool.prepare(Ingress::Events {frame:identity,events,failures:Vec::new()})?
        };
        let identity = prepared.frame.frame().to_owned();
        let pending = prepared.acknowledgment.is_none();
        drop(spool);
        Ok(PreparedWrite { identity, pending })
    }

    pub(crate) fn complete_capture(
        &self,
        identity: &str,
        writes: &[ProductWrite],
    ) -> Result<PreparedWrite> {
        let previous = self
            .spool
            .lock()
            .map_err(|_| Error::Invalid("producer spool panicked".into()))?
            .get(identity)?
            .ok_or_else(|| Error::Invalid("capture is not prepared".into()))?;
        if let Some(dependency) = previous.dependency {
            let prepared = self
                .spool
                .lock()
                .map_err(|_| Error::Invalid("producer spool panicked".into()))?
                .get(&dependency)?
                .ok_or_else(|| Error::Invalid("capture products are missing".into()))?;
            return Ok(PreparedWrite {
                identity: dependency,
                pending: prepared.acknowledgment.is_none(),
            });
        }
        let prepared = self.prepare_products(writes)?;
        self.spool
            .lock()
            .map_err(|_| Error::Invalid("producer spool panicked".into()))?
            .depend(identity, &prepared.identity)?;
        Ok(prepared)
    }

    pub(crate) fn prepare_request(
        &self,
        request: &UserRequest,
    ) -> Result<polylogue_store::UserOutcome> {
        let identity = format!("request:{}", request.id);
        let spool = self
            .spool
            .lock()
            .map_err(|_| Error::Invalid("producer spool panicked".into()))?;
        let original = if spool.get(&identity)?.is_some() {
            let previous: UserRequest = serde_json::from_reader(spool.body(&identity)?)?;
            if previous.write.intent() != request.write.intent() {
                return Err(Error::Invalid("request identity changed contents".into()));
            }
            previous
        } else {
            request.clone()
        };
        let bytes = serde_json::to_vec(&original)?;
        let frame = Ingress::Material {
            frame: identity.clone(),
            material: Box::new(MaterialFrame {
                id: crate::material_id(&identity),
                input: InputRef {
                    kind: "polylogue".into(),
                    locator: "requests".into(),
                    host: self.host.clone(),
                },
                blob: format!("blake3:{}", blake3::hash(&bytes).to_hex()),
                length: bytes.len() as u64,
                parser: "polylogue-request".into(),
                parser_version: 3,
                part_index: 0,
                previous_part: None,
                manifest: Manifest {
                    role: sinex_model::MaterialRole::Records,
                    media_type: Some("application/json".into()),
                    ..Default::default()
                },
            }),
        };
        spool.prepare_material(frame, bytes.as_slice())?;
        drop(spool);
        let writes = [ProductWrite {
            product: Product::Request(original),
            location: Some(SourceLocation {
                material: crate::material_id(&identity).to_string(),
                anchor: 0,
            }),
            parents: Vec::new(),
            machine: Some(request.machine.clone()),
            output_index: 0,
        }];
        let prepared = self.complete_capture(&identity, &writes)?;
        Ok(polylogue_store::UserOutcome {
            id: request.id.clone(),
            repeat: false,
            pending: prepared.pending,
        })
    }
}
