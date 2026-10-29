//! Durable acquisition and product preparation for the primary writer.
use crate::{Error, Result, Sinex};
use polylogue_model::{BlobHash, authored::PreparedRequest};
use polylogue_store::primary::{
    Capture, CapturedInput, Material, PendingCapture, PreparedInput, PreparedWrite, Product,
    ProductWrite, SourceLocation,
};
use sinex_model::{InputRef, Manifest, MaterialId, MaterialRole};
use sinex_protocol::{Ingress, MaterialFrame};
use std::io::Read;
mod reading;

impl Sinex {
    pub(crate) fn resume_publication(&self) {
        let result = self.with(async |client| client.retry_prepared().await);
        if result.is_ok() {
            let result = self
                .drain_requests()
                .and_then(|()| self.drain_observations())
                .and_then(|()| self.drain_readings());
            if let Err(error) = result {
                if let Ok(mut health) = self.health.lock() {
                    health.error = Some(error.to_string());
                }
            }
        }
    }

    pub(super) fn queue(&self) -> Result<std::sync::MutexGuard<'_, crate::spool::Spool>> {
        self.spool
            .lock()
            .map_err(|_| Error::Invalid("producer spool panicked".into()))
    }

    pub(super) fn visit_prepared(
        &self,
        mut visit: impl FnMut(&crate::spool::Spool, crate::spool::Prepared) -> Result<()>,
    ) -> Result<()> {
        let queue = self.queue()?;
        let mut after = None;
        loop {
            let page = queue.pending_after(after, 64)?;
            if page.is_empty() {
                return Ok(());
            }
            for entry in page {
                after = Some(entry.order);
                visit(&queue, entry)?;
            }
        }
    }

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
        let queue = self.queue()?;
        if queue.get(&identity)?.is_some() {
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
        let observation = matches!(&captured,Capture::Input(input) if input.rows.extent.is_none());
        let observation_bytes = if observation {
            Some(serde_json::to_vec(&captured)?)
        } else {
            None
        };
        let (hash, length) = observation_bytes.as_ref().map_or((hash, length), |bytes| {
            (BlobHash::of(bytes), bytes.len() as u64)
        });
        let frame = Ingress::Material {
            frame: identity.clone(),
            material: Box::new(MaterialFrame {
                id: crate::material_id(&identity),
                input,
                blob: format!("blake3:{hash}"),
                length,
                parser: if observation {
                    "polylogue-observation".into()
                } else {
                    parser
                },
                parser_version: 3,
                part_index: if observation { 0 } else { part_index },
                previous_part: if observation { None } else { previous_part },
                manifest: Manifest {
                    details: [("polylogue_capture".into(), serde_json::to_value(&captured)?)]
                        .into_iter()
                        .collect(),
                    role: if observation {
                        MaterialRole::Records
                    } else {
                        MaterialRole::default()
                    },
                    media_type: observation.then(|| "application/json".into()),
                    ..Default::default()
                },
            }),
        };
        match observation_bytes {
            Some(bytes) => queue.prepare_material(frame, bytes.as_slice())?,
            None => queue.prepare_material(frame, bytes)?,
        };
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
        Ok(PreparedInput::Pending {
            identity: self
                .prepared_capture(Capture::Input(captured.clone()), bytes)?
                .identity,
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

    fn capture_material(
        queue: &crate::spool::Spool,
        entry: &crate::spool::Prepared,
    ) -> Result<Option<Material>> {
        let Some(ack) = queue.acknowledgment(&entry.identity)? else {
            return Ok(None);
        };
        crate::client::checked(ack.clone())?;
        let descriptor = entry
            .material
            .as_ref()
            .ok_or_else(|| Error::Invalid("capture has no material descriptor".into()))?;
        let material = Material {
            id: ack
                .material
                .ok_or_else(|| Error::Invalid("material acknowledgment has no material".into()))?
                .to_string(),
            hash: descriptor
                .blob
                .strip_prefix("blake3:")
                .and_then(|hash| BlobHash::parse(hash).ok())
                .ok_or_else(|| Error::Invalid("invalid material digest".into()))?,
            length: descriptor.length,
            extents: ack.material_extents,
        };
        crate::material::validate(&material)?;
        Ok(Some(material))
    }

    pub(crate) fn pending_captures(&self) -> Result<Vec<PendingCapture>> {
        let mut pending = Vec::new();
        self.visit_prepared(|queue,entry|{
            let Some(captured)=entry.capture.clone() else {return Ok(());};
            let acknowledged=Self::capture_material(queue,&entry)?;
            let anchor=acknowledged.as_ref().map(|material|SourceLocation{material:material.id.clone(),anchor:0});
            let material=if matches!(&captured,Capture::Input(input) if input.rows.extent.is_none()) {None} else {acknowledged};
            pending.push(PendingCapture{identity:entry.identity,captured,material,anchor});
            Ok(())
        })?;
        Ok(pending)
    }

    /// Later A→B→A observations get a new operation; retrying the most recent
    /// observation keeps its original identity without reading large frames.
    pub(crate) fn prepare_products(&self, writes: &[ProductWrite]) -> Result<PreparedWrite> {
        let keys = writes
            .iter()
            .map(|write| crate::products::key(&write.product, write.output_index))
            .collect::<Result<Vec<_>>>()?;
        let scope = blake3::hash(&serde_json::to_vec(&keys)?)
            .to_hex()
            .to_string();
        let digest = blake3::hash(&serde_json::to_vec(writes)?)
            .to_hex()
            .to_string();
        let prefix = format!("products:{scope}:");
        let mut latest = None;
        self.visit_prepared(|_, entry| {
            if entry.identity.starts_with(&prefix) {
                latest = Some(entry);
            }
            Ok(())
        })?;
        let identity = match latest
            .filter(|entry| entry.identity.starts_with(&format!("{prefix}{digest}:")))
        {
            Some(entry) => {
                return Ok(PreparedWrite {
                    identity: entry.identity,
                    pending: true,
                });
            }
            None => format!("{prefix}{digest}:{}", uuid::Uuid::now_v7()),
        };
        self.prepare_named_products(&identity, writes)
    }

    /// The transactional derived outbox supplies this stable operation identity.
    pub(crate) fn prepare_named_products(
        &self,
        identity: &str,
        writes: &[ProductWrite],
    ) -> Result<PreparedWrite> {
        let queue = self.queue()?;
        if queue.get(identity)?.is_some() {
            return Ok(PreparedWrite {
                identity: identity.into(),
                pending: true,
            });
        }
        let unlocated = writes
            .iter()
            .any(|write| write.location.is_none() && write.parents.is_empty());
        if unlocated {
            let mut writes = writes.to_vec();
            let observation = format!("{identity}:observation");
            let body = serde_json::to_vec(&writes)?;
            queue.prepare_material(
                Ingress::Material {
                    frame: observation.clone(),
                    material: Box::new(MaterialFrame {
                        id: crate::material_id(&observation),
                        input: InputRef {
                            kind: "polylogue-observation".into(),
                            locator: identity.into(),
                            host: self.host.clone(),
                        },
                        blob: format!("blake3:{}", blake3::hash(&body).to_hex()),
                        length: body.len() as u64,
                        parser: "polylogue-products".into(),
                        parser_version: 3,
                        part_index: 0,
                        previous_part: None,
                        manifest: Manifest {
                            role: MaterialRole::Records,
                            media_type: Some("application/json".into()),
                            ..Default::default()
                        },
                    }),
                },
                body.as_slice(),
            )?;
            for write in &mut writes {
                if write.location.is_none() && write.parents.is_empty() {
                    write.location = Some(SourceLocation {
                        material: crate::material_id(&observation).to_string(),
                        anchor: 0,
                    });
                }
            }
            if queue
                .get(&observation)?
                .is_some_and(|entry| entry.acknowledged && !entry.refused)
            {
                queue.prepare_group(
                    Ingress::Events {
                        frame: identity.into(),
                        events: crate::products::events(&writes)?,
                        failures: Vec::new(),
                    },
                    &[observation],
                )?;
            }
            return Ok(PreparedWrite {
                identity: identity.into(),
                pending: true,
            });
        }
        queue.prepare(Ingress::Events {
            frame: identity.into(),
            events: crate::products::events(&writes)?,
            failures: Vec::new(),
        })?;
        Ok(PreparedWrite {
            identity: identity.into(),
            pending: true,
        })
    }

    pub(crate) fn drain_observations(&self) -> Result<()> {
        let mut ready = Vec::new();
        self.visit_prepared(|queue, entry| {
            if entry.kind == crate::spool::Kind::Material
                && entry.acknowledged
                && !entry.refused
                && entry.capture.is_none()
                && !entry.request
                && queue.capture_group(&entry.identity)?.is_none()
            {
                if let Ingress::Material { material, .. } = queue.decode_frame(&entry.identity)? {
                    if material.parser == "polylogue-products" {
                        ready.push(entry.identity);
                    }
                }
            }
            Ok(())
        })?;
        for observation in ready {
            let writes: Vec<ProductWrite> = {
                let queue = self.queue()?;
                serde_json::from_reader(std::io::BufReader::new(queue.body(&observation)?))?
            };
            let identity = observation.strip_suffix(":observation").ok_or_else(|| {
                Error::Invalid("observation identity has no product operation".into())
            })?;
            self.prepare_named_products(identity, &writes)?;
        }
        Ok(())
    }

    pub(crate) fn complete_captures(
        &self,
        identities: &[String],
        writes: &[ProductWrite],
    ) -> Result<PreparedWrite> {
        let queue = self.queue()?;
        if let Some(group) = queue.group_for(identities)? {
            return Ok(PreparedWrite {
                identity: group.identity,
                pending: true,
            });
        }
        for identity in identities {
            let entry = queue
                .get(identity)?
                .ok_or_else(|| Error::Invalid("capture is not prepared".into()))?;
            if Self::capture_material(&queue, &entry)?.is_none() {
                return Err(Error::Unreachable(
                    "capture awaits material confirmation".into(),
                ));
            }
        }
        let identity = format!(
            "capture-products:{}",
            blake3::hash(&serde_json::to_vec(identities)?).to_hex()
        );
        queue.prepare_group(
            Ingress::Events {
                frame: identity.clone(),
                events: crate::products::events(writes)?,
                failures: Vec::new(),
            },
            identities,
        )?;
        Ok(PreparedWrite {
            identity,
            pending: true,
        })
    }

    pub(crate) fn prepare_request(
        &self,
        request: &PreparedRequest,
    ) -> Result<polylogue_store::UserOutcome> {
        let identity = format!("request:{}", request.request.id);
        let queue = self.queue()?;
        let repeat = if queue.get(&identity)?.is_some() {
            let original: PreparedRequest =
                serde_json::from_reader(std::io::BufReader::new(queue.body(&identity)?))?;
            if original != *request {
                return Err(Error::Invalid(
                    "request identity changed immutable prepared effects".into(),
                ));
            }
            true
        } else {
            false
        };
        let bytes = serde_json::to_vec(request)?;
        queue.prepare_material(
            Ingress::Material {
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
                        role: MaterialRole::Records,
                        media_type: Some("application/json".into()),
                        ..Default::default()
                    },
                }),
            },
            bytes.as_slice(),
        )?;
        let mut outcome = request.outcome.clone();
        outcome.repeat |= repeat;
        outcome.pending = true;
        Ok(outcome)
    }

    pub(crate) fn visit_pending_requests(
        &self,
        visit: &mut dyn FnMut(&PreparedRequest) -> polylogue_store::Result<()>,
    ) -> polylogue_store::Result<()> {
        Ok(self.visit_prepared(|queue, entry| {
            if entry.request {
                let request =
                    serde_json::from_reader(std::io::BufReader::new(queue.body(&entry.identity)?))?;
                visit(&request).map_err(|error| Error::Invalid(error.to_string()))?;
            }
            Ok(())
        })?)
    }
    pub(crate) fn visit_pending_blobs(
        &self,
        visit: &mut dyn FnMut(&polylogue_store::NewBlob) -> polylogue_store::Result<()>,
    ) -> polylogue_store::Result<()> {
        Ok(self.visit_prepared(|_, entry| {
            if let Some(Capture::Blob { hash, length }) = entry.capture {
                visit(&polylogue_store::NewBlob { hash, size: length })
                    .map_err(|error| Error::Invalid(error.to_string()))?;
            }
            Ok(())
        })?)
    }

    pub(crate) fn drain_requests(&self) -> Result<()> {
        let mut ready = Vec::new();
        self.visit_prepared(|queue, entry| {
            if entry.request
                && entry.acknowledged
                && queue.capture_group(&entry.identity)?.is_none()
            {
                ready.push(entry.identity);
            }
            Ok(())
        })?;
        for identity in ready {
            let request: PreparedRequest = {
                let queue = self.queue()?;
                serde_json::from_reader(std::io::BufReader::new(queue.body(&identity)?))?
            };
            let mut dependencies = vec![identity.clone()];
            let mut writes = Vec::new();
            let mut waiting = false;
            for attachment in request
                .effect
                .chats
                .iter()
                .flat_map(|change| &change.action.attachments)
            {
                self.visit_prepared(|queue,entry|{
                    if matches!(&entry.capture,Some(Capture::Blob{hash,..}) if *hash==attachment.blob) {
                        if let Some(material)=Self::capture_material(queue,&entry)? {
                            writes.push(ProductWrite{location:Some(SourceLocation{material:material.id.clone(),anchor:0}),product:Product::Blob(material),parents:Vec::new(),machine:Some(request.request.machine.clone()),output_index:writes.len() as u64});
                            if queue.capture_group(&entry.identity)?.is_none() {dependencies.push(entry.identity);}
                        } else {waiting=true;}
                    }
                    Ok(())
                })?;
            }
            if waiting {
                continue;
            }
            dependencies.sort();
            dependencies.dedup();
            writes.push(ProductWrite {
                product: Product::Request(request.clone()),
                location: Some(SourceLocation {
                    material: crate::material_id(&identity).to_string(),
                    anchor: 0,
                }),
                parents: Vec::new(),
                machine: Some(request.request.machine),
                output_index: writes.len() as u64,
            });
            self.complete_captures(&dependencies, &writes)?;
        }
        Ok(())
    }
}
