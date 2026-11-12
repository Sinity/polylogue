//! Durable acquisition and product preparation for the primary writer.
use crate::{Error, Result, Sinex};
use polylogue_model::{
    BlobHash,
    authored::{PendingAuthored, PreparedRequest},
};
use polylogue_store::primary::{
    Capture, CapturedInput, Material, PendingCapture, PreparedBody, PreparedInput, PreparedWrite,
    Product, ProductWrite, SourceLocation,
    queue::{Completion, Lookup, Order},
};
use sinex_model::{InputRef, Manifest, MaterialId, MaterialRole};
use sinex_protocol::{Ingress, MaterialFrame};
mod acquisition;
mod exports;
mod reading;
mod requests;

impl Sinex {
    /// Optional advice must not mistake unpublished or unapplied producer
    /// ownership for empty delivery history, including after restart.
    pub fn advisory_history_ready(&self) -> bool {
        self.spool
            .try_lock()
            .is_ok_and(|queue| queue.is_known_empty())
    }

    pub(crate) fn send_pending(&self) {
        if let Some(sender) = &self.sender {
            sender.wake();
        }
    }

    pub(crate) fn drain_pending(&self) -> Result<()> {
        let mut result = self.client_operation(async |client| client.retry_prepared().await);
        if result.is_ok() || matches!(result, Err(Error::Refused { .. })) {
            for complete in [
                Self::drain_requests,
                Self::drain_observations,
                Self::drain_readings,
            ] {
                match complete(self) {
                    Ok(()) => {}
                    Err(error @ Error::Refused { .. }) => {
                        if result.is_ok() {
                            result = Err(error);
                        }
                    }
                    Err(error) => {
                        result = Err(error);
                        break;
                    }
                }
            }
        }
        // Publication owns this complete retry result. Reading the confirmed
        // stream remains independent and cannot erase a failed publication.
        if let Ok(mut health) = self.health.lock() {
            health.publication_error = result.as_ref().err().map(ToString::to_string);
        }
        result
    }

    pub(super) fn queue(&self) -> Result<std::sync::MutexGuard<'_, crate::spool::Spool>> {
        self.spool
            .lock()
            .map_err(|_| Error::Invalid("producer spool panicked".into()))
    }

    pub(super) fn visit_matching(
        &self,
        lookup: Lookup<'_>,
        mut visit: impl FnMut(&crate::spool::Spool, crate::spool::Prepared) -> Result<()>,
    ) -> Result<()> {
        let queue = self.queue()?;
        let mut after = None;
        loop {
            let page = queue.lookup(lookup, after, None, Order::OldestFirst, 64)?;
            if page.is_empty() {
                return Ok(());
            }
            for entry in page {
                after = Some(entry.order);
                visit(&queue, entry)?;
            }
        }
    }
}

/// The chunks a body's encodings name, when there are several.
pub(crate) fn chunks_of(
    parts: impl ExactSizeIterator<Item = hashed_blobs::EncodedBlob>,
) -> Vec<sinex_protocol::MaterialChunk> {
    if parts.len() < 2 {
        return Vec::new();
    }
    parts
        .map(|descriptor| sinex_protocol::MaterialChunk {
            blob: descriptor.hash.to_string(),
            length: descriptor.decoded_len,
        })
        .collect()
}

impl Sinex {
    fn prepared_capture(&self, captured: Capture, body: PreparedBody) -> Result<PreparedWrite> {
        let (expected_hash, expected_length) = match &captured {
            Capture::Input(input) => (input.hash, input.length),
            Capture::Blob { hash, length } => (*hash, *length),
        };
        if body.descriptor.hash.0 != expected_hash.0
            || body.descriptor.decoded_len != expected_length
        {
            return Err(Error::Invalid(
                "prepared capture descriptor differs from its declared source bytes".into(),
            ));
        }
        let mut intent = serde_json::to_value(&captured)?;
        if let Some(observation) = intent
            .pointer_mut("/rows/observation")
            .and_then(serde_json::Value::as_object_mut)
        {
            observation.remove("id");
            observation.remove("seen_at");
        }
        let source = match &captured {
            Capture::Input(input) => serde_json::to_vec(&input.input)?,
            Capture::Blob { hash, .. } => serde_json::to_vec(hash)?,
        };
        let prefix = format!("capture:{}:", blake3::hash(&source).to_hex());
        let desired = format!(
            "{prefix}{}:",
            blake3::hash(&serde_json::to_vec(&intent)?).to_hex()
        );
        let latest = self
            .queue()?
            .lookup(
                Lookup::IdentityPrefix(&prefix),
                None,
                None,
                Order::NewestFirst,
                1,
            )?
            .into_iter()
            .next();
        if let Some(entry) = latest.filter(|entry| entry.identity.starts_with(&desired)) {
            return Ok(PreparedWrite {
                identity: entry.record.identity,
                pending: true,
            });
        }
        let identity = format!("{desired}{}", uuid::Uuid::now_v7());
        let queue = self.queue()?;
        let frame = self.capture_frame(&captured, &identity, Vec::new())?;
        queue.prepare_encoded_capture(frame, vec![body], &captured)?;
        Ok(PreparedWrite {
            identity,
            pending: true,
        })
    }

    /// `chunks` names the body's content-defined chunks when it is stored as
    /// chunks.
    fn capture_frame(
        &self,
        captured: &Capture,
        identity: &str,
        chunks: Vec<sinex_protocol::MaterialChunk>,
    ) -> Result<Ingress> {
        let (input, hash, length, parser, part_index, previous_part) = match captured {
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
        let header = crate::spool::capture_header(captured);
        let observation = matches!(captured,Capture::Input(input) if input.raw_range.is_none());
        if matches!(captured, Capture::Input(input) if input.raw_range.is_none() && !input.rows.extents.is_empty())
        {
            return Err(Error::Invalid(
                "provider acquisition has no original raw range".into(),
            ));
        }
        Ok(Ingress::Material {
            frame: identity.into(),
            material: Box::new(MaterialFrame {
                source: crate::source_reference(),
                id: crate::material_id(&self.config.producer, identity),
                input,
                blob: format!("blake3:{hash}"),
                length,
                chunks,
                parser: if observation {
                    "polylogue-observation".into()
                } else {
                    parser
                },
                parser_version: 3,
                part_index: if observation { 0 } else { part_index },
                previous_part: if observation { None } else { previous_part },
                manifest: Manifest {
                    details: [("polylogue_capture".into(), serde_json::to_value(&header)?)]
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
        })
    }

    pub(crate) fn prepare_input(
        &self,
        captured: &CapturedInput,
        body: PreparedBody,
    ) -> Result<PreparedInput> {
        Ok(PreparedInput::Pending {
            identity: self
                .prepared_capture(Capture::Input(Box::new(captured.clone())), body)?
                .identity,
        })
    }
    pub(crate) fn prepare_blob(
        &self,
        hash: BlobHash,
        length: u64,
        body: PreparedBody,
    ) -> Result<PreparedWrite> {
        self.prepared_capture(Capture::Blob { hash, length }, body)
    }

    fn capture_material(
        queue: &crate::spool::Spool,
        entry: &crate::spool::Prepared,
    ) -> Result<Option<Material>> {
        let Some(ack) = queue.acknowledgment(&entry.identity)? else {
            return Ok(None);
        };
        let ack = crate::client::checked(ack)?;
        let descriptor = entry
            .material()
            .ok_or_else(|| Error::Invalid("capture has no material descriptor".into()))?;
        let sinex_protocol::FrameResult::Admitted {
            material: Some(id),
            material_extents,
            ..
        } = ack.result
        else {
            return Err(Error::Invalid(
                "material acknowledgment has no material".into(),
            ));
        };
        let material = Material {
            id: id.to_string(),
            hash: descriptor
                .blob
                .strip_prefix("blake3:")
                .and_then(|hash| BlobHash::parse(hash).ok())
                .ok_or_else(|| Error::Invalid("invalid material digest".into()))?,
            length: descriptor.length,
            extents: material_extents,
        };
        crate::material::validate(&material)?;
        Ok(Some(material))
    }

    pub(crate) fn pending_captures(&self) -> Result<Vec<PendingCapture>> {
        let mut pending = Vec::new();
        self.visit_matching(Lookup::Captures, |queue, entry| {
            if entry.unresolved {
                return Ok(());
            }
            let Some(captured) = entry.capture().cloned() else {
                return Ok(());
            };
            let acknowledged = if entry.phase.is_refused() {
                None
            } else {
                Self::capture_material(queue, &entry)?
            };
            let anchor = acknowledged.as_ref().map(|material| SourceLocation {
                material: material.id.clone(),
                anchor: 0,
            });
            let material = if matches!(&captured,Capture::Input(input) if input.raw_range.is_none())
            {
                None
            } else {
                acknowledged
            };
            pending.push(PendingCapture {
                identity: entry.record.identity,
                captured,
                material,
                anchor,
            });
            Ok(())
        })?;
        Ok(pending)
    }

    pub(crate) fn capture(&self, identity: &str) -> Result<Capture> {
        let pinned = self.queue()?.pin_capture(identity)?;
        pinned.decode()
    }

    /// Later A→B→A observations get a new operation; retrying the most recent
    /// observation keeps its original identity without reading large frames.
    pub(crate) fn prepare_products(&self, writes: &[ProductWrite]) -> Result<PreparedWrite> {
        for write in writes {
            crate::products::require_source(&write.product)?;
        }
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
        let latest = self
            .queue()?
            .lookup(
                Lookup::IdentityPrefix(&prefix),
                None,
                None,
                Order::NewestFirst,
                1,
            )?
            .into_iter()
            .next();
        let identity = match latest
            .filter(|entry| entry.identity.starts_with(&format!("{prefix}{digest}:")))
        {
            Some(entry) => {
                return Ok(PreparedWrite {
                    identity: entry.record.identity,
                    pending: true,
                });
            }
            None => format!("{prefix}{digest}:{}", uuid::Uuid::now_v7()),
        };
        self.prepare_named_products(&identity, writes)
    }

    pub(crate) fn prepare_named_products(
        &self,
        identity: &str,
        writes: &[ProductWrite],
    ) -> Result<PreparedWrite> {
        self.prepare_products_bound(identity, writes, None)
    }

    fn prepare_products_bound(
        &self,
        identity: &str,
        writes: &[ProductWrite],
        export_key: Option<&polylogue_store::primary::ExportReadingKey>,
    ) -> Result<PreparedWrite> {
        for write in writes {
            match export_key {
                Some(key) => crate::products::require_export(&write.product, key)?,
                None => crate::products::require_source(&write.product)?,
            }
        }
        let queue = self.queue()?;
        if queue.get(identity)?.is_some() {
            let original = queue.decode_frame(identity)?;
            let mut expected = crate::products::events(writes)?;
            let Ingress::Events { source, events, .. } = &original else {
                return Err(Error::Invalid(
                    "named product intent is not an event group".into(),
                ));
            };
            if source.as_ref().is_none_or(|source| {
                source.name != "polylogue"
                    || !matches!(source.version, 5 | 6 | crate::SOURCE_VERSION)
            }) {
                return Err(Error::Invalid(
                    "named product intent has an unsupported source".into(),
                ));
            }
            let observation = format!("{identity}:observation");
            if queue.get(&observation)?.is_some() {
                if queue.export_material_key(&observation)?.as_ref() != export_key {
                    return Err(Error::Invalid(
                        "export identity changed its immutable reading key".into(),
                    ));
                }
                let retained: Vec<ProductWrite> =
                    crate::spool::decode_json(queue.body(&observation)?)?;
                if retained != writes {
                    return Err(Error::Invalid(
                        "named product identity changed immutable facts".into(),
                    ));
                }
                return Ok(PreparedWrite {
                    identity: identity.into(),
                    pending: true,
                });
            }
            for (new, old) in expected.iter_mut().zip(events) {
                if source.as_ref().is_some_and(|source| source.version == 5)
                    && old.kind.name == "polylogue.parse_progress"
                    && old.kind.version == 5
                    && new.kind.name == old.kind.name
                    && new.payload == old.payload
                {
                    new.kind = old.kind.clone();
                }
            }
            if &expected != events {
                return Err(Error::Invalid(
                    "named product identity changed immutable facts".into(),
                ));
            }
            return Ok(PreparedWrite {
                identity: identity.into(),
                pending: true,
            });
        }
        let unlocated = export_key.is_some()
            || writes
                .iter()
                .any(|write| write.location.is_none() && write.parents.is_empty());
        if unlocated {
            let observation = format!("{identity}:observation");
            let frozen_key = if queue.get(&observation)?.is_some() {
                let original = queue.export_material_key(&observation)?;
                if original.as_ref() != export_key {
                    return Err(Error::Invalid(
                        "export identity changed its immutable reading key".into(),
                    ));
                }
                original
            } else {
                export_key.cloned()
            };
            let export_key = frozen_key.as_ref();
            let mut writes = if queue.get(&observation)?.is_some() {
                let previous: Vec<ProductWrite> =
                    crate::spool::decode_json(queue.body(&observation)?)?;
                if previous != writes {
                    return Err(Error::Invalid(
                        "named product identity changed immutable facts".into(),
                    ));
                }
                previous
            } else {
                writes.to_vec()
            };
            let body = serde_json::to_vec(&writes)?;
            let observation_source = if queue.get(&observation)?.is_some() {
                let Ingress::Material { material, .. } = queue.decode_frame(&observation)? else {
                    return Err(Error::Invalid("observation has no material".into()));
                };
                if material.source.name != "polylogue"
                    || !matches!(material.source.version, 5 | 6 | crate::SOURCE_VERSION)
                {
                    return Err(Error::Invalid(
                        "observation has an unsupported source".into(),
                    ));
                }
                material.source
            } else {
                crate::source_reference()
            };
            queue.prepare_material(
                Ingress::Material {
                    frame: observation.clone(),
                    material: Box::new(MaterialFrame {
                        source: observation_source,
                        id: crate::material_id(&self.config.producer, &observation),
                        input: InputRef {
                            kind: "polylogue-observation".into(),
                            locator: identity.into(),
                            host: self.host.clone(),
                        },
                        blob: format!("blake3:{}", blake3::hash(&body).to_hex()),
                        length: body.len() as u64,
                        chunks: Vec::new(),
                        parser: if writes.iter().any(|write| {
                            matches!(
                                write.product,
                                Product::RestoredRequest { .. } | Product::RestoredAssertion { .. }
                            )
                        }) {
                            "polylogue-request-restore".into()
                        } else {
                            "polylogue-products".into()
                        },
                        parser_version: 3,
                        part_index: 0,
                        previous_part: None,
                        manifest: Manifest {
                            role: MaterialRole::Records,
                            media_type: Some("application/json".into()),
                            details: export_key
                                .map(|key| {
                                    serde_json::to_value(key).map(|value| {
                                        [("polylogue_export_key".into(), value)]
                                            .into_iter()
                                            .collect()
                                    })
                                })
                                .transpose()?
                                .unwrap_or_default(),
                            ..Default::default()
                        },
                    }),
                },
                body.as_slice(),
            )?;
            for write in &mut writes {
                if write.location.is_none() && write.parents.is_empty() {
                    write.location = Some(SourceLocation {
                        material: crate::material_id(&self.config.producer, &observation)
                            .to_string(),
                        anchor: 0,
                    });
                }
            }
            if queue
                .get(&observation)?
                .is_some_and(|entry| entry.phase.is_acknowledged())
            {
                queue.prepare_group(
                    Ingress::Events {
                        source: Some(crate::source_reference()),
                        coverage: None,
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
            source: Some(crate::source_reference()),
            coverage: None,
            frame: identity.into(),
            events: crate::products::events(writes)?,
            failures: Vec::new(),
        })?;
        Ok(PreparedWrite {
            identity: identity.into(),
            pending: true,
        })
    }

    pub(crate) fn drain_observations(&self) -> Result<()> {
        let mut ready = Vec::new();
        self.visit_matching(
            Lookup::Completion(Completion::Observation),
            |queue, entry| {
                if entry.material().is_some()
                    && entry.phase.is_acknowledged()
                    && entry.capture().is_none()
                    && queue.capture_group(&entry.identity)?.is_none()
                    && let Ingress::Material { material, .. } =
                        queue.decode_frame(&entry.identity)?
                    && matches!(
                        material.parser.as_str(),
                        "polylogue-products" | "polylogue-request-restore"
                    )
                {
                    ready.push(entry.record.identity);
                }
                Ok(())
            },
        )?;
        for observation in ready {
            let writes: Vec<ProductWrite> = {
                let queue = self.queue()?;
                crate::spool::decode_json(queue.body(&observation)?)?
            };
            let identity = observation.strip_suffix(":observation").ok_or_else(|| {
                Error::Invalid("observation identity has no product operation".into())
            })?;
            let export_key = self.queue()?.export_material_key(&observation)?;
            self.prepare_products_bound(identity, &writes, export_key.as_ref())?;
        }
        Ok(())
    }

    pub(crate) fn complete_captures(
        &self,
        identities: &[String],
        writes: &[ProductWrite],
    ) -> Result<PreparedWrite> {
        let mut retained = Vec::new();
        if writes.iter().any(|write| {
            write
                .product
                .reading()
                .is_some_and(|reading| reading.delivery.is_some())
        }) {
            for identity in identities {
                let pinned = {
                    let queue = self.queue()?;
                    let entry = queue
                        .get(identity)?
                        .ok_or_else(|| Error::Invalid("capture is not prepared".into()))?;
                    if entry.unresolved || entry.phase.is_refused() {
                        return Err(Error::Refused {
                            name: identity.clone(),
                            reason: "capture has no accepted semantic custody".into(),
                        });
                    }
                    queue.pin_capture(identity)?
                };
                if let Capture::Input(input) = pinned.decode()? {
                    for reading in input.parsed {
                        if let (Some(key), Some(delivery)) =
                            (&reading.export_key, &reading.delivery)
                        {
                            reading.clone().into_step().validate_export()?;
                            retained.push((key.clone(), *delivery));
                        }
                    }
                }
            }
        }
        let mut group = Vec::new();
        let mut group_key = None;
        for write in writes {
            if let Some(reading) = write
                .product
                .reading()
                .filter(|reading| reading.delivery.is_some())
            {
                let key = retained
                    .iter()
                    .find_map(|(key, delivery)| {
                        (reading.delivery.as_ref() == Some(delivery)
                            && reading.input.id == key.input.id
                            && reading.input.format == key.input.format
                            && reading.chain == key.chain
                            && reading.reading_version == key.reading_version
                            && reading.source_length == Some(key.length))
                        .then_some(key)
                    })
                    .ok_or_else(|| {
                        Error::Invalid(
                            "export product is not retained by its actual capture sidecar".into(),
                        )
                    })?;
                crate::products::require_export(&write.product, key)?;
                if group_key.as_ref().is_some_and(|previous| previous != key) {
                    return Err(Error::Invalid(
                        "capture callback spans different reading keys".into(),
                    ));
                }
                group_key = Some(key.clone());
                group.push(&write.product);
                if matches!(&write.product, Product::ParseProgress { .. }) {
                    polylogue_store::primary::CapturedReading::from_product_group(
                        key,
                        group.iter().copied(),
                    )?;
                    group.clear();
                    group_key = None;
                }
            } else {
                crate::products::require_source(&write.product)?;
                if !group.is_empty() {
                    return Err(Error::Invalid(
                        "capture callback is interrupted before its control".into(),
                    ));
                }
            }
        }
        if !group.is_empty() {
            return Err(Error::Invalid(
                "capture callback has no final control".into(),
            ));
        }
        let queue = self.queue()?;
        if let Some(group) = queue.group_for(identities)? {
            return Ok(PreparedWrite {
                identity: group.record.identity,
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
                source: Some(crate::source_reference()),
                coverage: None,
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
    ) -> Result<polylogue_model::user::UserOutcome> {
        let identity = format!("request:{}", request.request.id);
        let queue = self.queue()?;
        let mut source = crate::source_reference();
        let repeat = if queue.get(&identity)?.is_some() {
            let Ingress::Material { material, .. } = queue.decode_frame(&identity)? else {
                return Err(Error::Invalid("authored intent has no material".into()));
            };
            if material.source.name != "polylogue"
                || !matches!(material.source.version, 5 | 6 | crate::SOURCE_VERSION)
            {
                return Err(Error::Invalid(
                    "authored intent belongs to an unsupported adapter source".into(),
                ));
            }
            source = material.source.clone();
            let original: PreparedRequest = crate::spool::decode_json(queue.body(&identity)?)?;
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
                    source: source.clone(),
                    id: crate::material_id(&self.config.producer, &identity),
                    input: InputRef {
                        kind: "polylogue".into(),
                        locator: "requests".into(),
                        host: self.host.clone(),
                    },
                    blob: format!("blake3:{}", blake3::hash(&bytes).to_hex()),
                    length: bytes.len() as u64,
                    chunks: Vec::new(),
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

    /// Named custody remains visible through publication until its confirmed
    /// barrier is applied. Only matching immutable product bodies are hydrated.
    pub(crate) fn visit_named_products(
        &self,
        selector: &polylogue_store::primary::NamedProductSelector,
        visit: &mut dyn FnMut(&Product) -> polylogue_store::Result<()>,
    ) -> Result<()> {
        use polylogue_store::primary::NamedProductSelector;
        let queue = self.queue()?;
        let mut names = std::collections::BTreeSet::new();
        match selector {
            NamedProductSelector::Exact(name) => {
                names.insert(name.clone());
            }
            NamedProductSelector::Prefix(prefix) => {
                queue.visit_names(prefix, &mut |identity| {
                    let entry = queue.get(identity)?.ok_or_else(|| {
                        Error::Invalid("named operation disappeared while locked".into())
                    })?;
                    let name = entry.logical_name().ok_or_else(|| {
                        Error::Invalid("named index points to an unnamed operation".into())
                    })?;
                    names.insert(name.to_owned());
                    Ok(())
                })?;
            }
        }
        for name in names {
            let observation = format!("{name}:observation");
            if let Some(entry) = queue.get(&observation)? {
                if !entry.is_named() {
                    return Err(Error::Invalid(
                        "named observation has another operation type".into(),
                    ));
                }
                let writes: Vec<ProductWrite> =
                    crate::spool::decode_json(queue.body(&observation)?)?;
                for write in writes {
                    visit(&write.product).map_err(Error::Store)?;
                }
            } else if let Some(entry) = queue.get(&name)? {
                if !entry.is_named() {
                    return Err(Error::Invalid(
                        "named product has another operation type".into(),
                    ));
                }
                let Ingress::Events { events, .. } = queue.decode_frame(&name)? else {
                    return Err(Error::Invalid("named product has no event facts".into()));
                };
                for event in events {
                    let product = crate::products::decode(&event.kind, event.payload)?;
                    visit(&product).map_err(Error::Store)?;
                }
            }
        }
        Ok(())
    }

    pub(crate) fn visit_pending_requests(
        &self,
        visit: &mut dyn FnMut(&PendingAuthored) -> polylogue_store::Result<()>,
    ) -> polylogue_store::Result<()> {
        Ok(self.visit_matching(Lookup::Requests, |queue, entry| {
            requests::visit(queue, &entry, visit)?;
            Ok(())
        })?)
    }

    pub(crate) fn visit_pending_blobs(
        &self,
        visit: &mut dyn FnMut(&polylogue_store::NewBlob) -> polylogue_store::Result<()>,
    ) -> polylogue_store::Result<()> {
        Ok(self.visit_matching(Lookup::Blobs(None), |_, entry| {
            if let Some(Capture::Blob { hash, length }) = entry.capture().cloned() {
                visit(&polylogue_store::NewBlob::whole(hash, length))
                    .map_err(|error| Error::Invalid(error.to_string()))?;
            }
            Ok(())
        })?)
    }

    pub(crate) fn drain_requests(&self) -> Result<()> {
        let mut ready = Vec::new();
        self.visit_matching(Lookup::Completion(Completion::Request), |queue, entry| {
            if entry.is_request()
                && entry.identity.starts_with("request:")
                && matches!(
                    entry.phase,
                    crate::spool::Phase::Acknowledged { .. } | crate::spool::Phase::Refused
                )
                && queue.capture_group(&entry.identity)?.is_none()
            {
                ready.push(entry.record.identity);
            }
            Ok(())
        })?;
        let mut refusal = None;
        for identity in ready {
            match self.complete_request(&identity) {
                Ok(()) => {}
                Err(error @ Error::Refused { .. }) => {
                    if refusal.is_none() {
                        refusal = Some(error);
                    }
                }
                Err(error) => return Err(error),
            }
        }
        refusal.map_or(Ok(()), Err)
    }

    fn complete_request(&self, identity: &str) -> Result<()> {
        let request: PreparedRequest = {
            let queue = self.queue()?;
            let Ingress::Material { material, .. } = queue.decode_frame(identity)? else {
                return Err(Error::Invalid("authored intent has no material".into()));
            };
            if material.source.name != "polylogue"
                || !matches!(material.source.version, 5 | 6 | crate::SOURCE_VERSION)
            {
                return Err(Error::Invalid(
                    "authored intent belongs to an unsupported adapter source".into(),
                ));
            }
            crate::spool::decode_json(queue.body(identity)?)?
        };
        let mut dependencies = vec![identity.to_owned()];
        let mut writes = Vec::new();
        let mut waiting = false;
        for attachment in request
            .effect
            .chats
            .iter()
            .flat_map(|change| &change.action.attachments)
        {
            self.visit_matching(Lookup::Blobs(Some(&attachment.blob.to_string())), |queue,entry|{
                    if matches!(entry.capture(),Some(Capture::Blob{hash,..}) if *hash==attachment.blob) {
                        if let Some(material)=Self::capture_material(queue,&entry)? {
                            writes.push(ProductWrite{location:Some(SourceLocation{material:material.id.clone(),anchor:0}),product:Product::Blob(material),parents:Vec::new(),machine:Some(request.request.machine.clone()),output_index:writes.len() as u64});
                            if queue.capture_group(&entry.identity)?.is_none() {dependencies.push(entry.record.identity);}
                        } else {waiting=true;}
                    }
                    Ok(())
                })?;
        }
        if waiting {
            return Ok(());
        }
        dependencies.sort();
        dependencies.dedup();
        writes.push(ProductWrite {
            product: Product::Request(request.clone()),
            location: Some(SourceLocation {
                material: crate::material_id(&self.config.producer, identity).to_string(),
                anchor: 0,
            }),
            parents: Vec::new(),
            machine: Some(request.request.machine),
            output_index: writes.len() as u64,
        });
        self.complete_captures(&dependencies, &writes)?;
        Ok(())
    }
}
