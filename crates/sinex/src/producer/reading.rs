//! A completed replacement is prepared as streamed domain facts, then admitted
//! through Sinex's one atomic replay operation after its scope is confirmed.
use super::*;
use std::io::{BufRead, Write};

#[derive(serde::Serialize, serde::Deserialize)]
struct ReadingReplacement {
    root: String,
    key: polylogue_store::primary::ExportReadingKey,
}

impl Sinex {
    pub(crate) fn prepare_reading(
        &self,
        root: &str,
        key: &polylogue_store::primary::ExportReadingKey,
        visit: &mut polylogue_store::primary::ReadingProducts<'_>,
    ) -> polylogue_store::Result<PreparedWrite> {
        Ok((|| -> Result<_> {
            key.validate()?;
            let scratch = self.config.store.join("polylogue-staging");
            let preparation = hashed_blobs::Preparation::new(&scratch)?;
            let mut body = preparation.plain_writer(tempfile::tempfile_in(scratch)?, 3)?;
            {
                let mut out = std::io::BufWriter::new(&mut body);
                let mut group = Vec::new();
                visit(&mut |product| {
                    crate::products::require_reading(&product.product, key)?;
                    let complete = matches!(&product.product, Product::ParseProgress { .. });
                    if product
                        .product
                        .reading()
                        .is_some_and(|reading| reading.delivery.is_some())
                    {
                        group.push(product.product.clone());
                    }
                    if complete {
                        if group.iter().any(|product| {
                            product
                                .reading()
                                .is_some_and(|reading| reading.delivery.is_some())
                        }) {
                            polylogue_store::primary::CapturedReading::from_product_group(
                                key, &group,
                            )?;
                        }
                        group.clear();
                    }
                    serde_json::to_writer(&mut out, &product)
                        .map_err(|error| polylogue_store::StoreError::Invalid(error.to_string()))?;
                    out.write_all(b"\n")?;
                    Ok(())
                })?;
                if !group.is_empty() {
                    return Err(Error::Invalid(
                        "reading ends before its callback control".into(),
                    ));
                }
                out.flush()?;
            }
            let body = body.finish()?;
            let length = body.decoded_len();
            let hash = body.hash();
            let scope = blake3::hash(root.as_bytes()).to_hex().to_string();
            let content = blake3::hash(&serde_json::to_vec(&(key.identity(), hash.to_string()))?)
                .to_hex()
                .to_string();
            let prefix = format!("reading:{scope}:");
            let desired = format!("{prefix}{content}:");
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
            let replacement = ReadingReplacement {
                root: root.into(),
                key: key.clone(),
            };
            self.queue()?.prepare_encoded_material(
                Ingress::Material {
                    frame: identity.clone(),
                    material: Box::new(MaterialFrame {
                        source: crate::source_reference(),
                        id: crate::material_id(&self.config.producer, &identity),
                        input: InputRef {
                            kind: "polylogue-reading".into(),
                            locator: root.into(),
                            host: self.host.clone(),
                        },
                        blob: hash.to_string(),
                        length,
                        chunks: Vec::new(),
                        parser: "polylogue-reading".into(),
                        parser_version: 3,
                        part_index: 0,
                        previous_part: None,
                        manifest: Manifest {
                            role: MaterialRole::Records,
                            media_type: Some("application/x-ndjson".into()),
                            details: [
                                (
                                    "polylogue_replacement".into(),
                                    serde_json::to_value(replacement)?,
                                ),
                                ("polylogue_export_key".into(), serde_json::to_value(key)?),
                            ]
                            .into_iter()
                            .collect(),
                            ..Default::default()
                        },
                    }),
                },
                PreparedBody {
                    descriptor: body.descriptor(),
                    file: body.into_file(),
                    path: None,
                },
            )?;
            Ok(PreparedWrite {
                identity,
                pending: true,
            })
        })()?)
    }

    pub(super) fn drain_readings(&self) -> Result<()> {
        let mut ready = Vec::new();
        self.visit_matching(Lookup::Completion(Completion::Reading), |queue, entry| {
            if entry.material().is_some()
                && entry.phase.is_acknowledged()
                && entry.capture().is_none()
                && !entry.is_request()
                && queue.capture_group(&entry.identity)?.is_none()
                && let Ingress::Material { material, .. } = queue.decode_frame(&entry.identity)?
                && let Some(replacement) = material.manifest.details.get("polylogue_replacement")
            {
                ready.push((
                    entry.record.identity,
                    serde_json::from_value::<ReadingReplacement>(replacement.clone())?,
                ));
            }
            Ok(())
        })?;
        let mut refusal = None;
        for (identity, replacement) in ready {
            match self.complete_reading(identity, replacement) {
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

    fn complete_reading(&self, identity: String, replacement: ReadingReplacement) -> Result<()> {
        let frame = format!("{identity}:replay");
        if self.queue()?.get(&frame)?.is_some() {
            return Ok(());
        }
        let events = {
            let queue = self.queue()?;
            let mut events = Vec::new();
            let mut line = String::new();
            let mut body = std::io::BufReader::new(queue.body(&identity)?);
            while body.read_line(&mut line)? > 0 {
                let write: ProductWrite = crate::spool::decode_json(line.as_bytes())?;
                crate::products::require_reading(&write.product, &replacement.key)?;
                events.extend(crate::products::events(&[write])?);
                line.clear();
            }
            events
        };
        let root = MaterialId(
            replacement
                .root
                .parse()
                .map_err(|error| Error::Invalid(format!("reading root: {error}")))?,
        );
        let event_types = Some(
            crate::kinds::declarations("polylogue")
                .into_iter()
                .map(|declaration| declaration.kind.name)
                .filter(|name| {
                    matches!(
                        name.as_str(),
                        "polylogue.session"
                            | "polylogue.message"
                            | "polylogue.document"
                            | "polylogue.fragment"
                            | "polylogue.event"
                            | "polylogue.link"
                            | "polylogue.background"
                            | "polylogue.repository"
                            | "polylogue.parse_progress"
                    )
                })
                .collect(),
        );
        let query = sinex_protocol::ReplayScopeQuery {
            materials: vec![root],
            event_types: event_types.clone(),
            from: None,
            until: None,
            after: None,
        };
        let (selection, declarations, source) = self.client_operation(async |client| {
            let mut selections = client.replay_scope(&query).await?;
            if selections.len() != 1 || selections[0].root != root {
                return Err(Error::Invalid(
                    "replacement scope does not identify one complete raw chain".into(),
                ));
            }
            let selection = selections.remove(0);
            let mut source = None;
            for material in &selection.materials {
                let actual = client.fetch_material(material.id).await?.source;
                if source.as_ref().is_some_and(|source| source != &actual) {
                    return Err(Error::Invalid(
                        "reading materials span different registered sources".into(),
                    ));
                }
                source = Some(actual);
            }
            let source = source
                .ok_or_else(|| Error::Invalid("reading has no material source evidence".into()))?;
            let declarations = crate::kinds::declarations(client.family())
                .iter()
                .map(|declaration| sinex_protocol::DeclarationHash {
                    kind: declaration.kind.clone(),
                    hash: declaration.definition_hash(),
                })
                .collect();
            Ok((selection, declarations, source))
        })?;
        let operation_hash = blake3::hash(identity.as_bytes());
        let mut operation = [0u8; 16];
        operation.copy_from_slice(&operation_hash.as_bytes()[..16]);
        let sequence = sinex_protocol::ProducerReplay {
            source,
            operation: sinex_model::OperationId(uuid::Uuid::from_bytes(operation)),
            root: selection.root,
            materials: selection.materials,
            event_types,
            from: None,
            until: None,
            declarations,
            parser_build: build_id()?,
            parser_version: replacement.key.reading_version,
            expected_live_hash: selection.expected_live_hash,
            batches: vec![sinex_protocol::ReplayBatch {
                events,
                failures: Vec::new(),
            }],
        };
        self.queue()?.prepare_group(
            Ingress::ReplaySequence {
                frame,
                sequence: Box::new(sequence),
            },
            &[identity],
        )?;
        Ok(())
    }
}

/// The running binary's hash, which names the parser build of a replay.
fn build_id() -> Result<String> {
    static BUILD: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    if let Some(build) = BUILD.get() {
        return Ok(build.clone());
    }
    let mut file = std::fs::File::open("/proc/self/exe")?;
    let mut hash = blake3::Hasher::new();
    std::io::copy(&mut file, &mut hash)?;
    let build = hash.finalize().to_hex().to_string();
    let _ = BUILD.set(build.clone());
    Ok(build)
}
