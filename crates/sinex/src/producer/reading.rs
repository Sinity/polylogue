//! Native replacement facts are admitted against the original material scope.
//! Local reconstruction reads those originals through the ordinary parser.
use super::*;

impl Sinex {
    pub(crate) fn prepare_reading(
        &self,
        root: &str,
        key: &polylogue_store::primary::ExportReadingKey,
        visit: &mut polylogue_store::primary::ReadingProducts<'_>,
    ) -> polylogue_store::Result<polylogue_store::primary::ParsedPublication> {
        use polylogue_store::primary::{ParsedPublication, ParsedSource};
        Ok((|| -> Result<_> {
            key.validate()?;
            let mut events = Vec::new();
            let mut group = Vec::new();
            let mut reading = None;
            visit(&mut |write| {
                crate::products::require_reading(&write.product, key)?;
                if write.location.is_none() {
                    return Err(polylogue_store::StoreError::Invalid(
                        "a native reading needs its original source anchor".into(),
                    ));
                }
                let complete = matches!(&write.product, Product::ParseProgress { .. });
                reading = write.product.reading().cloned();
                group.push(write);
                if complete {
                    if group.iter().any(|write| {
                        write
                            .product
                            .reading()
                            .is_some_and(|reading| reading.delivery.is_some())
                    }) {
                        let products = group
                            .iter()
                            .map(|write| write.product.clone())
                            .collect::<Vec<_>>();
                        polylogue_store::primary::CapturedReading::from_product_group(
                            key, &products,
                        )?;
                    }
                    events.extend(
                        crate::products::events(&group)
                            .map_err(polylogue_store::StoreError::from)?,
                    );
                    group.clear();
                }
                Ok(())
            })?;
            if !group.is_empty() {
                return Err(Error::Invalid(
                    "reading ends before its parser progress".into(),
                ));
            }
            let mut reading = reading
                .ok_or_else(|| Error::Invalid("replacement has no interpretation".into()))?;
            reading.start = 0;
            let parsed = ParsedSource {
                root_material: root.into(),
                reading,
            };
            let scope = (root, &key.input.id, key.chain);
            let prefix = format!(
                "parsed:{}:",
                blake3::hash(&serde_json::to_vec(&scope)?).to_hex()
            );
            let desired = format!(
                "{prefix}{}:",
                blake3::hash(&serde_json::to_vec(&(
                    &parsed,
                    &events,
                    crate::source_reference()
                ))?)
                .to_hex()
            );
            {
                let queue = self.queue()?;
                if let Some(entry) = queue
                    .lookup(
                        Lookup::IdentityPrefix(&prefix),
                        None,
                        None,
                        Order::NewestFirst,
                        1,
                    )?
                    .into_iter()
                    .next()
                {
                    if entry.identity.starts_with(&desired) {
                        return queue.publication(&entry.identity, &self.config.producer);
                    }
                    if matches!(entry.phase, polylogue_store::primary::queue::Phase::Pending) {
                        return Ok(ParsedPublication::Pending {
                            identity: entry.identity,
                        });
                    }
                }
            }
            let identity = format!("{desired}{}", uuid::Uuid::now_v7());
            let root = MaterialId(
                root.parse()
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
                                | "polylogue.prefix_copy"
                                | "polylogue.prefix_member"
                                | "polylogue.background"
                                | "polylogue.repository"
                                | "polylogue.parse_progress"
                                | "polylogue.tool_call"
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
                let source = source.ok_or_else(|| {
                    Error::Invalid("reading has no material source evidence".into())
                })?;
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
                parser_version: key.reading_version.clone(),
                expected_live_hash: selection.expected_live_hash,
                batches: vec![sinex_protocol::ReplayBatch {
                    events,
                    failures: Vec::new(),
                }],
            };
            let queue = self.queue()?;
            if let Some(entry) = queue
                .lookup(
                    Lookup::IdentityPrefix(&prefix),
                    None,
                    None,
                    Order::NewestFirst,
                    1,
                )?
                .into_iter()
                .next()
                && matches!(entry.phase, polylogue_store::primary::queue::Phase::Pending)
            {
                return Ok(ParsedPublication::Pending {
                    identity: entry.identity,
                });
            }
            queue.prepare_parsed(
                Ingress::ReplaySequence {
                    frame: identity.clone(),
                    sequence: Box::new(sequence),
                },
                &parsed,
            )?;
            drop(queue);
            self.send_pending();
            Ok(ParsedPublication::Pending { identity })
        })()?)
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
