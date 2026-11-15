//! A parser upgrade replaces Sinex's copy of one chain's parsed facts with a
//! producer replay of the chain's original materials.
use super::*;

impl Sinex {
    /// Prepares the replay frame, or answers `None` while an earlier frame of
    /// the same chain is unacknowledged: the replay names the live set Sinex
    /// holds for the chain, which is known only once that frame is admitted.
    pub(crate) fn prepare_reading(
        &self,
        root: &str,
        key: &polylogue_store::primary::ExportReadingKey,
        visit: &mut polylogue_store::primary::ReadingProducts<'_>,
    ) -> polylogue_store::Result<Option<PreparedWrite>> {
        Ok((|| -> Result<_> {
            key.validate()?;
            let prefix = polylogue_store::primary::queue::parsed_chain_prefix(
                root,
                &key.input.id,
                key.chain,
            )?;
            if let Some(entry) = self
                .queue()?
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
                return Ok(None);
            }
            let root_id = MaterialId(
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
                materials: vec![root_id],
                event_types: event_types.clone(),
                from: None,
                until: None,
                after: None,
            };
            let (selection, declarations, source) = self.client_operation(async |client| {
                let mut selections = client.replay_scope(&query).await?;
                if selections.len() != 1 || selections[0].root != root_id {
                    return Err(Error::Invalid(
                        "replacement scope does not identify one complete raw chain".into(),
                    ));
                }
                let selection = selections.remove(0);
                let source = crate::source_reference();
                if selection.materials.is_empty() {
                    return Err(Error::Invalid(
                        "reading has no material source evidence".into(),
                    ));
                }
                for material in &selection.materials {
                    let original = client.fetch_material(material.id).await?.source;
                    if original.name != source.name {
                        return Err(Error::Invalid(
                            "an interpretation cannot relabel its original material source".into(),
                        ));
                    }
                }
                let declarations = crate::kinds::declarations(client.family())
                    .iter()
                    .map(|declaration| sinex_protocol::DeclarationHash {
                        kind: declaration.kind.clone(),
                        hash: declaration.definition_hash(),
                    })
                    .collect();
                Ok((selection, declarations, source))
            })?;

            let mut events = Vec::new();
            let mut group = Vec::new();
            let mut interpreted = false;
            let mut original = blake3::Hasher::new();
            visit(&mut |write| {
                crate::products::require_reading(&write.product, key)?;
                if write.location.is_none() {
                    return Err(polylogue_store::StoreError::Invalid(
                        "a parsed fact needs its original source anchor".into(),
                    ));
                }
                serde_json::to_writer(
                    &mut original,
                    &(
                        &write.product,
                        &write.location,
                        &write.parents,
                        &write.machine,
                        write.output_index,
                    ),
                )?;
                original.update(b"\n");
                let complete = matches!(&write.product, Product::ParseProgress { .. });
                interpreted = true;
                group.push(write);
                if complete {
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
            if !interpreted {
                return Err(Error::Invalid("replacement has no interpretation".into()));
            }
            let desired = format!(
                "{prefix}{}:",
                blake3::hash(&serde_json::to_vec(&(
                    key,
                    original.finalize().to_hex().as_str(),
                    crate::source_reference()
                ))?)
                .to_hex()
            );
            let queue = self.queue()?;
            if let Some(entry) = queue
                .lookup(
                    Lookup::IdentityPrefix(&desired),
                    None,
                    None,
                    Order::NewestFirst,
                    1,
                )?
                .into_iter()
                .next()
            {
                return Ok(Some(PreparedWrite {
                    identity: entry.record.identity,
                    pending: entry.phase.is_pending(),
                }));
            }
            let identity = format!("{desired}{}", uuid::Uuid::now_v7());
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
            queue.prepare(Ingress::ReplaySequence {
                frame: identity.clone(),
                sequence: Box::new(sequence),
            })?;
            drop(queue);
            self.send_pending();
            Ok(Some(PreparedWrite {
                identity,
                pending: true,
            }))
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
