//! A completed replacement is prepared as streamed domain facts, then admitted
//! through Sinex's one atomic replay operation after its scope is confirmed.
use super::*;
use std::io::{BufRead, Seek, Write};

#[derive(serde::Serialize, serde::Deserialize)]
struct ReadingReplacement {
    root: String,
    reading_version: String,
}

impl Sinex {
    pub(crate) fn prepare_reading(
        &self,
        root: &str,
        version: &str,
        visit: &mut dyn FnMut(
            &mut dyn FnMut(ProductWrite) -> polylogue_store::Result<()>,
        ) -> polylogue_store::Result<()>,
    ) -> polylogue_store::Result<PreparedWrite> {
        Ok((|| -> Result<_> {
            let mut body = tempfile::tempfile()?;
            {
                let mut out = std::io::BufWriter::new(&mut body);
                visit(&mut |product| {
                    serde_json::to_writer(&mut out, &product)
                        .map_err(|error| polylogue_store::StoreError::Invalid(error.to_string()))?;
                    out.write_all(b"\n")?;
                    Ok(())
                })?;
                out.flush()?;
            }
            body.rewind()?;
            let mut digest = blake3::Hasher::new();
            let length = std::io::copy(&mut body, &mut digest)?;
            let hash = digest.finalize();
            body.rewind()?;
            let scope = blake3::hash(root.as_bytes()).to_hex().to_string();
            let content = blake3::hash(&serde_json::to_vec(&(version, hash.to_hex().as_str()))?)
                .to_hex()
                .to_string();
            let prefix = format!("reading:{scope}:");
            let desired = format!("{prefix}{content}:");
            let mut latest = None;
            self.visit_prepared(|_, entry| {
                if entry.identity.starts_with(&prefix) {
                    latest = Some(entry);
                }
                Ok(())
            })?;
            if let Some(entry) = latest.filter(|entry| entry.identity.starts_with(&desired)) {
                return Ok(PreparedWrite {
                    identity: entry.identity,
                    pending: true,
                });
            }
            let identity = format!("{desired}{}", uuid::Uuid::now_v7());
            let replacement = ReadingReplacement {
                root: root.into(),
                reading_version: version.into(),
            };
            self.queue()?.prepare_material(
                Ingress::Material {
                    frame: identity.clone(),
                    material: Box::new(MaterialFrame {
                        id: crate::material_id(&identity),
                        input: InputRef {
                            kind: "polylogue-reading".into(),
                            locator: root.into(),
                            host: self.host.clone(),
                        },
                        blob: format!("blake3:{}", hash.to_hex()),
                        length,
                        parser: "polylogue-reading".into(),
                        parser_version: 3,
                        part_index: 0,
                        previous_part: None,
                        manifest: Manifest {
                            role: MaterialRole::Records,
                            media_type: Some("application/x-ndjson".into()),
                            details: [(
                                "polylogue_replacement".into(),
                                serde_json::to_value(replacement)?,
                            )]
                            .into_iter()
                            .collect(),
                            ..Default::default()
                        },
                    }),
                },
                body,
            )?;
            Ok(PreparedWrite {
                identity,
                pending: true,
            })
        })()?)
    }

    pub(super) fn drain_readings(&self) -> Result<()> {
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
                    if let Some(replacement) =
                        material.manifest.details.get("polylogue_replacement")
                    {
                        ready.push((
                            entry.identity,
                            serde_json::from_value::<ReadingReplacement>(replacement.clone())?,
                        ));
                    }
                }
            }
            Ok(())
        })?;
        for (identity, replacement) in ready {
            let frame = format!("{identity}:replay");
            if self.queue()?.get(&frame)?.is_some() {
                continue;
            }
            let events = {
                let queue = self.queue()?;
                let mut events = Vec::new();
                let mut line = String::new();
                let mut body = std::io::BufReader::new(queue.body(&identity)?);
                while body.read_line(&mut line)? > 0 {
                    let write: ProductWrite = serde_json::from_str(&line)?;
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
            let (selection, declarations) = self.with(async |client| {
                let mut selections = client.replay_scope(&query).await?;
                if selections.len() != 1 || selections[0].root != root {
                    return Err(Error::Invalid(
                        "replacement scope does not identify one complete raw chain".into(),
                    ));
                }
                let selection = selections.remove(0);
                for material in &selection.materials {
                    client.fetch_material(material.id).await?;
                }
                let declarations = crate::kinds::declarations(client.family())
                    .iter()
                    .map(|declaration| sinex_protocol::DeclarationHash {
                        kind: declaration.kind.clone(),
                        hash: declaration.definition_hash(),
                    })
                    .collect();
                Ok((selection, declarations))
            })?;
            let operation_hash = blake3::hash(identity.as_bytes());
            let mut operation = [0u8; 16];
            operation.copy_from_slice(&operation_hash.as_bytes()[..16]);
            let sequence = sinex_protocol::ProducerReplay {
                operation: sinex_model::OperationId(uuid::Uuid::from_bytes(operation)),
                root: selection.root,
                materials: selection.materials,
                event_types,
                from: None,
                until: None,
                declarations,
                parser_build: replacement.reading_version.clone(),
                parser_version: replacement.reading_version,
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
        }
        Ok(())
    }
}
