use super::*;
use crate::encoding::unpack_block;
use polylogue_model::{Attachment, BlobHash, Block, MessageId};
use std::collections::BTreeSet;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum AttachmentState {
    Available,
    Ambiguous,
    Unavailable,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct AssetMatch {
    pub input: String,
    pub chain: i64,
    pub blob: Option<BlobHash>,
    pub source_version: String,
    pub size: u64,
    pub members: Vec<String>,
    pub names: Vec<String>,
    pub media_types: Vec<String>,
    pub declared_sizes: Vec<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct AttachmentResolution {
    pub message: MessageId,
    pub block: usize,
    pub attachment_index: usize,
    pub source: Attachment,
    pub state: AttachmentState,
    pub matches: Vec<AssetMatch>,
    pub binding: String,
}

impl Reader<'_> {
    /// Resolve a recorded attachment through only current sources of the shown
    /// message version. Historical observations cannot authorize removed bytes.
    pub fn attachment(
        &self,
        message: &MessageId,
        block: usize,
        attachment_index: usize,
    ) -> Result<AttachmentResolution> {
        let conn = self.conn()?;
        let (version,body,source_input,source_chain):(Vec<u8>,Vec<u8>,String,i64)=conn.query_row(
            "SELECT ms.version_hash,b.body,i.id,ms.chain FROM message m JOIN message_source ms ON ms.message_pk=m.pk AND ms.input_pk=m.input_pk AND ms.chain=m.source_chain JOIN input_ref i ON i.pk=ms.input_pk JOIN block b ON b.message_pk=m.pk WHERE m.id=?1 AND coalesce(b.ordinal,CAST(b.idx AS TEXT))=?2",
            params![message.to_string(),block.to_string()],
            |r|Ok((r.get(0)?,r.get(1)?,r.get(2)?,r.get(3)?))).optional()?
            .ok_or_else(||StoreError::NotFound(format!("attachment block {message}:{block}")))?;
        let source = match unpack_block(&body)? {
            Block::Image(a) | Block::File(a) if attachment_index == 0 => a,
            Block::ToolResult { attachments, .. } => attachments
                .into_iter()
                .nth(attachment_index)
                .ok_or_else(|| StoreError::NotFound("tool result attachment index".into()))?,
            _ => {
                return Err(StoreError::Invalid(
                    "selected block has no such attachment".into(),
                ));
            }
        };
        let bound =
            self.attachment_binding_contexts(message, block as u64, attachment_index as u64)?;
        if bound.namespace_count != 0 {
            let mut matches = Vec::new();
            for context in &bound.contexts {
                if source.blob.is_some()
                    || source.source_id.as_deref() != Some(context.claim.source_id.as_str())
                {
                    return Err(StoreError::Invalid(
                        "binding claim differs from the selected attachment".into(),
                    ));
                }
                for member in &context.members {
                    matches.push(super::bindings::asset_match(conn, member)?);
                }
            }
            let binding = blake3::hash(&serde_json::to_vec(&(
                message,
                block,
                attachment_index,
                &version,
                &source,
                &bound,
            ))?)
            .to_hex()
            .to_string();
            return Ok(AttachmentResolution {
                message: message.clone(),
                block,
                attachment_index,
                source,
                state: bound.state,
                matches,
                binding,
            });
        }
        let scopes = import_scopes(conn, &source_input, source_chain)?;
        let mut matches = std::collections::BTreeMap::<String, AssetMatch>::new();
        let mut known = std::collections::HashMap::<String, Option<AssetSource>>::new();
        let identity = source.source_id.as_deref().filter(|id| !id.is_empty());
        if let Some(source_id) = identity {
            for scope in &scopes {
                scope_matches(conn, scope, source_id, &mut known, &mut matches)?;
            }
        }
        let mut matches = matches.into_values().collect::<Vec<_>>();
        for asset in &mut matches {
            unique(&mut asset.members);
            unique(&mut asset.names);
            unique(&mut asset.media_types);
            unique(&mut asset.declared_sizes);
        }
        let state = match matches.len() {
            1 => AttachmentState::Available,
            0 if scopes.is_empty() || identity.is_none() => AttachmentState::Unknown,
            0 => AttachmentState::Unavailable,
            _ => AttachmentState::Ambiguous,
        };
        let binding = blake3::hash(&serde_json::to_vec(&(
            message,
            block,
            attachment_index,
            &version,
            &source,
            &scopes,
            &matches,
        ))?)
        .to_hex()
        .to_string();
        Ok(AttachmentResolution {
            message: message.clone(),
            block,
            attachment_index,
            source,
            state,
            matches,
            binding,
        })
    }
}

/// The import namespaces under which a message's source input was read.
pub(super) fn import_scopes(conn: &Connection, input: &str, chain: i64) -> Result<Vec<String>> {
    let mut scopes = conn.prepare(
        "SELECT DISTINCT own.import_id FROM import_member own
        JOIN source_observation o ON o.id=own.observation AND o.input=own.input
        LEFT JOIN source_revision r ON r.input=o.input AND r.number=o.revision
        WHERE own.input=?1 AND coalesce(o.chain,r.chain)=?2
          AND json_extract(o.details_json,'$.import.import_id')=own.import_id
        ORDER BY own.import_id",
    )?;
    let scopes = scopes
        .query_map(params![input, chain], |r| r.get::<_, String>(0))?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    Ok(scopes)
}

/// Members of one import namespace that carry `source_id`, grouped by the
/// input holding their bytes, with the names the namespace's sidecars give.
pub(super) fn scope_matches(
    conn: &Connection,
    scope: &str,
    source_id: &str,
    known: &mut std::collections::HashMap<String, Option<AssetSource>>,
    matches: &mut std::collections::BTreeMap<String, AssetMatch>,
) -> Result<()> {
    let mut stmt = conn.prepare(
        "SELECT DISTINCT m.input,m.member,m.observation FROM import_member m
        WHERE m.import_id=?1 AND m.source_id=?2
          AND EXISTS(SELECT 1 FROM source_revision r WHERE r.input=m.input)
        ORDER BY m.input,m.member,m.observation",
    )?;
    for row in stmt.query_map(params![scope, source_id], |r| {
        Ok((
            r.get::<_, String>(0)?,
            r.get::<_, String>(1)?,
            r.get::<_, String>(2)?,
        ))
    })? {
        let (input, member, observation) = row?;
        if !known.contains_key(&input) {
            known.insert(input.clone(), asset_source(conn, conn, "source_", &input)?);
        }
        let Some(mut current) = known[&input].clone() else {
            continue;
        };
        if current.length == 0 {
            let Some(bound) = super::empty::source(
                conn,
                conn,
                "source_",
                &input,
                current.chain,
                Some(&observation),
                None,
            )?
            else {
                continue;
            };
            current = bound;
        }
        let asset = matches.entry(input.clone()).or_insert_with(|| AssetMatch {
            input,
            chain: current.chain,
            blob: current.blob,
            source_version: current.version.clone(),
            size: current.length,
            members: Vec::new(),
            names: Vec::new(),
            media_types: Vec::new(),
            declared_sizes: Vec::new(),
        });
        asset.members.push(member);
        let mut meta=conn.prepare_cached("SELECT x.name,x.media_type,x.size,m.input,s.source_version FROM import_member m
            JOIN import_asset_metadata x ON x.input=m.input AND x.sidecar=m.sidecar
            JOIN import_asset_sidecar s ON s.input=x.input AND s.sidecar=x.sidecar AND s.error IS NULL
            WHERE m.import_id=?1 AND x.source_id=?2 AND m.sidecar IS NOT NULL
              AND EXISTS(SELECT 1 FROM source_revision r WHERE r.input=m.input)
            ORDER BY m.input,x.ordinal")?;
        for row in meta.query_map(params![scope, source_id], |r| {
            Ok((
                r.get::<_, Option<String>>(0)?,
                r.get::<_, Option<String>>(1)?,
                r.get::<_, Option<String>>(2)?,
                r.get::<_, String>(3)?,
                r.get::<_, String>(4)?,
            ))
        })? {
            let (name, media, size, input, recorded) = row?;
            if !known.contains_key(&input) {
                known.insert(input.clone(), asset_source(conn, conn, "source_", &input)?);
            }
            if known[&input].as_ref().map(|source| source.version.as_str())
                != Some(recorded.as_str())
            {
                continue;
            }
            asset.names.extend(name);
            asset.media_types.extend(media);
            if let Some(size) = size {
                asset.declared_sizes.push(
                    size.parse()
                        .map_err(|_| StoreError::Invalid("invalid declared asset size".into()))?,
                );
            }
        }
    }
    Ok(())
}

impl Reader<'_> {
    /// Claims describing the stored bytes behind a message's attachments, for
    /// slots no retained binding covers. Each import namespace that read the
    /// message's shown source makes one claim per slot: its candidates are the
    /// complete bytes of the members carrying the slot's source ID, and none
    /// when no such member is stored.
    pub fn current_attachment_claims(
        &self,
        message: &MessageId,
        source_version: BlobHash,
        slots: &[polylogue_model::attachment_binding::AttachmentSlot],
    ) -> Result<Vec<polylogue_model::attachment_binding::AttachmentBinding>> {
        use polylogue_model::attachment_binding::{
            AttachmentBinding, AttachmentCandidate, AttachmentNamespace, AttachmentSourceClaim,
        };
        if slots.is_empty() {
            return Ok(Vec::new());
        }
        let conn = self.conn()?;
        let mut shown = conn.prepare(
            "SELECT i.id,s.chain FROM message m
             JOIN message_source s ON s.message_pk=m.pk
             JOIN input_ref i ON i.pk=s.input_pk
             WHERE m.id=?1 AND s.version_hash=?2 AND s.input_pk=m.input_pk AND s.chain=m.source_chain ORDER BY i.id,s.chain",
        )?;
        let shown = shown
            .query_map(
                params![message.to_string(), source_version.0.as_slice()],
                |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?)),
            )?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        let mut claims = Vec::new();
        for (input, chain) in shown {
            for scope in import_scopes(conn, &input, chain)? {
                let namespace = AttachmentNamespace {
                    input: input.clone(),
                    chain,
                    import_id: scope.clone(),
                };
                for slot in slots {
                    let mut known = std::collections::HashMap::new();
                    let mut matches = std::collections::BTreeMap::new();
                    scope_matches(conn, &scope, &slot.source_id, &mut known, &mut matches)?;
                    let mut candidates =
                        std::collections::BTreeMap::<([u8; 32], u64), AttachmentCandidate>::new();
                    for (_, mut asset) in matches {
                        unique(&mut asset.members);
                        unique(&mut asset.names);
                        unique(&mut asset.media_types);
                        unique(&mut asset.declared_sizes);
                        let hash = match asset.blob {
                            Some(hash) => hash,
                            None => {
                                let pin = AssetSource {
                                    chain: asset.chain,
                                    length: asset.size,
                                    version: asset.source_version.clone(),
                                    blob: None,
                                };
                                super::bindings::hash_member(self, &asset.input, asset.chain, &pin)?
                            }
                        };
                        candidates
                            .entry((hash.0, asset.size))
                            .or_insert_with(|| AttachmentCandidate {
                                hash,
                                size: asset.size,
                                sources: Vec::new(),
                            })
                            .sources
                            .push(AttachmentSourceClaim {
                                namespace: Some(AttachmentNamespace {
                                    input: asset.input.clone(),
                                    chain: asset.chain,
                                    import_id: scope.clone(),
                                }),
                                input: asset.input,
                                chain: asset.chain,
                                source_version: asset.source_version,
                                members: asset.members,
                                names: asset.names,
                                media_types: asset.media_types,
                                declared_sizes: asset.declared_sizes,
                            });
                    }
                    claims.push(AttachmentBinding {
                        source_version,
                        namespace: namespace.clone(),
                        exported_from: namespace.clone(),
                        prior_exports: Vec::new(),
                        block: slot.block,
                        attachment: slot.attachment,
                        source_id: slot.source_id.clone(),
                        candidates: candidates.into_values().collect(),
                    });
                }
            }
        }
        Ok(claims)
    }
}

fn unique<T: Ord + Clone>(values: &mut Vec<T>) {
    *values = values
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
}
