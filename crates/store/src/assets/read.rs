use super::*;
use polylogue_model::attachment_binding::{AttachmentBinding, AttachmentNamespace, AttachmentSlot};
use polylogue_model::bundle::{self, Record};
use polylogue_model::{Attachment, BlobHash, Block, MessageId};
use std::collections::{BTreeMap, BTreeSet, HashMap};

polylogue_model::stored_names! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum AttachmentState {
        Available => "available",
        Ambiguous => "ambiguous",
        Unavailable => "unavailable",
        Unknown => "unknown",
    }
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
        let (version,piece,source_input,source_chain,origin):(Vec<u8>,crate::content::Piece,String,i64,String)=conn.query_row(
            "SELECT m.version_hash,b.content_group,b.content_start,b.content_length,b.data_length,i.id,m.source_chain,i.origin FROM message m JOIN input_ref i ON i.pk=m.input_pk JOIN block b ON b.message_pk=m.pk WHERE m.id=?1 AND m.version_hash IS NOT NULL AND coalesce(b.source_position,CAST(b.position AS TEXT))=?2",
            params![message.to_string(),block.to_string()],
            |r| Ok((r.get(0)?, crate::content::Piece::of(r, 1)?, r.get(5)?, r.get(6)?, r.get(7)?))).optional()?
            .ok_or_else(||StoreError::NotFound(format!("attachment block {message}:{block}")))?;
        let source = match crate::ordered_blocks::body(
            conn,
            &mut crate::content::Reader::default(),
            piece,
            false,
        )? {
            Block::Image(a) | Block::File(a) if attachment_index == 0 => a,
            Block::ToolResult { attachments, .. } => attachments
                .into_iter()
                .nth(attachment_index)
                .ok_or_else(|| StoreError::NotFound("tool result attachment index".into()))?,
            _ => {
                return Err(polylogue_model::Reason::NoSuchAttachment {
                    message: message.to_string(),
                    block,
                    attachment: attachment_index,
                }
                .into());
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
                    return Err(StoreError::stored_data(
                        "a stated attachment binding differs from the selected attachment",
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
        let mut matches = BTreeMap::new();
        let identity = source.source_id.as_deref().filter(|id| !id.is_empty());
        let mut candidate_count = None;
        if let Some(source_id) = identity {
            if origin == bundle::FORMAT && source.blob.is_none() && !scopes.is_empty() {
                let slot = AttachmentSlot {
                    block: block as u64,
                    attachment: attachment_index as u64,
                    source_id: source_id.to_owned(),
                };
                let candidates: BTreeSet<_> = self
                    .bundle_bindings(message, &version)?
                    .into_iter()
                    .filter(|claim| binding_matches_slot(claim, &slot))
                    .flat_map(|claim| claim.candidates.into_iter().map(|c| (c.hash, c.size)))
                    .collect();
                for &(hash, size) in &candidates {
                    for scope in &scopes {
                        bundle_matches(conn, scope, hash, size, &mut matches)?;
                    }
                }
                candidate_count = Some(candidates.len());
            } else {
                let mut known = HashMap::new();
                for scope in &scopes {
                    scope_matches(conn, scope, source_id, &mut known, &mut matches)?;
                }
            }
        }
        let mut matches = matches.into_values().collect::<Vec<_>>();
        for asset in &mut matches {
            unique(&mut asset.members);
            unique(&mut asset.names);
            unique(&mut asset.media_types);
            unique(&mut asset.declared_sizes);
        }
        let state = match (candidate_count, matches.len()) {
            (Some(2..), _) | (_, 2..) => AttachmentState::Ambiguous,
            (_, 1) => AttachmentState::Available,
            (Some(0), _) => AttachmentState::Unknown,
            (_, 0) if scopes.is_empty() || identity.is_none() => AttachmentState::Unknown,
            _ => AttachmentState::Unavailable,
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

    fn bundle_bindings(
        &self,
        message: &MessageId,
        version: &[u8],
    ) -> Result<Vec<AttachmentBinding>> {
        let source = self
            .stored_record(crate::items::RecordTable::Message, &message.to_string())?
            .ok_or_else(|| StoreError::stored_data("bundle message has no stored source record"))?;
        match serde_json::from_slice::<Record>(&source.bytes)? {
            Record::Message(record)
                if record.message.id == *message
                    && record.source_version.0.as_slice() == version =>
            {
                Ok(record.bindings)
            }
            _ => Err(StoreError::stored_data(
                "bundle source record differs from the selected message",
            )),
        }
    }
}

fn binding_matches_slot(binding: &AttachmentBinding, slot: &AttachmentSlot) -> bool {
    binding.block == slot.block
        && binding.attachment == slot.attachment
        && binding.source_id == slot.source_id
}

/// A portable candidate grants access only through a matching member admitted
/// with the receiving import. Provider IDs and archive-wide blobs are not proof.
fn bundle_matches(
    conn: &Connection,
    scope: &str,
    hash: BlobHash,
    size: u64,
    matches: &mut BTreeMap<String, AssetMatch>,
) -> Result<()> {
    let input = hash.to_string();
    let path = format!("blobs/{input}");
    let mut stmt = conn.prepare_cached(
        "SELECT member,observation FROM import_member
         WHERE import_id=?1 AND input=?2
           AND (member=?3 OR substr(member,-length(?3)-1)='/'||?3)
         ORDER BY member,observation",
    )?;
    let mut members = stmt.query(params![scope, input, path])?;
    let Some(current) = asset_source(conn, conn, "source_", &input)? else {
        return Ok(());
    };
    if current.length != size || current.blob != Some(hash) {
        return Ok(());
    }
    while let Some(member) = members.next()? {
        let pin = if size == 0 {
            let Some(pin) = super::empty::source(
                conn,
                conn,
                "source_",
                &input,
                current.chain,
                Some(&member.get::<_, String>(1)?),
                None,
            )?
            else {
                continue;
            };
            pin
        } else {
            current.clone()
        };
        matches
            .entry(input.clone())
            .or_insert_with(|| AssetMatch {
                input: input.clone(),
                chain: pin.chain,
                blob: pin.blob,
                source_version: pin.version,
                size,
                members: Vec::new(),
                names: Vec::new(),
                media_types: Vec::new(),
                declared_sizes: Vec::new(),
            })
            .members
            .push(member.get(0)?);
    }
    Ok(())
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
        <(String, String, String)>::try_from(r)
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
            ORDER BY m.input,x.position")?;
        for row in meta.query_map(params![scope, source_id], |r| {
            <(
                Option<String>,
                Option<String>,
                Option<String>,
                String,
                String,
            )>::try_from(r)
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
                asset.declared_sizes.push(size.parse().map_err(|error| {
                    crate::StoreError::stored(
                        2,
                        format!("invalid declared asset size {size:?}: {error}"),
                    )
                })?);
            }
        }
    }
    Ok(())
}

impl Reader<'_> {
    /// Stated bindings for the stored bytes behind a message's attachments, for
    /// slots no stored binding covers. Each import namespace that read the
    /// message's shown source states one binding per slot: its candidates are the
    /// complete bytes of the members carrying the slot's source ID, and none
    /// when no such member is stored.
    pub fn current_attachment_bindings(
        &self,
        message: &MessageId,
        source_version: BlobHash,
        slots: &[AttachmentSlot],
    ) -> Result<Vec<AttachmentBinding>> {
        use polylogue_model::attachment_binding::{AttachmentCandidate, StatedAttachmentSource};
        if slots.is_empty() {
            return Ok(Vec::new());
        }
        let conn = self.conn()?;
        let mut shown = conn.prepare(
            "SELECT i.id,m.source_chain,i.origin FROM message m
             JOIN input_ref i ON i.pk=m.input_pk
             WHERE m.id=?1 AND m.version_hash=?2 ORDER BY i.id,m.source_chain",
        )?;
        let shown = shown
            .query_map(
                params![message.to_string(), source_version.0.as_slice()],
                |r| <(String, i64, String)>::try_from(r),
            )?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        let mut stated = Vec::new();
        for (input, chain, origin) in shown {
            let scopes = import_scopes(conn, &input, chain)?;
            if scopes.is_empty() {
                continue;
            }
            if origin == bundle::FORMAT {
                let bindings = self.bundle_bindings(message, &source_version.0)?;
                for scope in scopes {
                    let context = AttachmentNamespace {
                        input: input.clone(),
                        chain,
                        import_id: scope,
                    };
                    stated.extend(
                        bindings
                            .iter()
                            .filter(|claim| {
                                slots.iter().any(|slot| binding_matches_slot(claim, slot))
                            })
                            .cloned()
                            .map(|mut claim| {
                                claim.exported_from = context.clone();
                                claim
                            }),
                    );
                }
                continue;
            }
            for scope in scopes {
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
                            .push(StatedAttachmentSource {
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
                    stated.push(AttachmentBinding {
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
        Ok(stated)
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
