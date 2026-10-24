use super::*;
use crate::encoding::unpack_block;
use polylogue_model::{Attachment, BlobHash, Block, MessageId};
use std::collections::BTreeSet;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AttachmentState {
    Available,
    Ambiguous,
    Unavailable,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AssetCandidate {
    pub input: String,
    pub chain: i64,
    pub blob: Option<BlobHash>,
    pub basis: String,
    pub size: u64,
    pub members: Vec<String>,
    pub names: Vec<String>,
    pub media_types: Vec<String>,
    pub declared_sizes: Vec<u64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AttachmentResolution {
    pub message: MessageId,
    pub block: usize,
    pub attachment_index: usize,
    pub source: Attachment,
    pub state: AttachmentState,
    pub candidates: Vec<AssetCandidate>,
    pub binding: String,
}

impl Reader<'_> {
    /// Standalone display replays ordered source-prefix masking without
    /// changing the stored Attachment or its binding.
    pub fn masked_attachment(
        &self,
        message: &MessageId,
        block: usize,
        attachment_index: usize,
    ) -> Result<AttachmentResolution> {
        let mut resolution = self.attachment(message, block, attachment_index)?;
        let conn = self.conn()?;
        let (selected,session):(i64,i64)=conn.query_row("SELECT b.pk,m.session_pk FROM current_message m JOIN block b ON b.message_pk=m.pk WHERE m.id=?1 AND b.idx=?2",
            params![message.to_string(),block as i64],|row|Ok((row.get(0)?,row.get(1)?)))?;
        let mut masker = polylogue_model::masking::MessageMasker::default();
        crate::ordered_blocks::visit(conn, session, |mut stored| {
            masker.block(&mut stored.block);
            if stored.pk != selected {
                return Ok(std::ops::ControlFlow::Continue(()));
            }
            resolution.source = match stored.block {
                Block::Image(attachment) | Block::File(attachment) => attachment,
                Block::ToolResult { attachments, .. } => attachments
                    .into_iter()
                    .nth(attachment_index)
                    .ok_or_else(|| StoreError::NotFound("attachment index".into()))?,
                _ => {
                    return Err(StoreError::Invalid(
                        "selected block is not an attachment".into(),
                    ));
                }
            };
            Ok(std::ops::ControlFlow::Break(()))
        })?;
        Ok(resolution)
    }

    /// Resolve a recorded attachment through only current sources of the shown
    /// message version. Historical observations cannot authorize removed bytes.
    pub fn attachment(
        &self,
        message: &MessageId,
        block: usize,
        attachment_index: usize,
    ) -> Result<AttachmentResolution> {
        let conn = self.conn()?;
        let (pk,version,body):(i64,Vec<u8>,Vec<u8>)=conn.query_row(
            "SELECT m.pk,ms.version_hash,b.body FROM current_message m JOIN message_source ms ON ms.message_pk=m.pk AND ms.preferred=1 JOIN block b ON b.message_pk=m.pk WHERE m.id=?1 AND b.idx=?2",
            params![message.to_string(),i64::try_from(block).map_err(|_|StoreError::Invalid("block index is out of range".into()))?],
            |r|Ok((r.get(0)?,r.get(1)?,r.get(2)?))).optional()?
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
        let mut scopes=conn.prepare("SELECT DISTINCT own.acquisition FROM message_source ms
            JOIN input_ref ir ON ir.pk=ms.input_pk JOIN a.input i ON i.id=ir.id AND i.format='chatgpt'
            JOIN import_member own ON own.input=ir.id
            WHERE ms.message_pk=?1 AND ms.current=1 AND ms.version_hash=?2 ORDER BY own.acquisition")?;
        let scopes = scopes
            .query_map(params![pk, version], |r| r.get::<_, String>(0))?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        let mut candidates = std::collections::BTreeMap::<String, AssetCandidate>::new();
        let mut known = std::collections::HashMap::<String, Option<AssetBasis>>::new();
        if let Some(source_id) = source.source_id.as_deref() {
            let mut stmt = conn.prepare(
                "SELECT DISTINCT m.input,m.member FROM import_member m
                WHERE m.acquisition=?1 AND m.source_id=?2
                  AND EXISTS(SELECT 1 FROM a.revision r WHERE r.input=m.input)
                ORDER BY m.input,m.member",
            )?;
            for scope in &scopes {
                for row in stmt.query_map(params![scope, source_id], |r| {
                    Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
                })? {
                    let (input, member) = row?;
                    if !known.contains_key(&input) {
                        known.insert(input.clone(), basis(conn, "a.", &input)?);
                    }
                    let Some(current) = known[&input].as_ref() else {
                        continue;
                    };
                    let candidate =
                        candidates
                            .entry(input.clone())
                            .or_insert_with(|| AssetCandidate {
                                input,
                                chain: current.chain,
                                blob: current.blob,
                                basis: current.hash.clone(),
                                size: current.length,
                                members: Vec::new(),
                                names: Vec::new(),
                                media_types: Vec::new(),
                                declared_sizes: Vec::new(),
                            });
                    candidate.members.push(member);
                    let mut meta=conn.prepare_cached("SELECT x.name,x.media_type,x.size,m.input,s.basis FROM import_member m
                        JOIN import_asset_metadata x ON x.input=m.input AND x.sidecar=m.sidecar
                        JOIN import_asset_sidecar s ON s.input=x.input AND s.sidecar=x.sidecar AND s.error IS NULL
                        WHERE m.acquisition=?1 AND x.source_id=?2 AND m.sidecar IS NOT NULL
                          AND EXISTS(SELECT 1 FROM a.revision retained WHERE retained.input=m.input)
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
                            known.insert(input.clone(), basis(conn, "a.", &input)?);
                        }
                        if known[&input].as_ref().map(|basis| basis.hash.as_str())
                            != Some(recorded.as_str())
                        {
                            continue;
                        }
                        candidate.names.extend(name);
                        candidate.media_types.extend(media);
                        if let Some(size) = size {
                            candidate.declared_sizes.push(size.parse().map_err(|_| {
                                StoreError::Invalid("invalid declared asset size".into())
                            })?);
                        }
                    }
                }
            }
        }
        let mut candidates = candidates.into_values().collect::<Vec<_>>();
        for candidate in &mut candidates {
            unique(&mut candidate.members);
            unique(&mut candidate.names);
            unique(&mut candidate.media_types);
            unique(&mut candidate.declared_sizes);
        }
        let state = match candidates.len() {
            1 => AttachmentState::Available,
            0 if scopes.is_empty() || source.source_id.is_none() => AttachmentState::Unknown,
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
            &candidates,
        ))?)
        .to_hex()
        .to_string();
        Ok(AttachmentResolution {
            message: message.clone(),
            block,
            attachment_index,
            source,
            state,
            candidates,
            binding,
        })
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
