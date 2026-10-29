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

#[derive(Debug, Clone, Serialize)]
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
        let (selected,session):(i64,i64)=conn.query_row("SELECT b.pk,m.session_pk FROM message m JOIN block b ON b.message_pk=m.pk WHERE m.id=?1 AND coalesce(b.ordinal,CAST(b.idx AS TEXT))=?2",
            params![message.to_string(),block.to_string()],|row|Ok((row.get(0)?,row.get(1)?)))?;
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
            "SELECT m.pk,ms.version_hash,b.body FROM message m JOIN message_source ms ON ms.message_pk=m.pk AND ms.shown=1 JOIN block b ON b.message_pk=m.pk WHERE m.id=?1 AND coalesce(b.ordinal,CAST(b.idx AS TEXT))=?2",
            params![message.to_string(),block.to_string()],
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
        let mut scopes = conn.prepare(
            "SELECT DISTINCT own.import_id FROM message_source ms
            JOIN input_ref ir ON ir.pk=ms.input_pk JOIN source_input i ON i.id=ir.id
            JOIN import_member own ON own.input=ir.id
            WHERE ms.message_pk=?1 AND ms.version_hash=?2 ORDER BY own.import_id",
        )?;
        let scopes = scopes
            .query_map(params![pk, version], |r| r.get::<_, String>(0))?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        let mut matches = std::collections::BTreeMap::<String, AssetMatch>::new();
        let mut known = std::collections::HashMap::<String, Option<AssetSource>>::new();
        let identity = source.source_id.as_deref().filter(|id| !id.is_empty());
        if let Some(source_id) = identity {
            let mut stmt = conn.prepare(
                "SELECT DISTINCT m.input,m.member FROM import_member m
                WHERE m.import_id=?1 AND m.source_id=?2
                  AND EXISTS(SELECT 1 FROM source_revision r WHERE r.input=m.input)
                ORDER BY m.input,m.member",
            )?;
            for scope in &scopes {
                for row in stmt.query_map(params![scope, source_id], |r| {
                    Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
                })? {
                    let (input, member) = row?;
                    if !known.contains_key(&input) {
                        known.insert(input.clone(), asset_source(conn, "source_", &input)?);
                    }
                    let Some(current) = known[&input].as_ref() else {
                        continue;
                    };
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
                            known.insert(input.clone(), asset_source(conn, "source_", &input)?);
                        }
                        if known[&input].as_ref().map(|source| source.version.as_str())
                            != Some(recorded.as_str())
                        {
                            continue;
                        }
                        asset.names.extend(name);
                        asset.media_types.extend(media);
                        if let Some(size) = size {
                            asset.declared_sizes.push(size.parse().map_err(|_| {
                                StoreError::Invalid("invalid declared asset size".into())
                            })?);
                        }
                    }
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

fn unique<T: Ord + Clone>(values: &mut Vec<T>) {
    *values = values
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
}
