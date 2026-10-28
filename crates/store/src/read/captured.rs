//! What the index holds of conversations the browser extension captures:
//! enough for the extension to show a page's state, to skip, during a
//! backfill, conversations the provider has not changed since their last
//! capture, and to skip downloading files whose bytes are already stored.
use super::*;

/// One captured conversation's session in the index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapturedSession {
    pub messages: u64,
    pub downloading: bool,
    /// When the latest stored capture of it was taken (ms): the provider's
    /// conversation at that moment is in the archive.
    pub last_captured_at: Option<i64>,
}

/// A file of a conversation whose bytes the blob store holds.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct StoredFile {
    /// The provider's ID of the file.
    pub id: String,
    pub blob: polylogue_model::BlobHash,
    pub size: Option<u64>,
}

impl Reader<'_> {
    /// Each session's state, in the order given; `None` for a session the
    /// index does not hold.
    pub fn captured_sessions(
        &self,
        sessions: &[polylogue_model::SessionId],
    ) -> Result<Vec<Option<CapturedSession>>> {
        let mut stmt = self.conn()?.prepare_cached(
            "SELECT (SELECT count(*) FROM message m WHERE m.session_pk = s.pk),
                    (SELECT max(e.at) FROM event e
                     WHERE e.session_pk = s.pk AND e.kind = 'browser_capture')
             FROM session s WHERE s.id = ?1",
        )?;
        sessions
            .iter()
            .map(|id| {
                let mut held = stmt
                    .query_row([id.to_string()], |r| {
                        Ok(CapturedSession {
                            messages: u64::try_from(r.get::<_, i64>(0)?).unwrap_or(0),
                            last_captured_at: r.get(1)?,
                            downloading: false,
                        })
                    })
                    .optional()?;
                if let Some(held) = &mut held {
                    held.downloading = self.downloading_files(id)?;
                }
                Ok(held)
            })
            .collect()
    }

    /// Whether the newest capture is still waiting for attachment bytes.
    pub fn downloading_files(&self, session: &polylogue_model::SessionId) -> Result<bool> {
        let data: Option<Vec<u8>> = self
            .conn()?
            .query_row(
                "SELECT e.data FROM event e JOIN session s ON s.pk = e.session_pk
             WHERE s.id = ?1 AND e.kind = 'browser_capture'
             ORDER BY e.at DESC, e.pk DESC LIMIT 1",
                [session.to_string()],
                |r| r.get(0),
            )
            .optional()?;
        let Some(data) = data else {
            return Ok(false);
        };
        let data: polylogue_model::Json = crate::encoding::unpack_json(&data)?;
        Ok(data["attachments"]
            .as_array()
            .into_iter()
            .flatten()
            .any(|a| a["not_fetched"].as_str() == Some("downloading")))
    }

    /// The files of `session` whose bytes are stored, by the provider's
    /// file ID; one entry per ID.
    pub fn stored_files(&self, session: &polylogue_model::SessionId) -> Result<Vec<StoredFile>> {
        let mut stmt = self.conn()?.prepare_cached(
            "SELECT b.body FROM block b JOIN session s ON s.pk = b.session_pk
             WHERE s.id = ?1 AND b.kind IN ('image', 'file', 'tool_result')",
        )?;
        let mut rows = stmt.query([session.to_string()])?;
        let mut found: BTreeMap<String, StoredFile> = BTreeMap::new();
        while let Some(row) = rows.next()? {
            let block = unpack_block(&row.get::<_, Vec<u8>>(0)?)?;
            let attachments = match &block {
                Block::Image(a) | Block::File(a) => std::slice::from_ref(a),
                Block::ToolResult { attachments, .. } => attachments.as_slice(),
                _ => &[],
            };
            for a in attachments {
                if let (Some(id), Some(blob)) = (&a.source_id, a.blob) {
                    found.entry(id.clone()).or_insert(StoredFile {
                        id: id.clone(),
                        blob,
                        size: a.size,
                    });
                }
            }
        }
        Ok(found.into_values().collect())
    }
}
