//! What the index holds of conversations the browser extension captures:
//! enough for the extension to show a page's state, to skip, during a
//! provider import, conversations the provider has not changed since their last
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

row! {
    /// A session's message count and its latest capture.
    struct CaptureState {
        messages: u64,
        last_captured_at: Option<i64>,
    }
    statements {
        const OF_SESSION = "SELECT (SELECT count(*) FROM message m WHERE m.session_pk = s.pk) AS messages,
            (SELECT max(e.at) FROM event e WHERE e.session_pk = s.pk AND e.kind = 'browser_capture') AS last_captured_at
            FROM session s WHERE s.id = ?1";
    }
}

impl Reader<'_> {
    /// Each session's state, in the order given; `None` for a session the
    /// index does not hold.
    pub fn captured_sessions(
        &self,
        sessions: &[polylogue_model::SessionId],
    ) -> Result<Vec<Option<CapturedSession>>> {
        sessions
            .iter()
            .map(|id| {
                let mut held = self
                    .conn()?
                    .maybe(&CaptureState::OF_SESSION, [id.to_string()])?
                    .map(|state| CapturedSession {
                        messages: state.messages,
                        last_captured_at: state.last_captured_at,
                        downloading: false,
                    });
                if let Some(held) = &mut held {
                    held.downloading = self.downloading_files(id)?;
                }
                Ok(held)
            })
            .collect()
    }

    /// Whether the newest capture is still waiting for attachment bytes.
    pub fn downloading_files(&self, session: &polylogue_model::SessionId) -> Result<bool> {
        let data: Option<Vec<u8>> = self.conn()?.scalar_optional(
            &format!(
                "SELECT {} FROM event e JOIN session s ON s.pk = e.session_pk
                 WHERE s.id = ?1 AND e.kind = 'browser_capture'
                 ORDER BY e.at DESC, e.pk DESC LIMIT 1",
                crate::content::sql_for(crate::content::DATA_BYTES_SQL, "e")
            ),
            [session.to_string()],
        )?;
        let Some(data) = data else {
            return Ok(false);
        };
        let data: polylogue_model::Json = crate::encoding::read_json(&data)?;
        Ok(data["attachments"]
            .as_array()
            .into_iter()
            .flatten()
            .any(|a| a["not_fetched"].as_str() == Some("downloading")))
    }

    /// The files of `session` whose bytes are stored, by the provider's
    /// file ID; one entry per ID.
    pub fn stored_files(&self, session: &polylogue_model::SessionId) -> Result<Vec<StoredFile>> {
        let conn = self.conn()?;
        let mut stmt = conn.prepare_cached(
            "SELECT b.content_group,b.content_start,b.content_length,b.data_length FROM block b
             JOIN session s ON s.pk = b.session_pk
             WHERE s.id = ?1 AND b.kind IN ('image', 'file', 'tool_result')",
        )?;
        let mut rows = stmt.query([session.to_string()])?;
        let mut found: BTreeMap<String, StoredFile> = BTreeMap::new();
        let mut reader = crate::content::Reader::default();
        while let Some(row) = rows.next()? {
            let piece = crate::content::Piece::of(row, 0)?;
            let block = crate::ordered_blocks::body(conn, &mut reader, piece, false)?;
            let attachments = block.attachments();
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

/// The rows this module declares statements for, for the every-statement test.
#[cfg(test)]
pub(crate) fn declared_rows(check: &mut crate::row::Check<'_>) {
    check.row::<CaptureState>();
}
