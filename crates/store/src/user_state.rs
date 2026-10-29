//! User-authored state moves between archives as its original requests,
//! replayed in commit order through the writer. Primary mode keeps the same
//! sequence in Sinex.
use crate::primary::RegisterChange;
use crate::user_writes::UserRequest;
use crate::{Reader, Result, StoreError};

impl Reader<'_> {
    /// One committed request, including its resolved rows, targets or revision.
    pub fn user_request(&self, id: &str) -> Result<Option<UserRequest>> {
        use rusqlite::OptionalExtension;
        let packed: Option<Option<Vec<u8>>> = self
            .conn()?
            .query_row(
                "SELECT request FROM source_client_request WHERE id = ?1",
                [id],
                |row| row.get(0),
            )
            .optional()?;
        packed
            .map(|packed| {
                let packed = packed.ok_or_else(|| StoreError::Conflict(id.into()))?;
                Ok(serde_json::from_str(&crate::encoding::unpack_text(
                    &packed,
                )?)?)
            })
            .transpose()
    }

    /// Every request in this archive, in the order they were
    /// written.
    pub fn user_state(&self) -> Result<Vec<RegisterChange>> {
        let conn = self.conn()?;
        // (commit_seq, created_at, id, change): one write counter orders
        // requests.
        let mut entries: Vec<(i64, i64, String, RegisterChange)> = Vec::new();
        let mut requests =
            conn.prepare("SELECT id, request, commit_seq, created_at FROM source_client_request")?;
        let rows = requests.query_map([], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, Option<Vec<u8>>>(1)?,
                r.get::<_, i64>(2)?,
                r.get::<_, i64>(3)?,
            ))
        })?;
        for row in rows {
            let (id, packed, seq, at) = row?;
            let packed = packed.ok_or_else(|| {
                StoreError::Invalid(format!(
                    "request {id} was written before requests were kept whole, so it cannot be exported"
                ))
            })?;
            let request: UserRequest =
                serde_json::from_str(&crate::encoding::unpack_text(&packed)?)?;
            entries.push((
                seq,
                at,
                id,
                RegisterChange::Request {
                    request: Box::new(request),
                    first_note: None,
                },
            ));
        }
        entries.sort_by(|a, b| (a.0, a.1, &a.2).cmp(&(b.0, b.1, &b.2)));
        Ok(entries.into_iter().map(|(.., change)| change).collect())
    }
}
