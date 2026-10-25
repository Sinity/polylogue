//! Usage and quota readings in a measure's selection. The writer stores each
//! reading once per origin, so a copy in another input is not read twice.
use super::Selection;
use crate::{Reader, Result};
use serde::de::DeserializeOwned;

/// Visits every reading of `kind` in `selection`, in time order, with its
/// session's origin and the event time.
pub(super) fn visit<T: DeserializeOwned>(
    reader: &Reader<'_>,
    selection: &Selection,
    kind: &str,
    mut visit: impl FnMut(&str, Option<i64>, T) -> Result<()>,
) -> Result<()> {
    let (cond, mut params) = selection.condition();
    params.push(kind.to_owned().into());
    let conn = reader.conn()?;
    let mut stmt = conn.prepare(&format!(
        "SELECT s.origin, m.at, m.data FROM event m
         JOIN session s ON s.pk = m.session_pk WHERE {cond} AND m.kind = ?{} ORDER BY m.at, m.id",
        params.len()
    ))?;
    let mut rows = stmt.query(rusqlite::params_from_iter(params.iter()))?;
    while let Some(row) = rows.next()? {
        let value: serde_json::Value = crate::encoding::unpack_json(&row.get::<_, Vec<u8>>(2)?)?;
        if value
            .get("report_id")
            .and_then(serde_json::Value::as_str)
            .is_none()
        {
            continue;
        }
        visit(
            &row.get::<_, String>(0)?,
            row.get(1)?,
            serde_json::from_value(value)?,
        )?;
    }
    Ok(())
}
