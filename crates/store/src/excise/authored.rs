//! Text excision in authored payloads, preserving row and revision identities.

use super::*;

fn redact_value(value: &str, needle: &str, json: bool) -> Result<Option<String>> {
    let mut scan = if json && serde_json::from_str::<serde_json::Value>(value).is_ok() {
        text_matches::TextMatches::new(needle)?
    } else {
        text_matches::TextMatches::plain(needle)?
    };
    scan.feed(value.as_bytes())?;
    let ranges = scan.finish()?;
    if ranges.is_empty() {
        return Ok(None);
    }
    let fill = text_matches::replacement(needle)?;
    let mut bytes = value.as_bytes().to_vec();
    for (start, end) in ranges {
        for b in &mut bytes[start as usize..end as usize] {
            if *b != b'\n' {
                *b = fill;
            }
        }
    }
    Ok(Some(
        String::from_utf8(bytes).map_err(|e| StoreError::Invalid(e.to_string()))?,
    ))
}

fn fields(
    a: &Transaction<'_>,
    table: &str,
    columns: &[(&str, bool)],
    needle: &str,
    dry: bool,
) -> Result<u64> {
    let names = columns
        .iter()
        .map(|(name, _)| *name)
        .collect::<Vec<_>>()
        .join(", ");
    let mut stmt = a.prepare(&format!("SELECT id, {names} FROM {table} ORDER BY id"))?;
    let mut rows = stmt.query([])?;
    let mut count = 0;
    while let Some(row) = rows.next()? {
        let id: String = row.get(0)?;
        let mut changed = false;
        for (n, (column, json)) in columns.iter().enumerate() {
            let value: Option<String> = row.get(n + 1)?;
            if let Some(value) = value
                && let Some(new) = redact_value(&value, needle, *json)?
            {
                changed = true;
                if !dry {
                    a.execute(
                        &format!("UPDATE {table} SET {column} = ?2 WHERE id = ?1"),
                        params![id, new],
                    )?;
                }
            }
        }
        count += u64::from(changed);
    }
    Ok(count)
}

pub(super) fn redact(
    a: &Transaction<'_>,
    needle: &str,
    dry: bool,
    done: &mut Excised,
) -> Result<()> {
    done.notes_redacted = fields(
        a,
        "note",
        &[("key", false), ("value", true), ("check_json", true)],
        needle,
        dry,
    )?;
    done.variants_redacted = fields(a, "variant", &[("body", false)], needle, dry)?;
    done.kept_results_redacted = fields(
        a,
        "kept_result",
        &[("name", false), ("query", false), ("rows_json", true)],
        needle,
        dry,
    )?;
    fields(a, "excision", &[("reason", false)], needle, dry)?;
    let mut events = a.prepare(
        "SELECT delivery, what, at, detail FROM delivery_event ORDER BY delivery, what, at",
    )?;
    let mut rows = events.query([])?;
    while let Some(row) = rows.next()? {
        let delivery: String = row.get(0)?;
        let what: String = row.get(1)?;
        let at: i64 = row.get(2)?;
        let detail: Option<String> = row.get(3)?;
        if let Some(detail) = detail
            && let Some(new) = redact_value(&detail, needle, false)?
            && !dry
        {
            a.execute("UPDATE delivery_event SET detail = ?4 WHERE delivery = ?1 AND what = ?2 AND at = ?3",
                params![delivery, what, at, new])?;
        }
    }
    drop(rows);
    drop(events);
    let mut stmt = a.prepare("SELECT id, body FROM delivery ORDER BY id")?;
    let mut rows = stmt.query([])?;
    while let Some(row) = rows.next()? {
        let id: String = row.get(0)?;
        let body: Vec<u8> = row.get(1)?;
        if let Some(new) = redact_value(&unpack_text(&body)?, needle, false)? {
            done.deliveries_redacted += 1;
            if !dry {
                a.execute(
                    "UPDATE delivery SET body = ?2, content_hash = ?3 WHERE id = ?1",
                    params![
                        id,
                        zstd::bulk::compress(new.as_bytes(), 1)?,
                        &blake3::hash(new.as_bytes()).as_bytes()[..]
                    ],
                )?;
            }
        }
    }
    Ok(())
}
