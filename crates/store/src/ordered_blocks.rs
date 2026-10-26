//! Source blocks on an existing connection, one decoded value at a time,
//! without tool results' structured data (`block.data`).
use crate::Result;
use crate::encoding::{restore_text, unpack_block, unpack_text};
use polylogue_model::Block;
use rusqlite::Connection;
use std::ops::ControlFlow;

pub(crate) struct StoredBlock {
    pub pk: i64,
    pub block: Block,
}

pub(crate) fn visit(
    connection: &Connection,
    session: i64,
    mut visit: impl FnMut(StoredBlock) -> Result<ControlFlow<()>>,
) -> Result<()> {
    let mut statement = connection.prepare_cached(
        "SELECT b.pk, b.body, t.body
         FROM message m JOIN block b ON b.message_pk=m.pk
         LEFT JOIN text t ON t.id=b.text_id WHERE m.session_pk=?1 AND m.duplicate_of IS NULL
         ORDER BY m.order_input_id, m.seq, m.record_position, m.id, b.idx",
    )?;
    let mut rows = statement.query([session])?;
    while let Some(row) = rows.next()? {
        let mut block = unpack_block(&row.get::<_, Vec<u8>>(1)?)?;
        if let Some(text) = row.get::<_, Option<Vec<u8>>>(2)? {
            restore_text(&mut block, unpack_text(&text)?);
        }
        if visit(StoredBlock {
            pk: row.get(0)?,
            block,
        })?
        .is_break()
        {
            break;
        }
    }
    Ok(())
}

/// One block, decoded with its text.
pub(crate) fn one(connection: &Connection, block: i64) -> Result<Block> {
    let (body, text): (Vec<u8>, Option<Vec<u8>>) = connection
        .prepare_cached("SELECT b.body, t.body FROM block b LEFT JOIN text t ON t.id = b.text_id WHERE b.pk = ?1")?
        .query_row([block], |r| Ok((r.get(0)?, r.get(1)?)))?;
    let mut block = unpack_block(&body)?;
    if let Some(text) = text {
        restore_text(&mut block, unpack_text(&text)?);
    }
    Ok(block)
}

/// Whether masking leaves a PEM block open, and until what: `None` closed,
/// `Some(None)` open to the end, `Some(Some(end))` open until `end`.
type OpenPem = Option<Option<Vec<u8>>>;

/// Labels the masker keeps whole; a longer one never closes.
const PEM_LABEL: usize = 512;

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    memchr::memmem::find(haystack, needle)
}

/// What a block, stored as `body` with its text moved to `text`, holds of
/// PEM private key blocks, as `message.pem` records it: 0 nothing; 1 a PEM
/// header or END line; 2 also a header that masking the block from a fresh
/// start leaves open after it. Masking carries nothing else from one block
/// to the next, and a header or END line counts only whole within one block,
/// so a block holding neither leaves the masking state as it found it.
pub(crate) fn pem(body: &[u8], text: Option<&str>) -> i64 {
    let text = text.map(str::as_bytes);
    if after_block(None, body, text).is_some() {
        return 2;
    }
    let marked =
        |bytes: &[u8]| find(bytes, b"-----BEGIN ").is_some() || find(bytes, b"-----END ").is_some();
    i64::from(marked(body) || text.is_some_and(marked))
}

/// The masker's rule for a PEM header at `at` (just after `-----BEGIN `):
/// an uppercase label of single-spaced words ending `PRIVATE KEY`, then
/// five dashes. Returns where the header ends and what closes it.
fn pem_header(bytes: &[u8], at: usize) -> Option<(usize, Option<Vec<u8>>)> {
    let mut label = Vec::new();
    let mut i = at;
    while let Some(&c) = bytes.get(i) {
        let space_ok = !label.is_empty() && label.last() != Some(&b' ');
        if c.is_ascii_uppercase() || c == b' ' && space_ok {
            label.push(c);
            i += 1;
        } else {
            break;
        }
    }
    let ok = label == b"PRIVATE KEY" || label.ends_with(b" PRIVATE KEY");
    if !ok || !bytes[i..].starts_with(b"-----") {
        return None;
    }
    let end = (label.len() <= PEM_LABEL).then(|| {
        let mut end = b"-----END ".to_vec();
        end.extend(&label);
        end.extend(b"-----");
        end
    });
    Some((i + 5, end))
}

/// The PEM state after a block: headers count wherever its stored bytes
/// hold one; an END counts only in its text, which masking reads whole.
/// Either way the state is never shown closed when masking keeps it open.
fn after_block(mut state: OpenPem, body: &[u8], text: Option<&[u8]>) -> OpenPem {
    for (bytes, ends) in [(body, false), (text.unwrap_or_default(), true)] {
        let mut at = 0;
        loop {
            match &state {
                Some(None) => return state,
                Some(Some(end)) => {
                    if !ends {
                        break;
                    }
                    match find(&bytes[at..], end) {
                        Some(i) => {
                            at += i + end.len();
                            state = None;
                        }
                        None => break,
                    }
                }
                None => match find(&bytes[at..], b"-----BEGIN ") {
                    None => break,
                    Some(i) => {
                        let header = at + i + b"-----BEGIN ".len();
                        match pem_header(bytes, header) {
                            Some((after, end)) => {
                                state = Some(end);
                                at = after;
                            }
                            None => at = header,
                        }
                    }
                },
            }
        }
    }
    state
}

/// Of `blocks` in `session`, those masking reaches with no PEM block open:
/// masked alone, they mask the same as after the session's earlier blocks.
/// Reads stored bytes only, without decoding or masking blocks.
pub(crate) fn masked_alone(
    connection: &Connection,
    session: i64,
    blocks: &std::collections::HashSet<i64>,
) -> Result<Vec<i64>> {
    let opens: bool = connection
        .prepare_cached("SELECT EXISTS (SELECT 1 FROM message WHERE session_pk = ?1 AND pem = 2)")?
        .query_row([session], |r| r.get(0))?;
    if !opens {
        let mut alone: Vec<i64> = blocks.iter().copied().collect();
        alone.sort_unstable();
        return Ok(alone);
    }
    let mut statement = connection.prepare_cached(
        "SELECT b.pk, b.body, t.body
         FROM message m JOIN block b ON b.message_pk=m.pk
         LEFT JOIN text t ON t.id=b.text_id WHERE m.session_pk=?1 AND m.duplicate_of IS NULL
         ORDER BY m.order_input_id, m.seq, m.record_position, m.id, b.idx",
    )?;
    let mut rows = statement.query([session])?;
    let mut alone = Vec::new();
    let mut left = blocks.len();
    let mut state: OpenPem = None;
    while left > 0
        && let Some(row) = rows.next()?
    {
        let pk: i64 = row.get(0)?;
        if blocks.contains(&pk) {
            left -= 1;
            if state.is_none() {
                alone.push(pk);
            }
        }
        let body = crate::encoding::unpack_bytes(&row.get::<_, Vec<u8>>(1)?)?;
        let text = row
            .get::<_, Option<Vec<u8>>>(2)?
            .map(|t| crate::encoding::unpack_bytes(&t))
            .transpose()?;
        state = after_block(state, &body, text.as_deref());
    }
    Ok(alone)
}

#[cfg(test)]
mod tests;
