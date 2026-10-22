//! Check record boundaries and surviving uses before rewriting shared inputs.
use super::*;

pub(super) fn preflight(
    t: &Transactions<'_, '_>,
    input: &str,
    spans: &[Span],
    erasures: &super::erasures::Erasures,
    sessions: &[(i64, String)],
) -> Result<()> {
    let format: String =
        t.archive
            .query_row("SELECT format FROM input WHERE id=?1", [input], |r| {
                r.get(0)
            })?;
    let refuse = |detail: String| StoreError::ExcisionFormat {
        input: input.to_owned(),
        format: format.clone(),
        detail,
    };
    let excluded = serde_json::to_string(&sessions.iter().map(|(pk, _)| *pk).collect::<Vec<_>>())?;
    let chains: BTreeSet<i64> = spans.iter().map(|s| s.chain).collect();
    for &chain in &chains {
        let length: Option<i64> = t.archive.query_row(
            "SELECT MAX(length) FROM revision WHERE input=?1 AND chain=?2",
            params![input, chain],
            |r| r.get(0),
        )?;
        let length =
            length.ok_or_else(|| refuse("source chain has no retained revision".into()))?;
        if spans
            .iter()
            .filter(|s| s.chain == chain)
            .any(|s| s.start < 0 || s.end > length || s.end <= s.start)
        {
            return Err(refuse(
                "source citation is outside its retained chain".into(),
            ));
        }
        let mut statement=t.archive.prepare("SELECT e.offset,e.length,e.blob,e.blob_offset,b.size FROM extent e JOIN blob b ON b.hash=e.blob WHERE e.input=?1 AND e.chain=?2 AND e.offset<?3 ORDER BY e.offset")?;
        let mut rows = statement.query(params![input, chain, length])?;
        let mut position = 0;
        let mut documents = 0;
        let mut erased = 0;
        while let Some(row) = rows.next()? {
            let offset: i64 = row.get(0)?;
            let size: i64 = row.get(1)?;
            let hash: Vec<u8> = row.get(2)?;
            let blob_offset: i64 = row.get(3)?;
            let blob_size: i64 = row.get(4)?;
            if offset != position
                || size <= 0
                || offset.checked_add(size).is_none_or(|end| end > length)
                || blob_offset != 0
                || size != blob_size
            {
                return Err(refuse(
                    "source pieces do not provide complete independently decodable records".into(),
                ));
            }
            let (piece_documents, piece_erased) = match erasures.counts(&hash)? {
                Some(counts) => counts,
                None => {
                    let hash_bytes: [u8; 32] = hash
                        .as_slice()
                        .try_into()
                        .map_err(|_| refuse("invalid source blob hash".into()))?;
                    let bytes = super::plain_blob::PlainBlob::load(t, input, BlobHash(hash_bytes))?;
                    let ranges = erasures.requested(&hash)?;
                    let checked = bytes.plan(&ranges).map_err(|error| match error {
                        super::super::record_ranges::Error::Records(detail) => refuse(detail),
                        super::super::record_ranges::Error::Io(error) => StoreError::Io(error),
                    })?;
                    let counts = (checked.documents, checked.erased_documents);
                    erasures.checked(&hash, &checked)?;
                    counts
                }
            };
            documents += piece_documents;
            erased += piece_erased;
            position += size;
        }
        if position != length {
            return Err(refuse("retained source pieces have a gap".into()));
        }
        let surviving:bool=t.index.query_row("SELECT EXISTS(SELECT 1 FROM parsed p JOIN input_ref r ON r.pk=p.input_pk WHERE r.id=?1 AND p.chain=?2 AND p.session_pk NOT IN(SELECT value FROM json_each(?3)))",params![input,chain,excluded],|r|r.get(0))?;
        if documents > 0 && erased == documents && surviving {
            return Err(refuse(
                "removing every document would also remove surviving session facts".into(),
            ));
        }
    }
    if erasures.has_unchecked()? {
        return Err(refuse("source citation has no retained piece".into()));
    }
    erasures.visit(|hash, ranges| {
        let mut uses = t
            .archive
            .prepare("SELECT input,chain,offset,length,blob_offset FROM extent WHERE blob=?1")?;
        let mut rows = uses.query([hash])?;
        while let Some(row) = rows.next()? {
            let source: String = row.get(0)?;
            let chain: i64 = row.get(1)?;
            let offset: i64 = row.get(2)?;
            let length: i64 = row.get(3)?;
            let blob_offset: i64 = row.get(4)?;
            if source != input || !chains.contains(&chain) {
                return Err(refuse(
                    "source bytes are also held by a source outside this rewrite".into(),
                ));
            }
            for &(start, end) in ranges {
                let start = start as i64;
                let end = end as i64;
                let a = start.max(blob_offset) - blob_offset + offset;
                let b = end.min(blob_offset + length) - blob_offset + offset;
                if a >= b {
                    continue;
                }
                let overlaps:bool=t.index.query_row("SELECT EXISTS(
                    SELECT 1 FROM message_source s JOIN input_ref r ON r.pk=s.input_pk
                    WHERE r.id=?1 AND s.chain=?2 AND s.source_start<?4 AND s.source_end>?3
                      AND s.session_pk NOT IN(SELECT value FROM json_each(?5))
                    UNION ALL SELECT 1 FROM event_source s JOIN input_ref r ON r.pk=s.input_pk JOIN event e ON e.id=s.event_id
                    WHERE r.id=?1 AND s.chain=?2 AND s.source_start<?4 AND s.source_end>?3
                      AND e.session_pk NOT IN(SELECT value FROM json_each(?5)))",params![input,chain,a,b,excluded],|r|r.get(0))?;
                if overlaps {
                    return Err(refuse(
                        "source removal overlaps a surviving session record".into(),
                    ));
                }
            }
        }
        Ok(())
    })
}
