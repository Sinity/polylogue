//! One retained export cursor admits callback semantics before parsed effects.
use super::*;

#[derive(Clone)]
pub(crate) struct ExportPrefix {
    cursor: Option<ExportCursor>,
    first: ReadingFootprint,
    last: ReadingFootprint,
    length: u64,
    body_hash: String,
    refused: bool,
}

pub(crate) fn is_refusal(step: &ParseStep) -> bool {
    matches!(
        step.progress,
        polylogue_model::batch::StepProgress::Refused { .. }
    )
}

pub(crate) fn load_prefix(
    conn: &Connection,
    input: &str,
    chain: i64,
    version: &str,
) -> Result<Option<ExportPrefix>> {
    let previous: Option<String> = conn.query_row(
        "SELECT w.state FROM wire_reading w JOIN input_ref i ON i.pk=w.input_pk WHERE i.id=?1 AND w.chain=?2 AND w.reading_version=?3",
        params![input,chain,version], |row| row.get(0),
    ).optional()?;
    previous
        .map(|json| {
            let old: Reading = serde_json::from_str(&json)?;
            Ok(ExportPrefix {
                cursor: old.export_cursor,
                first: old.first.footprint,
                last: old.last.footprint,
                length: old.length,
                body_hash: old.last_hash,
                refused: old.refused_export,
            })
        })
        .transpose()
}

pub(crate) fn admit_export(
    conn: &Connection,
    input: &str,
    chain: i64,
    length: u64,
    step: &ParseStep,
) -> Result<bool> {
    admit(
        load_prefix(conn, input, chain, &step.reading_version)?.as_ref(),
        length,
        step,
    )
}

fn admit(previous: Option<&ExportPrefix>, length: u64, step: &ParseStep) -> Result<bool> {
    // A failure carries no records and moves no cursor, so it switches
    // neither an export reading nor ordinary byte control.
    if matches!(
        step.progress,
        polylogue_model::batch::StepProgress::Failed { .. }
    ) {
        return Ok(true);
    }
    let Some(delivery) = step.delivery.as_ref() else {
        if previous.is_some_and(|old| old.cursor.is_some() || old.refused) {
            return invalid("an export reading cannot switch to ordinary byte control");
        }
        return Ok(true);
    };
    step.validate_export()?;
    match delivery.source {
        ExportDeliverySource::Item { end, .. } if end > length => {
            return invalid("export item lies outside its immutable stored source");
        }
        ExportDeliverySource::Complete {
            length: scanned, ..
        } => {
            if scanned != length
                || step.start != 0
                || step.parsed_upto != length
                || !step.progress.reached_end()
            {
                return invalid("export completion differs from its verified source length");
            }
        }
        _ if step.start != 0 || step.parsed_upto != 0 || step.progress.reached_end() => {
            return invalid("an export callback cannot acknowledge a partial byte cursor");
        }
        _ => {}
    }
    let refused = is_refusal(step);
    if refused && (step.output != polylogue_model::Output::default() || step.checkpoint.is_some()) {
        return invalid(
            "an unreadable export header cannot carry facts or a completion checkpoint",
        );
    }
    let previous = match previous {
        Some(old) if old.length < length && delivery.ordinal == 0 => None,
        Some(old) if old.length != length => {
            return invalid("export control belongs to another stored source length");
        }
        other => other,
    };
    if let Some(old) = previous
        && old.refused
    {
        if refused && old.body_hash == delivery.body_hash.to_string() {
            return Ok(false);
        }
        return invalid(
            "a refused immutable source cannot change its callback without a new actual source length",
        );
    }
    let first = if let Some(old) = previous {
        let cursor = old.cursor.as_ref().ok_or_else(|| {
            StoreError::Invalid(
                "historical byte control cannot authenticate an export callback prefix".into(),
            )
        })?;
        cursor
            .validate()
            .map_err(|error| StoreError::Invalid(error.to_string()))?;
        if refused {
            return invalid("an unreadable scan cannot replace a retained export callback prefix");
        }
        if delivery.ordinal <= cursor.last_ordinal {
            if delivery.ordinal == cursor.last_ordinal && delivery.prefix_hash != cursor.prefix_hash
            {
                return invalid("replayed export prefix differs from the committed callback body");
            }
            if matches!(delivery.source, ExportDeliverySource::Complete { .. })
                && delivery.ordinal != cursor.last_ordinal
            {
                return invalid("export replay completed before its retained callback prefix");
            }
            return Ok(false);
        }
        if matches!(
            old.last,
            ReadingFootprint::Export {
                source: ExportDeliverySource::Complete { .. }
            }
        ) {
            return invalid("a completed immutable export cannot acquire another callback");
        }
        cursor
            .advance(delivery)
            .map_err(|error| StoreError::Invalid(error.to_string()))?;
        if step.reset {
            return invalid("an export continuation cannot reset committed callback effects");
        }
        old.first
    } else {
        ExportCursor::from_first(delivery)
            .map_err(|error| StoreError::Invalid(error.to_string()))?;
        if !step.reset {
            return invalid("the first export callback must begin its reading");
        }
        ReadingFootprint::Export {
            source: delivery.source,
        }
    };
    if let ExportDeliverySource::Complete {
        item_count,
        delivery_count,
        ..
    } = delivery.source
    {
        let headers = u64::from(matches!(
            first,
            ReadingFootprint::Export {
                source: ExportDeliverySource::ContainerHeader
            }
        ));
        if delivery_count != delivery.ordinal
            || item_count.checked_add(headers) != Some(delivery.ordinal)
        {
            return invalid("export completion does not account for its committed callbacks");
        }
    }
    Ok(true)
}

#[cfg(test)]
mod tests;
