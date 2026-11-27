//! One stored export cursor checks each callback before its parsed rows apply.
use super::*;
use crate::ConnExt as _;

pub(crate) fn is_refusal(step: &ParsePart) -> bool {
    step.failure
        .as_ref()
        .is_some_and(|failure| failure.kind == polylogue_model::ChainFailureKind::Refused)
}

fn load_prefix(
    conn: &Connection,
    input: &str,
    chain: i64,
    version: &str,
) -> Result<Option<Reading>> {
    let previous: Option<String> = conn.scalar_optional(
        "SELECT w.state FROM wire_parse w JOIN input_ref i ON i.pk=w.input_pk WHERE i.id=?1 AND w.chain=?2 AND w.parser_version=?3",
        params![input,chain,version])?;
    previous
        .map(|json| Ok(serde_json::from_str(&json)?))
        .transpose()
}

pub(crate) fn admit_export(
    conn: &Connection,
    input: &str,
    chain: i64,
    length: u64,
    step: &ParsePart,
) -> Result<bool> {
    accept(
        load_prefix(conn, input, chain, &step.parse_version)?.as_ref(),
        length,
        step,
    )
}

fn accept(previous: Option<&Reading>, length: u64, step: &ParsePart) -> Result<bool> {
    // A failure carries no records and moves no cursor, so it switches
    // neither an export reading nor a byte-range reading.
    if step.failure.is_some() && !is_refusal(step) {
        return Ok(true);
    }
    let Some(delivery) = step.delivery.as_ref() else {
        if previous.is_some_and(|old| old.export_cursor.is_some() || old.refused_export) {
            return Err(StoreError::internal(
                "an export reading cannot switch to byte ranges",
            ));
        }
        return Ok(true);
    };
    step.validate_export()?;
    match delivery.source {
        ExportDeliverySource::Item { end, .. } if end > length => {
            return Err(StoreError::internal(
                "export item lies outside its immutable stored source",
            ));
        }
        ExportDeliverySource::Complete {
            length: scanned, ..
        } => {
            if scanned != length || step.start != 0 || step.parsed_upto != length || !step.at_end {
                return Err(StoreError::internal(
                    "export completion differs from its verified source length",
                ));
            }
        }
        _ if step.start != 0 || step.parsed_upto != 0 || step.at_end => {
            return Err(StoreError::internal(
                "an export callback cannot acknowledge a partial byte cursor",
            ));
        }
        _ => {}
    }
    let refused = is_refusal(step);
    if refused && (step.output != polylogue_model::Output::default() || step.resume.is_some()) {
        return Err(StoreError::internal(
            "an unreadable export header cannot carry facts or a completion checkpoint",
        ));
    }
    let previous = match previous {
        Some(old) if old.length < length && delivery.ordinal == 0 => None,
        Some(old) if old.length != length => {
            return Err(StoreError::internal(
                "export control belongs to another stored source length",
            ));
        }
        other => other,
    };
    if let Some(old) = previous
        && old.refused_export
    {
        if refused && old.last_hash == delivery.body_hash.to_string() {
            return Ok(false);
        }
        return Err(StoreError::internal(
            "a refused immutable source cannot change its callback without a new source length",
        ));
    }
    let first = if let Some(old) = previous {
        let cursor = old.export_cursor.as_ref().ok_or_else(|| {
            StoreError::internal(
                "historical byte control cannot authenticate an export callback prefix",
            )
        })?;
        cursor.validate().map_err(crate::StoreError::internal)?;
        if refused {
            return Err(StoreError::internal(
                "an unreadable scan cannot replace a stored export callback prefix",
            ));
        }
        if delivery.ordinal <= cursor.last_ordinal {
            if delivery.ordinal == cursor.last_ordinal && delivery.prefix_hash != cursor.prefix_hash
            {
                return Err(StoreError::internal(
                    "replayed export prefix differs from the committed callback body",
                ));
            }
            if matches!(delivery.source, ExportDeliverySource::Complete { .. })
                && delivery.ordinal != cursor.last_ordinal
            {
                return Err(StoreError::internal(
                    "export replay completed before its kept callback prefix",
                ));
            }
            return Ok(false);
        }
        if matches!(
            old.last.covered,
            ReadRange::Export {
                source: ExportDeliverySource::Complete { .. }
            }
        ) {
            return Err(StoreError::internal(
                "a completed immutable export cannot acquire another callback",
            ));
        }
        cursor
            .advance(delivery)
            .map_err(crate::StoreError::internal)?;
        if step.reset {
            return Err(StoreError::internal(
                "an export continuation cannot reset committed callback effects",
            ));
        }
        old.first.covered
    } else {
        ExportCursor::from_first(delivery).map_err(crate::StoreError::internal)?;
        if !step.reset {
            return Err(StoreError::internal(
                "the first export callback must begin its reading",
            ));
        }
        ReadRange::Export {
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
            ReadRange::Export {
                source: ExportDeliverySource::ContainerHeader
            }
        ));
        if delivery_count != delivery.ordinal
            || item_count.checked_add(headers) != Some(delivery.ordinal)
        {
            return Err(StoreError::internal(
                "export completion does not account for its committed callbacks",
            ));
        }
    }
    Ok(true)
}

#[cfg(test)]
mod tests;
