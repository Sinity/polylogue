//! Validate the complete immutable package before any source or authored write.
use super::*;
use polylogue_model::bundle::{Counts, InventoryRecord, MemberKind, Record};
use std::collections::{BTreeMap, BTreeSet};

fn verify(
    member: &package::Member,
    file: &polylogue_model::bundle::FileDescriptor,
) -> std::io::Result<()> {
    let mut source = member.bytes.reader().map_err(std::io::Error::other)?;
    let mut hash = blake3::Hasher::new();
    let size = std::io::copy(&mut source, &mut hash)?;
    if hash.finalize().as_bytes() != &file.hash.0 || size != file.size {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "bundle member {} differs from its declared hash or size",
                file.path
            ),
        ));
    }
    Ok(())
}
fn stage_record(
    items: &mut polylogue_store::export_items::ExportItems,
    range: std::ops::Range<u64>,
    mut record: Record,
) -> Result<(), String> {
    if let Record::Annotation(annotation) = &mut record {
        annotation.annotation = annotation
            .annotation
            .canonicalized()
            .map_err(|error| error.to_string())?;
    }
    if let Record::Authored(authored) = &record {
        if authored.prepared.outcome.pending {
            return Err("a bundle cannot contain an unconfirmed authored outcome".into());
        }
        polylogue_store::authored_records::validate(&authored.prepared)
            .map_err(|error| error.to_string())?;
    }
    let Some(reference) = record.reference() else {
        return Err("a bundle contains a second manifest".into());
    };
    let mut key = format!("{}:{}", record.commit().unwrap_or(0), reference);
    if let Record::TextFragment(fragment) = &record {
        // A span is shared by source versions and independent declarations.
        // Keep conflicting totals and sources as stated values for the Store
        // reader; selected-text identity alone must not replace or veto either.
        let raw = serde_json::to_vec(&record).map_err(|error| error.to_string())?;
        key.push_str(&format!(
            ":{}:{}",
            fragment.fragment.content_key(),
            polylogue_model::BlobHash::of(&raw)
        ));
    }
    if let Some(previous) = items.source_item(&key).map_err(|error| error.to_string())? {
        let previous: Record =
            serde_json::from_reader(previous.reader).map_err(|error| error.to_string())?;
        match (&mut record, previous) {
            (Record::Event(current), Record::Event(previous))
                if current.event == previous.event
                    && current.identity.id == previous.identity.id =>
            {
                current.identity.sources.extend(previous.identity.sources);
                current.identity.sources.sort();
                current.identity.sources.dedup();
            }
            (current, previous) if *current == previous => {}
            _ => return Err(format!("bundle has conflicting records for {key}")),
        }
    }
    let raw = serde_json::to_vec(&record).map_err(|error| error.to_string())?;
    items
        .push(range, key.as_bytes().to_vec(), Some(&key), &raw)
        .map_err(|error| error.to_string())
}
fn count(counts: &mut Counts, record: &Record) {
    match record {
        Record::Session(_) => counts.sessions += 1,
        Record::Message(_) => counts.messages += 1,
        Record::Event(_) => counts.events += 1,
        Record::Document(_) => counts.documents += 1,
        Record::Link(_) => counts.links += 1,
        Record::Authored(_) => counts.authored += 1,
        Record::Annotation(_) => counts.annotations += 1,
        Record::TextFragment(_) => counts.text_fragments += 1,
        Record::Alternate(_) => counts.alternates += 1,
        Record::AlternatePart(_) => counts.alternate_parts += 1,
        Record::Blob(_) => counts.blobs += 1,
        Record::Manifest(_) => {}
    }
}

pub(super) fn preflight(members: &mut [package::Member]) -> std::io::Result<()> {
    let is_bundle_member = |member: &package::Member| {
        member
            .origin
            .is_some_and(|origin| origin.name == polylogue_sources::polylogue_bundle::ORIGIN)
    };
    let primary: Vec<_> = members
        .iter()
        .enumerate()
        .filter(|(_, member)| is_bundle_member(member))
        .filter_map(|(index, member)| {
            member
                .bytes
                .reader()
                .ok()
                .and_then(|file| polylogue_sources::polylogue_bundle::manifest(file).ok())
                .map(|_| index)
        })
        .collect();
    if primary.is_empty() && members.iter().any(is_bundle_member) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "a bundle is missing its primary manifest",
        ));
    }
    let names: BTreeMap<_, _> = members
        .iter()
        .enumerate()
        .map(|(index, member)| (member.path.clone(), index))
        .collect();
    let mut claimed = BTreeSet::new();
    for primary in primary {
        if claimed.contains(&primary) {
            continue;
        }
        let root = members[primary]
            .path
            .parent()
            .unwrap_or(Path::new(""))
            .to_path_buf();
        let manifest = polylogue_sources::polylogue_bundle::manifest(
            members[primary]
                .bytes
                .reader()
                .map_err(std::io::Error::other)?,
        )
        .map_err(|message| std::io::Error::new(std::io::ErrorKind::InvalidData, message))?;
        let mut records = vec![primary];
        let mut listed = BTreeSet::from([primary]);
        if let Some(inventory) = &manifest.inventory {
            let inventory_index = *names.get(&root.join(&inventory.path)).ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "bundle inventory is missing",
                )
            })?;
            verify(&members[inventory_index], inventory)?;
            listed.insert(inventory_index);
            members[inventory_index].origin = polylogue_sources::origin::by_name(KEPT);
            let input = std::io::BufReader::new(
                members[inventory_index]
                    .bytes
                    .reader()
                    .map_err(std::io::Error::other)?,
            );
            for entry in serde_json::Deserializer::from_reader(input).into_iter::<InventoryRecord>()
            {
                let entry = entry.map_err(std::io::Error::other)?;
                entry.file.validate().map_err(|error| {
                    std::io::Error::new(std::io::ErrorKind::InvalidData, error.to_string())
                })?;
                let index = *names.get(&root.join(&entry.file.path)).ok_or_else(|| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("bundle member {} is missing", entry.file.path),
                    )
                })?;
                if !listed.insert(index) {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "bundle inventory repeats a member or lists its primary/inventory",
                    ));
                }
                verify(&members[index], &entry.file)?;
                if entry.kind == MemberKind::Records {
                    records.push(index);
                    members[index].origin = polylogue_sources::origin::by_name(
                        polylogue_sources::polylogue_bundle::ORIGIN,
                    );
                } else {
                    members[index].origin = polylogue_sources::origin::by_name(KEPT);
                }
            }
        }
        let mut staged =
            polylogue_store::export_items::ExportItems::new().map_err(std::io::Error::other)?;
        for index in records {
            let mut first = index == primary;
            let mut reader = std::io::BufReader::new(
                members[index]
                    .bytes
                    .reader()
                    .map_err(std::io::Error::other)?,
            );
            polylogue_sources::polylogue_bundle::visit(&mut reader, |range, record, _| {
                if first {
                    first = false;
                    if matches!(record, Record::Manifest(_)) {
                        return Ok(());
                    }
                    return Err("bundle primary does not start with its manifest".into());
                }
                if matches!(record, Record::Alternate(_) | Record::AlternatePart(_)) {
                    return Err(
                        "bundle alternate restoration requires the historical alternate import"
                            .into(),
                    );
                }
                if let Record::Blob(blob) = &record {
                    let member = *names
                        .get(&root.join(&blob.file.path))
                        .ok_or_else(|| format!("bundle blob {} is missing", blob.file.path))?;
                    verify(&members[member], &blob.file).map_err(|error| error.to_string())?;
                    if manifest.inventory.is_some() && !listed.contains(&member) {
                        return Err("bundle blob is absent from its pinned inventory".into());
                    }
                    listed.insert(member);
                }
                stage_record(&mut staged, range, record)
            })
            .map_err(std::io::Error::other)?;
        }
        let mut actual = Counts::default();
        while let Some(item) = staged.next_item().map_err(std::io::Error::other)? {
            let record: Record =
                serde_json::from_reader(item.reader).map_err(std::io::Error::other)?;
            count(&mut actual, &record);
        }
        if actual != manifest.counts {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "bundle record counts differ from the manifest",
            ));
        }
        for (index, member) in members.iter().enumerate() {
            if !member.container
                && member.path.starts_with(&root)
                && member.parent.map(|p| p.0) == members[primary].parent.map(|p| p.0)
                && !listed.contains(&index)
            {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("bundle member {} is not declared", member.path.display()),
                ));
            }
        }
        claimed.extend(listed);
    }
    Ok(())
}

#[cfg(test)]
mod tests;
