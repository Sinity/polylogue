//! Validate the complete immutable package before any source or authored write.
use super::*;
use polylogue_model::bundle::{Counts, InventoryRecord, MemberKind, Record};
use std::collections::{BTreeMap, BTreeSet};
use std::io::Read;

fn invalid(message: impl Into<String>) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, message.into())
}
fn verify(
    member: &package::Member,
    file: &polylogue_model::bundle::FileDescriptor,
) -> std::io::Result<()> {
    let mut source = std::fs::File::open(&member.bytes)?;
    let mut hash = blake3::Hasher::new();
    let mut bytes = [0; 65536];
    let mut size = 0u64;
    loop {
        let n = source.read(&mut bytes)?;
        if n == 0 {
            break;
        }
        hash.update(&bytes[..n]);
        size += n as u64;
    }
    if hash.finalize().as_bytes() != &file.hash.0 || size != file.size {
        return Err(invalid(format!(
            "bundle member {} differs from its declared hash or size",
            file.path
        )));
    }
    Ok(())
}
fn stage_record(
    items: &mut polylogue_store::export_items::ExportItems,
    range: std::ops::Range<u64>,
    mut record: Record,
) -> Result<(), String> {
    if let Record::Authored(authored) = &record {
        if authored.prepared.outcome.pending {
            return Err("a bundle cannot contain an unconfirmed authored outcome".into());
        }
        polylogue_store::primary::authored::validate(&authored.prepared)
            .map_err(|error| error.to_string())?;
    }
    let Some(reference) = record.reference() else {
        return Err("a bundle contains a second manifest".into());
    };
    let key = format!("{}:{}", record.commit().unwrap_or(0), reference);
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
            (Record::Authored(current), Record::Authored(previous))
                if current.prepared == previous.prepared =>
            {
                for claim in previous.imports {
                    if !current.imports.contains(&claim) {
                        current.imports.push(claim);
                    }
                }
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
        Record::Variant(_) => counts.variants += 1,
        Record::VariantPart(_) => counts.variant_parts += 1,
        Record::Blob(_) => counts.blobs += 1,
        Record::Manifest(_) => {}
    }
}

pub(super) fn preflight(members: &mut [package::Member]) -> std::io::Result<()> {
    let primary: Vec<_> = members
        .iter()
        .enumerate()
        .filter(|(_, member)| {
            member
                .origin
                .is_some_and(|origin| origin.name == "polylogue-bundle")
        })
        .filter_map(|(index, member)| {
            std::fs::File::open(&member.bytes)
                .ok()
                .and_then(|file| polylogue_sources::polylogue_bundle::manifest(file).ok())
                .map(|_| index)
        })
        .collect();
    if primary.is_empty()
        && members.iter().any(|member| {
            member
                .origin
                .is_some_and(|origin| origin.name == "polylogue-bundle")
        })
    {
        return Err(invalid("a bundle is missing its primary manifest"));
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
        let manifest = polylogue_sources::polylogue_bundle::manifest(std::fs::File::open(
            &members[primary].bytes,
        )?)
        .map_err(invalid)?;
        let mut records = vec![primary];
        let mut listed = BTreeSet::from([primary]);
        if let Some(inventory) = &manifest.inventory {
            let inventory_index = *names
                .get(&root.join(&inventory.path))
                .ok_or_else(|| invalid("bundle inventory is missing"))?;
            verify(&members[inventory_index], inventory)?;
            listed.insert(inventory_index);
            members[inventory_index].origin = polylogue_sources::origin::by_name(KEPT);
            let input =
                std::io::BufReader::new(std::fs::File::open(&members[inventory_index].bytes)?);
            for entry in serde_json::Deserializer::from_reader(input).into_iter::<InventoryRecord>()
            {
                let entry = entry.map_err(std::io::Error::other)?;
                entry
                    .file
                    .validate()
                    .map_err(|error| invalid(error.to_string()))?;
                let index = *names.get(&root.join(&entry.file.path)).ok_or_else(|| {
                    invalid(format!("bundle member {} is missing", entry.file.path))
                })?;
                if !listed.insert(index) {
                    return Err(invalid(
                        "bundle inventory repeats a member or lists its primary/inventory",
                    ));
                }
                verify(&members[index], &entry.file)?;
                if entry.kind == MemberKind::Records {
                    records.push(index);
                    members[index].origin = polylogue_sources::origin::by_name("polylogue-bundle");
                } else {
                    members[index].origin = polylogue_sources::origin::by_name(KEPT);
                }
            }
        }
        let mut staged =
            polylogue_store::export_items::ExportItems::new().map_err(std::io::Error::other)?;
        for index in records {
            let mut first = index == primary;
            let mut reader = std::io::BufReader::new(std::fs::File::open(&members[index].bytes)?);
            polylogue_sources::polylogue_bundle::visit(&mut reader, |range, record, _| {
                if first {
                    first = false;
                    if matches!(record, Record::Manifest(_)) {
                        return Ok(());
                    }
                    return Err("bundle primary does not start with its manifest".into());
                }
                if matches!(record, Record::Variant(_) | Record::VariantPart(_)) {
                    return Err(
                        "bundle variant restoration requires the canonical historical variant seam"
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
            return Err(invalid("bundle record counts differ from the manifest"));
        }
        for (index, member) in members.iter().enumerate() {
            if !member.container
                && member.path.starts_with(&root)
                && member.parent.map(|p| p.0) == members[primary].parent.map(|p| p.0)
                && !listed.contains(&index)
            {
                return Err(invalid(format!(
                    "bundle member {} is not declared",
                    member.path.display()
                )));
            }
        }
        claimed.extend(listed);
    }
    Ok(())
}
