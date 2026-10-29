//! Client-owned, staged directory delivery over canonical read records.
use polylogue_model::{BlobHash, SessionId, bundle::*, read::*};
use polylogue_store::export_items::{ExportItems, Order};
use std::{
    fs::{self, File, OpenOptions},
    io::{self, BufRead, BufReader, Read, Write},
    path::{Path, PathBuf},
};

#[derive(Debug, thiserror::Error)]
#[error("{message}; staged bundle retained at {staging}")]
pub struct Error {
    pub message: String,
    pub staging: PathBuf,
}
type Result<T> = std::result::Result<T, Error>;
#[derive(Debug)]
pub struct Summary {
    pub destination: PathBuf,
    pub counts: Counts,
    pub previous: Option<PathBuf>,
}

/// The staging directory is retained on every unsuccessful path, including
/// disconnect/drop. Only publication transfers its ownership to the destination.
pub struct Bundle {
    destination: PathBuf,
    staging: PathBuf,
    begin: ReadBegin,
    format: Format,
    with_blobs: bool,
    items: ExportItems,
    received: Counts,
    failed: bool,
}
impl Bundle {
    pub fn new(
        destination: &Path,
        begin: ReadBegin,
        format: Format,
        with_blobs: bool,
    ) -> Result<Self> {
        let parent = destination
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        let staging = tempfile::Builder::new()
            .prefix(".polylogue-bundle-")
            .tempdir_in(parent)
            .map_err(|e| Error {
                message: e.to_string(),
                staging: parent.to_owned(),
            })?
            .keep();
        let build = || -> std::result::Result<ExportItems, String> {
            begin.options.validate().map_err(|e| e.to_string())?;
            Manifest::from_snapshot(
                begin.snapshot.clone(),
                Counts::default(),
                Flags {
                    format,
                    with_blobs,
                    read: begin.options.clone(),
                },
                None,
            )
            .validate()
            .map_err(|e| e.to_string())?;
            check_destination(destination)?;
            ExportItems::new().map_err(|e| e.to_string())
        };
        let items = build().map_err(|message| Error {
            message,
            staging: staging.clone(),
        })?;
        Ok(Self {
            destination: destination.to_owned(),
            staging,
            begin,
            format,
            with_blobs,
            items,
            received: Counts::default(),
            failed: false,
        })
    }
    pub fn staging(&self) -> &Path {
        &self.staging
    }
    fn error(&mut self, message: impl ToString) -> Error {
        self.failed = true;
        Error {
            message: message.to_string(),
            staging: self.staging.clone(),
        }
    }
    pub fn push(&mut self, mut item: ReadRecord) -> Result<()> {
        if self.failed {
            return Err(self.error("bundle delivery already failed"));
        }
        if let Err(e) = item.record.validate() {
            return Err(self.error(e));
        }
        if matches!(item.record, Record::Manifest(_)) {
            return Err(self.error("manifest inside record stream"));
        }
        increment(&mut self.received, &item.record);
        let reference = item.record.reference().expect("non-manifest").to_owned();
        let old = self
            .items
            .source_item(&reference)
            .map_err(|e| self.error(e))?;
        if let Some(old) = old {
            let previous: ReadRecord =
                serde_json::from_reader(old.reader).map_err(|e| self.error(e))?;
            match (&mut item.record, &previous.record) {
                (Record::Message(new), Record::Message(old))
                    if new.commit == old.commit && new.order == old.order =>
                {
                    let merged = old
                        .projection()
                        .merge(&old.message, &new.projection(), &new.message)
                        .map_err(|e| self.error(e))?;
                    let Some((message, projection)) = merged else {
                        return Err(self.error("conflicting message source versions"));
                    };
                    new.message = message;
                    new.source_version = projection.source_version;
                    new.block_count = projection.block_count;
                    new.block_ordinals = projection.block_ordinals;
                    // Repeated content cannot add a second rendering of the same item.
                    let mut parts = previous.parts;
                    for part in item.parts {
                        if !parts.contains(&part) {
                            parts.push(part);
                        }
                    }
                    parts.sort_by_key(|part| match part {
                        ReadPart::Header(_) => (0, 0),
                        ReadPart::Item(item) => (1, item.source.block.unwrap_or(0)),
                        ReadPart::Footer(_) => (2, 0),
                    });
                    item.parts = parts;
                }
                (new, old) if *new == *old && item.parts == previous.parts => {
                    return Ok(());
                }
                _ => return Err(self.error("conflicting duplicate bundle reference")),
            }
        }
        let bytes = serde_json::to_vec(&item).map_err(|e| self.error(e))?;
        let key = order_key(&item.record);
        let mut alternate = group(&item.record).into_bytes();
        alternate.push(0);
        alternate.extend_from_slice(&key);
        self.items
            .push_ordered(
                0..bytes.len() as u64,
                key,
                alternate,
                Some(&reference),
                &bytes,
            )
            .map_err(|e| self.error(e))
    }
    /// Chunks are contiguous per hash; each descriptor is checked at finish.
    pub fn blob_chunk(&mut self, hash: BlobHash, offset: u64, bytes: &[u8]) -> Result<()> {
        if !self.with_blobs || self.failed {
            return Err(self.error("unexpected blob bytes"));
        }
        let path = self.staging.join("blobs").join(hash.to_string());
        fs::create_dir_all(path.parent().expect("blob parent")).map_err(|e| self.error(e))?;
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map_err(|e| self.error(e))?;
        if file.metadata().map_err(|e| self.error(e))?.len() != offset {
            return Err(self.error("noncontiguous blob chunk"));
        }
        file.write_all(bytes).map_err(|e| self.error(e))
    }
    pub fn finish(mut self, complete: ReadComplete) -> Result<Summary> {
        if self.failed {
            return Err(self.error("bundle delivery already failed"));
        }
        if complete.counts != self.received {
            return Err(self.error("final read counts differ from received records"));
        }
        if complete.footer.next.is_some() {
            return Err(self.error("read completed with a continuation"));
        }
        self.write_members(&complete).map_err(|e| self.error(e))?;
        check_destination(&self.destination).map_err(|e| self.error(e))?;
        let parent = self
            .destination
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        let parent_file = File::open(parent).map_err(|e| self.error(e))?;
        let replacing = self.destination.exists();
        if replacing {
            rustix::fs::renameat_with(
                rustix::fs::CWD,
                &self.staging,
                rustix::fs::CWD,
                &self.destination,
                rustix::fs::RenameFlags::EXCHANGE,
            )
            .map_err(|e| self.error(e))?;
            if let Err(error) = parent_file.sync_all() {
                if let Err(rollback) = rustix::fs::renameat_with(
                    rustix::fs::CWD,
                    &self.staging,
                    rustix::fs::CWD,
                    &self.destination,
                    rustix::fs::RenameFlags::EXCHANGE,
                ) {
                    return Err(self.error(format!("published directory sync failed: {error}; rollback failed: {rollback}; inspect destination and staging")));
                }
                return Err(self.error(error));
            }
            // The previous complete bundle remains recoverable at staging.
        } else {
            rustix::fs::renameat_with(
                rustix::fs::CWD,
                &self.staging,
                rustix::fs::CWD,
                &self.destination,
                rustix::fs::RenameFlags::NOREPLACE,
            )
            .map_err(|e| self.error(e))?;
            if let Err(error) = parent_file.sync_all() {
                if let Err(rollback) = fs::rename(&self.destination, &self.staging) {
                    return Err(self.error(format!("published directory sync failed: {error}; rollback failed: {rollback}; inspect destination and staging")));
                }
                return Err(self.error(error));
            }
        }
        Ok(Summary {
            destination: self.destination,
            counts: complete.counts,
            previous: replacing.then_some(self.staging),
        })
    }
    fn next(&mut self) -> std::result::Result<Option<ReadRecord>, String> {
        self.items
            .next_item()
            .map_err(|e| e.to_string())?
            .map(|i| serde_json::from_reader(i.reader).map_err(|e| e.to_string()))
            .transpose()
    }
    fn write_members(&mut self, complete: &ReadComplete) -> std::result::Result<(), String> {
        let mut inventory = File::create(self.staging.join("members.jsonl")).map_err(string)?;
        let mut counts = Counts::default();
        self.items.reset(Order::Primary).map_err(string)?;
        while let Some(item) = self.next()? {
            increment(&mut counts, &item.record);
            if let Record::Blob(blob) = &item.record {
                if !self.with_blobs {
                    return Err("blob descriptor without with_blobs".into());
                }
                let expected = format!("blobs/{}", blob.file.hash);
                if blob.file.path != expected {
                    return Err("noncanonical blob path".into());
                }
                let actual = describe(&self.staging, &expected)?;
                if actual != blob.file {
                    return Err("blob bytes do not match descriptor".into());
                }
                line(
                    &mut inventory,
                    &InventoryRecord {
                        kind: MemberKind::Blob,
                        reference: Some(blob.reference.clone()),
                        file: actual,
                    },
                )?;
            }
        }
        if self.staging.join("blobs").exists() {
            for entry in fs::read_dir(self.staging.join("blobs")).map_err(string)? {
                let entry = entry.map_err(string)?;
                let name = entry
                    .file_name()
                    .into_string()
                    .map_err(|_| "invalid blob name")?;
                let reference = format!("blob:{name}");
                if !entry.file_type().map_err(string)?.is_file()
                    || self
                        .items
                        .source_item(&reference)
                        .map_err(string)?
                        .is_none()
                {
                    return Err("blob bytes have no canonical descriptor".into());
                }
            }
        }
        let mut manifest = Manifest::from_snapshot(
            self.begin.snapshot.clone(),
            counts,
            Flags {
                format: self.format,
                with_blobs: self.with_blobs,
                read: self.begin.options.clone(),
            },
            None,
        );
        manifest.coverage = complete.footer.coverage.clone();
        if matches!(self.format, Format::Markdown | Format::Html) {
            let mut records = File::create(self.staging.join("records.jsonl")).map_err(string)?;
            self.items.reset(Order::Primary).map_err(string)?;
            while let Some(item) = self.next()? {
                line(&mut records, &item.record)?;
            }
            records.sync_all().map_err(string)?;
            if self.format != Format::Jsonl {
                line(
                    &mut inventory,
                    &InventoryRecord {
                        kind: MemberKind::Records,
                        reference: None,
                        file: describe(&self.staging, "records.jsonl")?,
                    },
                )?;
            }
        }
        if self.format != Format::Jsonl {
            self.items.reset(Order::Alternate).map_err(string)?;
            let mut pending = self.next()?;
            let rendered = matches!(self.format, Format::Markdown | Format::Html);
            let mut index = if rendered {
                Some(
                    File::create(self.staging.join(if self.format == Format::Html {
                        "index.html"
                    } else {
                        "index.md"
                    }))
                    .map_err(string)?,
                )
            } else {
                None
            };
            if let Some(index) = &mut index {
                index
                    .write_all(if self.format == Format::Html {
                        b"<!doctype html><meta charset=\"utf-8\"><ul>"
                    } else {
                        b"# Sessions\n\n"
                    })
                    .map_err(string)?;
            }
            while let Some(first) = pending.take() {
                let name = group(&first.record);
                let records_path = format!("{name}.json");
                let absolute = self.staging.join(&records_path);
                fs::create_dir_all(absolute.parent().expect("member parent")).map_err(string)?;
                let mut file = if rendered {
                    None
                } else {
                    Some(File::create(absolute).map_err(string)?)
                };
                if let Some(file) = &mut file {
                    file.write_all(b"[").map_err(string)?;
                }
                let mut rendered_file = if rendered {
                    Some(
                        File::create(self.staging.join(format!(
                            "{name}.{}",
                            if self.format == Format::Html {
                                "html"
                            } else {
                                "md"
                            }
                        )))
                        .map_err(string)?,
                    )
                } else {
                    None
                };
                let mut header = self.begin.header.clone();
                header.target = first.record.reference().unwrap_or("").to_owned();
                if let Record::Session(s) = &first.record {
                    header.kind = HeaderKind::Session {
                        session: Box::new(s.session.clone()),
                    };
                    header.target = s.reference.clone();
                    header.title = s
                        .session
                        .title
                        .as_ref()
                        .map(|t| ValueState::known(t.text.clone()))
                        .unwrap_or_default();
                }
                let mut encoder = rendered_file
                    .as_mut()
                    .map(|file| {
                        polylogue_render::Encoder::new(
                            &header,
                            if self.format == Format::Html {
                                polylogue_render::Format::Html
                            } else {
                                polylogue_render::Format::Markdown
                            },
                            polylogue_render::EncodeOptions {
                                standalone_html: true,
                                ..Default::default()
                            },
                            file,
                        )
                    })
                    .transpose()
                    .map_err(string)?;
                let mut current = Some(first);
                let mut separator = false;
                while let Some(item) = current.take() {
                    if let Some(file) = &mut file {
                        if separator {
                            file.write_all(b",").map_err(string)?;
                        }
                        serde_json::to_writer(file, &item.record).map_err(string)?;
                    }
                    separator = true;
                    if let Some(encoder) = &mut encoder {
                        for part in item.parts {
                            if let ReadPart::Item(item) = part {
                                encoder.push(&item).map_err(string)?;
                            }
                        }
                    }
                    let next = self.next()?;
                    match next {
                        Some(next) if group(&next.record) == name => current = Some(next),
                        other => pending = other,
                    }
                }
                if let Some(encoder) = encoder {
                    encoder.finish(&complete.footer).map_err(string)?;
                }
                if let Some(mut file) = file {
                    file.write_all(b"]\n").map_err(string)?;
                    file.sync_all().map_err(string)?;
                    line(
                        &mut inventory,
                        &InventoryRecord {
                            kind: MemberKind::Records,
                            reference: None,
                            file: describe(&self.staging, &records_path)?,
                        },
                    )?;
                }
                if let Some(file) = rendered_file {
                    file.sync_all().map_err(string)?;
                    let path = format!(
                        "{name}.{}",
                        if self.format == Format::Html {
                            "html"
                        } else {
                            "md"
                        }
                    );
                    line(
                        &mut inventory,
                        &InventoryRecord {
                            kind: MemberKind::Rendered,
                            reference: Some(header.target.clone()),
                            file: describe(&self.staging, &path)?,
                        },
                    )?;
                    if let Some(index) = &mut index {
                        if self.format == Format::Html {
                            writeln!(index, "<li><a href=\"{path}\">{path}</a></li>")
                                .map_err(string)?;
                        } else {
                            writeln!(index, "- [{path}]({path})").map_err(string)?;
                        }
                    }
                }
            }
            if let Some(mut index) = index {
                let path = if self.format == Format::Html {
                    index.write_all(b"</ul>\n").map_err(string)?;
                    "index.html"
                } else {
                    "index.md"
                };
                index.sync_all().map_err(string)?;
                line(
                    &mut inventory,
                    &InventoryRecord {
                        kind: MemberKind::Rendered,
                        reference: None,
                        file: describe(&self.staging, path)?,
                    },
                )?;
            }
        }
        inventory.sync_all().map_err(string)?;
        manifest.inventory = Some(describe(&self.staging, "members.jsonl")?);
        manifest.validate().map_err(string)?;
        let primary_path = if self.format == Format::Jsonl {
            "records.jsonl"
        } else {
            "manifest.json"
        };
        let mut primary = File::create(self.staging.join(primary_path)).map_err(string)?;
        line(&mut primary, &Record::Manifest(Box::new(manifest)))?;
        if self.format == Format::Jsonl {
            self.items.reset(Order::Primary).map_err(string)?;
            while let Some(item) = self.next()? {
                line(&mut primary, &item.record)?;
            }
        }
        primary.sync_all().map_err(string)?;
        validate_inventory(&self.staging)?;
        sync_directories(&self.staging)?;
        Ok(())
    }
}
fn string(e: impl ToString) -> String {
    e.to_string()
}
fn line(file: &mut File, value: &impl serde::Serialize) -> std::result::Result<(), String> {
    serde_json::to_writer(&mut *file, value).map_err(string)?;
    file.write_all(b"\n").map_err(string)
}
fn order_key(record: &Record) -> Vec<u8> {
    let mut key = record.commit().unwrap_or(u64::MAX).to_be_bytes().to_vec();
    key.extend_from_slice(record.reference().unwrap_or("").as_bytes());
    key
}
fn group(record: &Record) -> String {
    let session = match record {
        Record::Session(r) => Some(&r.session.id),
        Record::Message(r) => Some(&r.message.session),
        Record::Event(r) => r.event.session.as_ref(),
        Record::Link(r) => Some(&r.link.child),
        _ => None,
    };
    session.map(session_path).unwrap_or_else(|| {
        format!(
            "_records/{}",
            blake3::hash(record.reference().unwrap_or("").as_bytes()).to_hex()
        )
    })
}
fn session_path(id: &SessionId) -> String {
    let mut name = String::new();
    for byte in id.native().as_str().bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_') {
            name.push(char::from(byte));
        } else {
            name.push_str(&format!("%{byte:02X}"));
        }
    }
    if name.len() > 180 {
        name = format!(
            "~{}",
            blake3::hash(id.native().as_str().as_bytes()).to_hex()
        );
    }
    format!("{}/{name}", id.origin())
}
fn increment(counts: &mut Counts, record: &Record) {
    match record {
        Record::Manifest(_) => {}
        Record::Session(_) => counts.sessions += 1,
        Record::Message(_) => counts.messages += 1,
        Record::Document(_) => counts.documents += 1,
        Record::Event(_) => counts.events += 1,
        Record::Link(_) => counts.links += 1,
        Record::Authored(_) => counts.authored += 1,
        Record::Variant(_) => counts.variants += 1,
        Record::VariantPart(_) => counts.variant_parts += 1,
        Record::Blob(_) => counts.blobs += 1,
    }
}
fn describe(root: &Path, path: &str) -> std::result::Result<FileDescriptor, String> {
    let mut file = File::open(root.join(path)).map_err(string)?;
    let mut hash = blake3::Hasher::new();
    let mut buffer = [0; 65536];
    let mut size = 0;
    loop {
        let n = file.read(&mut buffer).map_err(string)?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
        size += n as u64;
    }
    Ok(FileDescriptor {
        path: path.into(),
        hash: BlobHash::parse(hash.finalize().to_hex().as_ref()).map_err(string)?,
        size,
    })
}
fn validate_inventory(root: &Path) -> std::result::Result<(), String> {
    let reader = BufReader::new(File::open(root.join("members.jsonl")).map_err(string)?);
    for line in reader.lines() {
        let item: InventoryRecord = serde_json::from_str(&line.map_err(string)?).map_err(string)?;
        item.file.validate().map_err(string)?;
        if item.file.path == "members.jsonl" || item.file.path == "manifest.json" {
            return Err("recursive inventory member".into());
        }
        if describe(root, &item.file.path)? != item.file {
            return Err("inventory content mismatch".into());
        }
    }
    Ok(())
}
fn sync_directories(root: &Path) -> std::result::Result<(), String> {
    for entry in fs::read_dir(root).map_err(string)? {
        let entry = entry.map_err(string)?;
        if entry.file_type().map_err(string)?.is_dir() {
            sync_directories(&entry.path())?;
        }
    }
    File::open(root).map_err(string)?.sync_all().map_err(string)
}
fn check_destination(path: &Path) -> std::result::Result<(), String> {
    match fs::symlink_metadata(path) {
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(string(e)),
        Ok(meta) => {
            if !meta.is_dir() || meta.file_type().is_symlink() {
                return Err("destination must be a bundle directory".into());
            }
            let file = File::open(path.join("manifest.json"))
                .or_else(|_| File::open(path.join("records.jsonl")))
                .map_err(|_| "destination is not a current bundle directory".to_owned())?;
            let record: Record = serde_json::Deserializer::from_reader(file)
                .into_iter()
                .next()
                .ok_or("empty bundle")?
                .map_err(string)?;
            let Record::Manifest(manifest) = record else {
                return Err("destination has no manifest".into());
            };
            manifest.validate().map_err(string)
        }
    }
}

#[cfg(test)]
#[path = "bundle/tests.rs"]
mod tests;
