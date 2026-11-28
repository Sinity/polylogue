//! Prepare from an indexed cut of one native standalone archive, then take
//! that writer's final cut in the same invocation.

use super::*;
use polylogue_store::archive::Mode;
use std::io::{Seek, SeekFrom, Write};
use std::os::unix::fs::MetadataExt;

pub(super) fn run(mut args: impl Iterator<Item = OsString>) -> Result<(), Error> {
    let (Some(earlier), Some(source), Some(candidate), None) =
        (args.next(), args.next(), args.next(), args.next())
    else {
        return Err(
            "usage: reuse_archive_index continue <checked cut> <native source> <new candidate>"
                .into(),
        );
    };
    let pending = Pending::prepare(
        Path::new(&earlier),
        Path::new(&source),
        Path::new(&candidate),
    )?;
    println!(
        "Prepared the index at {} and checked the source through {}. Stop the source writer, then enter 'finish'.",
        pending.index_at, pending.checked_at
    );
    std::io::stdout().flush()?;
    let mut request = String::new();
    std::io::stdin().read_line(&mut request)?;
    if request.trim() != "finish" {
        return Err("cancelled before taking the final cut".into());
    }
    let (from, to) = pending.finish()?;
    println!("Installed the index at {from}; the archive is at {to}.");
    Ok(())
}

struct Pinned {
    path: PathBuf,
    file: File,
}

impl Pinned {
    fn open(path: PathBuf) -> Result<Self, Error> {
        let file = File::open(&path)?;
        let pin = Self { path, file };
        pin.check()?;
        Ok(pin)
    }

    fn check(&self) -> Result<(), Error> {
        let opened = self.file.metadata()?;
        let current = self.path.symlink_metadata()?;
        if (opened.dev(), opened.ino()) != (current.dev(), current.ino()) || current.is_symlink() {
            return Err(format!("source path {} was replaced", self.path.display()).into());
        }
        Ok(())
    }
}

struct Segment {
    source: Pinned,
    through: u64,
}

type DatabaseCopy = fn(&Path, &Path) -> Result<(), Error>;

pub(super) struct Pending {
    source: Pinned,
    lock: Pinned,
    record: Pinned,
    vectors: Pinned,
    segments: Vec<(String, Segment)>,
    candidate: PathBuf,
    staging: tempfile::TempDir,
    copy_database: DatabaseCopy,
    identity: ArchiveId,
    pub(super) index_at: u64,
    pub(super) checked_at: u64,
}

impl Pending {
    pub(super) fn prepare(earlier: &Path, source: &Path, candidate: &Path) -> Result<Self, Error> {
        Self::prepare_using(earlier, source, candidate, reflink_database)
    }

    pub(super) fn prepare_using(
        earlier: &Path,
        source: &Path,
        candidate: &Path,
        copy_database: DatabaseCopy,
    ) -> Result<Self, Error> {
        let parent = candidate
            .parent()
            .filter(|path| !path.as_os_str().is_empty())
            .unwrap_or(Path::new("."))
            .canonicalize()?;
        let candidate = parent.join(
            candidate
                .file_name()
                .ok_or("name the new candidate directory")?,
        );
        absent(&candidate)?;
        if Archive::recorded_mode(earlier)? != Some(Mode::Standalone)
            || Archive::recorded_mode(source)? != Some(Mode::Standalone)
        {
            return Err("both archives must use the current standalone mode".into());
        }
        let _earlier_lock = super::lock(earlier)?;
        let source = Pinned::open(std::path::absolute(source)?)?;
        if parent.starts_with(source.path.canonicalize()?)
            || parent.starts_with(earlier.canonicalize()?)
        {
            return Err("put the candidate outside both archive directories".into());
        }
        let lock = Pinned::open(source.path.join("lock"))?;
        let live = Archive::at(&source.path);
        let record = Pinned::open(live.archive_db())?;
        let vectors = Pinned::open(live.embeddings_db())?;
        let _ = open_archive(&live)?;
        let checked = Archive::at(earlier);
        let (old, index_at) = open_archive(&checked)?;
        let identity = archive_id(&old)?;
        checked_index(earlier, identity.clone(), index_at)?;
        let staging = tempfile::tempdir_in(&parent)?;
        // Check the actual files and destination before asking the operator
        // to stop. These copies may include in-flight writes and are discarded.
        {
            let probe = tempfile::tempdir_in(staging.path())?;
            copy_database(&record.path, &probe.path().join("archive.db"))?;
            copy_database(&vectors.path, &probe.path().join("embeddings.db"))?;
        }
        fs::create_dir(staging.path().join("segments"))?;
        let (checked_at, mut segments) = {
            attach(&old, &live.archive_db())?;
            let read = old.unchecked_transaction()?;
            let (live_id, checked_at) = cut(&read, "later.")?;
            if live_id != identity || checked_at < index_at {
                return Err(
                    "checked cut must come from this source at or before its current position"
                        .into(),
                );
            }
            prove_record_prefix(&read, index_at)?;
            let mut segments = Vec::new();
            for (name, end) in segment_ends(&read, "later.")? {
                let path = segment_path(&name)?;
                let source = Pinned::open(live.root.join(path))?;
                segments.push((
                    name,
                    Segment {
                        source,
                        through: end,
                    },
                ));
            }
            (checked_at, segments)
        };
        // Committed byte ranges cannot change while native writers append.
        // Release the SQLite reader before comparing or staging raw bytes.
        let earlier_segments = segment_ends(&old, "main.")?;
        for (name, Segment { source, through }) in &mut segments {
            let target = staging.path().join(segment_path(name)?);
            let before = earlier_segments
                .iter()
                .find(|(old_name, _)| old_name == name);
            let from = match before {
                Some((_, end)) => {
                    let path = earlier.join(segment_path(name)?);
                    let mut original = File::open(&path)?;
                    if original.metadata()?.len() != *end || *end > *through {
                        return Err(
                            format!("segment {name} has incompatible committed lengths").into()
                        );
                    }
                    compare_segment(name, &mut original, &mut source.file, *end)?;
                    copy_file(&path, &target)?;
                    *end
                }
                None => {
                    File::create(&target)?;
                    0
                }
            };
            append(source, &target, from, *through)?;
        }
        if earlier_segments
            .iter()
            .any(|(name, _)| !segments.iter().any(|(found, _)| found == name))
        {
            return Err("source archive is missing an earlier segment".into());
        }
        prove_vectors(&checked, &live)?;
        install_index(earlier, staging.path(), &identity, index_at)?;
        for pin in [&source, &lock, &record, &vectors] {
            pin.check()?;
        }
        Ok(Self {
            source,
            lock,
            record,
            vectors,
            segments,
            candidate,
            staging,
            copy_database,
            identity,
            index_at,
            checked_at,
        })
    }

    pub(super) fn finish(mut self) -> Result<(u64, u64), Error> {
        self.lock
            .file
            .try_lock()
            .map_err(|error| format!("cannot obtain source writer lock: {error}"))?;
        for pin in [&self.source, &self.lock, &self.record, &self.vectors] {
            pin.check()?;
        }
        let archive = Archive::at(&self.source.path);
        if Archive::recorded_mode(&self.source.path)? != Some(Mode::Standalone) {
            return Err("source archive mode changed after preparation".into());
        }
        let (conn, position) = open_archive(&archive)?;
        if archive_id(&conn)? != self.identity || position < self.checked_at {
            return Err("source archive identity changed or its record position regressed".into());
        }
        let ends = segment_ends(&conn, "main.")?;
        drop(conn);
        if self.segments.len() != ends.len() {
            return Err("source segment files changed after preparation".into());
        }
        let stage = self.staging.path();
        for ((expected, segment), (name, end)) in self.segments.iter_mut().zip(ends) {
            if *expected != name || segment.through > end {
                return Err("source segment files changed or their positions regressed".into());
            }
            segment.source.check()?;
            if segment.source.file.metadata()?.len() != end {
                return Err(format!(
                    "source segment {name} does not end at its committed position"
                )
                .into());
            }
            append(
                &mut segment.source,
                &stage.join(segment_path(&name)?),
                segment.through,
                end,
            )?;
            // Keep the source's native check against a future mismatched
            // restore of the catalog and segment.
            let committed = format!("{name}.committed");
            let marker = self.source.path.join(&committed);
            if marker.try_exists()? {
                let target = stage.join(committed);
                copy_file(&marker, &target)?;
                File::open(target)?.sync_all()?;
            }
        }
        // The vector store commits independently of the record counter. Copy
        // its final database and WAL even when the record position is unchanged.
        (self.copy_database)(&self.record.path, &stage.join("archive.db"))?;
        (self.copy_database)(&self.vectors.path, &stage.join("embeddings.db"))?;
        copy_file(
            &self.source.path.join("meta.json"),
            &stage.join("meta.json"),
        )?;
        File::create_new(stage.join("lock"))?;
        for entry in fs::read_dir(stage)? {
            let entry = entry?;
            if entry.file_type()?.is_file() {
                File::open(entry.path())?.sync_all()?;
            }
        }
        File::open(stage.join("segments"))?.sync_all()?;
        File::open(stage)?.sync_all()?;
        absent(&self.candidate)?;
        fs::rename(stage, &self.candidate)?;
        File::open(self.candidate.parent().expect("absolute candidate"))?.sync_all()?;
        Ok((self.index_at, position))
    }
}

fn cut(conn: &Connection, db: &str) -> Result<(ArchiveId, u64), Error> {
    let (identity, position): (String, String) = conn.query_row(
        &format!(
            "SELECT value, coalesce((SELECT value FROM {db}meta WHERE key='write_counter'), '0')
                  FROM {db}meta WHERE key='archive_id'"
        ),
        [],
        |row| Ok((row.get(0)?, row.get(1)?)),
    )?;
    Ok((ArchiveId(identity), position.parse()?))
}

fn segment_ends(conn: &Connection, db: &str) -> Result<Vec<(String, u64)>, Error> {
    Ok(conn
        .prepare(&format!(
            "SELECT name, committed_end FROM {db}segment ORDER BY name"
        ))?
        .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?
        .collect::<rusqlite::Result<_>>()?)
}

fn append(source: &mut Pinned, target: &Path, from: u64, to: u64) -> Result<(), Error> {
    source.file.seek(SeekFrom::Start(from))?;
    let mut target = fs::OpenOptions::new().append(true).open(target)?;
    let n = std::io::copy(&mut (&source.file).take(to - from), &mut target)?;
    if n != to - from {
        return Err(format!(
            "source segment {} is shorter than its committed end",
            source.path.display()
        )
        .into());
    }
    target.sync_all()?;
    Ok(())
}

fn absent(path: &Path) -> Result<(), Error> {
    match path.symlink_metadata() {
        Ok(_) => Err(format!("candidate path {} already exists", path.display()).into()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.into()),
    }
}
