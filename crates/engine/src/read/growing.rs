//! A second read after detected growth. Pin the planned bytes with a
//! reflink where supported, or check the new file facts around a fresh read.

use super::{File, FileFacts, FileSnapshot, check_source, file_snapshot};
use std::io::{self, Seek, SeekFrom};
use std::path::Path;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub(crate) struct FrozenFile {
    file: Arc<File>,
    pub(crate) observed: FileSnapshot,
}

impl FrozenFile {
    pub(crate) fn open(&self) -> io::Result<File> {
        self.file.try_clone()
    }

    pub(crate) fn read(&self, start: u64, end: u64) -> io::Result<Vec<u8>> {
        let mut file = self.open()?;
        file.seek(SeekFrom::Start(start))?;
        let mut bytes = vec![0; (end - start) as usize];
        super::read_exact_checked(&mut file, &mut bytes)?;
        Ok(bytes)
    }
}

pub(super) struct GrowingFile {
    pub(super) file: File,
    pub(super) observed: FileSnapshot,
    source: FileFacts,
    frozen: bool,
}

impl GrowingFile {
    pub(super) fn open(
        path: &Path,
        expected: FileFacts,
        staging: &Path,
    ) -> io::Result<Option<Self>> {
        let source_file = match File::open(path) {
            Ok(source) => source,
            Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(err) => return Err(err),
        };
        let metadata = source_file.metadata()?;
        let source = FileFacts::of(&metadata);
        if source.inode != expected.inode
            || source.device != expected.device
            || source.size <= expected.size
        {
            return Ok(None);
        }
        let observed = file_snapshot(&metadata)?;
        let copy = reflink(&source_file, staging, expected.size);
        if copy.is_some() {
            let at_path = FileFacts::of(&std::fs::metadata(path)?);
            if at_path.device != source.device || at_path.inode != source.inode {
                return Ok(None);
            }
        }
        let frozen = copy.is_some();
        Ok(Some(Self {
            file: copy.unwrap_or(source_file),
            observed,
            source,
            frozen,
        }))
    }

    pub(super) fn finish(self, path: &Path) -> io::Result<Option<FrozenFile>> {
        if self.frozen {
            return Ok(Some(FrozenFile {
                file: Arc::new(self.file),
                observed: self.observed,
            }));
        }
        check_source(&self.file, path, self.source)?;
        Ok(None)
    }

    pub(super) fn read_range(&mut self, start: u64, end: u64) -> io::Result<Vec<u8>> {
        self.file.seek(SeekFrom::Start(start))?;
        let mut bytes = vec![0; (end - start) as usize];
        super::read_exact_checked(&mut self.file, &mut bytes)?;
        Ok(bytes)
    }
}

#[cfg(target_os = "linux")]
fn reflink(source: &File, staging: &Path, through: u64) -> Option<File> {
    if through == 0 {
        return None;
    }
    let clone = tempfile::tempfile_in(staging).ok()?;
    rustix::fs::ioctl_ficlone(&clone, source).ok()?;
    if clone.metadata().ok()?.len() < through {
        return None;
    }
    // Keep only the planned sample. The clone shares the source's blocks
    // until either file writes them.
    clone.set_len(through).ok()?;
    Some(clone)
}

#[cfg(not(target_os = "linux"))]
fn reflink(_: &File, _: &Path, _: u64) -> Option<File> {
    None
}
