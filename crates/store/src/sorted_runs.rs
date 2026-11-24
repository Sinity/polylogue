//! One external sorter for every sorted stream a build writes: records are
//! (key, value) byte strings, buffered up to a memory budget, sorted by key
//! and spilled as zstd-compressed runs; a k-way merge reads the runs
//! forward in key order. Memory is the budget plus one read buffer per run
//! merged, whatever the number of records.
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

/// Runs merged at once; more are merged in passes first.
const FAN_IN: usize = 128;
const READ_BUFFER: usize = 128 << 10;

pub(crate) fn put_varint(out: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 {
        out.push((v as u8) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

fn read_varint(r: &mut dyn Read) -> io::Result<Option<u64>> {
    let (mut v, mut shift, mut byte) = (0u64, 0, [0u8]);
    loop {
        if r.read(&mut byte)? == 0 {
            return if shift == 0 {
                Ok(None)
            } else {
                Err(io::ErrorKind::UnexpectedEof.into())
            };
        }
        v |= u64::from(byte[0] & 0x7f) << shift;
        if byte[0] < 0x80 {
            return Ok(Some(v));
        }
        shift += 7;
    }
}

/// A sequential, compressed record file.
pub(crate) struct RecordWriter(zstd::stream::write::Encoder<'static, BufWriter<File>>);

impl RecordWriter {
    pub(crate) fn create(path: &Path) -> io::Result<Self> {
        let file = BufWriter::with_capacity(1 << 20, File::create(path)?);
        Ok(Self(zstd::stream::write::Encoder::new(file, 1)?))
    }

    pub(crate) fn record(&mut self, key: &[u8], value: &[u8]) -> io::Result<()> {
        let mut head = Vec::with_capacity(20);
        put_varint(&mut head, key.len() as u64);
        put_varint(&mut head, value.len() as u64);
        self.0.write_all(&head)?;
        self.0.write_all(key)?;
        self.0.write_all(value)
    }

    pub(crate) fn finish(self) -> io::Result<()> {
        self.0.finish()?.flush()
    }
}

/// Reads a record file forward.
pub(crate) struct RecordReader {
    inner: Box<dyn Read + Send>,
    key: Vec<u8>,
    value: Vec<u8>,
}

impl RecordReader {
    pub(crate) fn open(path: &Path) -> io::Result<Self> {
        let file = BufReader::with_capacity(READ_BUFFER, File::open(path)?);
        Ok(Self {
            inner: Box::new(BufReader::with_capacity(
                READ_BUFFER,
                zstd::stream::read::Decoder::with_buffer(file)?,
            )),
            key: Vec::new(),
            value: Vec::new(),
        })
    }

    /// Reads the next record; false at the end.
    fn advance(&mut self) -> io::Result<bool> {
        let Some(key) = read_varint(&mut self.inner)? else {
            return Ok(false);
        };
        let value = read_varint(&mut self.inner)?.ok_or(io::ErrorKind::UnexpectedEof)?;
        self.key.resize(key as usize, 0);
        self.inner.read_exact(&mut self.key)?;
        self.value.resize(value as usize, 0);
        self.inner.read_exact(&mut self.value)?;
        Ok(true)
    }
}

/// Buffers records up to `budget` bytes, then spills them sorted as a run.
pub(crate) struct Sorter {
    dir: PathBuf,
    name: String,
    budget: usize,
    arena: Vec<u8>,
    /// (offset, key length, value length) of each buffered record.
    index: Vec<(usize, u32, u32)>,
    runs: Vec<PathBuf>,
}

impl Sorter {
    pub(crate) fn new(dir: &Path, name: String, budget: usize) -> Self {
        Self {
            dir: dir.to_owned(),
            name,
            budget,
            arena: Vec::new(),
            index: Vec::new(),
            runs: Vec::new(),
        }
    }

    pub(crate) fn push(&mut self, key: &[u8], value: &[u8]) -> io::Result<()> {
        let at = self.arena.len();
        self.arena.extend_from_slice(key);
        self.arena.extend_from_slice(value);
        self.index.push((at, key.len() as u32, value.len() as u32));
        if self.arena.len() + self.index.len() * 16 >= self.budget {
            self.spill()?;
        }
        Ok(())
    }

    fn spill(&mut self) -> io::Result<()> {
        if self.index.is_empty() {
            return Ok(());
        }
        let arena = &self.arena;
        let key = |&(at, k, _): &(usize, u32, u32)| &arena[at..at + k as usize];
        // Stable, so equal keys keep the order they were pushed in.
        self.index.sort_by(|a, b| key(a).cmp(key(b)));
        let path = self
            .dir
            .join(format!("{}-{:05}.run", self.name, self.runs.len()));
        let mut w = RecordWriter::create(&path)?;
        for &(at, k, v) in &self.index {
            let (k, v) = (k as usize, v as usize);
            w.record(&arena[at..at + k], &arena[at + k..at + k + v])?;
        }
        w.finish()?;
        self.runs.push(path);
        self.arena.clear();
        self.index.clear();
        Ok(())
    }

    /// Spills what is buffered and returns the runs, in the order written.
    pub(crate) fn finish(mut self) -> io::Result<Vec<PathBuf>> {
        self.spill()?;
        Ok(self.runs)
    }
}

/// K-way merge over sorted runs. Equal keys come out in run order.
pub(crate) struct Merger {
    readers: Vec<RecordReader>,
    heap: Vec<usize>,
    advance_top: bool,
}

impl Merger {
    /// Opens `runs`; past [`FAN_IN`], groups are first merged into larger
    /// runs in `dir`, and the runs they replace are removed.
    pub(crate) fn open(mut runs: Vec<PathBuf>, dir: &Path) -> io::Result<Self> {
        let mut pass = 0;
        while runs.len() > FAN_IN {
            let mut next = Vec::new();
            for (i, group) in runs.chunks(FAN_IN).enumerate() {
                let path = dir.join(format!("pass{pass}-{i:05}.run"));
                let mut merged = Merger::open(group.to_vec(), dir)?;
                let mut w = RecordWriter::create(&path)?;
                while let Some((k, v)) = merged.next()? {
                    w.record(k, v)?;
                }
                w.finish()?;
                for run in group {
                    std::fs::remove_file(run)?;
                }
                next.push(path);
            }
            runs = next;
            pass += 1;
        }
        let mut readers = Vec::new();
        for path in &runs {
            let mut r = RecordReader::open(path)?;
            if r.advance()? {
                readers.push(r);
            }
        }
        let mut m = Merger {
            heap: (0..readers.len()).collect(),
            readers,
            advance_top: false,
        };
        for i in (0..m.heap.len() / 2).rev() {
            m.sift_down(i);
        }
        Ok(m)
    }

    fn less(&self, a: usize, b: usize) -> bool {
        match self.readers[a].key.cmp(&self.readers[b].key) {
            std::cmp::Ordering::Equal => a < b,
            order => order.is_lt(),
        }
    }

    fn sift_down(&mut self, mut i: usize) {
        let n = self.heap.len();
        loop {
            let (l, r) = (2 * i + 1, 2 * i + 2);
            let mut m = i;
            if l < n && self.less(self.heap[l], self.heap[m]) {
                m = l;
            }
            if r < n && self.less(self.heap[r], self.heap[m]) {
                m = r;
            }
            if m == i {
                return;
            }
            self.heap.swap(i, m);
            i = m;
        }
    }

    fn settle(&mut self) -> io::Result<()> {
        if std::mem::take(&mut self.advance_top) {
            let top = self.heap[0];
            if self.readers[top].advance()? {
                self.sift_down(0);
            } else {
                let last = self.heap.pop().expect("a reader on the heap");
                if !self.heap.is_empty() {
                    self.heap[0] = last;
                    self.sift_down(0);
                }
            }
        }
        Ok(())
    }

    /// The next record.
    pub(crate) fn next(&mut self) -> io::Result<Option<(&[u8], &[u8])>> {
        self.settle()?;
        if self.heap.is_empty() {
            return Ok(None);
        }
        self.advance_top = true;
        let r = &self.readers[self.heap[0]];
        Ok(Some((r.key.as_slice(), r.value.as_slice())))
    }
}

#[cfg(test)]
mod tests;
