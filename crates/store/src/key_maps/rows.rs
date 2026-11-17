//! Bounded key rows: recent keys in memory, older keys in sorted files.
use crate::{OptionExt as _, Result};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::ops::ControlFlow;
use std::os::unix::fs::FileExt;

const LEVELS: usize = 64;
const RECORD_BYTES: usize = 48;

#[derive(Clone, Copy)]
struct Record {
    key: [u8; 32],
    value: i64,
    sequence: u64,
}
impl Record {
    fn order(self) -> ([u8; 32], i64) {
        (self.key, self.value)
    }
    fn encode(self) -> [u8; RECORD_BYTES] {
        let mut bytes = [0; RECORD_BYTES];
        bytes[..32].copy_from_slice(&self.key);
        bytes[32..40].copy_from_slice(&self.value.to_le_bytes());
        bytes[40..].copy_from_slice(&self.sequence.to_le_bytes());
        bytes
    }
    fn decode(bytes: [u8; RECORD_BYTES]) -> Self {
        Self {
            key: bytes[..32].try_into().expect("key bytes"),
            value: i64::from_le_bytes(bytes[32..40].try_into().expect("row bytes")),
            sequence: u64::from_le_bytes(bytes[40..].try_into().expect("sequence bytes")),
        }
    }
}
struct Run {
    file: tempfile::NamedTempFile,
    length: u64,
    first: [u8; 32],
    last: [u8; 32],
    samples: Vec<([u8; 32], u64)>,
    #[cfg(test)]
    reads: std::sync::atomic::AtomicU64,
}
impl Run {
    fn read(&self, at: u64) -> Result<Record> {
        let mut bytes = [0; RECORD_BYTES];
        self.file
            .as_file()
            .read_exact_at(&mut bytes, at * RECORD_BYTES as u64)?;
        #[cfg(test)]
        self.reads
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(Record::decode(bytes))
    }
    fn visit(
        &self,
        key: &[u8; 32],
        visit: &mut impl FnMut(Record) -> Result<ControlFlow<()>>,
    ) -> Result<ControlFlow<()>> {
        if key < &self.first || key > &self.last {
            return Ok(ControlFlow::Continue(()));
        }
        // Start before the first sample equal to the key: its candidates can
        // begin before that sample and continue across several samples.
        let sample = self.samples.partition_point(|(value, _)| value < key);
        let mut lo = sample.checked_sub(1).map_or(0, |n| self.samples[n].1);
        let mut hi = self
            .samples
            .get(sample)
            .map_or(self.length, |(_, at)| at + 1);
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if self.read(mid)?.key < *key {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        while lo < self.length {
            let record = self.read(lo)?;
            if record.key != *key {
                break;
            }
            if visit(record)?.is_break() {
                return Ok(ControlFlow::Break(()));
            }
            lo += 1;
        }
        Ok(ControlFlow::Continue(()))
    }
}
struct RunReader {
    file: BufReader<File>,
    left: u64,
}
impl RunReader {
    fn new(run: &Run, buffer: usize) -> Result<Self> {
        Ok(Self {
            file: BufReader::with_capacity(buffer, File::open(run.file.path())?),
            left: run.length,
        })
    }
    fn next(&mut self) -> Result<Option<Record>> {
        if self.left == 0 {
            return Ok(None);
        }
        let mut bytes = [0; RECORD_BYTES];
        self.file.read_exact(&mut bytes)?;
        self.left -= 1;
        Ok(Some(Record::decode(bytes)))
    }
}
fn sample_bytes(buffer: usize) -> usize {
    (buffer / 8).max(std::mem::size_of::<([u8; 32], u64)>())
}
struct RunWriter {
    samples: Vec<([u8; 32], u64)>,
    stride: u64,
    file: BufWriter<tempfile::NamedTempFile>,
    length: u64,
    first: [u8; 32],
    last: [u8; 32],
}
impl RunWriter {
    fn new(buffer: usize, maximum: u64) -> Result<Self> {
        let capacity = (sample_bytes(buffer) / std::mem::size_of::<([u8; 32], u64)>()).max(1);
        let stride = maximum.div_ceil(capacity as u64).max(1);
        Ok(Self {
            samples: Vec::with_capacity(capacity),
            stride,
            file: BufWriter::with_capacity(buffer, tempfile::NamedTempFile::new()?),
            length: 0,
            first: [0; 32],
            last: [0; 32],
        })
    }
    fn push(&mut self, record: Record) -> Result<()> {
        if self.length.is_multiple_of(self.stride) {
            self.samples.push((record.key, self.length));
        }
        self.file.write_all(&record.encode())?;
        if self.length == 0 {
            self.first = record.key;
        }
        self.last = record.key;
        self.length += 1;
        Ok(())
    }
    fn finish(self) -> Result<Run> {
        Ok(Run {
            file: self.file.into_inner().map_err(|error| error.into_error())?,
            length: self.length,
            first: self.first,
            last: self.last,
            samples: self.samples,
            #[cfg(test)]
            reads: std::sync::atomic::AtomicU64::new(0),
        })
    }
}

/// Key histories use fixed resident space. A negative Bloom check reads no file;
/// false positives and stored keys are checked against the sorted files.
pub(crate) struct KeyRows {
    entries: BTreeMap<([u8; 32], i64), u64>,
    runs: [Option<Run>; LEVELS],
    filter: Vec<u64>,
    limit: usize,
    buffer: usize,
    sequence: u64,
}
impl KeyRows {
    pub(crate) fn new(bytes: u64) -> Result<Self> {
        // The directory and merge readers have fixed overhead. A small caller's
        // allowance still makes progress by spilling after each row.
        let bytes = usize::try_from(bytes).unwrap_or(usize::MAX).max(64 * 1024);
        let filter_bytes = (bytes / 8).clamp(64, 8 * 1024 * 1024);
        let directory = std::mem::size_of::<[Option<Run>; LEVELS]>();
        let buffer = (bytes.saturating_sub(filter_bytes + directory) / (LEVELS + LEVELS / 8 + 4))
            .clamp(RECORD_BYTES, 64 * 1024);
        let fixed = filter_bytes
            + (buffer + sample_bytes(buffer)) * (LEVELS + 1)
            + std::mem::size_of::<[Option<Run>; LEVELS]>();
        Ok(Self {
            entries: BTreeMap::new(),
            runs: std::array::from_fn(|_| None),
            filter: vec![0; filter_bytes.div_ceil(8)],
            // A B-tree entry holds 48 bytes; 128 also covers node occupancy and links.
            limit: bytes.saturating_sub(fixed).saturating_div(128).max(1),
            buffer,
            sequence: 0,
        })
    }
    fn positions(&self, key: &[u8; 32]) -> impl Iterator<Item = usize> {
        let a = u64::from_le_bytes(key[..8].try_into().expect("hash bytes"));
        let b = u64::from_le_bytes(key[8..16].try_into().expect("hash bytes")) | 1;
        let bits = self.filter.len() as u64 * 64;
        (0..7u64).map(move |i| a.wrapping_add(i.wrapping_mul(b)).wrapping_rem(bits) as usize)
    }
    fn possible(&self, key: &[u8; 32]) -> bool {
        self.positions(key)
            .all(|bit| self.filter[bit / 64] & (1 << (bit % 64)) != 0)
    }
    pub(crate) fn set(&mut self, key: [u8; 32], value: i64) -> Result<()> {
        self.sequence = self
            .sequence
            .checked_add(1)
            .or_invalid("key history exhausted")?;
        self.entries.insert((key, value), self.sequence);
        let mut positions = [0; 7];
        for (slot, position) in positions.iter_mut().zip(self.positions(&key)) {
            *slot = position;
        }
        for bit in positions {
            self.filter[bit / 64] |= 1 << (bit % 64);
        }
        if self.entries.len() >= self.limit {
            self.spill()?;
        }
        Ok(())
    }
    pub(crate) fn get(&self, key: &[u8; 32]) -> Result<Option<i64>> {
        if let Some(((_, value), _)) = self
            .entries
            .range((*key, i64::MIN)..=(*key, i64::MAX))
            .max_by_key(|(_, sequence)| **sequence)
        {
            return Ok(Some(*value));
        }
        let mut latest: Option<Record> = None;
        if self.possible(key) {
            for run in self.runs.iter().flatten() {
                let _ = run.visit(key, &mut |record| {
                    if latest.is_none_or(|prior| record.sequence > prior.sequence) {
                        latest = Some(record);
                    }
                    Ok(ControlFlow::Continue(()))
                })?;
            }
        }
        Ok(latest.map(|record| record.value))
    }
    pub(crate) fn visit(
        &self,
        key: &[u8; 32],
        mut visit: impl FnMut(i64) -> Result<ControlFlow<()>>,
    ) -> Result<()> {
        for ((_, value), _) in self.entries.range((*key, i64::MIN)..=(*key, i64::MAX)) {
            if visit(*value)?.is_break() {
                return Ok(());
            }
        }
        if self.possible(key) {
            for run in self.runs.iter().flatten() {
                if run
                    .visit(key, &mut |record| visit(record.value))?
                    .is_break()
                {
                    break;
                }
            }
        }
        Ok(())
    }
    fn spill(&mut self) -> Result<()> {
        if self.entries.is_empty() {
            return Ok(());
        }
        let mut output = RunWriter::new(self.buffer, self.entries.len() as u64)?;
        for ((key, value), sequence) in &self.entries {
            output.push(Record {
                key: *key,
                value: *value,
                sequence: *sequence,
            })?;
        }
        let mut run = output.finish()?;
        for level in 0..LEVELS {
            if let Some(prior) = &self.runs[level] {
                run = merge(prior, &run, self.buffer)?;
            } else {
                self.entries.clear();
                for slot in &mut self.runs[..level] {
                    *slot = None;
                }
                self.runs[level] = Some(run);
                return Ok(());
            }
        }
        Err(crate::StoreError::Invalid(
            "key history exceeds its address space".into(),
        ))
    }
    /// Rewrites one current value per key, in key order, without retaining the
    /// dictionary's complete contents in memory.
    pub(crate) fn transform_values(&mut self, transform: impl Fn(i64) -> i64) -> Result<()> {
        self.spill()?;
        let mut readers: Vec<_> = self
            .runs
            .iter()
            .flatten()
            .map(|run| RunReader::new(run, self.buffer))
            .collect::<Result<_>>()?;
        if readers.is_empty() {
            return Ok(());
        }
        let mut current: Vec<_> = readers
            .iter_mut()
            .map(RunReader::next)
            .collect::<Result<_>>()?;
        let maximum = self.runs.iter().flatten().map(|run| run.length).sum();
        let mut output = RunWriter::new(self.buffer, maximum)?;
        let mut latest: Option<Record> = None;
        while let Some((at, record)) = current
            .iter()
            .enumerate()
            .filter_map(|(at, record)| record.map(|record| (at, record)))
            .min_by_key(|(_, record)| record.order())
        {
            if latest.is_some_and(|prior| prior.key != record.key) {
                let prior = latest.take().expect("current key");
                output.push(Record {
                    value: transform(prior.value),
                    ..prior
                })?;
            }
            if latest.is_none_or(|prior| record.sequence > prior.sequence) {
                latest = Some(record);
            }
            current[at] = readers[at].next()?;
        }
        if let Some(prior) = latest {
            output.push(Record {
                value: transform(prior.value),
                ..prior
            })?;
        }
        let run = output.finish()?;
        let level = (run.length.saturating_sub(1) / self.limit as u64)
            .checked_ilog2()
            .map_or(0, |n| n as usize + 1)
            .min(LEVELS - 1);
        self.runs = std::array::from_fn(|_| None);
        self.runs[level] = Some(run);
        Ok(())
    }
    pub(crate) fn clear(&mut self) {
        self.entries.clear();
        self.runs = std::array::from_fn(|_| None);
        self.filter.fill(0);
        self.sequence = 0;
    }
}
fn merge(first: &Run, second: &Run, buffer: usize) -> Result<Run> {
    let (mut a, mut b) = (
        RunReader::new(first, buffer)?,
        RunReader::new(second, buffer)?,
    );
    let (mut x, mut y) = (a.next()?, b.next()?);
    let mut output = RunWriter::new(buffer, first.length + second.length)?;
    while let (Some(left), Some(right)) = (x, y) {
        match left.order().cmp(&right.order()) {
            std::cmp::Ordering::Less => {
                output.push(left)?;
                x = a.next()?;
            }
            std::cmp::Ordering::Greater => {
                output.push(right)?;
                y = b.next()?;
            }
            std::cmp::Ordering::Equal => {
                output.push(if left.sequence > right.sequence {
                    left
                } else {
                    right
                })?;
                x = a.next()?;
                y = b.next()?;
            }
        }
    }
    while let Some(record) = x {
        output.push(record)?;
        x = a.next()?;
    }
    while let Some(record) = y {
        output.push(record)?;
        y = b.next()?;
    }
    output.finish()
}

#[cfg(test)]
mod tests;
