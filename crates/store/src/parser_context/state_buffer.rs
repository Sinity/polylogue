//! Pending state belongs to this attempt until its existing scratch checkpoint.
use super::*;
use polylogue_model::Json;
use std::collections::{BTreeMap, VecDeque};

// The other half of the existing 256 KiB scratch allowance remains SQLite's
// page cache. Responses and aliases keep their existing disk-backed owner.
const TOTAL_BYTES: usize = 256 * 1024;
pub(super) const LIMIT: usize = TOTAL_BYTES / 2;
pub(super) const PAGE_CACHE_KIB: i64 = -(((TOTAL_BYTES - LIMIT) / 1024) as i64);

#[derive(Default)]
pub(super) struct States {
    entries: BTreeMap<Key, Pending>,
    bytes: usize,
    order: VecDeque<Key>,
}
struct Pending {
    value: Json,
    bytes: usize,
}
impl States {
    #[cfg(test)]
    pub(super) fn usage(&self) -> usize {
        self.bytes
    }
    pub(super) fn get(&self, key: &Key) -> Option<&Json> {
        self.entries.get(key).map(|entry| &entry.value)
    }
    pub(super) fn remove(&mut self, key: &Key) {
        if let Some(entry) = self.entries.remove(key) {
            self.bytes -= entry.bytes;
            self.order.retain(|found| found != key);
        }
    }
    pub(super) fn clear(&mut self, namespace: &str) {
        self.entries.retain(|key, entry| {
            if matches!(key, Key::State { namespace: found, .. } if found == namespace) {
                self.bytes -= entry.bytes;
                false
            } else {
                true
            }
        });
        self.order.retain(|key| self.entries.contains_key(key));
    }
    fn spill_first(&mut self, scratch: &Connection) -> Result<()> {
        let Some(key) = self.order.front() else {
            return Ok(());
        };
        let entry = &self.entries[key];
        super::scratch::write_state(scratch, key, &entry.value)?;
        self.bytes -= entry.bytes;
        let key = self
            .order
            .pop_front()
            .expect("pending state has a queue entry");
        self.entries.remove(&key);
        Ok(())
    }
    pub(super) fn flush(&mut self, scratch: &Connection) -> Result<()> {
        while !self.order.is_empty() {
            self.spill_first(scratch)?;
        }
        Ok(())
    }
    pub(super) fn put(&mut self, scratch: &Connection, key: Key, value: Json) -> Result<()> {
        let Some(bytes) = weight(&key, &value) else {
            // Write before removing an older pending value: a refusal leaves
            // the attempt's last successful value available for lookup.
            super::scratch::write_state(scratch, &key, &value)?;
            self.remove(&key);
            return Ok(());
        };
        let previous = self.entries.get(&key).map_or(0, |entry| entry.bytes);
        while self.bytes - previous + bytes > LIMIT {
            // The value being replaced is already credited in `previous`.
            // Skip it without scanning every other buffered state.
            if self.order.front() == Some(&key) {
                self.order.rotate_left(1);
            }
            self.spill_first(scratch)?;
        }
        let present = self.entries.remove(&key);
        if let Some(old) = present {
            self.bytes -= old.bytes;
        } else {
            self.order.push_back(key.clone());
        }
        self.bytes += bytes;
        self.entries.insert(key, Pending { value, bytes });
        Ok(())
    }
}

// Charge tree/container overhead conservatively, including spare string and
// array capacity. Stop before walking an oversized tree or a deep stack.
fn weight(key: &Key, value: &Json) -> Option<usize> {
    let Key::State { namespace, key } = key else {
        return None;
    };
    let key_bytes = namespace
        .capacity()
        .checked_add(key.capacity())?
        .checked_mul(2)?
        .checked_add(256)?;
    let mut remaining = LIMIT.checked_sub(key_bytes)?;
    fn visit(value: &Json, remaining: &mut usize, depth: usize) -> Option<()> {
        if depth > 64 {
            return None;
        }
        *remaining = remaining.checked_sub(256)?;
        match value {
            Json::String(value) => *remaining = remaining.checked_sub(value.capacity())?,
            Json::Array(values) => {
                *remaining = remaining
                    .checked_sub(values.capacity().checked_mul(std::mem::size_of::<Json>())?)?;
                for value in values {
                    visit(value, remaining, depth + 1)?;
                }
            }
            Json::Object(values) => {
                for (key, value) in values {
                    *remaining = remaining.checked_sub(key.capacity())?;
                    visit(value, remaining, depth + 1)?;
                }
            }
            Json::Number(value) => {
                struct Bound {
                    remaining: usize,
                }
                impl std::io::Write for Bound {
                    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                        self.remaining = self
                            .remaining
                            .checked_sub(bytes.len())
                            .ok_or_else(|| std::io::Error::other("number exceeds state buffer"))?;
                        Ok(bytes.len())
                    }
                    fn flush(&mut self) -> std::io::Result<()> {
                        Ok(())
                    }
                }
                let mut bound = Bound {
                    remaining: *remaining / 2,
                };
                let before = bound.remaining;
                serde_json::to_writer(&mut bound, value).ok()?;
                *remaining = remaining.checked_sub((before - bound.remaining) * 2)?;
            }
            Json::Null | Json::Bool(_) => {}
        }
        Some(())
    }
    visit(value, &mut remaining, 0)?;
    Some(LIMIT - remaining)
}
