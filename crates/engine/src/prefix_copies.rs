//! Older chains recorded from a newer chain's reading where their bytes
//! begin alike ([docs/archive.md], versioned rows). Parsing an older chain's
//! bytes that a newer chain of the same input also holds gives the reading
//! the newer chain has once its parse has read those bytes. So a rebuild
//! reads each byte of an input's history once: while the newer chain is
//! parsed, its reading is copied to each older chain at the point the two
//! chains stop being alike, and the older chain's parse continues from there
//! over the bytes only it holds.
//!
//! Each older chain takes its reading from the newest chain holding all its
//! bytes in the same blobs, else from the newer chain whose bytes it shares
//! the most of (the newest of equals), among the chains that can
//! still be at that point: chains not read yet, and chains read no further
//! than it. A chain identical to a newer one is copied once that chain is
//! read, after the newest chains of the pass ([`crate::Engine::copy_identical_readings`]);
//! any other copy is taken as the newer chain's parse passes its stop.
//!
//! [docs/archive.md]: ../../../docs/archive.md
use crate::{Engine, EngineError, Pass, Plan};
use polylogue_store::ChainLayout;
use std::collections::HashMap;

/// An older chain recorded from the planned chain's reading.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PrefixCopy {
    /// The older chain.
    pub chain: i64,
    /// The planned chain's parse stops here for the copy: the end of its
    /// last complete record before the two chains stop being alike or the
    /// older chain ends.
    pub stop: u64,
    /// The copy reads the older chain to its end.
    pub reaches_end: bool,
}

/// Where an older chain of `length` bytes takes its reading from a newer
/// chain of `newer` bytes when the two share their first `shared` bytes:
/// `None` for identical chains, else the stop and whether the copy reads the
/// older chain to its end. The end of a finished input (an import) ends its
/// last record, which the newer chain reads as an incomplete one, so such
/// an older chain stops one byte early and parses its last bytes itself.
pub(crate) fn copy_point(
    shared: u64,
    length: u64,
    newer: u64,
    finished: bool,
) -> Option<(u64, bool)> {
    if shared >= length && length == newer {
        None
    } else if shared >= length {
        Some(if finished {
            (length.saturating_sub(1), false)
        } else {
            (length, true)
        })
    } else {
        Some((shared, false))
    }
}

/// Bytes two stored chains of an input share from their starts, by the
/// input and each chain with the length it had: a chain that grows is
/// compared again.
pub(crate) type SharedPrefixes = HashMap<(String, (i64, u64), (i64, u64)), u64>;

/// Bytes two chains of one input share from their starts.
struct Shared<'a> {
    engine: &'a Engine,
    input: &'a str,
    layouts: &'a [ChainLayout],
}

impl Shared<'_> {
    /// Bytes chains `a` and `b` (`a` older) share from their starts. Bytes
    /// in the same blobs are alike without being read. Otherwise the chain
    /// after `a` decides most of it: if `a` shares a different number of
    /// bytes with it than it shares with `b`, `a` and `b` share the smaller
    /// number, and only where the two are equal are bytes read, from there.
    /// So each chain is read against the next one at most once per pass,
    /// however many newer chains it is compared with.
    fn between(&self, a: usize, b: usize) -> Result<u64, EngineError> {
        let (x, y) = (&self.layouts[a], &self.layouts[b]);
        let key = (
            self.input.to_owned(),
            (x.chain, x.length),
            (y.chain, y.length),
        );
        if let Some(n) = self.engine.shared_prefixes.borrow().get(&key) {
            return Ok(*n);
        }
        let mut n = x.shared_extents(y);
        if n < x.length.min(y.length) {
            let mut from = n;
            if b > a + 1 {
                let (next, onward) = (self.between(a, a + 1)?, self.between(a + 1, b)?);
                if next != onward {
                    from = x.length.min(y.length);
                    n = next.min(onward);
                } else {
                    from = from.max(next);
                }
            }
            if from < x.length.min(y.length) {
                // Bytes alike in different blobs: a chain rewritten as one
                // new blob, or cut into other ranges.
                let reader = self.engine.reads.reader()?;
                let mine = reader.stored_chain(self.input, x.chain, x.length)?;
                let theirs = reader.stored_chain(self.input, y.chain, y.length)?;
                n = mine.shared_prefix(&theirs, &self.engine.blobs, from)?;
            }
        }
        self.engine.shared_prefixes.borrow_mut().insert(key, n);
        Ok(n)
    }

    /// Whether newer chain `newer` can give an older chain its reading at
    /// `shared` bytes: being planned from at most there, or not read past it
    /// under the current reading.
    fn can_stop_at(
        &self,
        newer: usize,
        shared: u64,
        reading_version: &str,
        planned: Option<(usize, u64)>,
    ) -> bool {
        match planned {
            Some((chain, from)) if chain == newer => from <= shared,
            _ => self.layouts[newer].reading.as_ref().is_none_or(|reading| {
                reading.continues(reading_version) && reading.parsed_upto <= shared
            }),
        }
    }

    /// The chain older chain `older` takes its reading from, and the bytes
    /// they share. `planned` is a chain being planned now, with the position
    /// its parse starts from: it can stop anywhere after that.
    fn source(
        &self,
        older: usize,
        reading_version: &str,
        planned: Option<(usize, u64)>,
    ) -> Result<Option<(usize, u64)>, EngineError> {
        let length = self.layouts[older].length;
        // A newer chain holding all its bytes in the same blobs is found
        // without reading; no chain shares more.
        for newer in (older + 1..self.layouts.len()).rev() {
            if self.layouts[older].shared_extents(&self.layouts[newer]) >= length
                && length > 0
                && self.can_stop_at(newer, length, reading_version, planned)
            {
                return Ok(Some((newer, length)));
            }
        }
        let mut best: Option<(usize, u64)> = None;
        for newer in (older + 1..self.layouts.len()).rev() {
            let shared = self.between(older, newer)?;
            if best.is_some_and(|(_, most)| shared <= most) {
                continue;
            }
            if shared > 0 && self.can_stop_at(newer, shared, reading_version, planned) {
                best = Some((newer, shared));
                if shared >= length {
                    break;
                }
            }
        }
        Ok(best)
    }
}

/// Whether a plan's parse can stop for copies: a parser reading ranges in
/// order (lines or capture frames) over the stored bytes of a known chain.
fn copies_from(engine: &Engine, plan: &Plan) -> Option<i64> {
    if engine.archive.is_primary()
        || plan.replace
        || plan.byte_read.is_some()
        || plan.parser_deferred
        || !crate::has_reading(plan.origin)
        || plan.origin.is_row_dump(&plan.path)
        || matches!(
            plan.change,
            polylogue_store::Change::First | polylogue_store::Change::Rewritten
        )
    {
        return None;
    }
    parses_in_order(plan).then_some(plan.parser_chain).flatten()
}

/// Whether a plan's parser reads ranges in order (lines or capture frames),
/// so that its parse can stop at a position and continue from it.
pub(crate) fn parses_in_order(plan: &Plan) -> bool {
    matches!(
        (plan.origin.open)(&plan.path),
        Some(polylogue_sources::Parse::Lines(_) | polylogue_sources::Parse::CaptureLog(_))
    )
}

impl Engine {
    /// Gives `plan` the older chains of its input to record from its reading
    /// as its parse passes their stops.
    pub(crate) fn plan_prefix_copies(&self, plan: &mut Plan) -> Result<(), EngineError> {
        plan.prefix_copies.clear();
        let Some(chain) = copies_from(self, plan) else {
            return Ok(());
        };
        let layouts = self.reads.reader()?.chain_layouts(&plan.input.id)?;
        let Some(planned) = layouts.iter().position(|layout| layout.chain == chain) else {
            return Ok(());
        };
        if !layouts[..planned]
            .iter()
            .any(|layout| layout.reading.is_none())
        {
            return Ok(());
        }
        let version = plan.origin.reading_version();
        let shared = Shared {
            engine: self,
            input: &plan.input.id,
            layouts: &layouts,
        };
        for older in 0..planned {
            if layouts[older].reading.is_some() || layouts[older].length == 0 {
                continue;
            }
            let Some((source, bytes)) =
                shared.source(older, &version, Some((planned, plan.parse_pos)))?
            else {
                continue;
            };
            if source != planned {
                continue;
            }
            let Some((point, reaches_end)) = copy_point(
                bytes,
                layouts[older].length,
                layouts[planned].length,
                plan.input_finished,
            ) else {
                continue;
            };
            if point < plan.parse_pos {
                continue;
            }
            // The parse stops at a record's end, so it reads no byte twice.
            let stop = self.last_record_end(plan, &layouts[planned], point)?;
            if stop > 0 {
                plan.prefix_copies.push(crate::prefix_copies::PrefixCopy {
                    chain: layouts[older].chain,
                    stop,
                    reaches_end,
                });
            }
        }
        plan.prefix_copies.sort_by_key(|copy| copy.stop);
        Ok(())
    }

    /// The end of the last complete record of `plan`'s chain at or before
    /// `point`, and at or after the position its parse starts from, which is
    /// a record's end: where its parse stands once it has read the bytes
    /// before `point`. Lines end at a newline, found by reading back from
    /// `point`; capture frames are walked by their headers.
    fn last_record_end(
        &self,
        plan: &Plan,
        chain: &ChainLayout,
        point: u64,
    ) -> Result<u64, EngineError> {
        use polylogue_model::capture_log::{HEADER_LEN, Header, MAGIC};
        let from = plan.parse_pos;
        let reader = self.reads.reader()?;
        let bytes = reader.stored_chain(&plan.input.id, chain.chain, chain.length)?;
        let read = |start: u64, end: u64| bytes.read_range(&self.blobs, start, end);
        let end = if (plan.origin.open)(&plan.path).is_some_and(|parse| parse.is_capture_log()) {
            let mut at = from.max(MAGIC.len() as u64);
            loop {
                let header_end = at + HEADER_LEN as u64;
                if header_end > point {
                    break;
                }
                let Ok(header) = Header::decode(&read(at, header_end)?) else {
                    break;
                };
                match header_end.checked_add(header.length) {
                    Some(next) if next <= point => at = next,
                    _ => break,
                }
            }
            if at > MAGIC.len() as u64 { at } else { from }
        } else {
            let mut end = point;
            loop {
                if end <= from {
                    break from;
                }
                let start = from.max(end.saturating_sub(1 << 20));
                if let Some(newline) = read(start, end)?.iter().rposition(|&b| b == b'\n') {
                    break start + newline as u64 + 1;
                }
                end = start;
            }
        };
        bytes.release();
        Ok(end)
    }

    /// Records each older chain whose stop the plan's parse has reached from
    /// the reading the plan's chain has now. The pool reports a range only
    /// after the writer committed it, and the plan's next range is queued
    /// only after these copies, so each copy takes the reading at the plan's
    /// position. A copy that would differ is left out; that
    /// chain is parsed whole later in the pass.
    pub(crate) fn make_due_copies(
        &mut self,
        plan: &mut Plan,
        pass: &mut Pass,
    ) -> Result<(), EngineError> {
        while let Some(copy) = plan.prefix_copies.first() {
            let due = plan.failed || copy.stop <= plan.parse_pos || plan.read_to >= copy.stop;
            if !due {
                break;
            }
            let copy = plan.prefix_copies.remove(0);
            let Some(from) = plan.parser_chain else {
                continue;
            };
            // Nothing is recorded yet: the older chain is simply parsed.
            if plan.failed || plan.parse_pos == 0 {
                continue;
            }
            let answer = self.writer.copy_reading(polylogue_store::ReadingCopy {
                input: plan.input.id.clone(),
                from,
                to: copy.chain,
                reading_version: plan.origin.reading_version(),
                upto: plan.parse_pos,
                reaches_end: copy.reaches_end,
            })?;
            if matches!(answer, Ok(true)) {
                pass.readings_copied += 1;
                pass.bytes_copied += plan.parse_pos;
            }
        }
        Ok(())
    }

    /// The newer chain an older chain not read yet takes its reading from,
    /// with the bytes they share and the older chain's length.
    fn source_of(
        &self,
        input: &str,
        chain: i64,
        reading_version: &str,
    ) -> Result<Option<(ChainLayout, u64, u64)>, EngineError> {
        let layouts = self.reads.reader()?.chain_layouts(input)?;
        let Some(older) = layouts.iter().position(|layout| layout.chain == chain) else {
            return Ok(None);
        };
        if layouts[older].reading.is_some() {
            return Ok(None);
        }
        let shared = Shared {
            engine: self,
            input,
            layouts: &layouts,
        };
        Ok(shared
            .source(older, reading_version, None)?
            .map(|(newer, bytes)| (layouts[newer].clone(), bytes, layouts[older].length)))
    }

    /// Whether an older chain not read yet is copied after this pass's
    /// parsing ([`Engine::copy_now`]) rather than parsed: its source is
    /// identical to it, or is read as far as it will be. Any other source is
    /// still to be parsed, earlier in the pass, and copies to it on the way.
    pub(crate) fn copied_later(
        &self,
        input: &str,
        chain: i64,
        reading_version: &str,
        finished: bool,
    ) -> Result<bool, EngineError> {
        Ok(self
            .source_of(input, chain, reading_version)?
            .is_some_and(|(newer, bytes, length)| {
                copy_point(bytes, length, newer.length, finished).is_none()
                    || newer
                        .reading
                        .as_ref()
                        .is_some_and(|reading| reading.settled(newer.length))
            }))
    }

    /// The copy to take now for an older chain not read yet, from the
    /// reading its source has: at the source's position, which lies within
    /// the bytes they share. It reads the older chain to its end only when
    /// the source's parse has read through that end; otherwise the older
    /// chain's parse continues from the copy.
    pub(crate) fn copy_now(
        &self,
        input: &str,
        chain: i64,
        reading_version: &str,
        finished: bool,
    ) -> Result<Option<polylogue_store::ReadingCopy>, EngineError> {
        let Some((newer, bytes, length)) = self.source_of(input, chain, reading_version)? else {
            return Ok(None);
        };
        let Some(reading) = &newer.reading else {
            return Ok(None);
        };
        let (stop, reaches_end) =
            copy_point(bytes, length, newer.length, finished).unwrap_or((length, true));
        if reading.parsed_upto == 0 || reading.parsed_upto > stop {
            return Ok(None);
        }
        Ok(Some(polylogue_store::ReadingCopy {
            input: input.to_owned(),
            from: newer.chain,
            to: chain,
            reading_version: reading_version.to_owned(),
            upto: reading.parsed_upto,
            reaches_end: reaches_end && reading.settled(newer.length),
        }))
    }
}

#[cfg(test)]
mod tests;
