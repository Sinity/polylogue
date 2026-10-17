//! Cooperative acquisition quanta. A yield parks the existing worker, retaining
//! its one SQLite view, encoder and checked-object readers. It is not publication.
use crate::{Error, Result};
use std::{
    cell::RefCell,
    rc::Rc,
    time::{Duration, Instant},
};

#[derive(Clone, Copy, Debug)]
pub struct Budget {
    pub bytes: u64,
    pub checkpoints: u64,
}
impl Default for Budget {
    fn default() -> Self {
        Self {
            bytes: 8 * 1024 * 1024,
            checkpoints: 512,
        }
    }
}
#[derive(Clone, Copy, Debug)]
pub struct Progress {
    pub completed_turns: u64,
    pub charged_bytes: u64,
    pub checkpoints: u64,
    pub phase: &'static str,
    /// Work observed so far, including rereads; never a selected frontier.
    pub provider_bytes_read: u64,
}
pub(super) type Control<'a> = Rc<RefCell<Turns<'a>>>;
pub(super) struct Turns<'a> {
    budget: Budget,
    visit: &'a mut dyn FnMut(Progress) -> Result<()>,
    bytes: u64,
    checkpoints: u64,
    completed: u64,
    provider_bytes_read: u64,
    started: Instant,
}
impl<'a> Turns<'a> {
    pub fn new(
        budget: Budget,
        visit: &'a mut dyn FnMut(Progress) -> Result<()>,
    ) -> Result<Control<'a>> {
        if budget.bytes < 65536 || budget.checkpoints == 0 {
            return Err(Error::ObservedSourceRefused(
                "history quantum needs at least 64 KiB and one checkpoint".into(),
            ));
        }
        Ok(Rc::new(RefCell::new(Self {
            budget,
            visit,
            bytes: 0,
            checkpoints: 0,
            completed: 0,
            provider_bytes_read: 0,
            started: Instant::now(),
        })))
    }
    pub fn page_read(&mut self, bytes: u64) -> Result<()> {
        self.provider_bytes_read = self
            .provider_bytes_read
            .checked_add(bytes)
            .ok_or(Error::Range)?;
        Ok(())
    }
    pub fn window(&self) -> u64 {
        self.budget.bytes.min(4 * 1024 * 1024)
    }
    pub fn work(&mut self, phase: &'static str, bytes: u64) -> Result<()> {
        if bytes > self.budget.bytes {
            return Err(Error::Invariant(
                "history work unit exceeds its quantum".into(),
            ));
        }
        if self.started.elapsed() >= Duration::from_secs(300) {
            return Err(Error::ObservedSourceRefused(
                "history active turn exceeded 300 seconds".into(),
            ));
        }
        if self.checkpoints == self.budget.checkpoints
            || self
                .bytes
                .checked_add(bytes)
                .is_none_or(|sum| sum > self.budget.bytes)
        {
            self.completed = self.completed.checked_add(1).ok_or(Error::Range)?;
            (self.visit)(Progress {
                completed_turns: self.completed,
                charged_bytes: self.bytes,
                checkpoints: self.checkpoints,
                phase,
                provider_bytes_read: self.provider_bytes_read,
            })?;
            // Time awaiting another owner grant is not active work and does not
            // turn a per-turn budget/deadline into an input-size admission.
            self.started = Instant::now();
            self.bytes = 0;
            self.checkpoints = 0;
        }
        self.bytes += bytes;
        self.checkpoints += 1;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn quanta_reuse_capacity() {
        let mut visits = Vec::new();
        let mut visit = |p| {
            visits.push(p);
            Ok(())
        };
        let control = Turns::new(
            Budget {
                bytes: 65536,
                checkpoints: 2,
            },
            &mut visit,
        )
        .unwrap();
        for _ in 0..10 {
            control.borrow_mut().work("pages", 65536).unwrap();
        }
        assert_eq!(control.borrow().completed, 9);
        drop(control);
        assert_eq!(visits.len(), 9);
        assert!(
            visits
                .iter()
                .all(|p| p.charged_bytes == 65536 && p.checkpoints == 1)
        );
    }
}
