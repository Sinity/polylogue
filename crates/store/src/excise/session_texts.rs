//! The texts only the excised sessions hold, for blanking shared whole
//! documents and delivered briefs, and which of them a blanking found.
use crate::Result;
use std::cell::RefCell;
use std::collections::BTreeSet;

#[derive(Default)]
pub(super) struct SessionTexts {
    texts: BTreeSet<String>,
    found: RefCell<BTreeSet<String>>,
}

impl SessionTexts {
    pub fn insert(&mut self, value: &str) -> Result<()> {
        self.texts.insert(value.to_owned());
        Ok(())
    }

    /// The texts of at least `minimum_bytes`, in byte order.
    pub fn visit(&self, minimum_bytes: usize, mut f: impl FnMut(&str) -> Result<()>) -> Result<()> {
        for value in self.texts.iter().filter(|v| v.len() >= minimum_bytes) {
            f(value)?;
        }
        Ok(())
    }

    pub fn reset_found(&self) -> Result<()> {
        self.found.borrow_mut().clear();
        Ok(())
    }

    pub fn found(&self, value: &str) -> Result<()> {
        if self.texts.contains(value) {
            self.found.borrow_mut().insert(value.to_owned());
        }
        Ok(())
    }

    pub fn missing(&self) -> Result<u64> {
        Ok((self.texts.len() - self.found.borrow().len()) as u64)
    }
}

#[cfg(test)]
mod tests;
