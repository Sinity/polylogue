//! Count, digest and display sample of a Session's contributing inputs.
use super::*;
#[derive(Clone)]
pub(super) struct ContributorDigest {
    pub count: u64,
    pub observed: bool,
    hash: blake3::Hasher,
    sampled: usize,
    sample_bytes: usize,
    required_bytes: usize,
}
impl std::fmt::Debug for ContributorDigest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ContributorDigest")
            .field("count", &self.count)
            .field("observed", &self.observed)
            .finish()
    }
}
impl Default for ContributorDigest {
    fn default() -> Self {
        let mut hash = blake3::Hasher::new();
        hash.update(b"polylogue.session.input-bases.v1\0");
        Self {
            count: 0,
            observed: false,
            hash,
            sampled: 0,
            sample_bytes: 0,
            required_bytes: 0,
        }
    }
}
impl ContributorDigest {
    pub fn include(
        &mut self,
        basis: &ParsedFrom,
        required: bool,
        allowance: usize,
    ) -> Result<bool, Error> {
        let bytes = serde_json::to_vec(basis).map_err(|e| Error::Invariant(e.to_string()))?;
        self.count = self.count.checked_add(1).ok_or(Error::Range)?;
        self.observed |= basis.original_hash.is_none();
        self.hash.update(&(bytes.len() as u64).to_be_bytes());
        self.hash.update(&bytes);
        if required {
            self.required_bytes = self.required_bytes.saturating_add(bytes.len());
            if self.required_bytes > allowance {
                return Err(Error::ParserAdmission {
                    resource: "Session relevant input bases",
                    required: self.required_bytes as u64,
                });
            }
            Ok(true)
        } else if self.sampled < 8 && self.sample_bytes.saturating_add(bytes.len()) <= allowance / 4
        {
            self.sampled += 1;
            self.sample_bytes += bytes.len();
            Ok(true)
        } else {
            Ok(false)
        }
    }
    pub fn digest(&self) -> String {
        self.hash.finalize().to_hex().to_string()
    }
}
impl SessionStatus {
    pub(super) fn add_contributor(
        &mut self,
        basis: ParsedFrom,
        member: bool,
        allowance: usize,
    ) -> Result<(), Error> {
        // Every contributor is in the digest; only membership sources count
        // toward the single-member exception.
        if member {
            self.membership_sources = self.membership_sources.checked_add(1).ok_or(Error::Range)?;
        }
        if self
            .contributor_digest
            .include(&basis, self.exact && member, allowance)?
        {
            self.contributors.push(basis);
        }
        Ok(())
    }
    pub(super) fn include_input(&mut self, input: Value, allowance: usize) -> Result<(), Error> {
        // Catalog checks do not return per-Session source paths.
        if !self.exact {
            return Ok(());
        }
        let bytes = serde_json::to_vec(&input)
            .map_err(|e| Error::Invariant(e.to_string()))?
            .len();
        self.input_bytes = self.input_bytes.saturating_add(bytes);
        if self.input_bytes > allowance {
            return Err(Error::ParserAdmission {
                resource: "Session source-path metadata",
                required: self.input_bytes as u64,
            });
        }
        self.inputs.push(input);
        Ok(())
    }
}

#[cfg(test)]
#[path = "contributors_tests.rs"]
mod tests;
