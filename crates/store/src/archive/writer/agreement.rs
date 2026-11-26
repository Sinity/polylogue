//! Where each new chain of an input stops agreeing with what came before.
//!
//! When a chain is stored, the record keeps the earlier chain of the same
//! input its bytes begin like for longest (the newest of equals), and
//! through which byte. Each chain names one earlier chain, so an input's
//! chains form a tree, and how far any two of them are known to agree is
//! the smallest agreement on the path between them: a lower bound, never
//! longer than the true one.
//!
//! The comparison that finds a new chain's agreement reads bytes only where
//! arithmetic over recorded values cannot decide. The new bytes are
//! compared with the input's latest chain first, which gives their exact
//! agreement `d`. For every other earlier chain the recorded agreement `a`
//! with the latest is known: when `a` and `d` differ, the new bytes agree
//! with it through the smaller of the two; only when they are equal can the
//! new bytes agree with it further, and those bytes are compared from `d` on.
use crate::Result;
use crate::row::{Rows as _, row};
use std::collections::BTreeMap;
use std::io::Read as _;

/// One chain's record: the earlier chain it begins like and through which
/// byte.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize, schemars::JsonSchema,
)]
pub struct Agreement {
    pub begins_like: Option<i64>,
    pub agrees_to: u64,
}

impl Agreement {
    pub const NONE: Self = Self {
        begins_like: None,
        agrees_to: 0,
    };
}

/// The recorded chains of one input, by chain number.
#[derive(Debug, Clone, Default)]
pub struct Chains(pub BTreeMap<i64, Agreement>);

impl Chains {
    /// How far chains `a` and `b` are known to agree: the smallest
    /// agreement on the path between them; `None` when either is unknown.
    pub fn known(&self, a: i64, b: i64) -> Option<u64> {
        if a == b {
            return Some(u64::MAX);
        }
        let path = |mut chain: i64| {
            let mut path = vec![(chain, u64::MAX)];
            while let Some(Agreement {
                begins_like: Some(parent),
                agrees_to,
            }) = self.0.get(&chain).copied()
            {
                path.push((parent, agrees_to));
                chain = parent;
            }
            path
        };
        let (from_a, from_b) = (path(a), path(b));
        // The weights on each path from a chain up to an ancestor; the first
        // ancestor both reach joins them.
        let mut least = u64::MAX;
        for (n, (chain, weight)) in from_a.iter().enumerate() {
            if n > 0 {
                least = least.min(*weight);
            }
            let mut other = u64::MAX;
            for (m, (shared, weight)) in from_b.iter().enumerate() {
                if m > 0 {
                    other = other.min(*weight);
                }
                if shared == chain {
                    return Some(least.min(other));
                }
            }
        }
        None
    }

    /// The newer chains of the input that `chain` is known to agree with,
    /// and through which byte: the agreement any newer chain's reading
    /// holds the same rows for.
    pub fn newer(&self, chain: i64) -> impl Iterator<Item = (i64, u64)> + '_ {
        self.0
            .range(chain + 1..)
            .filter_map(move |(&newer, _)| Some((newer, self.known(chain, newer)?)))
            .filter(|(_, known)| *known > 0)
    }

    /// What the record says of `chain`, stored to `stored` bytes: the
    /// newest chain known to begin with all of it, and the byte through
    /// which its records are a newer chain's (the most any newer chain is
    /// known to agree with it).
    pub fn repeats(&self, chain: i64, stored: u64) -> (Option<i64>, u64) {
        let mut repeated_by = None;
        let mut own_from = 0;
        for (newer, known) in self.newer(chain) {
            if stored > 0 && known >= stored {
                repeated_by = Some(newer);
            }
            own_from = own_from.max(known.min(stored));
        }
        (repeated_by, own_from)
    }

    /// The agreement of new bytes with the earlier chains of this input.
    /// `latest` is the input's newest earlier chain and `compare(chain,
    /// from)` the first byte at which the new bytes and `chain` differ,
    /// comparing from `from` (bytes before it are known to be alike), or the
    /// shorter length when one begins the other.
    pub fn agreement(
        &self,
        latest: Option<i64>,
        mut compare: impl FnMut(i64, u64) -> Result<u64>,
    ) -> Result<Agreement> {
        let Some(latest) = latest else {
            return Ok(Agreement::NONE);
        };
        let exact = compare(latest, 0)?;
        let mut best = (exact, latest);
        for (&chain, _) in self.0.iter().rev() {
            if chain == latest {
                continue;
            }
            // Chains in another tree of the input are known to agree on nothing.
            let known = self.known(latest, chain).unwrap_or(0);
            let value = if known == exact {
                compare(chain, exact)?
            } else {
                known.min(exact)
            };
            // Newest of equals: chains are visited newest first.
            if value > best.0 {
                best = (value, chain);
            }
        }
        Ok(if best.0 == 0 {
            Agreement::NONE
        } else {
            Agreement {
                begins_like: Some(best.1),
                agrees_to: best.0,
            }
        })
    }
}

row! {
    /// One chain's stored agreement with the earlier chain it begins like.
    struct ChainRow {
        input: String,
        number: i64,
        begins_like: Option<i64>,
        agrees_to: i64,
    }
}

/// The recorded chains, on any connection that reaches the archive
/// ([`crate::source::Source::reading`]).
const RECORDED: &str = "SELECT input, number, begins_like, agrees_to FROM @source.chain";

fn insert(chains: &mut Chains, row: ChainRow) -> Result<()> {
    let agrees_to = u64::try_from(row.agrees_to)
        .map_err(|_| crate::StoreError::stored(2, "negative chain agreement"))?;
    chains.0.insert(
        row.number,
        Agreement {
            begins_like: row.begins_like,
            agrees_to,
        },
    );
    Ok(())
}

/// The recorded chains of `input`, every one of them: the comparison that
/// finds a new chain's agreement visits each earlier chain.
pub fn recorded(conn: &rusqlite::Connection, input: &str) -> Result<Chains> {
    let source = crate::source::Source::reading(conn)?;
    let statement = crate::row::Statement::<ChainRow>::built(
        source.sql(&format!("{RECORDED} WHERE input = ?1")),
    );
    let mut chains = Chains::default();
    for row in conn.all(&statement, [input])? {
        insert(&mut chains, row)?;
    }
    Ok(chains)
}

/// The recorded chains that begin like an earlier one, by input, of every
/// input or of the `selected` inputs only. A chain that begins like none
/// is left out: no newer chain is known to agree with it through it, and
/// [`Chains::known`] walks no further than the chains listed.
pub fn recorded_by_input(
    conn: &rusqlite::Connection,
    selected: Option<&[String]>,
) -> Result<std::collections::HashMap<String, Chains>> {
    let source = crate::source::Source::reading(conn)?;
    let (sql, json) = match selected {
        Some(inputs) => (
            format!(
                "{RECORDED} WHERE begins_like IS NOT NULL AND input IN (SELECT value FROM json_each(?1))"
            ),
            Some(serde_json::to_string(inputs)?),
        ),
        None => (
            format!("{RECORDED} WHERE begins_like IS NOT NULL AND ?1 IS NULL"),
            None,
        ),
    };
    let statement = crate::row::Statement::<ChainRow>::built(source.sql(&sql));
    let mut by_input = std::collections::HashMap::<String, Chains>::new();
    for row in conn.all(&statement, [json])? {
        let chains = by_input.entry(row.input.clone()).or_default();
        insert(chains, row)?;
    }
    Ok(by_input)
}

#[cfg(test)]
pub(crate) fn declared_rows(check: &mut crate::row::Check<'_>) {
    check.row::<ChainRow>();
}

/// Where two streams of bytes first differ, both read from `from` on (the
/// bytes before it are known to be alike); the shorter length when one
/// begins the other.
pub fn first_difference(
    mut a: impl std::io::Read,
    mut b: impl std::io::Read,
    from: u64,
) -> Result<u64> {
    std::io::copy(&mut (&mut a).take(from), &mut std::io::sink())?;
    std::io::copy(&mut (&mut b).take(from), &mut std::io::sink())?;
    let (mut left, mut right) = (vec![0; 64 << 10], vec![0; 64 << 10]);
    let mut at = from;
    loop {
        let n = fill(&mut a, &mut left)?;
        let m = fill(&mut b, &mut right)?;
        let both = n.min(m);
        if let Some(differs) = left[..both]
            .iter()
            .zip(&right[..both])
            .position(|(x, y)| x != y)
        {
            return Ok(at + differs as u64);
        }
        at += both as u64;
        if n != m || n < left.len() {
            return Ok(at);
        }
    }
}

/// Reads until `buffer` is full or the stream ends.
fn fill(reader: &mut impl std::io::Read, buffer: &mut [u8]) -> Result<usize> {
    let mut filled = 0;
    while filled < buffer.len() {
        match reader.read(&mut buffer[filled..])? {
            0 => break,
            n => filled += n,
        }
    }
    Ok(filled)
}

#[cfg(test)]
mod tests;
