//! Canonical assertion facts and indexed graph lookups. Write policy resolves
//! selectors before this layer receives immutable records.
use crate::{Result, StoreError, source::Source};
use polylogue_model::{
    Author, BlobHash, Reference, RevisionReference, Timestamp,
    assertion::{
        Assertion, AssertionFact, AssertionKind, AssertionLink, LinkRole,
        currency::{Currency, CurrencyError, RevisionLinks},
    },
    authorship::Channel,
};
use rusqlite::{Connection, params};

pub mod conversion;
pub(crate) mod currency;
mod historical;
mod imports;
pub use historical::{apply_historical, get_evidence, restore_historical, visit_historical};
mod saved;
pub(crate) mod sql;
pub(crate) use saved::{
    CurrentSavedDefinition, SavedInput, SavedPreparation, SavedResolver, saved_definition,
    saved_queries, saved_query_snapshot,
};
mod read;
mod settings;
pub use imports::restore;
mod write;
pub(crate) use write::{WriteContext, revise_preparing, write_preparing};

pub const SCHEMA: &str = include_str!("schema/assertions.sql");
const COLUMNS: &str = "id,root,kind,key,value,author,via,machine,created_at,valid_until,commit_seq,imports,historical";

fn invalid(message: &str) -> StoreError {
    StoreError::Invalid(message.into())
}
fn time(value: String) -> Result<Timestamp> {
    Timestamp::parse_rfc3339(&value).ok_or_else(|| invalid("assertion time is out of range"))
}
fn record(row: &rusqlite::Row<'_>) -> Result<Assertion> {
    let kind: String = row.get(2)?;
    let author: String = row.get(5)?;
    let via: Option<String> = row.get(6)?;
    let value: Option<String> = row.get(4)?;
    Ok(Assertion {
        fact: AssertionFact {
            id: row.get(0)?,
            root: row.get(1)?,
            kind: AssertionKind::from_name(&kind)
                .ok_or_else(|| invalid("unknown assertion kind"))?,
            key: row.get(3)?,
            value: value
                .map(|value| serde_json::from_str(&value))
                .transpose()?,
            author: Author::parse(&author).map_err(|_| invalid("unknown assertion author"))?,
            via: via
                .map(|via| {
                    Channel::from_name(&via).ok_or_else(|| invalid("unknown assertion channel"))
                })
                .transpose()?,
            machine: row.get(7)?,
            created_at: row.get::<_, Option<String>>(8)?.map(time).transpose()?,
            valid_until: row.get::<_, Option<String>>(9)?.map(time).transpose()?,
            links: Vec::new(),
        },
        commit_seq: row
            .get::<_, i64>(10)?
            .try_into()
            .map_err(|_| invalid("negative assertion counter"))?,
        imports: serde_json::from_str(&row.get::<_, String>(11)?)?,
    })
}
fn links(conn: &Connection, id: &str) -> Result<Vec<AssertionLink>> {
    let source = Source::reading(conn)?;
    let mut statement = conn.prepare_cached(&source.sql(
        "SELECT role,ref,seen_hash FROM @source.assertion_link WHERE assertion=?1 ORDER BY role,ref",
    ))?;
    let mut rows = statement.query([id])?;
    let mut result = Vec::new();
    while let Some(row) = rows.next()? {
        let role: String = row.get(0)?;
        let reference: String = row.get(1)?;
        let hash: Option<Vec<u8>> = row.get(2)?;
        result.push(AssertionLink {
            role: LinkRole::from_name(&role)
                .ok_or_else(|| invalid("unknown assertion link role"))?,
            reference: Reference::parse(&reference)
                .map_err(|_| invalid("invalid stored assertion reference"))?,
            seen_hash: hash
                .map(|hash| {
                    hash.try_into()
                        .map(BlobHash)
                        .map_err(|_| invalid("assertion hash must contain 32 bytes"))
                })
                .transpose()?,
        });
    }
    Ok(result)
}

pub fn get(conn: &Connection, id: &str) -> Result<Option<Assertion>> {
    match get_evidence(conn, id)? {
        Some(polylogue_model::historical::AssertionEvidence::Current(assertion)) => {
            Ok(Some(assertion))
        }
        Some(historical) => Err(StoreError::ReferenceUnavailable {
            reference: historical.reference().to_string(),
            reason: "the recorded historical value has no validated current-kind schema".into(),
        }),
        None => Ok(None),
    }
}

/// Streams canonical facts, independently of currency, in deterministic order.
pub fn visit(
    conn: &Connection,
    after: Option<u64>,
    mut visit: impl FnMut(Assertion) -> Result<()>,
) -> Result<()> {
    let after = after
        .map(i64::try_from)
        .transpose()
        .map_err(|_| invalid("assertion counter exceeds SQLite's integer range"))?;
    let source = Source::reading(conn)?;
    let mut statement = conn.prepare(&source.sql(&format!(
        "SELECT {COLUMNS} FROM @source.assertion WHERE historical IS NULL AND (?1 IS NULL OR commit_seq>?1) ORDER BY commit_seq,id",
    )))?;
    let mut rows = statement.query([after])?;
    while let Some(row) = rows.next()? {
        let mut assertion = record(row)?;
        assertion.fact.links = links(conn, &assertion.fact.id)?;
        visit(assertion)?;
    }
    Ok(())
}

pub(crate) fn canonical_fact(fact: &AssertionFact) -> Result<AssertionFact> {
    fact.canonicalized()
        .map_err(|error| invalid(&error.to_string()))
}

/// The caller owns the transaction, including its local change counter. Exact
/// retransmission preserves the first admission; changed immutable facts fail.
pub fn apply(conn: &Connection, assertions: &[AssertionFact], counter: i64) -> Result<()> {
    if counter < 0 {
        return Err(invalid("negative assertion counter"));
    }
    let source = Source::of(conn)?;
    for assertion in assertions {
        let assertion = canonical_fact(assertion)?;
        if let Some(existing) = get_evidence(conn, &assertion.id)? {
            if !matches!(existing,polylogue_model::historical::AssertionEvidence::Current(existing) if existing.fact==assertion)
            {
                return Err(StoreError::Conflict(
                    exact(&assertion.root, &assertion.id).to_string(),
                ));
            }
            continue;
        }
        conn.execute(&source.sql("INSERT INTO @source.assertion(id,root,kind,key,value,author,via,machine,created_at,valid_until,commit_seq) VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11)"), params![
            assertion.id, assertion.root, assertion.kind.as_str(), assertion.key,
            assertion.value.map(|value| serde_json::to_string(&value)).transpose()?,
            assertion.author.to_string(), assertion.via.map(|via|via.as_str()), assertion.machine,
            assertion.created_at.map(|time|time.to_string()), assertion.valid_until.map(|time| time.to_string()), counter,
        ])?;
        for link in assertion.links {
            conn.execute(&source.sql("INSERT INTO @source.assertion_link(assertion,role,ref,seen_hash) VALUES(?1,?2,?3,?4)"), params![
                assertion.id, link.role.as_str(), link.reference.to_string(), link.seen_hash.map(|hash| hash.0.to_vec()),
            ])?;
        }
    }
    Ok(())
}

/// A private preparation overlay has its own counter; copied assertions keep
/// their original identities, authorship, timestamps and link facts.
pub fn copy(from: &Connection, to: &Connection) -> Result<()> {
    visit(from, None, |assertion| {
        apply(to, std::slice::from_ref(&assertion.fact), 0)?;
        imports::save(to, &assertion.fact.id, &assertion.imports)
    })?;
    visit_historical(from, None, |fact, _, history| {
        apply_historical(to, fact, 0)?;
        imports::save(to, &fact.id, history)
    })
}

fn exact(root: &str, revision: &str) -> Reference {
    Reference::Assertion(RevisionReference {
        root: root.into(),
        revision: Some(revision.into()),
    })
}

/// One graph adapter serves assertions and variants; all lookups use natural
/// references and the incoming-link/root indexes, never local row identifiers.
pub struct Graph<'a> {
    conn: &'a Connection,
    source: Source,
    evidence_until: Option<Timestamp>,
    exclusive: bool,
}
fn check_read() -> Result<()> {
    if let Some(limit) = crate::read::limit::ReadLimit::current() {
        limit.check()?;
    }
    Ok(())
}
impl<'a> Graph<'a> {
    pub fn new(conn: &'a Connection) -> Result<Self> {
        Ok(Self {
            conn,
            source: Source::reading(conn)?,
            evidence_until: None,
            exclusive: false,
        })
    }
    /// Historical evidence bounds the graph; it never chooses a revision.
    pub(crate) fn as_of(conn: &'a Connection, until: Timestamp) -> Result<Self> {
        let mut graph = Self::new(conn)?;
        graph.evidence_until = Some(until);
        Ok(graph)
    }
    pub(crate) fn before(conn: &'a Connection, until: Timestamp) -> Result<Self> {
        let mut graph = Self::as_of(conn, until)?;
        graph.exclusive = true;
        Ok(graph)
    }
    fn eligible(&self, reference: &Reference, created_at: Option<String>) -> Result<bool> {
        let Some(until) = self.evidence_until else {
            return Ok(true);
        };
        let Some(created_at) = created_at else {
            return Err(StoreError::ReferenceUnavailable {
                reference: reference.to_string(),
                reason:
                    "historical applicability is unknown because the affecting assertion is undated"
                        .into(),
            });
        };
        let created_at = time(created_at)?;
        Ok(if self.exclusive {
            created_at < until
        } else {
            created_at <= until
        })
    }
    fn incoming(&self, reference: &Reference, role: LinkRole) -> Result<Vec<Reference>> {
        check_read()?;
        let mut statement = self.conn.prepare_cached(&self.source.sql(
            "SELECT a.root,a.id,a.created_at FROM @source.assertion_link l JOIN @source.assertion a ON a.id=l.assertion WHERE l.ref=?1 AND l.role=?2 ORDER BY a.root,a.id",
        ))?;
        let mut rows = statement.query(params![reference.to_string(), role.as_str()])?;
        let mut incoming = Vec::new();
        while let Some(row) = rows.next()? {
            check_read()?;
            let reference = exact(&row.get::<_, String>(0)?, &row.get::<_, String>(1)?);
            if self.eligible(&reference, row.get(2)?)? {
                incoming.push(reference);
            }
        }
        Ok(incoming)
    }
}
impl RevisionLinks for Graph<'_> {
    type Error = StoreError;
    fn revisions(&mut self, root: &Reference) -> Result<Vec<Reference>> {
        check_read()?;
        let (table, name) = match root {
            Reference::Assertion(RevisionReference {
                root,
                revision: None,
            }) => ("assertion", root),
            Reference::Variant {
                revision:
                    RevisionReference {
                        root,
                        revision: None,
                    },
                part: None,
            } => ("variant", root),
            _ => {
                return Err(invalid(
                    "revision lookup requires an assertion or variant root",
                ));
            }
        };
        let mut statement = self.conn.prepare_cached(&self.source.sql(&format!(
            "SELECT id,created_at FROM @source.{table} WHERE root=?1 ORDER BY id",
        )))?;
        let mut rows = statement.query([name])?;
        let mut result = Vec::new();
        while let Some(row) = rows.next()? {
            let id: String = row.get(0)?;
            let reference = if table == "assertion" {
                exact(name, &id)
            } else {
                Reference::Variant {
                    revision: RevisionReference {
                        root: name.clone(),
                        revision: Some(id),
                    },
                    part: None,
                }
            };
            if self.eligible(&reference, row.get(1)?)? {
                result.push(reference);
            }
        }
        Ok(result)
    }
    fn revisers(&mut self, revision: &Reference) -> Result<Vec<Reference>> {
        check_read()?;
        match revision {
            Reference::Assertion(_) => self.incoming(revision, LinkRole::Revises),
            Reference::Variant {
                revision:
                    RevisionReference {
                        root,
                        revision: Some(id),
                    },
                part: None,
            } => {
                let mut statement =
                    self.conn.prepare_cached(&self.source.sql(
                        "SELECT id,created_at FROM @source.variant WHERE root=?1 AND base=?2 ORDER BY id",
                    ))?;
                let mut rows = statement.query(params![root, id])?;
                let mut revisers = Vec::new();
                while let Some(row) = rows.next()? {
                    let reference = Reference::Variant {
                        revision: RevisionReference {
                            root: root.clone(),
                            revision: Some(row.get(0)?),
                        },
                        part: None,
                    };
                    if self.eligible(&reference, row.get(1)?)? {
                        revisers.push(reference);
                    }
                }
                Ok(revisers)
            }
            _ => Err(invalid(
                "reviser lookup requires an exact assertion or variant",
            )),
        }
    }
    fn withdrawers(&mut self, target: &Reference) -> Result<Vec<Reference>> {
        self.incoming(target, LinkRole::Withdraws)
    }
}
fn graph_error(error: CurrencyError<StoreError>) -> StoreError {
    match error {
        CurrencyError::Source(error) => error,
        CurrencyError::NotRevision(reference) => invalid(&format!(
            "{reference} is not an assertion or variant revision"
        )),
        CurrencyError::Cycle(reference) => invalid(&format!("cyclic withdrawal at {reference}")),
    }
}
pub fn heads(conn: &Connection, root: &Reference) -> Result<Vec<Reference>> {
    Currency::new(&mut Graph::new(conn)?)
        .heads(root)
        .map_err(graph_error)
}

#[cfg(test)]
mod tests;
