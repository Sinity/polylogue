//! Annotation records, their links and the stored current-revision index. Write policy
//! resolves selectors before this layer receives immutable records; every
//! record written here updates `revision_state` for exactly the roots it can
//! affect, in the same transaction.
use crate::encoding::Columns;
use crate::{Result, StoreError, source::Source};
use authored::{Currency, CurrencyError, Holders, Links, Revision, Target, UniqueSlot};
use polylogue_model::{
    Author, BlobHash, Reference, RevisionReference, Timestamp,
    annotation::{Annotation, AnnotationFact, AnnotationKind, AnnotationLink, LinkRole, Quote},
    authorship::Channel,
};
use rusqlite::{Connection, params};
use std::collections::BTreeSet;

pub(crate) mod current_revisions;
mod saved;
pub(crate) mod sql;
pub(crate) use saved::{
    CurrentSavedDefinition, SavedInput, SavedPreparation, SavedResolver, named_current,
    named_revision, saved_query_snapshot, saved_revision_number, single_named_current,
};
pub(crate) mod imports;
mod read;
pub(crate) mod seen;
pub(crate) mod settings;
mod write;
pub(crate) use write::{WriteContext, revise_preparing, write_parts_preparing, write_preparing};

pub const SCHEMA: &str = include_str!("schema/annotations.sql");
const COLUMNS: &str = "id,root,kind,key,value,author,via,machine,created_at,valid_until,commit_seq";

fn record(row: &rusqlite::Row<'_>) -> Result<Annotation> {
    let value: Option<String> = row.get(4)?;
    Ok(Annotation {
        fact: AnnotationFact {
            id: row.get(0)?,
            root: row.get(1)?,
            kind: row.name::<AnnotationKind>(2)?,
            key: row.get(3)?,
            value: value
                .map(|value| serde_json::from_str(&value))
                .transpose()?,
            author: row.id::<Author>(5)?,
            via: row.name::<Channel>(6)?,
            machine: row.get(7)?,
            created_at: row.optional_time_ms(8)?,
            valid_until: row.optional_time_ms(9)?,
            links: Vec::new(),
        },
        commit_seq: row.get(10)?,
    })
}
fn links(conn: &Connection, id: &str) -> Result<Vec<AnnotationLink>> {
    let source = Source::reading(conn)?;
    let mut statement = conn.prepare_cached(&source.sql(
        "SELECT role,ref,seen_hash,quote FROM @source.annotation_link WHERE annotation=?1 ORDER BY role,ref",
    ))?;
    let mut rows = statement.query([id])?;
    let mut result = Vec::new();
    while let Some(row) = rows.next()? {
        let reference: String = row.get(1)?;
        let quote: Option<String> = row.get(3)?;
        result.push(AnnotationLink {
            role: row.name::<LinkRole>(0)?,
            reference: Reference::parse(&reference).map_err(|error| {
                StoreError::stored(1, format!("reference {reference:?}: {error}"))
            })?,
            seen_hash: row.optional_hash(2)?.map(BlobHash),
            quote: quote
                .map(|quote| serde_json::from_str::<Quote>(&quote))
                .transpose()?,
        });
    }
    Ok(result)
}

/// One stored revision by its ID, with its links.
pub fn get(conn: &Connection, id: &str) -> Result<Option<Annotation>> {
    let source = Source::reading(conn)?;
    let mut statement = conn.prepare_cached(&source.sql(&format!(
        "SELECT {COLUMNS} FROM @source.annotation WHERE id=?1"
    )))?;
    let mut rows = statement.query([id])?;
    let Some(row) = rows.next()? else {
        return Ok(None);
    };
    let mut annotation = record(row)?;
    annotation.fact.links = links(conn, id)?;
    Ok(Some(annotation))
}

/// Whether any revision of an annotation family is stored.
pub(crate) fn family_present(conn: &Connection, root: &str) -> Result<bool> {
    let source = Source::reading(conn)?;
    Ok(conn
        .prepare_cached(&source.sql("SELECT 1 FROM @source.annotation WHERE root=?1 LIMIT 1"))?
        .exists([root])?)
}

/// Streams stored facts, current or not, in deterministic order.
pub fn visit(
    conn: &Connection,
    after: Option<u64>,
    mut visit: impl FnMut(Annotation) -> Result<()>,
) -> Result<()> {
    let after = after.map(i64::try_from).transpose().map_err(|_| {
        crate::StoreError::internal("commit position exceeds SQLite's integer range")
    })?;
    let source = Source::reading(conn)?;
    let predicate = if after.is_some() {
        "commit_seq>?1"
    } else {
        "?1 IS NULL"
    };
    let mut statement = conn.prepare(&source.sql(&format!(
        "SELECT {COLUMNS} FROM @source.annotation WHERE {predicate} ORDER BY commit_seq,id",
    )))?;
    let mut rows = statement.query([after])?;
    while let Some(row) = rows.next()? {
        let mut annotation = record(row)?;
        annotation.fact.links = links(conn, &annotation.fact.id)?;
        visit(annotation)?;
    }
    Ok(())
}

/// Families whose current revisions can change when these newly committed facts arrive.
/// Includes withdrawal/reinstatement targets through the owning link rules.
pub(crate) fn changed_roots(conn: &Connection, after: i64) -> Result<BTreeSet<String>> {
    let after = u64::try_from(after)
        .map_err(|_| crate::StoreError::stored_data("negative commit position"))?;
    let mut links = StoredLinks::new(conn)?;
    let mut roots = BTreeSet::new();
    visit(conn, Some(after), |annotation| {
        roots.extend(authored::affected_roots(
            &mut links,
            &header(&annotation.fact),
        )?);
        Ok(())
    })?;
    Ok(roots)
}

pub(crate) fn canonical_fact(fact: &AnnotationFact) -> Result<AnnotationFact> {
    fact.canonicalized().map_err(crate::StoreError::internal)
}

/// The exact rubric revision an annotation was derived from: the one
/// `derived_from` link to a stored rubric.
fn rubric_of<'a>(
    conn: &Connection,
    links: impl IntoIterator<Item = (LinkRole, &'a Reference)>,
) -> Result<Option<Reference>> {
    let mut rubrics = BTreeSet::new();
    for (role, reference) in links {
        if role != LinkRole::DerivedFrom {
            continue;
        }
        let Reference::Annotation(RevisionReference {
            root,
            revision: Some(id),
        }) = reference
        else {
            continue;
        };
        if get(conn, id)?
            .is_some_and(|row| row.fact.root == *root && row.fact.kind == AnnotationKind::Rubric)
        {
            rubrics.insert(reference.clone());
        }
    }
    if rubrics.len() > 1 {
        return Err(StoreError::Ambiguous {
            reference: "annotation rubric".into(),
            candidates: rubrics.iter().map(ToString::to_string).collect(),
        });
    }
    Ok(rubrics.pop_first())
}

/// The uniqueness slots a stored fact occupies.
fn slots_of(conn: &Connection, fact: &AnnotationFact) -> Result<Vec<UniqueSlot>> {
    let rubric = if fact.kind == AnnotationKind::RubricValue {
        // A stored fact was validated when written, and its rubric may arrive
        // later (a bundle, the confirmed stream): with no stored rubric, the
        // one exact annotation it was derived from is the rubric.
        match rubric_of(
            conn,
            fact.links.iter().map(|link| (link.role, &link.reference)),
        )? {
            Some(rubric) => Some(rubric),
            None => {
                let mut derived = fact.links.iter().filter(|link| {
                    link.role == LinkRole::DerivedFrom
                        && matches!(
                            link.reference,
                            Reference::Annotation(RevisionReference {
                                revision: Some(_),
                                ..
                            })
                        )
                });
                match (derived.next(), derived.next()) {
                    (Some(link), None) => Some(link.reference.clone()),
                    _ => None,
                }
            }
        }
    } else {
        None
    };
    Ok(fact
        .unique(rubric.as_ref())
        .map_err(crate::StoreError::stored_data)?
        .map(|unique| unique.slots)
        .unwrap_or_default())
}

/// The shared header of a stored fact: its revisions and withdrawals of
/// annotations. Withdrawals of alternates name no annotation root.
pub(crate) fn header(fact: &AnnotationFact) -> authored::Header {
    let mut revises = BTreeSet::new();
    let mut withdraws = BTreeSet::new();
    for link in &fact.links {
        match (link.role, &link.reference) {
            (
                LinkRole::Revises,
                Reference::Annotation(RevisionReference {
                    root,
                    revision: Some(id),
                }),
            ) => {
                revises.insert(Revision::new(root, id));
            }
            (LinkRole::Withdraws, Reference::Annotation(revision)) => {
                withdraws.insert(target(revision));
            }
            _ => {}
        }
    }
    authored::Header {
        root: fact.root.clone(),
        rev: fact.id.clone(),
        kind: fact.kind.as_str().into(),
        key: fact.key.clone(),
        value: fact.value.clone(),
        author: fact.author.to_string(),
        via: fact.via.as_str().into(),
        at: fact.created_at.map(Timestamp::unix_ms),
        revises,
        withdraws,
    }
}

fn target(revision: &RevisionReference) -> Target {
    match &revision.revision {
        Some(id) => Target::Revision(Revision::new(&revision.root, id)),
        None => Target::Root(revision.root.clone()),
    }
}

/// Stores facts with their links and slots and brings `revision_state` up to
/// date for every root they can affect. The caller owns the transaction,
/// including its write counter. An exact repeat of a stored fact changes
/// nothing; a different fact under a stored ID is refused.
pub fn apply(conn: &Connection, annotations: &[AnnotationFact], counter: i64) -> Result<()> {
    if counter < 0 {
        return Err(crate::StoreError::stored_data("negative commit position"));
    }
    let source = Source::of(conn)?;
    let mut written = Vec::new();
    for annotation in annotations {
        let annotation = canonical_fact(annotation)?;
        if let Some(existing) = get(conn, &annotation.id)? {
            if existing.fact != annotation {
                return Err(StoreError::Conflict(
                    exact(&annotation.root, &annotation.id).to_string(),
                ));
            }
            continue;
        }
        let slots = slots_of(conn, &annotation)?;
        insert(conn, source, &annotation, &slots, counter)?;
        written.push(header(&annotation));
    }
    let mut stored = StoredLinks::new(conn)?;
    let mut roots = BTreeSet::new();
    for record in &written {
        roots.extend(authored::affected_roots(&mut stored, record)?);
    }
    current_revisions::recompute(conn, roots)
}

/// Inserts one fact, its links and its slots; the caller recomputes current revisions.
pub(crate) fn insert(
    conn: &Connection,
    source: Source,
    annotation: &AnnotationFact,
    slots: &[UniqueSlot],
    counter: i64,
) -> Result<()> {
    conn.execute(&source.sql("INSERT INTO @source.annotation(id,root,kind,key,value,author,via,machine,created_at,valid_until,commit_seq) VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11)"), params![
        annotation.id, annotation.root, annotation.kind.as_str(), annotation.key,
        annotation.value.as_ref().map(serde_json::to_string).transpose()?,
        annotation.author.to_string(), annotation.via.as_str(), annotation.machine,
        annotation.created_at.map(Timestamp::unix_ms), annotation.valid_until.map(Timestamp::unix_ms), counter,
    ])?;
    for link in &annotation.links {
        conn.execute(
            &source.sql(
                "INSERT INTO @source.annotation_link(annotation,role,ref,seen_hash,quote) VALUES(?1,?2,?3,?4,?5)",
            ),
            params![
                annotation.id,
                link.role.as_str(),
                link.reference.to_string(),
                link.seen_hash.map(|hash| hash.0.to_vec()),
                link.quote.as_ref().map(serde_json::to_string).transpose()?,
            ],
        )?;
    }
    for slot in slots {
        conn.execute(
            &source.sql(
                "INSERT OR IGNORE INTO @source.annotation_unique_key(key_hash,id) VALUES(?1,?2)",
            ),
            params![slot.digest().as_slice(), annotation.id],
        )?;
    }
    Ok(())
}

/// A private preparation overlay has its own counter; copied annotations keep
/// their original identities, authorship, timestamps and links.
pub fn copy(from: &Connection, to: &Connection) -> Result<()> {
    visit(from, None, |annotation| {
        apply(to, std::slice::from_ref(&annotation.fact), 0)
    })
}

fn exact(root: &str, revision: &str) -> Reference {
    Reference::Annotation(RevisionReference {
        root: root.into(),
        revision: Some(revision.into()),
    })
}

fn reference(revision: &Revision) -> Reference {
    exact(&revision.root, &revision.rev)
}

fn target_reference(target: &Target) -> Reference {
    match target {
        Target::Root(root) => Reference::Annotation(RevisionReference {
            root: root.clone(),
            revision: None,
        }),
        Target::Revision(revision) => reference(revision),
    }
}

fn check_read() -> Result<()> {
    if let Some(limit) = crate::read::limit::ReadLimit::current() {
        limit.check()?;
    }
    Ok(())
}

/// The links of stored annotations, for the shared current-revision and request rules.
/// Reads go through the connection's source tables; an as-of view keeps only
/// records written by a time.
pub(crate) struct StoredLinks<'a> {
    conn: &'a Connection,
    source: Source,
    until: Option<Until>,
    through: Option<i64>,
}

#[derive(Clone, Copy)]
struct Until {
    millis: i64,
    inclusive: bool,
}

impl<'a> StoredLinks<'a> {
    pub(crate) fn new(conn: &'a Connection) -> Result<Self> {
        Ok(Self {
            conn,
            source: Source::reading(conn)?,
            until: None,
            through: None,
        })
    }
    fn at_annotation(
        conn: &'a Connection,
        at: polylogue_model::read::AnnotationAt,
    ) -> Result<Self> {
        match at {
            polylogue_model::read::AnnotationAt::Position(through) => {
                Self::at_commit(conn, through)
            }
            polylogue_model::read::AnnotationAt::Time(until) => Self::as_of(conn, until),
        }
    }
    /// Facts accepted through one local archive write counter.
    pub(crate) fn at_commit(conn: &'a Connection, through: i64) -> Result<Self> {
        Ok(Self {
            through: Some(through),
            ..Self::new(conn)?
        })
    }
    fn committed(&self, commit: i64) -> bool {
        self.through.is_none_or(|through| commit <= through)
    }
    /// Records written at or before `until`, by the writer's clock.
    pub(crate) fn as_of(conn: &'a Connection, until: Timestamp) -> Result<Self> {
        Ok(Self {
            until: Some(Until {
                millis: until.unix_ms(),
                inclusive: true,
            }),
            ..Self::new(conn)?
        })
    }
    /// Records written strictly before `until`.
    pub(crate) fn before(conn: &'a Connection, until: Timestamp) -> Result<Self> {
        Ok(Self {
            until: Some(Until {
                millis: until.unix_ms(),
                inclusive: false,
            }),
            ..Self::new(conn)?
        })
    }
    fn in_view(&self, created_at: Option<i64>) -> bool {
        match self.until {
            None => true,
            Some(Until {
                millis,
                inclusive: true,
            }) => created_at.is_some_and(|at| at <= millis),
            Some(Until {
                millis,
                inclusive: false,
            }) => created_at.is_some_and(|at| at < millis),
        }
    }
    fn incoming(&self, reference: &Reference, role: LinkRole) -> Result<Vec<Revision>> {
        check_read()?;
        let mut statement = self.conn.prepare_cached(&self.source.sql(
            "SELECT a.root,a.id,a.created_at,a.commit_seq FROM @source.annotation_link l JOIN @source.annotation a ON a.id=l.annotation WHERE l.ref=?1 AND l.role=?2 ORDER BY a.root,a.id",
        ))?;
        let mut rows = statement.query(params![reference.to_string(), role.as_str()])?;
        let mut incoming = Vec::new();
        while let Some(row) = rows.next()? {
            check_read()?;
            if self.in_view(row.get(2)?) && self.committed(row.get(3)?) {
                incoming.push(Revision::new(
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                ));
            }
        }
        Ok(incoming)
    }
}

impl Links for StoredLinks<'_> {
    type Error = StoreError;
    fn revisions(&mut self, root: &str) -> Result<Vec<Revision>> {
        check_read()?;
        let mut statement = self.conn.prepare_cached(&self.source.sql(
            "SELECT id,created_at,commit_seq FROM @source.annotation WHERE root=?1 ORDER BY id",
        ))?;
        let mut rows = statement.query([root])?;
        let mut result = Vec::new();
        while let Some(row) = rows.next()? {
            if self.in_view(row.get(1)?) && self.committed(row.get(2)?) {
                result.push(Revision::new(root, row.get::<_, String>(0)?));
            }
        }
        Ok(result)
    }
    fn revisers(&mut self, revision: &Revision) -> Result<Vec<Revision>> {
        self.incoming(&reference(revision), LinkRole::Revises)
    }
    fn withdrawers(&mut self, target: &Target) -> Result<Vec<Revision>> {
        self.incoming(&target_reference(target), LinkRole::Withdraws)
    }
    fn withdraws(&mut self, record: &Revision) -> Result<Vec<Target>> {
        check_read()?;
        let mut statement = self.conn.prepare_cached(&self.source.sql(
            "SELECT ref FROM @source.annotation_link WHERE annotation=?1 AND role='withdraws' ORDER BY ref",
        ))?;
        let mut rows = statement.query([&record.rev])?;
        let mut targets = Vec::new();
        while let Some(row) = rows.next()? {
            let text: String = row.get(0)?;
            if let Reference::Annotation(revision) = Reference::parse(&text)
                .map_err(|error| StoreError::stored(0, format!("reference {text:?}: {error}")))?
            {
                targets.push(target(&revision));
            }
        }
        Ok(targets)
    }
}

impl Holders for StoredLinks<'_> {
    fn holders(&mut self, slot: &UniqueSlot) -> Result<Vec<Revision>> {
        check_read()?;
        let mut statement = self.conn.prepare_cached(&self.source.sql(
            "SELECT a.root,a.id FROM @source.annotation_unique_key s JOIN @source.annotation a ON a.id=s.id WHERE s.key_hash=?1 ORDER BY a.root,a.id",
        ))?;
        let mut rows = statement.query([slot.digest().as_slice()])?;
        let mut holders = Vec::new();
        while let Some(row) = rows.next()? {
            holders.push(Revision::new(
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
            ));
        }
        Ok(holders)
    }
}

/// Alternate revisions follow the same current-revision rule: an alternate's `base` is the
/// revision it revises, and a `withdrawal` annotation withdraws it. A withdrawal
/// of a withdrawal is an annotation, so lookups that leave the alternate's own
/// revisions go to the stored annotations.
pub(crate) struct AlternateLinks<'a> {
    annotations: StoredLinks<'a>,
}

impl<'a> AlternateLinks<'a> {
    pub(crate) fn new(conn: &'a Connection) -> Result<Self> {
        Ok(Self {
            annotations: StoredLinks::new(conn)?,
        })
    }
    fn alternate(&self, root: &str) -> Result<bool> {
        Ok(self
            .annotations
            .conn
            .prepare_cached(
                &self
                    .annotations
                    .source
                    .sql("SELECT 1 FROM @source.alternate WHERE root=?1 LIMIT 1"),
            )?
            .exists([root])?)
    }
    fn alternate_reference(target: &Target) -> Reference {
        let (root, revision) = match target {
            Target::Root(root) => (root.clone(), None),
            Target::Revision(revision) => (revision.root.clone(), Some(revision.rev.clone())),
        };
        Reference::Alternate {
            revision: RevisionReference { root, revision },
            part: None,
        }
    }
}

impl Links for AlternateLinks<'_> {
    type Error = StoreError;
    fn revisions(&mut self, root: &str) -> Result<Vec<Revision>> {
        if !self.alternate(root)? {
            return self.annotations.revisions(root);
        }
        check_read()?;
        let mut statement = self.annotations.conn.prepare_cached(
            &self
                .annotations
                .source
                .sql("SELECT id FROM @source.alternate WHERE root=?1 ORDER BY id"),
        )?;
        Ok(statement
            .query_map([root], |row| {
                Ok(Revision::new(root, row.get::<_, String>(0)?))
            })?
            .collect::<rusqlite::Result<_>>()?)
    }
    fn revisers(&mut self, revision: &Revision) -> Result<Vec<Revision>> {
        if !self.alternate(&revision.root)? {
            return self.annotations.revisers(revision);
        }
        check_read()?;
        let mut statement = self.annotations.conn.prepare_cached(
            &self
                .annotations
                .source
                .sql("SELECT id FROM @source.alternate WHERE root=?1 AND base=?2 ORDER BY id"),
        )?;
        Ok(statement
            .query_map(params![revision.root, revision.rev], |row| {
                Ok(Revision::new(&revision.root, row.get::<_, String>(0)?))
            })?
            .collect::<rusqlite::Result<_>>()?)
    }
    fn withdrawers(&mut self, target: &Target) -> Result<Vec<Revision>> {
        if !self.alternate(target.root())? {
            return self.annotations.withdrawers(target);
        }
        self.annotations
            .incoming(&Self::alternate_reference(target), LinkRole::Withdraws)
    }
    fn withdraws(&mut self, record: &Revision) -> Result<Vec<Target>> {
        self.annotations.withdraws(record)
    }
}

pub(crate) fn currency_error(error: CurrencyError<StoreError>) -> StoreError {
    match error {
        CurrencyError::Source(error) => error,
        CurrencyError::Cycle(revision) => polylogue_model::Reason::WithdrawalCycle {
            revision: revision.to_string(),
        }
        .into(),
    }
}

/// The current revisions of an annotation or alternate root, sorted. More than
/// one is a conflict; none means every revision is withdrawn.
pub fn heads(conn: &Connection, root: &Reference) -> Result<Vec<Reference>> {
    match root {
        Reference::Annotation(RevisionReference {
            root,
            revision: None,
        }) => current_revisions::stored_heads(conn, root),
        Reference::Alternate {
            revision:
                RevisionReference {
                    root,
                    revision: None,
                },
            part: None,
        } => {
            let mut links = AlternateLinks::new(conn)?;
            Ok(Currency::new(&mut links)
                .heads(root)
                .map_err(currency_error)?
                .into_iter()
                .map(|revision| Reference::Alternate {
                    revision: RevisionReference {
                        root: revision.root,
                        revision: Some(revision.rev),
                    },
                    part: None,
                })
                .collect())
        }
        _ => Err(crate::StoreError::internal(
            "heads require an annotation or alternate root",
        )),
    }
}

#[cfg(test)]
mod tests;
