//! Read authored effects as their unchanged whole immutable request.
use crate::Error;
use polylogue_model::{
    Reference, RevisionReference,
    bundle::Record,
    read::{AuthoredCoverage, AuthoredSelection, LayoutFacts},
};
use polylogue_store::{
    Reader,
    read::authored_read::{AuthoredRead, AuthoredReadIssue},
};

type Visit<'a> = dyn FnMut(Record, LayoutFacts) -> Result<bool, Error> + 'a;

pub(super) fn reference(
    reader: &Reader<'_>,
    reference: &Reference,
    visit: &mut Visit<'_>,
) -> Result<bool, Error> {
    match reader.authored_record(reference)? {
        Some(AuthoredRead::Local { selected, record }) => visit(
            Record::Authored(record),
            LayoutFacts {
                authored: Some(AuthoredSelection {
                    selected,
                    coverage: AuthoredCoverage::WholeRequest,
                }),
                ..Default::default()
            },
        ),
        Some(AuthoredRead::Incomplete { issue }) => match (issue, reference) {
            (
                AuthoredReadIssue::MissingAssociation | AuthoredReadIssue::UnresolvedRevision,
                Reference::Assertion(revision),
            ) => direct_assertion(reader, revision, visit),
            (issue, _) => Err(Error::not_yet_available(issue)),
        },
        None => Err(Error::not_found(format!(
            "no authored record for {reference}"
        ))),
    }
}

fn direct_assertion(
    reader: &Reader<'_>,
    revision: &RevisionReference,
    visit: &mut Visit<'_>,
) -> Result<bool, Error> {
    let row = reader
        .assertion(revision)?
        .ok_or_else(|| Error::not_found(format!("no annotation:{revision}")))?;
    let record = Record::Assertion(Box::new(polylogue_model::assertion::AssertionRecord {
        reference: row.fact.reference().stored_form(),
        assertion: row.fact,
    }));
    visit(record, LayoutFacts::default())
}
