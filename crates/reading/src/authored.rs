//! Read authored effects as their unchanged whole immutable request.
use crate::Error;
use polylogue_model::{
    Reference,
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
        Some(AuthoredRead::Incomplete {
            issue: AuthoredReadIssue::MissingAssociation | AuthoredReadIssue::UnresolvedRevision,
        }) if matches!(reference, Reference::Assertion(_)) => {
            direct_assertion(reader, reference, visit)
        }
        Some(AuthoredRead::Incomplete { issue }) => Err(Error::not_yet_available(issue)),
        None => Err(Error::not_found(format!(
            "no authored record for {reference}"
        ))),
    }
}

fn direct_assertion(
    reader: &Reader<'_>,
    reference: &Reference,
    visit: &mut Visit<'_>,
) -> Result<bool, Error> {
    let Reference::Assertion(revision) = reference else {
        unreachable!()
    };
    let row = reader
        .assertion(revision)?
        .ok_or_else(|| Error::not_found(format!("no assertion {reference}")))?;
    let record = Record::Assertion(Box::new(polylogue_model::assertion::AssertionRecord {
        reference: format!("assertion:{}@{}", row.fact.root, row.fact.id),
        assertion: row.fact,
    }));
    visit(record, LayoutFacts::default())
}
