//! Read authored effects as their unchanged whole immutable request.
use super::super::Failure;
use polylogue_model::{
    Reference,
    bundle::Record,
    read::{AuthoredCoverage, AuthoredSelection, ProjectionEvidence},
};
use polylogue_store::{
    Reader,
    read::authored_evidence::{AuthoredReadEvidence, AuthoredReadIssue},
};

type Visit<'a> = dyn FnMut(Record, ProjectionEvidence) -> Result<bool, Failure> + 'a;

pub(super) fn reference(
    reader: &Reader<'_>,
    reference: &Reference,
    visit: &mut Visit<'_>,
) -> Result<bool, Failure> {
    match reader.authored_record(reference)? {
        Some(AuthoredReadEvidence::Local { selected, record }) => visit(
            Record::Authored(record),
            ProjectionEvidence {
                authored: Some(AuthoredSelection {
                    selected,
                    coverage: AuthoredCoverage::WholeRequest,
                }),
                ..Default::default()
            },
        ),
        Some(AuthoredReadEvidence::Incomplete {
            issue: AuthoredReadIssue::MissingAssociation | AuthoredReadIssue::UnresolvedRevision,
        }) if matches!(reference, Reference::Assertion(_)) => {
            direct_assertion(reader, reference, visit)
        }
        Some(AuthoredReadEvidence::Incomplete { issue }) => {
            Err(Failure::new("not_yet_available", issue))
        }
        None => Err(Failure::new(
            "not_found",
            format!("no authored record for {reference}"),
        )),
    }
}

fn direct_assertion(
    reader: &Reader<'_>,
    reference: &Reference,
    visit: &mut Visit<'_>,
) -> Result<bool, Failure> {
    let Reference::Assertion(revision) = reference else {
        unreachable!()
    };
    let row = reader
        .assertion(revision)?
        .ok_or_else(|| Failure::new("not_found", format!("no assertion {reference}")))?;
    let record = Record::Assertion(Box::new(polylogue_model::assertion::AssertionRecord {
        reference: format!("assertion:{}@{}", row.fact.root, row.fact.id),
        assertion: row.fact,
    }));
    visit(record, ProjectionEvidence::default())
}
