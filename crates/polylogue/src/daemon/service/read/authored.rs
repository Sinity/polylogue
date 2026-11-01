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
        Some(AuthoredReadEvidence::Historical { selected, record }) => visit(
            Record::Historical(record),
            ProjectionEvidence {
                authored: Some(AuthoredSelection {
                    selected,
                    coverage: AuthoredCoverage::WholeRequest,
                }),
                ..Default::default()
            },
        ),
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
        .assertion_evidence(revision)?
        .ok_or_else(|| Failure::new("not_found", format!("no assertion {reference}")))?;
    let record = match row {
        polylogue_model::historical::AssertionEvidence::Current(row) => {
            let reference = format!("assertion:{}@{}", row.fact.root, row.fact.id);
            let imports = row
                .imports
                .observed
                .into_iter()
                .chain(row.imports.declared)
                .collect();
            Record::Assertion(Box::new(polylogue_model::assertion::AssertionRecord {
                reference,
                assertion: row.fact,
                imports,
            }))
        }
        polylogue_model::historical::AssertionEvidence::Historical {
            fact,
            commit_seq,
            imports,
        } => {
            if commit_seq == 0 {
                return Err(Failure::new(
                    "not_yet_available",
                    "historical assertion has no retained positive local admission",
                ));
            }
            Record::Historical(Box::new(polylogue_model::bundle::HistoricalRecord {
                reference: format!("assertion:{}@{}", fact.root, fact.id),
                commit: commit_seq,
                evidence: polylogue_model::historical::HistoricalRecord::Assertion(Box::new(fact)),
                imports: imports
                    .observed
                    .into_iter()
                    .chain(imports.declared)
                    .collect(),
            }))
        }
    };
    visit(record, ProjectionEvidence::default())
}
