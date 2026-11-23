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
                Reference::Annotation(revision),
            ) => direct_annotation(reader, revision, visit),
            (issue, _) => Err(Error::not_yet_available(issue)),
        },
        None => Err(Error::not_found(format!(
            "no authored record for {reference}"
        ))),
    }
}

fn direct_annotation(
    reader: &Reader<'_>,
    revision: &RevisionReference,
    visit: &mut Visit<'_>,
) -> Result<bool, Error> {
    let row = reader
        .annotation(revision)?
        .ok_or_else(|| Error::not_found(format!("no annotation:{revision}")))?;
    let record = Record::Annotation(Box::new(polylogue_model::annotation::AnnotationRecord {
        reference: row.fact.reference().to_string(),
        annotation: row.fact,
    }));
    visit(record, LayoutFacts::default())
}
