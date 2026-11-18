//! Named-query identity and ordered, fragmentable history from one pinned reader.
use super::*;
use polylogue_model::saved_queries::{DependentSavedQuery, SavedQueryRecord};
use polylogue_model::{BlobHash, Reference, RevisionReference};

pub(super) struct SavedQuery {
    pub header: SavedQueryHeader,
    pub seed: BlobHash,
    query: SavedQueryRecord,
    selection_question: query_language::identity::QuestionId,
    dependents: Vec<DependentSavedQuery>,
}

fn seed(operation: &Context) -> Result<BlobHash, Error> {
    if let Some(request) = operation.accepted_request() {
        let hash = canonical_json::canonical_hash(
            "saved-query-reading",
            &(
                &operation.machine,
                &operation.authorship,
                &request.request_id,
            ),
        )
        .map_err(Error::internal)?;
        Ok(BlobHash(*hash.as_bytes()))
    } else {
        Ok(BlobHash::of(
            crate::operation::new_run_id().to_string().as_bytes(),
        ))
    }
}

pub(super) fn prepare(
    reader: &Reader<'_>,
    target: &str,
    selection: &PreparedSelection<'_>,
    operation: &Context,
) -> Result<Option<SavedQuery>, Error> {
    let Ok(Reference::Query { key, .. }) = Reference::parse(target) else {
        return Ok(None);
    };
    let Some(definition) = selection
        .definitions()
        .iter()
        .find(|definition| definition.name == key)
    else {
        return Ok(None);
    };
    let name = key.rsplit_once('@').map_or(key.as_str(), |(name, _)| name);
    let Some(mut query) = reader.saved_query_record(name)? else {
        return Ok(None);
    };
    let selected_revision = Reference::Assertion(RevisionReference::exact(
        definition.root.clone(),
        definition.revision.clone(),
    ))
    .to_string();
    if !query
        .revisions
        .iter()
        .any(|revision| revision.reference == selected_revision)
    {
        // A withdrawn authored override may have fallen back to a shipped definition.
        return Ok(None);
    }
    let dependents = reader.dependent_saved_queries(&query.dependents)?;
    let self_question = format!("query:{}", selection.question_id());
    if !dependents
        .iter()
        .any(|dependent| format!("query:{}", dependent.value.hash) == self_question)
    {
        // Queuing this read's anonymous alias must not change its own history.
        // A current authored alias remains a real dependent with its Save call.
        query
            .dependents
            .retain(|reference| *reference != self_question);
    }
    Ok(Some(SavedQuery {
        seed: seed(operation)?,
        header: SavedQueryHeader {
            name: query.name.clone(),
            selected_revision,
            question: Reference::Query {
                key: definition.question_id.to_string(),
                parameters: vec![],
            },
        },
        selection_question: selection.question_id(),
        query,
        dependents,
    }))
}

impl SavedQuery {
    /// Immutable authored revisions and kept/query IDs witness their complete
    /// bodies. Version work follows their count, never the length of their text.
    fn version_material(&self) -> Result<Vec<u8>, Error> {
        serde_json::to_vec(&(
            &self.header,
            self.selection_question,
            self.query
                .revisions
                .iter()
                .map(|revision| &revision.reference)
                .collect::<Vec<_>>(),
            self.query
                .kept
                .iter()
                .map(|kept| &kept.result)
                .collect::<Vec<_>>(),
            &self.query.watch,
            &self.query.dependents,
            self.dependents
                .iter()
                .map(|dependent| (&dependent.name, dependent.revision.to_string()))
                .collect::<Vec<_>>(),
        ))
        .map_err(Error::internal)
    }
    pub(super) fn version(&self) -> Result<BlobHash, Error> {
        Ok(BlobHash::of(&self.version_material()?))
    }
    pub(super) fn item_count(&self) -> usize {
        self.query.revisions.len() + self.query.kept.len() + 1 + self.query.dependents.len()
    }
    pub(super) fn item(&self, ordinal: usize, seed: BlobHash) -> Result<ReadItem, Error> {
        let (reference, item) = if ordinal < self.query.revisions.len() {
            let revision = self.query.revisions[ordinal].clone();
            (
                revision.reference.clone(),
                SavedQueryItem::Revision(revision),
            )
        } else {
            let ordinal = ordinal - self.query.revisions.len();
            if ordinal < self.query.kept.len() {
                let kept = self.query.kept[ordinal].clone();
                (kept.result.clone(), SavedQueryItem::Kept(kept))
            } else {
                let ordinal = ordinal - self.query.kept.len();
                if ordinal == 0 {
                    (
                        format!("query:{}", self.query.name),
                        SavedQueryItem::Watch {
                            reference: self.query.watch.clone(),
                        },
                    )
                } else {
                    let reference = self.query.dependents.get(ordinal - 1).ok_or_else(|| {
                        Error::stale_cursor("the saved-query history item no longer exists")
                    })?;
                    let mut resave = Vec::new();
                    for dependent in &self.dependents {
                        if *reference != format!("query:{}", dependent.value.hash) {
                            continue;
                        }
                        let mut step = serde_json::json!({"verb":"save","name":dependent.name});
                        if let Some(denominator) = &dependent.value.denominator {
                            step["denominator"] = serde_json::Value::String(denominator.clone());
                        }
                        let arguments =
                            serde_json::json!({"selection":dependent.value.text,"steps":[step]});
                        let identity = canonical_json::canonical_hash(
                            "saved-query-resave",
                            &(seed, dependent.revision.to_string(), &arguments),
                        )
                        .map_err(Error::internal)?;
                        resave.push(ReadyCall {
                            request: command_surface::RunnableRequest::new(
                                "rows",
                                arguments,
                                format!("saved-query-resave-{}", identity.to_hex()),
                            ),
                        });
                    }
                    (
                        reference.clone(),
                        SavedQueryItem::Dependent {
                            reference: reference.clone(),
                            resave,
                        },
                    )
                }
            }
        };
        Ok(ReadItem {
            source: Source::record(reference),
            kind: ItemKind::Card {
                card: Box::new(Card::SavedQuery {
                    item: Box::new(item),
                }),
            },
        })
    }
}

#[cfg(test)]
mod tests;
