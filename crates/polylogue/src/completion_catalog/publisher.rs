//! One startup snapshot, followed by indexed deltas after committed writes.
use crate::daemon::service::Daemon;
use polylogue_model::ChangePosition;
use polylogue_store::{
    Reader, Result,
    completion::{self, CatalogWriter, CommitSubscription, Metadata, SessionChange},
};
use query_language::catalog::Catalog;
use std::{
    collections::BTreeMap,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc::SyncSender,
    },
    thread::JoinHandle,
};

pub(crate) struct Publisher {
    stop: Arc<AtomicBool>,
    expected_exit: Arc<AtomicBool>,
    wake: Option<SyncSender<()>>,
    thread: Option<JoinHandle<()>>,
}
impl Publisher {
    pub fn start(
        daemon: Arc<Daemon>,
        subscription: Option<CommitSubscription>,
        worker_events: tokio::sync::watch::Sender<Option<&'static str>>,
    ) -> std::io::Result<Self> {
        let stop = Arc::new(AtomicBool::new(false));
        let stopped = Arc::clone(&stop);
        let expected_exit = Arc::new(AtomicBool::new(false));
        let completed = Arc::clone(&expected_exit);
        let wake = subscription.as_ref().map(CommitSubscription::wake_handle);
        let thread = std::thread::Builder::new()
            .name("polylogue-completion".into())
            .spawn(move || {
                let _exit = crate::daemon::lifecycle::WorkerExit::new(
                    worker_events,
                    "completion worker terminated",
                    Some(Arc::clone(&completed)),
                );
                let mut publisher =
                    State::new(daemon.archive.is_primary(), daemon.config.requests.clone());
                let mut last_error = None;
                loop {
                    if stopped.load(Ordering::Acquire) {
                        completed.store(true, Ordering::Release);
                        return;
                    }
                    let result = (|| {
                        let reader = daemon.reads.reader()?;
                        publisher.refresh(
                            &reader,
                            &daemon.completion_path,
                            &crate::runtime_dir::archive_key(&daemon.archive.root),
                        )
                    })();
                    match result {
                        Ok(()) => last_error = None,
                        Err(error) => {
                            if stopped.load(Ordering::Acquire) {
                                completed.store(true, Ordering::Release);
                                return;
                            }
                            let error = error.to_string();
                            if last_error.as_ref() != Some(&error) {
                                eprintln!("polylogue: completion catalog: {error}");
                                last_error = Some(error);
                            }
                            // Retry a failed publication without advancing its receipt.
                            // Successful idle publishers wait solely on commit events.
                            std::thread::sleep(std::time::Duration::from_millis(100));
                            continue;
                        }
                    }
                    match &subscription {
                        Some(subscription) => {
                            if subscription.recv().is_err() {
                                if stopped.load(Ordering::Acquire) {
                                    completed.store(true, Ordering::Release);
                                }
                                return;
                            }
                        }
                        None => {
                            completed.store(true, Ordering::Release);
                            return;
                        }
                    }
                }
            })?;
        Ok(Self {
            stop,
            expected_exit,
            wake,
            thread: Some(thread),
        })
    }

    /// A finite publisher without a subscription returns after its snapshot.
    /// Every other unrequested worker exit, including a panic, is a failure.
    pub(crate) fn worker_failure(&self) -> Option<&'static str> {
        (self.thread.as_ref().is_some_and(JoinHandle::is_finished)
            && !self.stop.load(Ordering::Acquire)
            && !self.expected_exit.load(Ordering::Acquire))
        .then_some("completion worker terminated")
    }

    /// Stop and join the exact worker before the runtime records its end.
    pub(crate) fn shutdown(&mut self) -> std::io::Result<()> {
        self.stop.store(true, Ordering::Release);
        if let Some(wake) = &self.wake {
            let _ = wake.try_send(());
        }
        if let Some(thread) = self.thread.take() {
            thread
                .join()
                .map_err(|_| std::io::Error::other("completion worker panicked"))?;
            if !self.expected_exit.load(Ordering::Acquire) {
                return Err(std::io::Error::other("completion worker terminated"));
            }
        }
        Ok(())
    }
}
impl Drop for Publisher {
    fn drop(&mut self) {
        if let Err(error) = self.shutdown() {
            eprintln!("polylogue: completion worker shutdown: {error}");
        }
    }
}

pub(super) struct State {
    writer: Option<(String, CatalogWriter)>,
    position: Option<ChangePosition>,
    catalog: Catalog,
    definitions: BTreeMap<String, completion::Saved>,
    kept: BTreeMap<String, polylogue_query::Kept>,
    requires_confirmed: bool,
    requests: polylogue_model::user::RequestMetadata,
}
impl State {
    pub(super) fn new(
        requires_confirmed: bool,
        requests: polylogue_model::user::RequestMetadata,
    ) -> Self {
        Self {
            writer: None,
            position: None,
            catalog: super::base_catalog(),
            definitions: BTreeMap::new(),
            kept: BTreeMap::new(),
            requires_confirmed,
            requests,
        }
    }
    pub(super) fn refresh(
        &mut self,
        reader: &Reader<'_>,
        path: &std::path::Path,
        archive: &str,
    ) -> Result<()> {
        let token = reader.scope_token();
        token.check()?;
        if self
            .writer
            .as_ref()
            .is_some_and(|(_, writer)| !writer.is_current_file())
        {
            self.writer = None;
            self.position = None;
            self.definitions.clear();
            self.kept.clear();
        }
        if self.requires_confirmed && reader.confirmed_checkpoint()?.is_none() {
            return Ok(());
        }
        let position = reader.change_position()?;
        if self.position.as_ref() == Some(&position) {
            return Ok(());
        }
        if self
            .writer
            .as_ref()
            .is_some_and(|(generation, _)| generation != &position.generation)
            || self
                .position
                .as_ref()
                .is_some_and(|previous| previous.generation != position.generation)
        {
            self.writer = None;
            self.position = None;
            self.definitions.clear();
            self.kept.clear();
        }
        let metadata = Metadata {
            archive: archive.into(),
            position: position.clone(),
            written_at: self.requests.now().unix_ms(),
        };
        let pointer = completion::CatalogPointer::new(path.to_owned());
        let generation_path = pointer.generation_path(&position.generation);
        let after = self.position.clone();
        let result = if let Some((generation, mut writer)) = self.writer.take() {
            let result = writer.update(&metadata, |writer| {
                self.apply(reader, writer, after.as_ref())
            });
            self.writer = Some((generation, writer));
            result
        } else {
            CatalogWriter::build(&generation_path, &metadata, |writer| {
                writer.put_catalog(&self.catalog)?;
                self.apply(reader, writer, None)
            })
            .map(|writer| {
                self.writer = Some((position.generation.clone(), writer));
            })
        };
        result?;
        pointer.activate(&position.generation, &token)?;
        self.position = Some(position);
        Ok(())
    }
    fn apply(
        &mut self,
        reader: &Reader<'_>,
        writer: &mut CatalogWriter,
        after: Option<&ChangePosition>,
    ) -> Result<()> {
        reader.visit_completion_sessions(after, |change| match change {
            SessionChange::Upsert(session) => writer.put_session(&session),
            SessionChange::Removed(id) => writer.remove_session(&id),
        })?;
        let mut changed = false;
        let cached = writer.definition_families()?;
        let mut seen = std::collections::BTreeSet::new();
        reader.visit_completion_definitions(after.map(|position| position.archive), |change| {
            seen.insert(change.family.clone());
            changed = true;
            self.apply_definition(writer, change)
        })?;
        for family in cached {
            if !seen.contains(&family) {
                changed = true;
                self.apply_definition(writer, reader.completion_definition(&family)?)?;
            }
        }
        reader.visit_completion_kept(after.map(|position| position.archive), |kept| {
            changed = true;
            let columns: Option<Vec<polylogue_query::Column>> = kept
                .columns
                .and_then(|columns| serde_json::from_value(columns).ok());
            let matches: Option<Vec<polylogue_query::Column>> = kept
                .match_columns
                .and_then(|columns| serde_json::from_value(columns).ok());
            let typed = kept
                .unit
                .as_ref()
                .zip(columns.as_ref())
                .zip(kept.output_metadata.as_ref())
                .map(|((unit, columns), metadata)| polylogue_query::Kept {
                    output_metadata: metadata.clone(),
                    shape: kept.shape,
                    native_identity: kept.native_identity,
                    unit: unit.clone(),
                    columns: columns.clone(),
                    match_columns: matches.clone().unwrap_or_default(),
                })
                .filter(|typed| typed.validate_output_metadata().is_ok());
            let schema = typed
                .as_ref()
                .map(|typed| typed.output_schema(query_language::ast::ResultPart::Rows));
            let match_schema = typed
                .as_ref()
                .filter(|_| matches.is_some())
                .map(|typed| typed.output_schema(query_language::ast::ResultPart::Matches));
            writer.put_kept(&completion::Kept {
                id: kept.id.clone(),
                name: kept.name,
                unit: kept.unit.clone(),
                row_count: kept.row_count,
                rows: schema,
                matches: match_schema,
            })?;
            self.kept.remove(&kept.id);
            if let Some(typed) = typed {
                self.kept.insert(kept.id, typed);
            }
            Ok(())
        })?;
        if changed {
            self.publish_definitions(reader, writer)?;
        }
        let mut keys: std::collections::BTreeSet<_> =
            reader.completion_setting_keys(None)?.into_iter().collect();
        keys.extend(
            polylogue_model::settings::CATALOG
                .iter()
                .filter(|spec| !spec.key.ends_with('*'))
                .map(|spec| spec.key.to_string()),
        );
        for key in writer.setting_keys()? {
            if !keys.contains(&key) {
                writer.remove_setting(&key)?;
            }
        }
        for key in keys {
            let current = reader.setting_current(&key, self.requests.machine())?;
            let choices = if current.owner == "sinex" {
                Vec::new()
            } else {
                polylogue_model::settings::declaration(&key)
                    .map_err(|error| polylogue_store::StoreError::Invalid(error.to_string()))?
                    .choices()
                    .iter()
                    .map(|value| value.to_string())
                    .collect()
            };
            writer.put_setting(&completion::Setting {
                key: key.clone(),
                value: current,
                choices,
            })?;
        }
        Ok(())
    }
    fn apply_definition(
        &mut self,
        writer: &mut CatalogWriter,
        change: completion::DefinitionChange,
    ) -> Result<()> {
        self.definitions.remove(&change.family);
        writer.remove_saved(&change.family)?;
        let values = change
            .tag
            .map(|(key, value)| {
                let mut tags = BTreeMap::from([(key.clone(), 1)]);
                if let Some(value) = value {
                    tags.insert(format!("{key}:{value}"), 1);
                }
                BTreeMap::from([("tag".into(), tags)])
            })
            .unwrap_or_default();
        writer.put_values(
            &polylogue_model::Reference::Assertion(polylogue_model::RevisionReference {
                root: change.family.clone(),
                revision: None,
            })
            .to_string(),
            &values,
        )?;
        if let Some(saved) = change.saved {
            self.definitions.insert(
                change.family.clone(),
                completion::Saved {
                    relation_scope: None,
                    family: change.family,
                    revision: saved.revision,
                    name: saved.name,
                    text: saved.text,
                    parameters: vec![],
                    help: String::new(),
                    schema: None,
                },
            );
        }
        Ok(())
    }
    fn publish_definitions(
        &mut self,
        reader: &Reader<'_>,
        writer: &mut CatalogWriter,
    ) -> Result<()> {
        let saved = reader
            .selection_saved_queries()
            .map_err(polylogue_store::selection::SelectionError::into_store_error)?;
        let bindings = polylogue_query::Bindings {
            origins: polylogue_sources::origin::ORIGINS
                .iter()
                .map(|origin| origin.name.to_owned())
                .collect(),
            saved,
            kept: self.kept.clone(),
        };
        for definition in self.definitions.values_mut() {
            let Some(entries) = bindings.saved.get(&definition.name) else {
                definition.parameters.clear();
                definition.schema = None;
                definition.relation_scope = None;
                definition.help = format!(
                    "{} (historical definition has no recorded semantic identity)",
                    definition.text
                );
                writer.put_saved(definition)?;
                continue;
            };
            let current = entries
                .iter()
                .find(|current| {
                    current.root == definition.family && current.revision == definition.revision
                })
                .ok_or_else(|| {
                    polylogue_store::StoreError::NotFound(format!(
                        "assertion:{}@{}",
                        definition.family, definition.revision
                    ))
                })?;
            definition.parameters.clear();
            definition.schema = None;
            definition.relation_scope = None;
            if entries.len() > 1 || !current.conflicts.is_empty() {
                definition.help = format!("{} (ambiguous query name)", definition.text);
            } else {
                match current.executable_text().and_then(|syntax| {
                    let parameters = bindings.definition_parameters(syntax)?;
                    let (schema, relation_scope) = bindings.output_metadata(syntax)?;
                    Ok((parameters, schema, relation_scope))
                }) {
                    Ok((parameters, schema, relation_scope)) => {
                        definition.parameters = parameters
                            .into_iter()
                            .map(|name| completion::Parameter { name, kind: None })
                            .collect();
                        definition.schema = Some(schema);
                        definition.relation_scope = Some(relation_scope);
                        definition.help = if definition.parameters.is_empty() {
                            definition.text.clone()
                        } else {
                            format!(
                                "{} (parameters: {})",
                                definition.text,
                                definition
                                    .parameters
                                    .iter()
                                    .map(|parameter| parameter.name.as_str())
                                    .collect::<Vec<_>>()
                                    .join(", ")
                            )
                        };
                    }
                    Err(error) => definition.help = format!("{} ({error})", definition.text),
                }
            }
            writer.put_saved(definition)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
