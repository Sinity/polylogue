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
        let thread = polylogue_model::spawn_named("polylogue-completion", move || {
            let _exit = crate::daemon::lifecycle::WorkerExit::new(
                worker_events,
                "completion worker terminated",
                Some(Arc::clone(&completed)),
            );
            let mut publisher = State::new(daemon.config.requests.clone());
            let expected = polylogue_store::follow::follow(
                "completion catalog",
                subscription.as_ref(),
                &stopped,
                || daemon.config.requests.now(),
                || daemon.status.changed(),
                || {
                    let reader = daemon.reads.reader()?;
                    publisher.refresh(
                        &reader,
                        &daemon.completion_path,
                        &crate::runtime_dir::archive_key(&daemon.archive.root),
                    )
                },
            );
            completed.store(expected, Ordering::Release);
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

    /// Stop and join the worker before the runtime records its end.
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
            polylogue_store::log!(error, "completion worker shutdown: {error}");
        }
    }
}

// Long-lived state holds completion metadata and dependency IDs, never saved AST bodies.
#[derive(Default)]
struct Vocabulary {
    families: BTreeMap<String, Family>,
    source_users: BTreeMap<String, std::collections::BTreeSet<String>>,
    name_users: BTreeMap<String, std::collections::BTreeSet<String>>,
    kept: BTreeMap<String, polylogue_query::Kept>,
}
type Family = completion::Dependencies;
impl Vocabulary {
    fn install_family(&mut self, id: String, family: Family) {
        if let Some(old) = self.families.remove(&id) {
            for (index, keys) in [
                (&mut self.source_users, old.sources),
                (&mut self.name_users, old.names.into_keys().collect()),
            ] {
                for key in keys {
                    let empty = index.get_mut(&key).is_some_and(|users| {
                        users.remove(&id);
                        users.is_empty()
                    });
                    if empty {
                        index.remove(&key);
                    }
                }
            }
        }
        for source in &family.sources {
            self.source_users
                .entry(source.clone())
                .or_default()
                .insert(id.clone());
        }
        for name in family.names.keys() {
            self.name_users
                .entry(name.clone())
                .or_default()
                .insert(id.clone());
        }
        if !family.names.is_empty() || !family.sources.is_empty() {
            self.families.insert(id, family);
        }
    }
}
// Prepared against one pinned source snapshot. Installing this after SQL commits
// cannot fail; a failed publication simply discards it.
struct VocabularyUpdate {
    replaced: bool,
    families: BTreeMap<String, Family>,
    definitions: BTreeMap<String, Option<completion::Saved>>,
    tags: BTreeMap<String, BTreeMap<String, BTreeMap<String, u64>>>,
    kept: BTreeMap<String, (completion::Kept, Option<polylogue_query::Kept>)>,
    settings: Vec<String>,
}
/// The sessions a publication writes: the startup snapshot streams every
/// session; a later one writes those changed since the last publication.
enum Sessions {
    All,
    Changed(Vec<SessionChange>),
}

impl VocabularyUpdate {
    fn is_empty(&self) -> bool {
        !self.replaced
            && self.families.is_empty()
            && self.definitions.is_empty()
            && self.tags.is_empty()
            && self.kept.is_empty()
            && self.settings.is_empty()
    }

    fn install(self, vocabulary: &mut Vocabulary) {
        if self.replaced {
            *vocabulary = Vocabulary::default();
        }
        for (id, family) in self.families {
            vocabulary.install_family(id, family);
        }
        for (id, (_, value)) in self.kept {
            if let Some(value) = value {
                vocabulary.kept.insert(id, value);
            } else {
                vocabulary.kept.remove(&id);
            }
        }
    }
}

pub(super) struct State {
    writer: Option<(String, CatalogWriter)>,
    position: Option<ChangePosition>,
    catalog: Catalog,
    vocabulary: Vocabulary,
    requests: polylogue_model::user::RequestMetadata,
}
impl State {
    pub(super) fn new(requests: polylogue_model::user::RequestMetadata) -> Self {
        Self {
            writer: None,
            position: None,
            catalog: super::base_catalog(),
            vocabulary: Vocabulary::default(),
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
            .is_none_or(|(_, writer)| !writer.is_current_file())
        {
            self.writer = None;
            self.position = None;
            self.vocabulary = Vocabulary::default();
        }
        let position = reader.change_position()?;
        if self
            .writer
            .as_ref()
            .is_some_and(|(generation, _)| generation != &position.index_instance)
            || self
                .position
                .as_ref()
                .is_some_and(|old| old.index_instance != position.index_instance)
        {
            self.writer = None;
            self.position = None;
            self.vocabulary = Vocabulary::default();
        }
        let metadata = Metadata {
            archive: archive.into(),
            position: position.clone(),
            written_at: self.requests.now().unix_ms(),
        };
        let pointer = completion::CatalogPointer::new(path.to_owned());
        // Missing, incompatible or damaged derived state gets a fresh snapshot.
        // A reader's readonly_shm mapping can temporarily prevent a writable
        // reopen: let the existing worker retry after that reader releases it.
        if self.writer.is_none()
            && self.position.is_none()
            && let Some((writer, saved, vocabulary)) =
                pointer.resume(&metadata, &self.catalog, self.requests.machine())?
        {
            token.check()?;
            self.vocabulary = Vocabulary::default();
            for (id, family) in vocabulary.families {
                self.vocabulary.install_family(id, family);
            }
            self.vocabulary.kept = vocabulary.kept;
            self.position = Some(saved.position);
            self.writer = Some((position.index_instance.clone(), writer));
        }
        if self.position.as_ref() == Some(&position) {
            pointer.activate(
                &self.writer.as_ref().expect("resumed publication").1,
                &token,
            )?;
            return Ok(());
        }
        let after = self.position.clone();
        let update = self.prepare_vocabulary(
            reader,
            reader.completion_vocabulary_changes(after.as_ref())?,
        )?;
        let result = if let Some((generation, mut writer)) = self.writer.take() {
            // Most commits change nothing the catalog holds (a build commits
            // continually); the file is rewritten only when its content changes.
            let mut sessions = Vec::new();
            reader.visit_completion_sessions(after.as_ref(), |change| {
                sessions.push(change);
                Ok(())
            })?;
            if sessions.is_empty() && update.is_empty() {
                self.writer = Some((generation, writer));
                self.position = Some(position);
                return Ok(());
            }
            let result = writer.update(&metadata, |writer| {
                self.apply(writer, Sessions::Changed(sessions), &update, reader)
            });
            self.writer = Some((generation, writer));
            result
        } else {
            pointer
                .build(&metadata, |writer| {
                    writer.put_catalog(&self.catalog)?;
                    self.apply(writer, Sessions::All, &update, reader)
                })
                .map(|writer| {
                    self.writer = Some((position.index_instance.clone(), writer));
                })
        };
        result?;
        let (_, writer) = self
            .writer
            .as_ref()
            .expect("successful catalog publication");
        pointer.activate(writer, &token)?;
        update.install(&mut self.vocabulary);
        self.position = Some(position);
        Ok(())
    }

    fn prepare_vocabulary(
        &self,
        reader: &Reader<'_>,
        changes: completion::VocabularyChanges,
    ) -> Result<VocabularyUpdate> {
        let mut families = changes
            .definitions
            .into_iter()
            .map(|change| (change.family.clone(), change))
            .collect::<BTreeMap<_, _>>();
        let affected = changes
            .queries
            .iter()
            .map(|id| format!("query:{id}"))
            .chain(
                changes
                    .kept
                    .iter()
                    .map(|value| format!("result:{}", value.id)),
            )
            .flat_map(|source| {
                self.vocabulary
                    .source_users
                    .get(&source)
                    .into_iter()
                    .flatten()
                    .cloned()
            })
            .collect::<std::collections::BTreeSet<_>>();
        for family in affected {
            if !changes.replaced && !families.contains_key(&family) {
                families.insert(family.clone(), reader.completion_definition(&family)?);
            }
        }
        // Only families sharing a changed name need ambiguity metadata refreshed.
        let mut pending = families.keys().cloned().collect::<Vec<_>>();
        while let Some(id) = pending.pop() {
            let next = &families[&id];
            let names = next.names.keys().chain(
                self.vocabulary
                    .families
                    .get(&id)
                    .into_iter()
                    .flat_map(|old| old.names.keys()),
            );
            let neighbors = names
                .flat_map(|name| {
                    self.vocabulary
                        .name_users
                        .get(name)
                        .into_iter()
                        .flatten()
                        .cloned()
                })
                .collect::<std::collections::BTreeSet<_>>();
            for neighbor in neighbors {
                if !changes.replaced && !families.contains_key(&neighbor) {
                    families.insert(neighbor.clone(), reader.completion_definition(&neighbor)?);
                    pending.push(neighbor);
                }
            }
        }
        let mut update = VocabularyUpdate {
            replaced: changes.replaced,
            families: BTreeMap::new(),
            definitions: BTreeMap::new(),
            tags: BTreeMap::new(),
            kept: BTreeMap::new(),
            settings: changes.settings,
        };
        for kept in changes.kept {
            let columns: Option<Vec<polylogue_query::Column>> = kept
                .columns
                .and_then(|value| serde_json::from_value(value).ok());
            let matches: Option<Vec<polylogue_query::Column>> = kept
                .match_columns
                .and_then(|value| serde_json::from_value(value).ok());
            let typed = kept
                .unit
                .as_ref()
                .zip(columns.as_ref())
                .zip(kept.output_metadata.as_ref())
                .map(|((unit, columns), metadata)| polylogue_query::Kept {
                    output_metadata: metadata.clone(),
                    shape: kept.shape,
                    row_identity: kept.row_identity,
                    unit: unit.clone(),
                    columns: columns.clone(),
                    match_columns: matches.clone().unwrap_or_default(),
                })
                .filter(|value| value.validate_output_metadata().is_ok());
            let rows = typed
                .as_ref()
                .map(|value| value.output_schema(query_language::ast::ResultPart::Rows));
            let match_schema = typed
                .as_ref()
                .filter(|_| matches.is_some())
                .map(|value| value.output_schema(query_language::ast::ResultPart::Matches));
            let cached = completion::Kept {
                id: kept.id.clone(),
                name: kept.name,
                unit: kept.unit,
                row_count: kept.row_count,
                rows,
                matches: match_schema,
            };
            update.kept.insert(kept.id, (cached, typed));
        }
        for (id, change) in &families {
            let mut sources = std::collections::BTreeSet::new();
            for value in change.queries.values().flatten() {
                sources.insert(format!("query:{}", value.hash));
                sources.extend(value.denominator.map(|id| format!("query:{id}")));
                // An unavailable body can still declare its dependencies. Read
                // that immutable object for invalidation only, never as a check that it can run.
                let object = match &value.body {
                    polylogue_query::SavedQueryBody::Available { object, .. } => object.clone(),
                    polylogue_query::SavedQueryBody::Unavailable { .. } => {
                        match reader.query_object(&value.hash.to_string()) {
                            Ok(found) => found.map(std::sync::Arc::new),
                            Err(
                                polylogue_store::StoreError::NotFound(_)
                                | polylogue_store::StoreError::Invalid(_)
                                | polylogue_store::StoreError::Refused(_)
                                | polylogue_store::StoreError::Internal(_)
                                | polylogue_store::StoreError::StoredValue { .. }
                                | polylogue_store::StoreError::ReferenceUnavailable { .. }
                                | polylogue_store::StoreError::Ambiguous { .. },
                            ) => None,
                            Err(error) => return Err(error),
                        }
                    }
                };
                if let Some(object) = object {
                    sources.extend(object.result_references().map(str::to_owned));
                }
            }
            update.families.insert(
                id.clone(),
                Family {
                    names: change.names.clone(),
                    sources,
                },
            );
            let values = change
                .tag
                .as_ref()
                .map(|(key, value)| {
                    let mut tags = BTreeMap::from([(key.clone(), 1)]);
                    if let Some(value) = value {
                        tags.insert(format!("{key}:{value}"), 1);
                    }
                    BTreeMap::from([("tag".into(), tags)])
                })
                .unwrap_or_default();
            update.tags.insert(id.clone(), values);
            let definition = if let Some(saved) = &change.saved {
                let count = families
                    .values()
                    .map(|family| family.names.get(&saved.name).copied().unwrap_or(0))
                    .sum::<usize>()
                    + if changes.replaced {
                        0
                    } else {
                        self.vocabulary
                            .name_users
                            .get(&saved.name)
                            .into_iter()
                            .flatten()
                            .filter(|family| !families.contains_key(*family))
                            .map(|family| self.vocabulary.families[family].names[&saved.name])
                            .sum::<usize>()
                    };
                Some(self.definition_metadata(saved, change, count, &update)?)
            } else {
                None
            };
            update.definitions.insert(id.clone(), definition);
        }
        Ok(update)
    }

    fn definition_metadata(
        &self,
        saved: &completion::SavedEntry,
        change: &completion::DefinitionChange,
        names: usize,
        update: &VocabularyUpdate,
    ) -> Result<completion::Saved> {
        let mut definition = completion::Saved {
            family: change.family.clone(),
            revision: saved.revision.clone(),
            name: saved.name.clone(),
            text: saved.text.clone(),
            parameters: vec![],
            help: saved.text.clone(),
            schema: None,
            relation_scope: None,
        };
        let current = change
            .queries
            .get(&saved.name)
            .into_iter()
            .flatten()
            .find(|value| value.root == change.family && value.revision == saved.revision)
            .ok_or_else(|| {
                polylogue_store::StoreError::NotFound(format!(
                    "annotation:{}@{}",
                    change.family, saved.revision
                ))
            })?;
        if names > 1 || !current.conflicts.is_empty() {
            definition.help = format!("{} (ambiguous query name)", definition.text);
            return Ok(definition);
        }
        let mut kept = BTreeMap::new();
        for value in change.queries.values().flatten() {
            if let polylogue_query::SavedQueryBody::Available {
                object: Some(object),
                ..
            } = &value.body
            {
                for reference in object.result_references() {
                    let result = reference.strip_prefix("result:").unwrap_or(reference);
                    let cached = match update.kept.get(result) {
                        Some((_, value)) => value.as_ref(),
                        None if !update.replaced => self.vocabulary.kept.get(result),
                        None => None,
                    };
                    if let Some(cached) = cached {
                        kept.insert(result.to_owned(), cached.clone());
                    }
                }
            }
        }
        let bindings = polylogue_query::Bindings {
            origins: super::origin_names(),
            saved: change.queries.clone(),
            kept,
            kept_names: Default::default(),
        };
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
                if !definition.parameters.is_empty() {
                    definition.help = format!(
                        "{} (parameters: {})",
                        definition.text,
                        definition
                            .parameters
                            .iter()
                            .map(|parameter| parameter.name.as_str())
                            .collect::<Vec<_>>()
                            .join(", ")
                    );
                }
            }
            Err(error) => definition.help = format!("{} ({error})", definition.text),
        }
        Ok(definition)
    }

    fn apply(
        &self,
        writer: &mut CatalogWriter,
        sessions: Sessions,
        update: &VocabularyUpdate,
        reader: &Reader<'_>,
    ) -> Result<()> {
        writer.put_publisher_machine(self.requests.machine())?;
        let mut put = |change| writer.apply(change);
        let first = match sessions {
            Sessions::All => {
                polylogue_store::fault::check(
                    "completion.full_snapshot",
                    &reader.change_position()?.index_instance,
                )?;
                reader.visit_completion_sessions(None, &mut put)?;
                true
            }
            Sessions::Changed(changes) => {
                for change in changes {
                    put(change)?;
                }
                false
            }
        };
        if update.replaced {
            writer.clear_authored()?;
        }
        for (family, values) in &update.tags {
            writer.put_values(
                &polylogue_model::Reference::Annotation(polylogue_model::RevisionReference {
                    root: family.clone(),
                    revision: None,
                })
                .to_string(),
                values,
            )?;
        }
        for (family, definition) in &update.definitions {
            if let Some(definition) = definition {
                writer.put_saved(definition)?;
            } else {
                writer.remove_saved(family)?;
            }
        }
        let origins = super::origin_names();
        for shipped in polylogue_model::saved_queries::SHIPPED_QUERIES {
            let changed = first
                || update.replaced
                || update.families.iter().any(|(id, family)| {
                    family.names.contains_key(shipped.name)
                        || self
                            .vocabulary
                            .families
                            .get(id)
                            .is_some_and(|old| old.names.contains_key(shipped.name))
                });
            if !changed {
                continue;
            }
            let authored = (!update.replaced
                && self
                    .vocabulary
                    .name_users
                    .get(shipped.name)
                    .is_some_and(|owners| {
                        owners.iter().any(|id| !update.families.contains_key(id))
                    }))
                || update
                    .families
                    .values()
                    .any(|family| family.names.contains_key(shipped.name));
            let family = format!("shipped:{}", shipped.name);
            if authored {
                writer.remove_saved(&family)?;
                continue;
            }
            let saved = polylogue_store::selection::shipped_bindings(&origins)
                .map_err(|error| error.into_store_error())?;
            let revision = saved[shipped.name][0].hash.to_string();
            let bindings = polylogue_query::Bindings {
                origins: origins.clone(),
                saved,
                ..Default::default()
            };
            let (schema, relation_scope) =
                bindings.output_metadata(&format!("query:{}", shipped.name))?;
            writer.put_saved(&completion::Saved {
                family,
                revision,
                name: shipped.name.into(),
                text: shipped.text.into(),
                parameters: vec![],
                help: shipped.help.into(),
                schema: Some(schema),
                relation_scope: Some(relation_scope),
            })?;
        }
        for (id, family) in &update.families {
            writer.put_dependencies(id, family)?;
        }
        for (id, (kept, binding)) in &update.kept {
            writer.put_kept(kept)?;
            writer.put_kept_binding(id, binding.as_ref())?;
        }
        let mut keys: std::collections::BTreeSet<_> = update.settings.iter().cloned().collect();
        if first || update.replaced {
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
        }
        for key in keys {
            if polylogue_model::settings::historical_hint(&key).is_some() {
                writer.remove_setting(&key)?;
                continue;
            }
            let current = reader.setting_current(&key, self.requests.machine())?;
            let choices = if current.supplier == "sinex" {
                Vec::new()
            } else {
                polylogue_model::settings::declaration(&key)
                    .map_err(|error| polylogue_store::StoreError::internal(error.to_string()))?
                    .choices()
                    .iter()
                    .map(|value| value.to_string())
                    .collect()
            };
            writer.put_setting(&completion::Setting {
                key,
                value: current,
                choices,
            })?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
