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

// Long-lived state holds completion metadata and dependency IDs, never saved AST bodies.
#[derive(Default)]
struct Vocabulary {
    families: BTreeMap<String, Family>,
    source_users: BTreeMap<String, std::collections::BTreeSet<String>>,
    name_users: BTreeMap<String, std::collections::BTreeSet<String>>,
    kept: BTreeMap<String, polylogue_query::Kept>,
}
#[derive(Default)]
struct Family {
    names: BTreeMap<String, usize>,
    sources: std::collections::BTreeSet<String>,
}
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
            vocabulary: Vocabulary::default(),
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
            self.vocabulary = Vocabulary::default();
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
                .is_some_and(|old| old.generation != position.generation)
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
                    self.writer = Some((position.generation.clone(), writer));
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
                    native_identity: kept.native_identity,
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
                // An unavailable body can still declare exact dependencies. Read
                // that immutable object for invalidation only, never as execution proof.
                let object = match &value.body {
                    polylogue_query::SavedQueryBody::Frozen { object, .. } => object.clone(),
                    polylogue_query::SavedQueryBody::Unavailable { .. } => {
                        match reader.query_object(&value.hash.to_string(), None) {
                            Ok(found) => found.map(|found| std::sync::Arc::new(found.value)),
                            Err(
                                polylogue_store::StoreError::NotFound(_)
                                | polylogue_store::StoreError::Invalid(_)
                                | polylogue_store::StoreError::ReferenceUnavailable { .. }
                                | polylogue_store::StoreError::Ambiguous { .. },
                            ) => None,
                            Err(error) => return Err(error),
                        }
                    }
                };
                if let Some(object) = object {
                    for output in object.source_outputs.iter().chain(
                        object
                            .definitions
                            .iter()
                            .flat_map(|node| &node.source_outputs),
                    ) {
                        if let query_language::catalog::OutputSource::Result { reference, .. } =
                            &output.source
                        {
                            sources.insert(reference.clone());
                        }
                    }
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
        saved: &completion::NativeSaved,
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
                    "assertion:{}@{}",
                    change.family, saved.revision
                ))
            })?;
        if names > 1 || !current.conflicts.is_empty() {
            definition.help = format!("{} (ambiguous query name)", definition.text);
            return Ok(definition);
        }
        let mut kept = BTreeMap::new();
        for value in change.queries.values().flatten() {
            if let polylogue_query::SavedQueryBody::Frozen {
                object: Some(object),
                ..
            } = &value.body
            {
                for output in object.source_outputs.iter().chain(
                    object
                        .definitions
                        .iter()
                        .flat_map(|node| &node.source_outputs),
                ) {
                    if let query_language::catalog::OutputSource::Result { reference, .. } =
                        &output.source
                    {
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
        }
        let bindings = polylogue_query::Bindings {
            origins: polylogue_sources::origin::ORIGINS
                .iter()
                .map(|origin| origin.name.to_owned())
                .collect(),
            saved: change.queries.clone(),
            kept,
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
        let mut put = |change| writer.apply(change);
        let first = match sessions {
            Sessions::All => {
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
                &polylogue_model::Reference::Assertion(polylogue_model::RevisionReference {
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
        for (kept, _) in update.kept.values() {
            writer.put_kept(kept)?;
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
