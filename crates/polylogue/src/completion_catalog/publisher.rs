//! One startup snapshot, followed by indexed deltas after committed writes.
use crate::daemon::service::Daemon;
use polylogue_model::ChangePosition;
use polylogue_store::{
    Reader, Result,
    completion::{self, CatalogWriter, CommitSubscription, Metadata, SessionChange},
};
use query_language::{
    catalog::Catalog,
};
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
    wake: Option<SyncSender<()>>,
    thread: Option<JoinHandle<()>>,
}
impl Publisher {
    pub fn start(
        daemon: Arc<Daemon>,
        subscription: Option<CommitSubscription>,
    ) -> std::io::Result<Self> {
        let stop = Arc::new(AtomicBool::new(false));
        let stopped = Arc::clone(&stop);
        let wake = subscription.as_ref().map(CommitSubscription::wake_handle);
        let thread = std::thread::Builder::new()
            .name("polylogue-completion".into())
            .spawn(move || {
                let mut publisher = State::new(daemon.archive.is_primary());
                let mut last_error = None;
                loop {
                    if stopped.load(Ordering::Acquire) {
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
                                return;
                            }
                        }
                        None => return,
                    }
                }
            })?;
        Ok(Self {
            stop,
            wake,
            thread: Some(thread),
        })
    }
}
impl Drop for Publisher {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(wake) = &self.wake {
            let _ = wake.try_send(());
        }
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

pub(super) struct State {
    writer: Option<CatalogWriter>,
    position: Option<ChangePosition>,
    catalog: Catalog,
    definitions: BTreeMap<String, completion::Saved>,
    kept: BTreeMap<String, polylogue_query::Kept>,
    requires_confirmed: bool,
}
impl State {
    pub(super) fn new(requires_confirmed: bool) -> Self {
        Self {
            writer: None,
            position: None,
            catalog: super::base_catalog(),
            definitions: BTreeMap::new(),
            kept: BTreeMap::new(),
            requires_confirmed,
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
            .is_some_and(|writer| !writer.is_current_file())
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
            written_at: polylogue_store::user_writes::now().unix_ms(),
        };
        let pointer = completion::CatalogPointer::new(path.to_owned());
        let generation_path = pointer.generation_path(&position.generation);
        let after = self.position.clone();
        let result = if let Some(mut writer) = self.writer.take() {
            let result = writer.update(&metadata, |writer| {
                self.apply(reader, writer, after.as_ref())
            });
            self.writer = Some(writer);
            result
        } else {
            CatalogWriter::build(&generation_path, &metadata, |writer| {
                writer.put_catalog(&self.catalog)?;
                self.apply(reader, writer, None)
            })
            .map(|writer| {
                self.writer = Some(writer);
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
        reader.visit_completion_definitions(after.map(|position| position.archive), |change| {
            changed = true;
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
            writer.put_values(&format!("note:{}", change.family), &values)?;
            if let Some(saved) = change.saved {
                self.definitions.insert(
                    change.family.clone(),
                    completion::Saved {
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
        })?;
        reader.visit_completion_kept(after.map(|position| position.archive), |kept| {
            changed = true;
            let columns: Option<Vec<polylogue_query::Column>> = kept
                .columns
                .and_then(|columns| serde_json::from_value(columns).ok());
            let matches: Option<Vec<polylogue_query::Column>> = kept
                .match_columns
                .and_then(|columns| serde_json::from_value(columns).ok());
            let schema = kept
                .unit
                .as_ref()
                .zip(columns.as_ref())
                .map(|(unit, columns)| {
                    let mut schema =
                        polylogue_query::compile::columns_schema(&self.catalog, unit, columns);
                    if self.catalog.unit(unit).is_none() {
                        schema.unit = None;
                    }
                    schema
                });
            let match_schema = kept
                .unit
                .as_ref()
                .zip(matches.as_ref())
                .map(|(unit, columns)| {
                    let mut schema =
                        polylogue_query::compile::columns_schema(&self.catalog, unit, columns);
                    if self.catalog.unit(unit).is_none() {
                        schema.unit = None;
                    }
                    schema
                });
            writer.put_kept(&completion::Kept {
                id: kept.id.clone(),
                name: kept.name,
                unit: kept.unit.clone(),
                row_count: kept.row_count,
                rows: schema,
                matches: match_schema,
            })?;
            self.kept.remove(&kept.id);
            if let (Some(unit), Some(columns)) = (kept.unit, columns) {
                self.kept.insert(
                    kept.id,
                    polylogue_query::Kept {
                        unit,
                        columns,
                        match_columns: matches.unwrap_or_default(),
                    },
                );
            }
            Ok(())
        })?;
        if changed {
            self.publish_definitions(writer)?;
        }
        let mut keys = reader.completion_setting_keys(after.map(|position| position.archive))?;
        if after.is_none() {
            keys.extend(KNOWN_SETTINGS.iter().map(|(key, _)| key.to_string()));
            keys.sort();
            keys.dedup();
        }
        let machine = polylogue_store::user_writes::machine_id();
        for key in keys {
            let value = reader.setting(&key, &machine)?;
            writer.put_setting(&completion::Setting {
                key: key.clone(),
                value: serde_json::to_value(value)?,
                choices: KNOWN_SETTINGS
                    .iter()
                    .find(|(name, _)| *name == key)
                    .map(|(_, choices)| choices.iter().map(|value| value.to_string()).collect())
                    .unwrap_or_default(),
            })?;
        }
        Ok(())
    }
    fn publish_definitions(&mut self, writer: &mut CatalogWriter) -> Result<()> {
        let mut saved: BTreeMap<String, Vec<polylogue_query::SavedQuery>> = BTreeMap::new();
        for definition in self.definitions.values() {
            saved
                .entry(definition.name.clone())
                .or_default()
                .push(polylogue_query::SavedQuery {
                    text: definition.text.clone(),
                    revision: definition.revision.clone(),
                    first_note: definition.family.clone(),
                });
        }
        let bindings=polylogue_query::Bindings {
            origins:polylogue_sources::origin::ORIGINS.iter().map(|origin|origin.name.to_owned()).collect(),
            command_words:crate::commands::command_words(),saved,kept:self.kept.clone(),
        };
        for definition in self.definitions.values_mut() {
            definition.parameters =
                bindings.definition_parameters(&definition.text)
                    .unwrap_or_default()
                    .into_iter()
                    .map(|name| completion::Parameter { name, kind: None })
                    .collect();
            definition.schema=bindings.output_schema(&definition.text).ok();
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
            writer.put_saved(definition)?;
        }
        Ok(())
    }
}
const KNOWN_SETTINGS: &[(&str, &[&str])] = &[
    ("briefing.session_start", &["on", "off"]),
    ("briefing.budget", &[]),
    ("browser.actions", &["on", "off"]),
    ("day.zone", &[]),
    ("day.logical_start", &[]),
    ("embeddings", &[]),
    ("notify.command", &[]),
    ("signal.interrupts_per_hour", &[]),
];
