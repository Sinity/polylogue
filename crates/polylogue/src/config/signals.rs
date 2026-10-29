//! Validated deployment outlets. Authored rules name these definitions; they
//! never supply programs, arguments or output paths of their own.
use super::Error;
use serde::{Deserialize, Serialize};
use shared_signals::notifications::{NotificationConfig, Route, RouteCommand};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    time::Duration,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ValueTarget {
    pub path: PathBuf,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ActionCommand {
    pub program: PathBuf,
    #[serde(default)]
    pub arguments: Vec<String>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NotificationSettings {
    #[serde(default)]
    pub route: Route,
    pub command: Option<ActionCommand>,
    #[serde(default = "deadline")]
    pub deadline_ms: u64,
}
fn deadline() -> u64 {
    5_000
}
impl Default for NotificationSettings {
    fn default() -> Self {
        Self {
            route: Route::Dbus,
            command: None,
            deadline_ms: deadline(),
        }
    }
}
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SignalDeployment {
    #[serde(default)]
    pub notifications: NotificationSettings,
    #[serde(default)]
    pub value_targets: BTreeMap<String, ValueTarget>,
    #[serde(default)]
    pub action_commands: BTreeMap<String, ActionCommand>,
}
impl SignalDeployment {
    pub fn load(path: &Path) -> Result<Self, Error> {
        let mut config: Self = super::optional_file(path)?
            .map(|text| {
                toml::from_str(&text)
                    .map_err(|error| Error::Invalid(format!("{}: {error}", path.display())))
            })
            .transpose()?
            .unwrap_or_default();
        let base = path
            .parent()
            .ok_or_else(|| Error::Invalid("signals path has no parent".into()))?;
        for (name, target) in &mut config.value_targets {
            valid_name(name)?;
            if target.path.as_os_str().is_empty() {
                return Err(Error::Invalid("value target path cannot be blank".into()));
            }
            target.path = super::absolute(target.path.clone(), base);
        }
        for (name, command) in &mut config.action_commands {
            valid_name(name)?;
            command.resolve(base)?;
        }
        if let Some(command) = &mut config.notifications.command {
            command.resolve(base)?;
        }
        if config.notifications.deadline_ms == 0 {
            return Err(Error::Invalid(
                "notification deadline must be positive".into(),
            ));
        }
        if matches!(config.notifications.route, Route::Command | Route::Both)
            && config.notifications.command.is_none()
        {
            return Err(Error::Invalid(
                "command notification routes require an explicit program and argument list".into(),
            ));
        }
        Ok(config)
    }
    pub fn identity(&self) -> Result<String, Error> {
        let bytes = serde_json::to_vec(self)
            .map_err(|error| Error::Invalid(format!("signal configuration: {error}")))?;
        Ok(blake3::hash(&bytes).to_hex().to_string())
    }
    pub fn rule_configuration(&self) -> shared_signals::Configuration {
        shared_signals::Configuration {
            value_targets: self.value_targets.keys().cloned().collect(),
            action_commands: self.action_commands.keys().cloned().collect(),
        }
    }
    pub fn notification_config(&self) -> NotificationConfig {
        NotificationConfig {
            route: self.notifications.route,
            command: self
                .notifications
                .command
                .as_ref()
                .map(|command| RouteCommand {
                    program: command.program.clone(),
                    arguments: command.arguments.clone(),
                }),
            deadline: Duration::from_millis(self.notifications.deadline_ms),
        }
    }
}
impl ActionCommand {
    fn resolve(&mut self, base: &Path) -> Result<(), Error> {
        if self.program.as_os_str().is_empty()
            || self
                .program
                .to_str()
                .is_some_and(|value| value.trim().is_empty() || value.contains('\0'))
            || self
                .arguments
                .iter()
                .any(|argument| argument.contains('\0'))
        {
            return Err(Error::Invalid(
                "configured command program cannot be blank".into(),
            ));
        }
        self.program = super::absolute(self.program.clone(), base);
        Ok(())
    }
}
fn valid_name(name: &str) -> Result<(), Error> {
    if name.trim().is_empty() || name.chars().any(char::is_control) {
        return Err(Error::Invalid(
            "outlet names cannot be blank or contain control characters".into(),
        ));
    }
    Ok(())
}
