/// Every subprocess receives a synthetic deployment and a small explicit
/// environment, including indirect children such as the Python client.
pub(crate) struct Fixture {
    root: std::path::PathBuf,
    archive: std::path::PathBuf,
    home: std::path::PathBuf,
    config: std::path::PathBuf,
}
impl Fixture {
    pub(crate) fn new(
        archive: &std::path::Path,
        home: &std::path::Path,
        config: &std::path::Path,
    ) -> Self {
        let root = config
            .parent()
            .expect("fixture configuration has a parent")
            .to_owned();
        assert!(archive.is_absolute() && home.is_absolute() && config.is_absolute());
        Self {
            root,
            archive: archive.into(),
            home: home.into(),
            config: config.into(),
        }
    }
    pub(crate) fn environment(&self, command: &mut std::process::Command) {
        use std::os::unix::fs::PermissionsExt;
        let runtime = self.root.join("runtime");
        for directory in [
            &self.home,
            &self.config,
            &self.root.join("data"),
            &self.root.join("state"),
            &self.root.join("cache"),
            &runtime,
            &self.root.join("tmp"),
        ] {
            std::fs::create_dir_all(directory).unwrap();
        }
        std::fs::set_permissions(&runtime, std::fs::Permissions::from_mode(0o700)).unwrap();
        let deployment = self.config.join("polylogue/polylogue.toml");
        std::fs::create_dir_all(deployment.parent().unwrap()).unwrap();
        if !deployment.exists() {
            let archive = serde_json::to_string(&self.archive).unwrap();
            std::fs::write(&deployment, format!("archive={archive}\n[day]\nzone='UTC'\nstart='04:00:00'\n[http]\nlisten='127.0.0.1:0'\n")).unwrap();
        }
        command
            .env_clear()
            .current_dir(&self.root)
            .env(
                "PATH",
                std::env::var_os("PATH").expect("test tools need PATH"),
            )
            .env("LANG", "C.UTF-8")
            .env("TMPDIR", self.root.join("tmp"))
            .env("HOME", &self.home)
            .env("XDG_CONFIG_HOME", &self.config)
            .env("XDG_DATA_HOME", self.root.join("data"))
            .env("XDG_STATE_HOME", self.root.join("state"))
            .env("XDG_CACHE_HOME", self.root.join("cache"))
            .env("XDG_RUNTIME_DIR", runtime);
    }
    pub(crate) fn command(&self) -> std::process::Command {
        let mut command = std::process::Command::new(crate::BIN);
        self.environment(&mut command);
        command
            .arg("--archive")
            .arg(&self.archive)
            .arg("--config")
            .arg(self.config.join("polylogue/polylogue.toml"));
        command
    }
    /// Chrome's native-host argument occupies argv[1]. Its explicit fixture
    /// deployment is selected through isolated XDG_CONFIG_HOME instead.
    pub(crate) fn native_command(&self) -> std::process::Command {
        let mut command = std::process::Command::new(crate::BIN);
        self.environment(&mut command);
        command
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn synthetic_children_drop_foreign_sources_bus_and_home() {
        use std::os::unix::fs::PermissionsExt;
        let root = tempfile::tempdir().unwrap();
        let fixture = Fixture::new(
            &root.path().join("archive"),
            &root.path().join("home"),
            &root.path().join("config"),
        );
        let mut command = std::process::Command::new("env");
        command
            .env("HOME", "/synthetic/foreign-home")
            .env("CLAUDE_CONFIG_DIR", "/synthetic/foreign-provider")
            .env(
                "DBUS_SESSION_BUS_ADDRESS",
                "unix:path=/synthetic/foreign-bus",
            )
            .env("CLAUDE_CODE_SESSION_ID", "foreign-session");
        fixture.environment(&mut command);
        let result = command.output().unwrap();
        assert!(result.status.success());
        let text = String::from_utf8(result.stdout).unwrap();
        let values: std::collections::BTreeMap<_, _> = text
            .lines()
            .filter_map(|line| line.split_once('='))
            .collect();
        assert_eq!(values["HOME"], fixture.home.to_str().unwrap());
        assert!(!values.contains_key("CLAUDE_CONFIG_DIR"));
        assert!(!values.contains_key("DBUS_SESSION_BUS_ADDRESS"));
        assert!(!values.contains_key("CLAUDE_CODE_SESSION_ID"));
        for (name, path) in [
            ("XDG_CONFIG_HOME", fixture.config.clone()),
            ("XDG_DATA_HOME", fixture.root.join("data")),
            ("XDG_STATE_HOME", fixture.root.join("state")),
            ("XDG_CACHE_HOME", fixture.root.join("cache")),
            ("XDG_RUNTIME_DIR", fixture.root.join("runtime")),
        ] {
            assert_eq!(values[name], path.to_str().unwrap());
        }
        assert_eq!(
            std::fs::metadata(fixture.root.join("runtime"))
                .unwrap()
                .permissions()
                .mode()
                & 0o777,
            0o700
        );
    }
}
