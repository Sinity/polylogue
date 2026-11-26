/// Every subprocess receives a synthetic deployment and a small explicit
/// environment, including indirect children such as the Python client.
pub(crate) struct Fixture {
    sandbox: polylogue_testkit::Sandbox,
    archive: std::path::PathBuf,
}
impl Fixture {
    /// The sandbox's root is `config`'s parent.
    pub(crate) fn new(
        archive: &std::path::Path,
        home: &std::path::Path,
        config: &std::path::Path,
    ) -> Self {
        let root = config.parent().expect("fixture configuration has a parent");
        assert!(archive.is_absolute() && home.is_absolute() && config.is_absolute());
        Self {
            sandbox: polylogue_testkit::Sandbox::with_home(root, home, config).unwrap(),
            archive: archive.into(),
        }
    }
    pub(crate) fn archive(&self) -> &std::path::Path {
        &self.archive
    }
    pub(crate) fn sandbox(&self) -> &polylogue_testkit::Sandbox {
        &self.sandbox
    }
    pub(crate) fn environment(&self, command: &mut std::process::Command) {
        self.sandbox.create_directories().unwrap();
        self.sandbox.write_deployment(&self.archive).unwrap();
        self.sandbox
            .apply(command)
            .current_dir(self.sandbox.root())
            // A terminal read pages through $PAGER; no one reads these pages.
            .env("PAGER", "cat");
    }
    pub(crate) fn command(&self) -> std::process::Command {
        let mut command = std::process::Command::new(crate::BIN);
        self.environment(&mut command);
        command
            .arg("--archive")
            .arg(&self.archive)
            .arg("--config")
            .arg(self.sandbox.deployment());
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
