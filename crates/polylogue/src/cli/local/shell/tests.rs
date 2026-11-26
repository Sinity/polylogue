use std::process::{Command, ExitCode};

fn fixture() -> (tempfile::TempDir, crate::config::Startup) {
    let directory = tempfile::tempdir().unwrap();
    let startup = crate::config::Startup::for_test(
        crate::test_daemon::engine_config(directory.path().join("home")),
        &directory.path().join("archive"),
    );
    (directory, startup)
}
fn invoke(startup: &crate::config::Startup, shell: &str, options: &[&str]) -> (ExitCode, Vec<u8>) {
    let mut words = vec!["polylogue", "install", shell];
    if !options.contains(&"--program") {
        words.extend(["--program", "plg"]);
    }
    words.extend(options);
    let (_, parsed) = crate::cli::parse(&words).unwrap();
    let mut output = vec![];
    let code = super::super::install_with_output(
        &parsed,
        startup,
        true,
        &mut output,
        &crate::auth::Drive::default(),
    );
    (code, output)
}
fn paths(
    startup: &crate::config::Startup,
    shell: &str,
) -> (std::path::PathBuf, Option<std::path::PathBuf>) {
    let paths = &startup.config.paths;
    match shell {
        "zsh" => (
            paths.data.join("zsh/site-functions/_plg"),
            Some(startup.client.zsh_directory.join(".zshrc")),
        ),
        "fish" => (
            paths.config.join("fish/completions/plg.fish"),
            Some(paths.config.join("fish/conf.d/plg-polylogue.fish")),
        ),
        "bash" => (paths.data.join("bash-completion/completions/plg"), None),
        _ => unreachable!(),
    }
}

#[test]
fn install_shell_remove_restores_original_bytes_and_repeats() {
    for shell in ["zsh", "bash", "fish"] {
        let (_directory, startup) = fixture();
        let (completion, init) = paths(&startup, shell);
        assert_eq!(invoke(&startup, shell, &["--dry-run"]).0, ExitCode::SUCCESS);
        assert!(!startup.config.paths.home.exists());
        let mut files = vec![completion.clone()];
        files.extend(init.iter().cloned());
        for path in &files {
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, b"# original shell configuration\n").unwrap();
        }
        let (code, script) = invoke(&startup, shell, &["--dry-run"]);
        assert_eq!(code, ExitCode::SUCCESS);
        assert_eq!(
            std::fs::read(&completion).unwrap(),
            b"# original shell configuration\n"
        );
        for _ in 0..2 {
            assert_eq!(invoke(&startup, shell, &[]).0, ExitCode::SUCCESS);
            assert_eq!(std::fs::read(&completion).unwrap(), script);
        }
        for _ in 0..2 {
            assert_eq!(invoke(&startup, shell, &["--remove"]).0, ExitCode::SUCCESS);
            for path in &files {
                assert_eq!(
                    std::fs::read(path).unwrap(),
                    b"# original shell configuration\n"
                );
                assert!(!crate::installation::backup_path(path).exists());
            }
        }
    }
}

#[test]
fn install_shell_remove_restores_startup_file_without_final_newline() {
    for shell in ["zsh", "fish"] {
        let (_directory, startup) = fixture();
        let (completion, startup_file) = paths(&startup, shell);
        let startup_file = startup_file.unwrap();
        for path in [&completion, &startup_file] {
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        }
        std::fs::write(&completion, b"# completion\n").unwrap();
        std::fs::write(&startup_file, b"# last line without newline").unwrap();
        assert_eq!(invoke(&startup, shell, &[]).0, ExitCode::SUCCESS);
        assert_eq!(invoke(&startup, shell, &["--remove"]).0, ExitCode::SUCCESS);
        assert_eq!(
            std::fs::read(&startup_file).unwrap(),
            b"# last line without newline"
        );
        assert!(!crate::installation::backup_path(&startup_file).exists());
    }
}

#[test]
fn install_zsh_preserves_later_unrelated_startup_edits_and_honors_zdotdir() {
    let (_directory, mut startup) = fixture();
    startup.client.zsh_directory = startup.config.paths.home.join("zsh-startup");
    let (_, init) = paths(&startup, "zsh");
    let init = init.unwrap();
    std::fs::create_dir_all(init.parent().unwrap()).unwrap();
    let original = "# unrelated startup\nexport USER_SETTING=kept\n";
    std::fs::write(&init, original).unwrap();
    assert_eq!(invoke(&startup, "zsh", &[]).0, ExitCode::SUCCESS);
    let mut installed = std::fs::read(&init).unwrap();
    installed.extend_from_slice(b"# a later user edit\n");
    std::fs::write(&init, installed).unwrap();
    assert_eq!(invoke(&startup, "zsh", &["--remove"]).0, ExitCode::SUCCESS);
    assert_eq!(
        std::fs::read_to_string(&init).unwrap(),
        format!("{original}# a later user edit\n")
    );
    assert_eq!(
        std::fs::read_to_string(crate::installation::backup_path(&init)).unwrap(),
        original
    );
    assert!(!startup.config.paths.home.join(".zshrc").exists());
}

#[test]
fn install_shell_preflights_managed_startup_before_writing_completion() {
    for shell in ["zsh", "fish"] {
        let (_directory, startup) = fixture();
        let (completion, init) = paths(&startup, shell);
        let init = init.unwrap();
        let source = startup.config.paths.home.join("managed");
        std::fs::create_dir_all(init.parent().unwrap()).unwrap();
        std::fs::write(&source, b"# managed\n").unwrap();
        std::os::unix::fs::symlink(&source, &init).unwrap();
        assert_eq!(invoke(&startup, shell, &[]).0, ExitCode::from(4));
        assert!(!completion.exists());
        assert_eq!(std::fs::read(source).unwrap(), b"# managed\n");
    }
}

#[test]
fn install_shell_normal_startup_loads_picker_binding() {
    for shell in ["zsh", "fish"] {
        let (_directory, startup) = fixture();
        assert_eq!(invoke(&startup, shell, &[]).0, ExitCode::SUCCESS);
        let check = if shell == "zsh" {
            r#"[[ "$(zle -l)" == *'_polylogue_pick_plg'* ]] || exit 5
[[ "$(bindkey '^[p')" == *'_polylogue_pick_plg' ]] || exit 6
print -r -- shell-ready"#
        } else {
            r#"functions --query _polylogue_pick_plg; or exit 5
string match --quiet '*_polylogue_pick_plg' (bind \ep); or exit 6
printf 'shell-ready\n'"#
        };
        let output = Command::new(shell)
            .args(["-i", "-c", check])
            .env("HOME", &startup.config.paths.home)
            .env("ZDOTDIR", &startup.client.zsh_directory)
            .env("XDG_CONFIG_HOME", &startup.config.paths.config)
            .env("XDG_DATA_HOME", &startup.config.paths.data)
            .env("XDG_CACHE_HOME", &startup.config.paths.cache)
            .env(
                "XDG_STATE_HOME",
                startup.config.paths.home.join(".local/state"),
            )
            .env("TERM", "xterm")
            .output()
            .unwrap();
        assert!(output.status.success(), "{shell}: {output:?}");
        assert_eq!(output.stdout, b"shell-ready\n", "{shell}: {output:?}");
    }
}

#[test]
fn install_shell_wrapper_function_names_replace_unsafe_program_characters() {
    for shell in ["zsh", "fish"] {
        let (_directory, startup) = fixture();
        assert_eq!(
            invoke(&startup, shell, &["--program", "work archive's"]).0,
            ExitCode::SUCCESS
        );
        let output = Command::new(shell)
            .args(["-i", "-c", if shell == "zsh" {
                "[[ $(bindkey '^[p') == *'_polylogue_pick_work_archive_s' ]] && print -r -- wrapper-ready"
            } else {
                "functions --query _polylogue_pick_work_archive_s; and printf 'wrapper-ready\n'"
            }])
            .env("HOME", &startup.config.paths.home)
            .env("ZDOTDIR", &startup.client.zsh_directory)
            .env("XDG_CONFIG_HOME", &startup.config.paths.config)
            .env("XDG_DATA_HOME", &startup.config.paths.data)
            .env("XDG_CACHE_HOME", &startup.config.paths.cache)
            .env("XDG_STATE_HOME", startup.config.paths.home.join(".local/state"))
            .env("TERM", "xterm")
            .output().unwrap();
        assert!(output.status.success(), "{shell}: {output:?}");
        assert_eq!(output.stdout, b"wrapper-ready\n", "{shell}: {output:?}");
    }
}
