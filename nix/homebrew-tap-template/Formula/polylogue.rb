class Polylogue < Formula
  include Language::Python::Virtualenv

  desc "Local archive for AI sessions (Claude, ChatGPT, Codex, Gemini)"
  homepage "https://github.com/Sinity/polylogue"
  # url, sha256, and version are bumped automatically by the
  # `.github/workflows/homebrew-bump.yml` workflow in the upstream repo on
  # each `vX.Y.Z` tag push. The placeholders below are the source of truth
  # at template installation time; do not edit them by hand once the bump
  # workflow has overwritten this file in the live tap.
  url "https://files.pythonhosted.org/packages/source/p/polylogue/polylogue-0.1.0.tar.gz"
  sha256 "0000000000000000000000000000000000000000000000000000000000000000"
  license "MIT"

  depends_on "python@3.13"
  depends_on "rust" => :build      # cryptography, watchfiles, nh3 source builds
  depends_on "pkg-config" => :build

  # Resource blocks are populated by `brew update-python-resources Formula/polylogue.rb`
  # in the homebrew-bump workflow. They mirror the `[project] dependencies`
  # list in pyproject.toml plus their transitive closure.
  #
  # The placeholder entries below let `brew audit --strict` discover the
  # virtualenv install path before the bump workflow has run; they are
  # rewritten in their entirety on every release.

  def install
    virtualenv_install_with_resources
  end

  service do
    run [opt_bin/"polylogued", "run"]
    keep_alive true
    log_path var/"log/polylogue/polylogued.log"
    error_log_path var/"log/polylogue/polylogued.err.log"
    environment_variables POLYLOGUE_FORCE_PLAIN: "1"
  end

  test do
    # Entrypoint smoke: each installed binary must answer `--version` or
    # `--help` cleanly against an empty XDG archive root, matching the
    # PyPI installed-smoke matrix in `.github/workflows/release.yml`.
    ENV["POLYLOGUE_ARCHIVE_ROOT"] = testpath/"archive"
    ENV["POLYLOGUE_CONFIG_DIR"] = testpath/"config"
    ENV["POLYLOGUE_FORCE_PLAIN"] = "1"
    (testpath/"archive").mkpath
    (testpath/"config").mkpath

    assert_match version.to_s, shell_output("#{bin}/polylogue --version")
    system bin/"polylogued", "--help"
    system bin/"polylogue-mcp", "--help"
  end
end
