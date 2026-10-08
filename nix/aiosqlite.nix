# Shared runtime dependency for Polylogue and its embedded package consumers.
{ pkgs, pythonPackages }:
pythonPackages.aiosqlite.overridePythonAttrs (_old: {
  version = "0.22.1";
  src = pkgs.fetchPypi {
    pname = "aiosqlite";
    version = "0.22.1";
    hash = "sha256-BD4L140yiIwKnKkPx4izh5aEM2DIVacmKlMoExM6BlA=";
  };
  dependencies = [ ];
})
