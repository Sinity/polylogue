# shell.nix
{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = [
    # Python interpreter itself
    pkgs.python3

    # Google API libraries
    pkgs.python3Packages.google-api-python-client
    pkgs.python3Packages.google-auth-oauthlib
    pkgs.python3Packages.google-auth-httplib2 # For sync auth flow
    pkgs.python3Packages.aiohttp             # For async transport

    # Utility libraries
    pkgs.python3Packages.tqdm                # Progress bars
    pkgs.python3Packages.pathvalidate        # Filename sanitization
    pkgs.python3Packages.aiofiles            # Async file I/O <--- ADDED
  ];
}
