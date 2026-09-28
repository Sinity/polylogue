"""Embedding storage, materialization, and readiness helpers.

Import from the owning submodule. The package deliberately re-exports
nothing: an eager re-export of ``derivation`` made importing any leaf here
(``identity``, from the archive-tier writer) re-enter the writer mid-import.
"""
