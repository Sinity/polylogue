# Vision prototypes on real data (level 3)

Each script answers one `intended` use from `vision/uses.toml` over the
operator's real archive, read-only, and prints aggregates only: counts,
distributions and anonymized shapes, never prompt text, titles, paths or
repository names. They read the pre-reset Python index (the only populated
archive today) in place with `immutable=1`; the computation is what the
rewrite's query would do, and each script's docstring says how it maps to
that use's level-1 query and where the query language could not express it.
Results and verdicts: `evidence/vision-prototypes.md`.

    ionice -c3 python3 prototypes/vision/<use-id>.py [--db PATH]

No model or embedding call is made by any of them.
