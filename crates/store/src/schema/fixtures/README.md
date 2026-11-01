Historical index DDL is retained unchanged for ordinary migration tests.
The tests add synthetic rows and verify source bytes and cached hash identities.

`archive_1.sql` freezes the source schema from `c8164086c7b3` before native path
bytes were retained. Migration tests keep its absent identity evidence unknown
and verify the same source change in the primary cache.

`archive_2.sql` and `index_5.sql` freeze the qualified native metadata carrier
`6b9628c965ed` before query, signal and derived evidence components. The latter
includes its coverage, variant and read-index declarations. Upgrade tests use
these actual historical shapes rather than assigning an old version to new tables.

`index_1.sql` freezes the version-1 index: the index and read-index
declarations from `a9782e7333^`, the wire-shape tables from `3822c9162c` and
the rendition search tables from `13acb1bbe7`, the three lines of work that
the integrated version 1 combines. Migrations 2 and 3 bring it to exactly the
`index_3.sql` tables, and the whole chain to the current index.

`index_15.sql` freezes the index declarations from `93da945931^`, the last
version 15 build. Migrations 14 and 15 change no declarations outside the
primary cache, so it is also the version-13 index outside that cache.
