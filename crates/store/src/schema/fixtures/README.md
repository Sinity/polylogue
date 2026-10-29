Historical index DDL is retained unchanged for ordinary migration tests.
The tests add synthetic rows and verify source bytes and cached hash identities.

`archive_1.sql` freezes the source schema from `c8164086c7b3` before native path
bytes were retained. Migration tests keep its absent identity evidence unknown
and verify the same source change in the primary cache.

`archive_2.sql` and `index_5.sql` freeze the qualified native metadata carrier
`6b9628c965ed` before query, signal and derived evidence components. The latter
includes its coverage, variant and read-index declarations. Upgrade tests use
these actual historical shapes rather than assigning an old version to new tables.
