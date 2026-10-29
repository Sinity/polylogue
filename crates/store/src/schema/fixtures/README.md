Historical index DDL is retained unchanged for ordinary migration tests.
The tests add synthetic rows and verify source bytes and cached hash identities.

`archive_1.sql` freezes the source schema from `c8164086c7b3` before native path
bytes were retained. Migration tests keep its absent identity evidence unknown
and verify the same source change in the primary cache.
