# Current read authority

Session catalogs, Sessions and Messages reuse prepared SQL statements on their existing read connections. Every visibility call still executes its current generation, identity and removal checks with the supplied parameters. Session and catalog reads also execute the current archive-excision check. Statement reuse does not retain an authorization result or pin these live authority connections in a read transaction.

The selected content snapshot remains separate from live removal authority. A hidden Message, changed removal epoch, changed generation or newly recorded archive excision can therefore refuse a reader that already pinned its content. Existing corruption, cancellation, deadlines and resource allowances remain in force.

This reduces repeated SQL compilation. It does not establish an elapsed-time bound for corpus search or explain every client timeout.
