-- Standing document selections are canonical rows, including input/text removals.
-- A new archive version prevents older writers from discarding this policy;
-- publication includes these rows in the existing excision unit.
CREATE TABLE excision_document (
  excision_id TEXT NOT NULL REFERENCES excision(id) ON DELETE CASCADE,
  document_id TEXT NOT NULL,
  PRIMARY KEY(excision_id,document_id)
) STRICT, WITHOUT ROWID;
