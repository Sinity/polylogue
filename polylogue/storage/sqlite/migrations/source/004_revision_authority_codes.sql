-- migration-safety: additive-with-backfill
-- Preserve the explanatory census detail while giving revision governance a
-- stable machine-readable authority. Existing marker rows are translated once;
-- arbitrary parser/error detail remains display-only with a NULL code.
ALTER TABLE raw_membership_census ADD COLUMN revision_authority TEXT;

UPDATE raw_membership_census
SET revision_authority = 'byte_proven'
WHERE detail = 'append fragments are governed by byte revision authority';

UPDATE raw_membership_census
SET revision_authority = 'quarantined'
WHERE detail IN (
    'historical non-prefix full revision governance',
    'cross-route full revision governance'
);
