-- polylogue-lb39z (Phase 1, item 2): the membership pipeline already decides
-- most quarantined raw_sessions rows correctly (decision IN ('applied',
-- 'superseded_equivalent', 'superseded_prefix'), membership revision_authority
-- = 'byte_proven' -- i.e. genuinely byte-proven, decided evidence) and never
-- propagates that verdict back onto raw_sessions.revision_authority. This is
-- pure fact transfer, not new judgment: the system already decided, the
-- decision was simply never written to the column the rest of the archive
-- reads authority from. A new, explicitly operator-invoked actuator
-- (devtools workspace raw-membership-writeback-apply) promotes exactly these
-- rows to revision_authority='byte_proven'.
--
-- Deliberately does NOT touch raw_sessions.predecessor_raw_id /
-- baseline_raw_id / acquisition_generation (revision-graph linkage fields
-- owned by classify_raw_revision_cohort) or revision_authority_evidence
-- (a closed CHECK vocabulary scoped to live-source-verification promotions,
-- migration 018) -- this is a different, independently-verifiable evidence
-- mechanism, and its own receipt row is the durable record of it, exactly
-- like raw_live_source_reconciliation_receipts.
CREATE TABLE IF NOT EXISTS raw_membership_writeback_receipts (
    raw_id                      TEXT PRIMARY KEY REFERENCES raw_sessions(raw_id) ON DELETE CASCADE,
    logical_source_key          TEXT NOT NULL,
    provider_session_id         TEXT NOT NULL,
    membership_decision         TEXT NOT NULL CHECK(membership_decision IN (
                                    'applied', 'superseded_equivalent', 'superseded_prefix'
                                )),
    previous_revision_authority TEXT NOT NULL CHECK(previous_revision_authority IN ('asserted', 'byte_proven', 'quarantined')),
    promoted_at_ms              INTEGER NOT NULL CHECK(promoted_at_ms >= 0),
    tool_version                TEXT NOT NULL,
    backup_manifest_path        TEXT NOT NULL,
    detail                      TEXT NOT NULL DEFAULT ''
) STRICT;

CREATE INDEX IF NOT EXISTS idx_raw_membership_writeback_receipts_promoted_at
ON raw_membership_writeback_receipts(promoted_at_ms);
