use super::*;
#[test]
fn codes_round_trip_the_wire_names() {
    for &code in Code::ALL {
        let text = serde_json::to_value(code).unwrap();
        assert_eq!(text.as_str(), Some(code.as_str()));
        assert_eq!(Code::from_name(code.as_str()), Some(code));
        assert_eq!(serde_json::from_value::<Code>(text).unwrap(), code);
    }
}

#[test]
fn every_reason_has_one_code_and_one_text() {
    let table = [
        (
            Reason::SingleValuedBulk {
                kind: "kind-x".into(),
                rows: 2,
            },
            Code::BadRequest,
            "`kind-x` holds one value per target and is not written over 2 rows; label rows with a rubric field (annotation) instead",
        ),
        (
            Reason::BadReference {
                text: "text-x".into(),
                problem: "problem-x".into(),
            },
            Code::BadRequest,
            "text-x is not a reference: problem-x",
        ),
        (
            Reason::RootNamesSession {},
            Code::BadRequest,
            "root: names a session, the root of a logical session",
        ),
        (
            Reason::RangeAcrossSessions {},
            Code::BadRequest,
            "a message range is read inside one physical session",
        ),
        (
            Reason::RangeEndsBeforeStart {},
            Code::BadRequest,
            "a message range ends before it starts",
        ),
        (
            Reason::CompactionCount {
                text: "text-x".into(),
            },
            Code::BadValue,
            "compaction:N takes a count, not `text-x`",
        ),
        (
            Reason::DiffTarget {},
            Code::BadRequest,
            "diff takes a message or compaction[:N]",
        ),
        (
            Reason::PhysicalNeedsRoot {},
            Code::BadRequest,
            "--physical applies to root:REF; a session reference is already one physical session",
        ),
        (
            Reason::UnknownReadFormat {
                format: "format-x".into(),
            },
            Code::BadValue,
            "unknown read format `format-x`",
        ),
        (
            Reason::FromSeenNeedsContentView {},
            Code::BadRequest,
            "read --from seen needs a session content view",
        ),
        (
            Reason::FromSeenNeedsOneSession {},
            Code::BadRequest,
            "read --from seen needs one session or logical-root reference",
        ),
        (
            Reason::TraceTakesNoAlternates {},
            Code::BadRequest,
            "trace shows original source evidence; alternates are unsupported",
        ),
        (
            Reason::StreamHasNoContinuation {},
            Code::BadRequest,
            "streaming reads do not have paged continuations",
        ),
        (
            Reason::PagedReadNeedsPageDelivery {},
            Code::BadRequest,
            "paged reads require page delivery",
        ),
        (
            Reason::StreamNeedsStreamDelivery {},
            Code::BadRequest,
            "streaming reads require stream delivery",
        ),
        (
            Reason::BlobsNeedStream {},
            Code::BadRequest,
            "with_blobs requires streaming delivery",
        ),
        (
            Reason::TwoMessageAnchors {},
            Code::BadRequest,
            "context has two different message anchors; select one message",
        ),
        (
            Reason::BadTime {
                text: "text-x".into(),
                problem: "problem-x".into(),
            },
            Code::BadRequest,
            "`text-x` is not a time: problem-x",
        ),
        (
            Reason::HistoricalContextTakesNoAround {},
            Code::BadRequest,
            "historical context does not accept around; select its context message",
        ),
        (
            Reason::PositionFromAnotherGeneration {},
            Code::BadRequest,
            "read --at position belongs to another index generation",
        ),
        (
            Reason::PositionAhead {},
            Code::BadRequest,
            "read --at position is ahead of this reader",
        ),
        (
            Reason::HistoricalReadTarget {},
            Code::BadRequest,
            "historical read --at takes a session or message",
        ),
        (
            Reason::HistoricalContextTarget {},
            Code::BadRequest,
            "historical context takes one message or a session; select its context message",
        ),
        (
            Reason::BudgetBelowFraming {},
            Code::BadRequest,
            "the page budget cannot hold the reading's metadata; raise the budget",
        ),
        (
            Reason::NoSuchAttachment {
                message: "message-x".into(),
                block: 2,
                attachment: 2,
            },
            Code::NotFound,
            "message-x block 2 has no attachment 2",
        ),
        (
            Reason::AttachmentDescriptorMismatch {
                message: "message-x".into(),
                block: 3,
                attachment: 3,
            },
            Code::Conflict,
            "the attachment descriptor for message-x block 3 attachment 3 differs from the one its source records",
        ),
        (
            Reason::OutsideCalendar {
                problem: "problem-x".into(),
            },
            Code::BadValue,
            "the time is outside the representable calendar: problem-x",
        ),
        (
            Reason::UnknownSetting {
                key: "key-x".into(),
            },
            Code::BadValue,
            "unknown runtime setting `key-x`",
        ),
        (
            Reason::BlankSetting {
                key: "key-x".into(),
            },
            Code::BadValue,
            "runtime setting `key-x` cannot be blank",
        ),
        (
            Reason::BadSettingValue {
                key: "key-x".into(),
                expected: "expected-x".into(),
            },
            Code::BadValue,
            "invalid value for runtime setting `key-x`: expected expected-x",
        ),
        (
            Reason::SettingOwnedBySinex {
                key: "key-x".into(),
            },
            Code::Refused,
            "runtime setting `key-x` is owned by Sinex in primary mode",
        ),
        (
            Reason::SettingConflict {
                key: "key-x".into(),
                revisions: Names(vec!["a".into(), "b".into()]),
            },
            Code::Ambiguous,
            "runtime setting `key-x` has conflicting current revisions: a, b",
        ),
        (
            Reason::WatchNeedsName {},
            Code::BadRequest,
            "a watch needs a saved query name",
        ),
        (
            Reason::FigureNeedsName {},
            Code::BadRequest,
            "a figure needs a saved query name",
        ),
        (
            Reason::WatchDuration {
                text: "text-x".into(),
            },
            Code::BadValue,
            "a watch needs a duration such as 20m, not `text-x`",
        ),
        (
            Reason::KeptRowsMissing {},
            Code::BadRequest,
            "a kept result needs its rows",
        ),
        (
            Reason::BadSavedName {
                name: "name-x".into(),
            },
            Code::BadValue,
            "`name-x` is not a saved query name",
        ),
        (
            Reason::NotAnExactRevision {
                reference: "reference-x".into(),
            },
            Code::BadRequest,
            "reference-x is not an exact annotation revision; name one revision",
        ),
        (
            Reason::WrongKind {
                reference: "reference-x".into(),
                expected: "expected-x".into(),
            },
            Code::BadRequest,
            "reference-x is not a expected-x",
        ),
        (
            Reason::BadAnnotationValue {
                kind: "kind-x".into(),
                problem: "problem-x".into(),
            },
            Code::BadValue,
            "the kind-x value is malformed: problem-x",
        ),
        (
            Reason::BadAnnotationLinks {
                problem: "problem-x".into(),
            },
            Code::BadRequest,
            "the annotation's links are refused: problem-x",
        ),
        (
            Reason::RubricValueNeedsRubric {},
            Code::BadRequest,
            "a rubric value needs its exact derived_from rubric",
        ),
        (
            Reason::NotAllowedByRubric {
                key: "key-x".into(),
                part: "part-x".into(),
            },
            Code::BadRequest,
            "the rubric does not allow part-x for `key-x`",
        ),
        (
            Reason::PassNotByAuthor {},
            Code::BadRequest,
            "a rubric value's origin must be a pass opened by its author",
        ),
        (
            Reason::PassForAnotherRubric {},
            Code::BadRequest,
            "the rubric value's pass names another rubric",
        ),
        (
            Reason::PassRequired {},
            Code::BadRequest,
            "a rubric value needs an exact pass origin",
        ),
        (
            Reason::BasesSpanFamilies {},
            Code::BadRequest,
            "one revision must name bases of one annotation family, or of families holding one slot",
        ),
        (
            Reason::SlotTaken {
                kind: "kind-x".into(),
                holders: Names(vec!["a".into(), "b".into()]),
            },
            Code::Conflict,
            "the revised kind-x would take a slot also held by a, b; a revision naming every head of every holder resolves it",
        ),
        (
            Reason::NoStatements {},
            Code::BadRequest,
            "assert needs at least one statement",
        ),
        (
            Reason::RevisionChangesKind {},
            Code::BadRequest,
            "a revision cannot change the annotation's kind",
        ),
        (
            Reason::NoExactBase {},
            Code::BadRequest,
            "revise needs at least one exact base",
        ),
        (
            Reason::WithdrawalRevised {},
            Code::BadRequest,
            "a withdrawal cannot be revised; withdraw it to restore what it withdrew",
        ),
        (
            Reason::BasesOfSeveralKinds {},
            Code::BadRequest,
            "one revision must name bases of one kind",
        ),
        (
            Reason::BasesInLinkPatch {},
            Code::BadRequest,
            "revision bases belong in bases, not the link patch",
        ),
        (
            Reason::QueryAnnotationAboutLinks {},
            Code::BadRequest,
            "a query annotation cannot supply its own about links",
        ),
        (
            Reason::BadAnnotation {
                problem: "problem-x".into(),
            },
            Code::BadRequest,
            "the annotation is refused: problem-x",
        ),
        (
            Reason::UnknownChoice {
                field: "field-x".into(),
                value: "value-x".into(),
                known: Names(vec!["a".into(), "b".into()]),
            },
            Code::BadValue,
            "field-x takes one of a, b, not `value-x`",
        ),
        (
            Reason::MeasureTakesNoTimeBounds {},
            Code::BadRequest,
            "per-session measures take no time bounds",
        ),
        (
            Reason::AlternateSource {},
            Code::BadRequest,
            "an alternate's source must name captured text or an exact authored revision",
        ),
        (
            Reason::AlternateRevisionChanges {},
            Code::BadRequest,
            "an alternate revision must keep its source revision, kind and languages",
        ),
        (
            Reason::BadSpan {
                problem: "problem-x".into(),
            },
            Code::BadRequest,
            "the span is refused: problem-x",
        ),
        (
            Reason::NotInSession {
                item: "item-x".into(),
                session: "session-x".into(),
            },
            Code::BadRequest,
            "item-x is not in session-x",
        ),
        (
            Reason::ReadStartOutsideSession {},
            Code::BadRequest,
            "the read starts at a message the session does not hold",
        ),
        (
            Reason::TraceTarget {},
            Code::BadRequest,
            "trace needs a message, block, tool call or text span",
        ),
        (
            Reason::NotPositive {
                field: "field-x".into(),
            },
            Code::BadValue,
            "field-x must be positive",
        ),
        (
            Reason::BadConfiguration {
                problem: "problem-x".into(),
            },
            Code::BadValue,
            "the configuration is refused: problem-x",
        ),
        (
            Reason::ImportOptions {
                problem: "problem-x".into(),
            },
            Code::BadRequest,
            "the import is refused: problem-x",
        ),
        (
            Reason::NotAFile {
                path: "path-x".into(),
            },
            Code::BadRequest,
            "path-x is not a file: name one of Python Polylogue's vector stores",
        ),
        (
            Reason::NotAVectorStore {
                path: "path-x".into(),
            },
            Code::BadRequest,
            "path-x is not one of Python Polylogue's vector stores: it has no message_embeddings tables in either layout",
        ),
        (
            Reason::PathHasNoFileName {
                path: "path-x".into(),
            },
            Code::BadRequest,
            "path-x has no file name",
        ),
        (
            Reason::ProducerQueueFormat {
                path: "path-x".into(),
            },
            Code::Refused,
            "producer queue path-x has format 1; let the earlier build send and confirm every operation, then set the empty queue aside",
        ),
        (
            Reason::ArchiveModeMismatch {
                archive: "archive-x".into(),
                recorded: "recorded-x".into(),
                requested: "requested-x".into(),
            },
            Code::Refused,
            "archive archive-x is recorded-x, but configuration requests requested-x; select its recorded mode or another archive",
        ),
        (
            Reason::ArchiveModeMissing {
                archive: "archive-x".into(),
            },
            Code::Refused,
            "archive archive-x has no recorded mode; fix its configuration before creating it",
        ),
        (
            Reason::ModeNotBuilt {
                mode: "mode-x".into(),
            },
            Code::NotBuilt,
            "mode-x mode is not built in this version of Polylogue",
        ),
        (
            Reason::BadQueryHash {
                prefix: "prefix-x".into(),
            },
            Code::BadValue,
            "a query hash takes 8 to 64 hexadecimal characters, not `prefix-x`",
        ),
        (
            Reason::SeenMarkNeedsItem {},
            Code::BadRequest,
            "a seen mark on a session names a message or block",
        ),
        (
            Reason::SeenFromAnotherGeneration {},
            Code::Conflict,
            "the seen mark belongs to another index generation; read the session again",
        ),
        (
            Reason::BadCoverage {
                problem: "problem-x".into(),
            },
            Code::BadRequest,
            "the capture coverage is refused: problem-x",
        ),
        (
            Reason::SinexRefused {
                name: "name-x".into(),
                reason: "reason-x".into(),
            },
            Code::Refused,
            "Sinex refused name-x: reason-x",
        ),
        (
            Reason::BudgetTooSmall {
                part: "part-x".into(),
            },
            Code::BadRequest,
            "the context budget cannot fit part-x; increase budget_tokens",
        ),
        (
            Reason::WithdrawalCycle {
                revision: "revision-x".into(),
            },
            Code::BadRequest,
            "withdrawals of revision-x form a cycle",
        ),
        (
            Reason::BadKeptResult {
                id: "id-x".into(),
                field: "field-x".into(),
                problem: "problem-x".into(),
            },
            Code::BadRequest,
            "kept result `id-x` field field-x: problem-x",
        ),
        (
            Reason::DaemonDidNotAnswer {},
            Code::DaemonRequired,
            "the Polylogue daemon did not answer; it is stopped or busy",
        ),
        (
            Reason::SaveOutcomeUnknown {},
            Code::DaemonRequired,
            "the note may or may not be saved; retry it with the same ID",
        ),
        (
            Reason::CaptureNotJson {},
            Code::BadRequest,
            "a capture is one JSON object",
        ),
        (
            Reason::CaptureNamesNoConversation {},
            Code::BadRequest,
            "a capture names session.provider and session.provider_session_id",
        ),
        (
            Reason::CaptureHasNoConversation {},
            Code::BadRequest,
            "a capture holds the provider's conversation JSON (raw_provider_payload) or session.turns",
        ),
        (
            Reason::CaptureUnreadable {
                problem: "problem-x".into(),
            },
            Code::BadRequest,
            "the capture cannot be read: problem-x",
        ),
        (
            Reason::AttachmentChanged {
                input: "input-x".into(),
            },
            Code::Conflict,
            "the attachment's stored bytes in input-x changed; select it again",
        ),
    ];
    let named: Vec<&str> = table.iter().map(|(reason, _, _)| reason.name()).collect();
    assert_eq!(
        named,
        Reason::NAMES,
        "every reason has one row, in declaration order"
    );
    for (reason, code, text) in table {
        assert_eq!(reason.code(), code, "{}", reason.name());
        assert_eq!(reason.to_string(), text, "{}", reason.name());
        let wire = serde_json::to_value(&reason).unwrap();
        assert_eq!(wire["name"], reason.name());
        assert_eq!(serde_json::from_value::<Reason>(wire).unwrap(), reason);
    }
}
