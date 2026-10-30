//! Synthetic definitions from frozen formats, never current catalog entries.
use crate::{Result, StoreError};
use polylogue_model::query_objects::QueryObject;
use std::collections::BTreeMap;

/// The bare Notes unit declared by Polylogue16708, using its named-object
/// shape and Commonddfe normalization rules. This is synthetic stored data,
/// not an observed request, receipt or original native execution.
pub fn synthetic_source3_notes() -> Result<QueryObject> {
    let normal_form = serde_json::from_str(
        r#"{"source":{"kind":"unit","unit":"notes","condition":null},"attach":[],"stages":[]}"#,
    )?;
    let identity_ast = serde_json::from_str(
        r#"{"source":{"kind":"unit","unit":"notes","condition":null,"implicit":false},"attach":[],"stages":[]}"#,
    )?;
    let parameters = BTreeMap::new();
    let id = query_language::identity::question_id_from_normal_form(&normal_form, &parameters)
        .map_err(|error| StoreError::Invalid(error.to_string()))?;
    Ok(QueryObject {
        id,
        normal_form,
        parameters,
        identity_ast,
        source_outputs: Vec::new(),
        definitions: Vec::new(),
        dependencies: Vec::new(),
        edges: Vec::new(),
    })
}

/// Synthetic old Notes execution, not an observed receipt or a current query.
/// Its literal question comes from the frozen predecessor definition above.
pub fn synthetic_source3_run() -> crate::Result<polylogue_model::query_objects::HistoricalQueryRun>
{
    use polylogue_model::{
        Author, ChangePosition, RunId, Timestamp,
        authorship::{Authorship, Channel},
        query_objects::*,
    };
    use query_language::{
        cursor::CursorContext,
        time::{Clock, DaySetting},
    };
    let object = synthetic_source3_notes()?;
    let clock = Clock::new(
        jiff::Timestamp::from_millisecond(1).unwrap(),
        DaySetting::new(jiff::tz::TimeZone::UTC, jiff::civil::time(4, 0, 0, 0)).unwrap(),
    );
    let position = ChangePosition {
        generation: "synthetic-old-index".into(),
        index: 2,
        archive: 1,
    };
    let run = QueryRun {
        id: RunId::parse("00000000-0000-4000-8000-000000000003").unwrap(),
        question: Some(object.id),
        at: Timestamp::from_unix_ms(1).unwrap(),
        read_at: position.clone(),
        machine: "synthetic-original-machine".into(),
        authorship: Authorship {
            author: Author::Operator,
            via: Channel::Cli,
        },
        binding: RunBinding::Original {
            execution: Box::new(ExecutionBinding {
                query_options: QueryExecutionOptions {
                    direct_reference: None,
                    fields: vec![],
                    self_session: None,
                    embeddings: false,
                    why: false,
                    ranking_identity: None,
                },
                presentation: ExecutionPresentation::Query,
                accepted_request: Some(
                    serde_json::json!({"operation":"query","arguments":{"query":"notes"},"request_id":"synthetic-old-run","deadline":null}),
                ),
                replay: ReplayBinding {
                    bound_ast: object.identity_ast,
                    source_outputs: object.source_outputs,
                    definitions: vec![],
                    context: CursorContext {
                        question: object.id,
                        clock: clock.wire(),
                        resolved_times: vec![],
                        random_seed: 7,
                        read_positions: std::collections::BTreeMap::from([
                            ("generation".into(), position.generation),
                            ("index".into(), position.index.to_string()),
                            ("archive".into(), position.archive.to_string()),
                        ]),
                        versions: Default::default(),
                    },
                    scope_fingerprint: None,
                    original_text: "notes".into(),
                },
            }),
        },
    };
    crate::query_objects::conserve_historical_run(
        polylogue_model::historical::SourceSchema::PolylogueArchive { version: 4 },
        &serde_json::to_vec(&run)?,
        None,
        None,
        Default::default(),
    )
}
