//! Shared independent cases through the product compiler, fixture store and pager.
use super::super::*;
use super::{fixture_clock, fixture_context, run};
use polylogue_model::{Precision, SessionId, TimeEvidence, TimeQuality, Timestamp, Usage};
use polylogue_store::fixture::{CallFacts, Fixture, MessageFacts, SessionFacts};
use query_language::cursor::{PageRequest, SealedCursor};
use query_language::{
    Error, ErrorKind,
    ast::*,
    compiler_cases,
    conformance::{Answer, Dataset, Extent, Row, Scalar},
};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

const OUTER_FIELDS: &[&str] = &["id", "input_tokens", "output_tokens", "repo", "title"];
const CHILD_FIELDS: &[&str] = &["message", "session", "exit_code", "at"];
struct Product;
struct Plan {
    text: String,
    unit: String,
    shape: Shape,
    fields: BTreeSet<String>,
    paired: bool,
}
fn error(message: impl ToString) -> Error {
    Error::new(ErrorKind::InvalidQuery, message.to_string(), 0)
}
fn options(fields: Vec<String>) -> Options {
    Options {
        clock: fixture_clock(0, "UTC", 240),
        origins: vec!["claude-code".into()],
        page_size: None,
        random_seed: Some(7),
        embeddings: false,
        self_session: None,
        saved: Default::default(),
        fields,
        kept: Default::default(),
        chat_pages: vec![],
    }
}
impl compiler_cases::Compiler for Product {
    type Plan = Plan;
    fn compile(&self, value: &Query) -> Result<Plan, Error> {
        let Source::Unit { unit, .. } = &value.source else {
            return Err(error("case fixture has no native source unit"));
        };
        let options = options(vec![]);
        let catalog = polylogue_query::catalog::catalog(&options.origins);
        let mut value = value.clone();
        project_fixture_sources(&mut value, &catalog)?;
        let text = query_language::format::format_query(&value, &catalog)?;
        let prepared = polylogue_query::compile(&text, &options).map_err(error)?;
        let bound = prepared.expanded_query();
        let typing = bound
            .typing()
            .query(&query_language::scope::NodePath::default())
            .unwrap();
        let output = typing
            .transforms
            .last()
            .map_or(&typing.source, |stage| &stage.after);
        let mut fields: BTreeSet<_> = output
            .fields
            .iter()
            .map(|info| info.field.name.clone())
            .collect();
        let program = polylogue_store::query::lower(&prepared, &options).map_err(error)?;
        fields.extend(
            program
                .columns
                .iter()
                .filter(|column| column.name.starts_with("unknown_"))
                .map(|column| column.name.clone()),
        );
        Ok(Plan {
            text,
            unit: unit.clone(),
            shape: program.shape,
            fields,
            paired: value
                .transforms
                .iter()
                .any(|stage| matches!(stage.kind, TransformKind::Join { .. })),
        })
    }
}
/// Give each native source the fixture's declared columns through the real
/// projection stage. Display options belong to the final answer and cannot
/// request generated witness/diff columns from an earlier native source.
fn project_fixture_sources(
    query: &mut Query,
    catalog: &query_language::catalog::Catalog,
) -> Result<(), Error> {
    fn condition(
        value: &mut Condition,
        catalog: &query_language::catalog::Catalog,
    ) -> Result<(), Error> {
        match &mut value.kind {
            ConditionKind::And { conditions } | ConditionKind::Or { conditions } => {
                for value in conditions {
                    condition(value, catalog)?;
                }
            }
            ConditionKind::Not { condition: value } => condition(value, catalog)?,
            ConditionKind::Exists {
                condition: Some(value),
                ..
            }
            | ConditionKind::Count {
                condition: Some(value),
                ..
            } => condition(value, catalog)?,
            ConditionKind::Sequence { steps, .. } => {
                for value in steps {
                    condition(value, catalog)?;
                }
            }
            ConditionKind::In { query, .. } | ConditionKind::Order { query, .. } => {
                project_fixture_sources(query, catalog)?;
            }
            _ => {}
        }
        Ok(())
    }
    let Source::Unit {
        unit,
        condition: source_condition,
        ..
    } = &mut query.source
    else {
        return Err(error("case fixture has no native source unit"));
    };
    let requested = if unit == "tool_calls" {
        CHILD_FIELDS
    } else {
        OUTER_FIELDS
    };
    if let Some(value) = source_condition {
        condition(value, catalog)?;
    }
    for stage in &mut query.transforms {
        match &mut stage.kind {
            TransformKind::Where { condition: value }
            | TransformKind::Scope {
                condition: Some(value),
                ..
            } => condition(value, catalog)?,
            TransformKind::Combine { query, .. }
            | TransformKind::Join { query, .. }
            | TransformKind::Diff { query } => project_fixture_sources(query, catalog)?,
            _ => {}
        }
    }
    let source_query = Query {
        source: query.source.clone(),
        transforms: vec![],
        attach: vec![],
    };
    let bound = query_language::bind::bind(&source_query, catalog)?;
    let source = &bound
        .typing()
        .query(&Default::default())
        .ok_or_else(|| error("case fixture source has no typed scope"))?
        .source;
    let mut columns = requested
        .iter()
        .map(|name| query_language::ast::Column::field(*name))
        .collect::<Vec<_>>();
    columns.extend(
        ["match.message", "match.exit_code", "match.distance"]
            .into_iter()
            .filter(|name| source.fields.iter().any(|info| info.field.name == *name))
            .map(query_language::ast::Column::field),
    );
    query.transforms.insert(
        0,
        Transform {
            span: Span::default(),
            kind: TransformKind::Fields { columns },
        },
    );
    Ok(())
}

/// Presence belongs to the fixture schema. Values below always come from the
/// real query page; clock evidence and order come back through native storage.
struct Metadata {
    fields: BTreeSet<String>,
    order: i64,
    time: Option<Extent>,
}
struct Loaded {
    fixture: Fixture,
    dataset: Dataset,
    metadata: BTreeMap<(String, String), Metadata>,
    parents: BTreeSet<String>,
    _directory: tempfile::TempDir,
}
#[derive(Default)]
struct Execution {
    loaded: Option<Loaded>,
}
fn tokens(value: Option<&Scalar>) -> Result<Option<u64>, Error> {
    match value {
        None | Some(Scalar::Unknown) => Ok(None),
        Some(Scalar::Number(value))
            if value.is_finite()
                && *value >= 0.0
                && value.fract() == 0.0
                && *value < u64::MAX as f64 =>
        {
            Ok(Some(*value as u64))
        }
        _ => Err(error(
            "fixture token counts must be representable whole nonnegative counts",
        )),
    }
}
fn exit_code(value: Option<&Scalar>) -> Result<Option<i64>, Error> {
    match value {
        None | Some(Scalar::Unknown) => Ok(None),
        Some(Scalar::Number(value))
            if value.is_finite()
                && value.fract() == 0.0
                && *value >= i64::MIN as f64
                && *value < i64::MAX as f64 =>
        {
            Ok(Some(*value as i64))
        }
        _ => Err(error(
            "fixture exit codes must be representable whole signed numbers",
        )),
    }
}
fn string(value: Option<&Scalar>) -> Option<&str> {
    match value {
        Some(Scalar::Text(value)) => Some(value),
        _ => None,
    }
}
fn scalar(value: &Value) -> Result<Scalar, Error> {
    if let Some(value) = value.as_f64() {
        Ok(Scalar::Number(value))
    } else if let Some(value) = value.as_str() {
        Ok(Scalar::Text(value.into()))
    } else if let Some(value) = value.as_bool() {
        Ok(Scalar::Bool(value))
    } else if value.is_null() {
        Ok(Scalar::Unknown)
    } else {
        Err(error("native fixture column is not a scalar"))
    }
}

fn native_id(value: &Value) -> Result<String, Error> {
    let text = value
        .as_str()
        .ok_or_else(|| error("native row has no identity"))?;
    let native = match polylogue_model::Reference::parse(text) {
        Ok(polylogue_model::Reference::Session(session)) => session.to_string(),
        Ok(polylogue_model::Reference::Message(message))
        | Ok(polylogue_model::Reference::Block { message, .. }) => message.to_string(),
        _ => text.to_owned(),
    };
    native
        .strip_prefix("claude-code:")
        .map(str::to_owned)
        .ok_or_else(|| error(format!("unexpected fixture identity {text}")))
}
/// One reference tick is one second. The native store's clock uses milliseconds.
fn timestamp(tick: i64) -> Result<Timestamp, Error> {
    tick.checked_mul(1000)
        .and_then(Timestamp::from_unix_ms)
        .ok_or_else(|| error("fixture time is outside the native timestamp range"))
}
fn time(extent: Option<Extent>) -> Result<(Option<Timestamp>, TimeEvidence), Error> {
    let Some(extent) = extent else {
        return Ok((None, TimeEvidence::UNKNOWN));
    };
    Ok((
        Some(timestamp(extent.start)?),
        TimeEvidence {
            quality: TimeQuality::Intrinsic,
            precision: Some(if extent.coarse {
                Precision::Bounded
            } else {
                Precision::Instant
            }),
            until: extent.coarse.then(|| timestamp(extent.end)).transpose()?,
        },
    ))
}
fn extent(at: Option<Timestamp>, evidence: TimeEvidence) -> Option<Extent> {
    at.map(|at| Extent {
        start: at.unix_ms() / 1000,
        end: evidence.until.unwrap_or(at).unix_ms() / 1000,
        coarse: evidence.precision != Some(Precision::Instant),
    })
}
impl Execution {
    fn load(&mut self, dataset: &Dataset) -> Result<(), Error> {
        let directory = tempfile::tempdir().map_err(error)?;
        let mut fixture = Fixture::new(&directory.path().join("archive")).map_err(error)?;
        let mut sessions = BTreeMap::new();
        let mut message_keys = BTreeMap::new();
        for row in &dataset["sessions"] {
            let name = format!("claude-code:{}", row.id);
            let (at, evidence) = time(row.time)?;
            let session = fixture
                .session(SessionFacts {
                    id: &name,
                    repo: string(row.fields.get("repo")),
                    title: string(row.fields.get("title")),
                    started: at.map(Timestamp::unix_ms),
                    last: at.map(Timestamp::unix_ms),
                    ..Default::default()
                })
                .map_err(error)?;
            fixture.session_time(session, at, evidence).map_err(error)?;
            sessions.insert(row.id.clone(), session);
            let message = fixture
                .message(
                    session,
                    MessageFacts {
                        id: &format!("claude-code:usage-{}", row.id),
                        role: "assistant",
                        written_by: "model",
                        at: at.map(Timestamp::unix_ms),
                        // Usage is a parent fact, before the abstract child rows.
                        seq: Some(-1),
                        ..Default::default()
                    },
                )
                .map_err(error)?;
            fixture.message_time(message, at, evidence).map_err(error)?;
            fixture
                .usage(
                    message,
                    &Usage {
                        input: tokens(row.fields.get("input_tokens"))?,
                        output: tokens(row.fields.get("output_tokens"))?,
                        ..Default::default()
                    },
                    None,
                )
                .map_err(error)?;
        }
        for row in dataset.get("tool_calls").into_iter().flatten() {
            let session = if let Some(parent) = &row.parent {
                sessions[parent]
            } else {
                // Tool calls still have native custody when the abstract relation
                // has no parent. Separate subagents cannot become order neighbours
                // or extra rows of the outer main-session relation.
                fixture
                    .session(SessionFacts {
                        id: &format!("claude-code:carrier-{}", row.id),
                        kind: "subagent",
                        ..Default::default()
                    })
                    .map_err(error)?
            };
            let (at, evidence) = time(row.time)?;
            let message = fixture
                .message(
                    session,
                    MessageFacts {
                        id: &format!("claude-code:{}", row.id),
                        role: "assistant",
                        written_by: "model",
                        at: at.map(Timestamp::unix_ms),
                        seq: Some(row.order),
                        ..Default::default()
                    },
                )
                .map_err(error)?;
            fixture.message_time(message, at, evidence).map_err(error)?;
            if let Some(parent) = row
                .parent
                .as_ref()
                .and_then(|parent| dataset["sessions"].iter().find(|row| &row.id == parent))
            {
                // Known zero child usage leaves an explicitly known parent total
                // exact; a wholly unknown parent total remains wholly unknown.
                fixture
                    .usage(
                        message,
                        &Usage {
                            input: tokens(parent.fields.get("input_tokens"))?.map(|_| 0),
                            output: tokens(parent.fields.get("output_tokens"))?.map(|_| 0),
                            ..Default::default()
                        },
                        None,
                    )
                    .map_err(error)?;
            }
            fixture
                .tool_call(
                    session,
                    message,
                    CallFacts {
                        tool: "Bash",
                        action: "shell",
                        command: Some(&row.id),
                        outcome: "error",
                        exit_code: exit_code(row.fields.get("exit_code"))?,
                        at: at.map(Timestamp::unix_ms),
                        ..Default::default()
                    },
                )
                .map_err(error)?;
            message_keys.insert(row.id.clone(), message);
        }
        let mut metadata = BTreeMap::new();
        {
            let reads = fixture.reads().map_err(error)?;
            let reader = reads.reader().map_err(error)?;
            for (unit, rows) in dataset {
                for row in rows {
                    let (order, recorded_time) = if unit == "sessions" {
                        let id =
                            SessionId::parse(&format!("claude-code:{}", row.id)).map_err(error)?;
                        let session = reader
                            .session(&id)
                            .map_err(error)?
                            .ok_or_else(|| error("fixture session missing"))?;
                        (0, extent(session.started, session.time_evidence()))
                    } else {
                        let parent = row
                            .parent
                            .as_ref()
                            .map(|p| format!("claude-code:{p}"))
                            .unwrap_or_else(|| format!("claude-code:carrier-{}", row.id));
                        let session = SessionId::parse(&parent).map_err(error)?;
                        let message = reader
                            .messages(&session, false)
                            .map_err(error)?
                            .into_iter()
                            .find(|message| {
                                message.id.to_string() == format!("claude-code:{}", row.id)
                            })
                            .ok_or_else(|| error("fixture message missing"))?;
                        (
                            fixture
                                .message_order(message_keys[&row.id])
                                .map_err(error)?,
                            extent(message.at, message.time_evidence()),
                        )
                    };
                    metadata.insert(
                        (unit.clone(), row.id.clone()),
                        Metadata {
                            fields: row.fields.keys().cloned().collect(),
                            order,
                            // No declared clock means no clock in this fixture's
                            // abstract projection, even if native derivations know more.
                            time: row.time.and(recorded_time),
                        },
                    );
                }
            }
        }
        self.loaded = Some(Loaded {
            fixture,
            dataset: dataset.clone(),
            metadata,
            parents: sessions.into_keys().collect(),
            _directory: directory,
        });
        Ok(())
    }
}
impl Loaded {
    fn row(&self, plan: &Plan, value: &Value) -> Result<Row, Error> {
        let source = native_id(
            &value[if plan.unit == "tool_calls" {
                "message"
            } else {
                "id"
            }],
        )?;
        let metadata = self
            .metadata
            .get(&(plan.unit.clone(), source.clone()))
            .ok_or_else(|| error(format!("query returned an uninstalled row {source}")))?;
        let mut row = Row::new(&source);
        row.fields = metadata
            .fields
            .iter()
            .map(|name| {
                let value = value
                    .get(name)
                    .ok_or_else(|| error(format!("query omitted fixture field {name}")))?;
                Ok((name.clone(), scalar(value)?))
            })
            .collect::<Result<_, Error>>()?;
        row.order = metadata.order;
        row.time = metadata.time;
        if plan.unit == "tool_calls" {
            let parent = native_id(&value["session"])?;
            row.parent = self.parents.contains(&parent).then_some(parent);
        }
        if let Some(matched) = value.get("match.message").filter(|value| !value.is_null()) {
            let matched = native_id(matched)?;
            let other = self
                .metadata
                .get(&("tool_calls".into(), matched.clone()))
                .ok_or_else(|| error("query returned an uninstalled match"))?;
            row.fields
                .insert("match.id".into(), Scalar::Text(matched.clone()));
            for field in &other.fields {
                let name = format!("match.{field}");
                let value = value
                    .get(&name)
                    .ok_or_else(|| error(format!("query omitted {name}")))?;
                row.fields.insert(name, scalar(value)?);
            }
            if plan.paired {
                row.id = format!("{source}:{matched}");
                row.matches = vec![matched];
            }
        }
        for name in ["match.distance", "change", "old_rank", "new_rank"] {
            if plan.fields.contains(name) {
                let value = value
                    .get(name)
                    .ok_or_else(|| error(format!("query omitted {name}")))?;
                // A negated order relation has no exposed match.
                if name != "match.distance" || row.fields.contains_key("match.id") {
                    row.fields.insert(name.into(), scalar(value)?);
                }
            }
        }
        Ok(row)
    }
}
impl compiler_cases::Executor<Plan> for Execution {
    fn execute(
        &mut self,
        plan: &Plan,
        fixture: &Dataset,
        request: &PageRequest,
    ) -> Result<compiler_cases::Page, Error> {
        // Each changed enumeration is installed through the native fixture.
        // The three page sizes read that same immutable installation.
        if request.cursor.is_none()
            && self
                .loaded
                .as_ref()
                .is_none_or(|loaded| loaded.dataset != *fixture)
        {
            self.load(fixture)?;
        }
        let loaded = self.loaded.as_ref().expect("first page loaded fixture");
        let reads = loaded.fixture.reads().map_err(error)?;
        let answer = run(
            &reads.reader().map_err(error)?,
            Ask {
                text: &plan.text,
                cursor: request.cursor.as_ref().map(SealedCursor::as_str),
                page_size: Some(request.page_size.get() as u64),
                why: false,
                session: None,
                context: fixture_context(0, "UTC", 240),
                embeddings: false,
                ranking_identity: None,
                near: None,
                fields: &[],
            },
        )
        .map_err(|failure| error(failure.message.clone()))?;
        let rows = answer["rows"]
            .as_array()
            .ok_or_else(|| error("query omitted rows"))?
            .iter()
            .map(|value| match plan.shape {
                Shape::Rows => loaded.row(plan, value),
                Shape::Summary | Shape::Groups => {
                    let mut row = if plan.shape == Shape::Summary {
                        Row::new("aggregate")
                    } else {
                        Row::new(format!(
                            "group:[Text({:?})]",
                            value["repo"]
                                .as_str()
                                .ok_or_else(|| error("group omitted repo"))?
                        ))
                    };
                    row.fields = plan
                        .fields
                        .iter()
                        .map(|field| {
                            let value = value.get(field).ok_or_else(|| {
                                error(format!("query omitted semantic column {field}"))
                            })?;
                            Ok((field.clone(), scalar(value)?))
                        })
                        .collect::<Result<_, Error>>()?;
                    Ok(row)
                }
                Shape::Matches => {
                    let parent = native_id(&value["parent"])?;
                    let step = native_id(&value["step"])?;
                    let n = value["match"]
                        .as_u64()
                        .ok_or_else(|| error("match omitted ordinal"))?;
                    let position = value["position"]
                        .as_u64()
                        .ok_or_else(|| error("match omitted position"))?;
                    let mut row = Row::new(format!("{parent}:match:{n}:{position}"))
                        .with("parent", parent.as_str())
                        .with("match", n as f64)
                        .with("position", position as f64)
                        .with("step", step.as_str());
                    if let Some(number) = value.get("step.exit_code") {
                        row = row.with("step.exit_code", scalar(number)?);
                    }
                    let metadata = loaded
                        .metadata
                        .get(&(plan.unit.clone(), parent.clone()))
                        .ok_or_else(|| error("match returned an uninstalled parent"))?;
                    for field in &metadata.fields {
                        let name = format!("parent.{field}");
                        let value = value
                            .get(&name)
                            .ok_or_else(|| error(format!("match omitted {name}")))?;
                        row = row.with(name, scalar(value)?);
                    }
                    row.parent = Some(parent);
                    row.matches = vec![step];
                    Ok(row)
                }
            })
            .collect::<Result<_, Error>>()?;
        Ok(compiler_cases::Page {
            answer: Answer {
                rows,
                total: answer["count"]
                    .as_u64()
                    .ok_or_else(|| error("query omitted total"))? as usize,
            },
            next: answer["next"].as_str().map(SealedCursor::from_wire),
        })
    }
}
#[test]
fn shared_language_laws_run_through_product_compilation_and_keyset_pages() {
    let cases = compiler_cases::cases(&compiler_cases::Vocabulary {
        outer: "sessions".into(),
        children: "tool_calls".into(),
        number: "input_tokens".into(),
        child_number: "exit_code".into(),
        category: "repo".into(),
        text: "title".into(),
        end: "output_tokens".into(),
    });
    let failures = compiler_cases::check(&Product, &mut Execution::default(), &cases);
    assert!(failures.is_empty(), "{failures:#?}");
}
