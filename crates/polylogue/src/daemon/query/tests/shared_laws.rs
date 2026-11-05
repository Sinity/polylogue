//! Shared independent cases through the product compiler, fixture store and pager.
use super::super::*;
use super::{fixture_clock, fixture_context, run};
use polylogue_store::fixture::{CallFacts, Fixture, MessageFacts, SessionFacts};
use query_language::cursor::{PageRequest, SealedCursor};
use query_language::{
    Error, ErrorKind,
    ast::*,
    laws,
    semantics::{Answer, Dataset, Row, Scalar},
};
use serde_json::Value;
use std::collections::BTreeMap;

struct Product;
struct Plan {
    text: String,
    shape: Shape,
    fields: std::collections::BTreeSet<String>,
}
fn error(message: impl ToString) -> Error {
    Error::new(ErrorKind::InvalidQuery, message.to_string(), 0)
}
fn options() -> Options {
    Options {
        clock: fixture_clock(0, "UTC", 240),
        origins: vec!["claude-code".into()],
        page_size: None,
        random_seed: Some(7),
        embeddings: false,
        self_session: None,
        saved: Default::default(),
        fields: vec!["id".into(), "cost".into(), "repo".into()],
        kept: Default::default(),
        chat_pages: vec![],
    }
}
fn operand(value: &mut Expression, child: bool) {
    match value {
        Expression::Field { name } if child && name == "cost" => *name = "exit_code".into(),
        Expression::Arithmetic { left, right, .. } => {
            operand(left, child);
            operand(right, child);
        }
        Expression::Function { arguments, .. } => {
            for argument in arguments {
                operand(argument, child);
            }
        }
        _ => {}
    }
}
fn condition(value: &mut Condition, child: bool) {
    match &mut value.kind {
        ConditionKind::And { conditions } | ConditionKind::Or { conditions } => {
            for value in conditions {
                condition(value, child);
            }
        }
        ConditionKind::Not { condition: value } => condition(value, child),
        ConditionKind::Compare { left, right, .. } => {
            operand(left, child);
            operand(right, child);
        }
        ConditionKind::Between { left: value, .. } | ConditionKind::In { left: value, .. } => {
            operand(value, child)
        }
        ConditionKind::Exists {
            condition: Some(value),
            ..
        }
        | ConditionKind::Count {
            condition: Some(value),
            ..
        } => condition(value, true),
        ConditionKind::Sequence { steps, .. } => {
            for value in steps {
                condition(value, true);
            }
        }
        _ => {}
    }
    if let ConditionKind::In { query: value, .. } | ConditionKind::Order { query: value, .. } =
        &mut value.kind
    {
        query(value);
    }
}
fn query(value: &mut Query) {
    if let Source::Unit {
        unit,
        condition: Some(value),
        ..
    } = &mut value.source
    {
        condition(value, unit == "tool_calls");
    }
    for stage in &mut value.transforms {
        match &mut stage.kind {
            TransformKind::Where { condition: value } => condition(value, false),
            TransformKind::Combine { query: value, .. }
            | TransformKind::Join { query: value, .. }
            | TransformKind::Diff { query: value } => query(value),
            _ => {}
        }
    }
}
impl laws::Compiler for Product {
    type Plan = Plan;
    fn compile(&self, value: &Query) -> Result<Plan, Error> {
        let mut value = value.clone();
        query(&mut value);
        let options = options();
        let catalog = polylogue_query::catalog::catalog(&options.origins);
        let text = query_language::format::format_query(&value, &catalog)?;
        let prepared = polylogue_query::compile(&text, &options).map_err(error)?;
        let program = polylogue_store::query::lower(&prepared, &options).map_err(error)?;
        // The producer binds Matches against its actual projection declarations.
        let bound = prepared.expanded_query();
        let typing = bound
            .typing()
            .query(&query_language::scope::NodePath::default())
            .unwrap();
        let output = typing
            .transforms
            .last()
            .map_or(&typing.source, |stage| &stage.after);
        let mut fields: std::collections::BTreeSet<_> = output
            .fields
            .iter()
            .map(|info| info.field.name.clone())
            .collect();
        // Unknown-value counts are part of the shared aggregate answer. Native
        // partial-cost evidence is checked separately from the shared fields.
        fields.extend(
            program
                .columns
                .iter()
                .filter(|column| column.name.starts_with("unknown_"))
                .map(|column| column.name.clone()),
        );
        Ok(Plan {
            text,
            shape: program.shape,
            fields,
        })
    }
}
struct Loaded {
    fixture: Fixture,
    _directory: tempfile::TempDir,
}
#[derive(Default)]
struct Execution {
    loaded: Option<Loaded>,
}
fn number(value: Option<&Scalar>) -> Option<f64> {
    match value {
        Some(Scalar::Number(value)) => Some(*value),
        _ => None,
    }
}
fn string(value: Option<&Scalar>) -> Option<&str> {
    match value {
        Some(Scalar::Text(value)) => Some(value),
        _ => None,
    }
}
fn scalar(value: &Value) -> Scalar {
    if let Some(value) = value.as_f64() {
        Scalar::Number(value)
    } else if let Some(value) = value.as_str() {
        Scalar::Text(value.into())
    } else if let Some(value) = value.as_bool() {
        Scalar::Bool(value)
    } else {
        Scalar::Unknown
    }
}
fn id(value: &Value) -> String {
    value
        .as_str()
        .unwrap()
        .strip_prefix("claude-code:")
        .unwrap()
        .into()
}
impl Execution {
    fn load(&mut self, dataset: &Dataset) -> Result<(), Error> {
        let directory = tempfile::tempdir().map_err(error)?;
        let mut fixture = Fixture::new(&directory.path().join("archive")).map_err(error)?;
        let mut sessions = BTreeMap::new();
        let mut ordered = dataset["sessions"].iter().collect::<Vec<_>>();
        ordered.sort_by(|a, b| a.order.cmp(&b.order).then(a.id.cmp(&b.id)));
        let positions: BTreeMap<_, _> = ordered
            .iter()
            .enumerate()
            .map(|(rank, row)| (row.id.as_str(), -(i64::try_from(rank).unwrap())))
            .collect();
        for row in &dataset["sessions"] {
            let name = format!("claude-code:{}", row.id);
            let session = fixture
                .session(SessionFacts {
                    id: &name,
                    repo: string(row.fields.get("repo")),
                    title: string(row.fields.get("title")),
                    // Map the oracle's source order into the actual native
                    // descending activity order before any page is selected.
                    last: Some(positions[row.id.as_str()]),
                    ..Default::default()
                })
                .map_err(error)?;
            sessions.insert(row.id.clone(), session);
            fixture
                .message(
                    session,
                    MessageFacts {
                        id: &format!("claude-code:cost-{}", row.id),
                        role: "assistant",
                        written_by: "model",
                        cost_nusd: number(row.fields.get("cost")).map(|n| (n * 1e9) as i64),
                        seq: Some(0),
                        ..Default::default()
                    },
                )
                .map_err(error)?;
        }
        for row in &dataset["tool_calls"] {
            let parent = row.parent.as_ref().expect("shared child parent");
            let session = sessions[parent];
            let at = row.time.map(|time| time.start);
            let message = fixture
                .message(
                    session,
                    MessageFacts {
                        id: &format!("claude-code:{}", row.id),
                        role: "assistant",
                        written_by: "model",
                        at,
                        seq: Some(row.order + 1),
                        ..Default::default()
                    },
                )
                .map_err(error)?;
            fixture
                .tool_call(
                    session,
                    message,
                    CallFacts {
                        tool: "Bash",
                        action: "shell",
                        command: Some(&row.id),
                        outcome: "error",
                        exit_code: number(row.fields.get("cost")).map(|n| n as i64),
                        at,
                        ..Default::default()
                    },
                )
                .map_err(error)?;
        }
        self.loaded = Some(Loaded {
            fixture,
            _directory: directory,
        });
        Ok(())
    }
}
impl laws::Executor<Plan> for Execution {
    fn execute(
        &mut self,
        plan: &Plan,
        fixture: &Dataset,
        request: &PageRequest,
    ) -> Result<laws::Page, Error> {
        if request.cursor.is_none() {
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
                fields: &["id".into(), "cost".into(), "repo".into()],
            },
        )
        .map_err(|failure| error(failure.message.clone()))?;
        let rows = answer["rows"]
            .as_array()
            .expect("query rows")
            .iter()
            .map(|value| match plan.shape {
                Shape::Rows => Row::new(id(&value["id"]))
                    .with("cost", scalar(&value["cost"]))
                    .with("repo", scalar(&value["repo"])),
                Shape::Summary | Shape::Groups => {
                    let mut row = if plan.shape == Shape::Summary {
                        Row::new("aggregate")
                    } else {
                        Row::new(format!(
                            "group:[Text({:?})]",
                            value["repo"].as_str().unwrap()
                        ))
                    };
                    if let Some(partial) = value.get("partial_cost") {
                        assert!(
                            partial.as_u64().is_some(),
                            "native partial-cost evidence remains present and typed"
                        );
                    }
                    row.fields = plan
                        .fields
                        .iter()
                        .map(|field| {
                            let value = value
                                .get(field)
                                .expect("every declared semantic field is projected");
                            (field.clone(), scalar(value))
                        })
                        .collect();
                    row
                }
                Shape::Matches => {
                    let reference = |field: &str| {
                        let parsed =
                            polylogue_model::Reference::parse(value[field].as_str().unwrap())
                                .unwrap();
                        match parsed {
                            polylogue_model::Reference::Session(session) => session
                                .to_string()
                                .strip_prefix("claude-code:")
                                .unwrap()
                                .to_owned(),
                            polylogue_model::Reference::Block { message, .. } => message
                                .to_string()
                                .strip_prefix("claude-code:")
                                .unwrap()
                                .to_owned(),
                            other => panic!("unexpected match reference: {other}"),
                        }
                    };
                    let parent = reference("parent");
                    let step = reference("step");
                    let n = value["match"].as_u64().unwrap();
                    let position = value["position"].as_u64().unwrap();
                    let mut row = Row::new(format!("{parent}:match:{n}:{position}"))
                        .with("parent", parent.as_str())
                        .with("match", n as f64)
                        .with("position", position as f64)
                        .with("step", step.as_str());
                    if let Some(number) = value.get("step.exit_code") {
                        row = row.with("step.cost", scalar(number));
                    }
                    row.parent = Some(parent);
                    row.matches = vec![step];
                    row
                }
            })
            .collect();
        Ok(laws::Page {
            answer: Answer {
                rows,
                total: answer["count"].as_u64().unwrap() as usize,
            },
            next: answer["next"].as_str().map(SealedCursor::from_wire),
        })
    }
}

#[test]
fn shared_language_laws_run_through_product_compilation_and_keyset_pages() {
    let cases = laws::cases(&laws::Vocabulary {
        outer: "sessions".into(),
        children: "tool_calls".into(),
        number: "cost".into(),
        category: "repo".into(),
        text: "title".into(),
        end: "last_active".into(),
    });
    let failures = laws::check(&Product, &mut Execution::default(), &cases);
    assert!(failures.is_empty(), "{failures:#?}");
}
