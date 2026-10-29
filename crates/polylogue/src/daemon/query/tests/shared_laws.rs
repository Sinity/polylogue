//! Shared independent cases through the product compiler, fixture store and pager.
use super::super::*;
use polylogue_store::fixture::{Fixture, SessionFacts, MessageFacts, CallFacts};
use query_language::{ast::*, laws, semantics::{Answer, Dataset, Row, Scalar}, Error, ErrorKind};
use query_language::cursor::{PageRequest, SealedCursor};
use std::collections::BTreeMap;

struct Product;
struct Plan { text: String, shape: Shape }
fn error(message: impl ToString) -> Error { Error::new(ErrorKind::InvalidQuery, message.to_string(), 0) }
fn options() -> Options {
    Options { clock: polylogue_query::time::clock(0, LogicalDay::named("UTC", 240).unwrap()).unwrap(), origins: vec!["claude-code".into()], command_words: vec![], page_size: None, random_seed: Some(7), embeddings: false, self_session: None, saved: Default::default(), fields: vec!["id".into(), "cost".into(), "repo".into()], kept: Default::default(), chat_pages: vec![] }
}
fn operand(value: &mut Operand, child: bool) {
    match value {
        Operand::Field { name } if child && name == "cost" => *name = "exit_code".into(),
        Operand::Function { arguments, .. } => for argument in arguments { operand(argument, child); },
        _ => {}
    }
}
fn condition(value: &mut Condition, child: bool) {
    match &mut value.kind {
        ConditionKind::And { conditions } | ConditionKind::Or { conditions } => for value in conditions { condition(value, child); },
        ConditionKind::Not { condition: value } => condition(value, child),
        ConditionKind::Compare { operand: value, .. } | ConditionKind::Between { operand: value, .. } | ConditionKind::In { operand: value, .. } => operand(value, child),
        ConditionKind::Exists { condition: Some(value), .. } | ConditionKind::Count { condition: Some(value), .. } => condition(value, true),
        ConditionKind::Sequence { steps, .. } => for value in steps { condition(value, true); },
        _ => {}
    }
    if let ConditionKind::In { query: value, .. } = &mut value.kind { query(value); }
}
fn query(value: &mut Query) {
    if let Source::Unit { unit, condition: Some(value) } = &mut value.source { condition(value, unit == "tool_calls"); }
    for stage in &mut value.stages {
        match &mut stage.kind {
            StageKind::Where { condition: value } => condition(value, false),
            StageKind::Combine { query: value, .. } => query(value),
            _ => {}
        }
    }
}
impl laws::Compiler for Product {
    type Plan = Plan;
    fn compile(&self, value: &Query) -> Result<Plan, Error> {
        let mut value = value.clone(); query(&mut value);
        let options = options();
        let catalog = polylogue_query::catalog::catalog(&options.origins, &options.command_words);
        let text = query_language::format::format_query(&value, &catalog)?;
        let program = polylogue_query::compile(&text, &options).map_err(error)?;
        Ok(Plan { text, shape: program.shape })
    }
}
struct Loaded { fixture: Fixture, _directory: tempfile::TempDir }
#[derive(Default)]
struct Execution { loaded: Option<Loaded> }
fn number(value: Option<&Scalar>) -> Option<f64> { match value { Some(Scalar::Number(value)) => Some(*value), _ => None } }
fn string(value: Option<&Scalar>) -> Option<&str> { match value { Some(Scalar::Text(value)) => Some(value), _ => None } }
fn scalar(value: &Value) -> Scalar {
    if let Some(value) = value.as_f64() { Scalar::Number(value) }
    else if let Some(value) = value.as_str() { Scalar::Text(value.into()) }
    else if let Some(value) = value.as_bool() { Scalar::Bool(value) }
    else { Scalar::Unknown }
}
fn id(value: &Value) -> String { value.as_str().unwrap().strip_prefix("claude-code:").unwrap().into() }
impl Execution {
    fn load(&mut self, dataset: &Dataset) -> Result<(), Error> {
        let directory = tempfile::tempdir().map_err(error)?;
        let mut fixture = Fixture::new(&directory.path().join("archive")).map_err(error)?;
        let mut sessions = BTreeMap::new();
        for row in &dataset["sessions"] {
            let name = format!("claude-code:{}", row.id);
            let session = fixture.session(SessionFacts { id: &name, repo: string(row.fields.get("repo")), ..Default::default() }).map_err(error)?;
            sessions.insert(row.id.clone(), session);
            fixture.message(session, MessageFacts { id: &format!("claude-code:cost-{}", row.id), role: "assistant", written_by: "model", cost_nusd: number(row.fields.get("cost")).map(|n| (n * 1e9) as i64), seq: Some(-1), ..Default::default() }).map_err(error)?;
        }
        for row in &dataset["tool_calls"] {
            let parent = row.parent.as_ref().expect("shared child parent");
            let session = sessions[parent];
            let at = row.time.map(|time| time.start);
            let message = fixture.message(session, MessageFacts { id: &format!("claude-code:{}", row.id), role: "assistant", written_by: "model", at, seq: Some(row.order), ..Default::default() }).map_err(error)?;
            fixture.tool_call(session, message, CallFacts { tool: "Bash", action: "shell", command: Some(&row.id), outcome: "error", exit_code: number(row.fields.get("cost")).map(|n| n as i64), at, ..Default::default() }).map_err(error)?;
        }
        self.loaded = Some(Loaded { fixture, _directory: directory });
        Ok(())
    }
}
impl laws::Executor<Plan> for Execution {
    fn execute(&mut self, plan: &Plan, fixture: &Dataset, request: &PageRequest) -> Result<laws::Page, Error> {
        if request.cursor.is_none() { self.load(fixture)?; }
        let loaded = self.loaded.as_ref().expect("first page loaded fixture");
        let reads = loaded.fixture.reads().map_err(error)?;
        let answer = run(&reads.reader().map_err(error)?, Ask { text: &plan.text, cursor: request.cursor.as_ref().map(SealedCursor::as_str), page_size: Some(request.page_size.get() as u64), why: false, session: None, now: 0, day: Some(LogicalDay::named("UTC", 240).unwrap()), embeddings: false, near: None, fields: &["id".into(), "cost".into(), "repo".into()] }).map_err(|failure| error(failure.message))?;
        let rows = answer["rows"].as_array().expect("query rows").iter().map(|value| {
            match plan.shape {
                Shape::Rows => Row::new(id(&value["id"])).with("cost", scalar(&value["cost"])).with("repo", scalar(&value["repo"])),
                Shape::Summary | Shape::Groups => {
                    let mut row = if plan.shape == Shape::Summary { Row::new("aggregate") } else { Row::new(format!("group:[Text({:?})]", value["repo"].as_str().unwrap())) };
                    row.fields = value.as_object().unwrap().iter().filter(|(key, _)| !key.starts_with('_')).map(|(key, value)| (key.clone(), scalar(value))).collect(); row
                }
                Shape::Matches => {
                    let parent = id(&value["session"]); let n = value["match"].as_u64().unwrap();
                    let first = id(&value["step1_message"]); let second = id(&value["step2_message"]);
                    let mut row = Row::new(format!("{parent}:match:{n}")).with("step1_id", first.as_str()).with("step2_id", second.as_str());
                    row.parent = Some(parent); row.matches = vec![first, second]; row
                }
            }
        }).collect();
        Ok(laws::Page { answer: Answer { rows, total: answer["count"].as_u64().unwrap() as usize }, next: answer["next"].as_str().map(SealedCursor::from_wire) })
    }
}

#[test]
fn shared_language_laws_run_through_product_compilation_and_keyset_pages() {
    let cases = laws::cases(&laws::Vocabulary { outer: "sessions".into(), children: "tool_calls".into(), number: "cost".into(), category: "repo".into() });
    let built: Vec<_> = cases.iter().filter(|case| case.law != "join-match-order-independent").cloned().collect();
    assert_eq!(built.len() + 1, cases.len());
    let failures = laws::check(&Product, &mut Execution::default(), &built);
    assert!(failures.is_empty(), "{failures:#?}");
    let future = cases.iter().find(|case| case.law == "join-match-order-independent").unwrap();
    let text = query_language::format::format_query(&future.query, &polylogue_query::catalog::catalog(&[], &[])).unwrap();
    assert_eq!(polylogue_query::compile(&text, &options()).unwrap_err().kind(), "not_built");
}
