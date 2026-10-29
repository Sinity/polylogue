//! Canonical object admission and replay use the product's native declarations.
use super::*;
use polylogue_model::query_objects::{FrozenQueryDefinition, QueryEdge, QueryObject};
use query_language::{catalog::OutputSchema, identity};
use std::collections::{BTreeSet, VecDeque};

fn invalid(message: impl Into<String>) -> QueryError {
    QueryError::Invalid {
        message: message.into(),
        position: 0,
    }
}
fn language(error: query_language::Error) -> QueryError {
    QueryError::language(error, "")
}
fn native_catalog() -> Catalog {
    crate::catalog::catalog(&[], &[])
}

/// A supported native declaration version, independent of transport vocabulary.
/// This belongs in execution bindings, never in semantic question identity.
pub fn catalog_fingerprint() -> String {
    // Catalog has no arbitrary JSON maps; derived struct serialization has stable
    // field order even when serde_json's preserve_order feature is enabled.
    let bytes = serde_json::to_vec(&native_catalog()).expect("native catalog is serializable");
    blake3::hash(&bytes).to_hex().to_string()
}
fn catalog(outputs: &[SourceOutput]) -> Catalog {
    let mut catalog = native_catalog();
    catalog.source_outputs = outputs.to_vec();
    catalog
}
fn parse_id(name: &str) -> Result<QuestionId, QueryError> {
    if name.len() != 64
        || !name
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
    {
        return Err(invalid(format!(
            "saved source {name:?} has no resolved question identity"
        )));
    }
    let mut bytes = [0; 32];
    for (at, byte) in bytes.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&name[at * 2..at * 2 + 2], 16)
            .map_err(|_| invalid("invalid question identity"))?;
    }
    Ok(QuestionId(bytes))
}
fn sources(query: &Query) -> Result<Vec<OutputSource>, QueryError> {
    let mut sources = Vec::new();
    let mut query = query.clone();
    walk(
        &mut query,
        &mut |query| {
            match &query.source {
                Source::Saved {
                    name, arguments, ..
                } => sources.push(OutputSource::Saved {
                    name: name.clone(),
                    arguments: arguments.clone(),
                }),
                Source::Result {
                    reference, part, ..
                } => sources.push(OutputSource::Result {
                    reference: reference.clone(),
                    part: *part,
                }),
                _ => {}
            }
            Ok(())
        },
        &mut |_| Ok(()),
    )?;
    Ok(sources)
}
fn used_outputs(query: &Query, source: &Catalog) -> Result<Vec<SourceOutput>, QueryError> {
    let mut outputs: Vec<SourceOutput> = Vec::new();
    for key in sources(query)? {
        let matches = source
            .source_outputs
            .iter()
            .filter(|output| output.source == key)
            .collect::<Vec<_>>();
        let [output] = matches.as_slice() else {
            return Err(invalid(
                "canonical source needs exactly one frozen output schema",
            ));
        };
        if !outputs.contains(output) {
            outputs.push((*output).clone());
        }
    }
    Ok(outputs)
}
fn edge_key(edge: &QueryEdge) -> String {
    match edge {
        QueryEdge::Saved { query } => format!("saved:{query}"),
        QueryEdge::Result { result } => format!("result:{result}"),
        QueryEdge::Combine { query } => format!("combine:{query}"),
        QueryEdge::In { query } => format!("in:{query}"),
    }
}
fn edges(query: &Query, native: &Catalog) -> Result<Vec<QueryEdge>, QueryError> {
    let output = std::cell::RefCell::new(Vec::new());
    let mut query = query.clone();
    walk(
        &mut query,
        &mut |query| {
            match &query.source {
                Source::Saved { name, .. } => output.borrow_mut().push(QueryEdge::Saved {
                    query: parse_id(name)?,
                }),
                Source::Result { reference, .. } => output.borrow_mut().push(QueryEdge::Result {
                    result: reference.clone(),
                }),
                _ => {}
            }
            for stage in &query.stages {
                if let StageKind::Combine { query, .. } | StageKind::Temporal { query, .. } =
                    &stage.kind
                {
                    output.borrow_mut().push(QueryEdge::Combine {
                        query: identity::question_id(query, native, &BTreeMap::new())
                            .map_err(language)?,
                    });
                }
            }
            Ok(())
        },
        &mut |condition| {
            match &condition.kind {
                ConditionKind::Saved { name, .. } => output.borrow_mut().push(QueryEdge::Saved {
                    query: parse_id(name)?,
                }),
                ConditionKind::In { query, .. } => output.borrow_mut().push(QueryEdge::In {
                    query: identity::question_id(query, native, &BTreeMap::new())
                        .map_err(language)?,
                }),
                _ => {}
            }
            Ok(())
        },
    )?;
    let mut output = output.into_inner();
    output.sort_by_key(edge_key);
    output.dedup();
    Ok(output)
}
fn dependencies(edges: &[QueryEdge]) -> Vec<QuestionId> {
    let mut ids = BTreeMap::new();
    for edge in edges {
        if let QueryEdge::Saved { query } | QueryEdge::Combine { query } | QueryEdge::In { query } =
            edge
        {
            ids.insert(query.to_string(), *query);
        }
    }
    ids.into_values().collect()
}
pub(super) fn from_plan(plan: &Plan) -> Result<QueryObject, QueryError> {
    let outputs = used_outputs(&plan.identity, &plan.identity_catalog)?;
    let native = catalog(&outputs);
    let mut frozen = plan.frozen.clone();
    // Inline composition bodies have canonical custody just like named inputs.
    let mut query = plan.identity.clone();
    let mut first = true;
    walk(
        &mut query,
        &mut |query| {
            if first {
                first = false;
                return Ok(());
            }
            let question = identity::question_id(query, &plan.identity_catalog, &BTreeMap::new())
                .map_err(language)?;
            if !frozen.iter().any(|node| node.question == question) {
                frozen.push(FrozenQueryDefinition {
                    question,
                    normal_form: identity::normal_form(query, &plan.identity_catalog)
                        .map_err(language)?,
                    parameters: BTreeMap::new(),
                    identity_ast: query.clone(),
                    source_outputs: used_outputs(query, &plan.identity_catalog)?,
                });
            }
            Ok(())
        },
        &mut |_| Ok(()),
    )?;
    frozen.sort_by_key(|node| node.question.to_string());
    frozen.dedup_by_key(|node| node.question);
    let edges = edges(&plan.identity, &native)?;
    let object = QueryObject {
        id: plan.question_id,
        normal_form: identity::normal_form(&plan.identity, &native).map_err(language)?,
        parameters: BTreeMap::new(),
        identity_ast: plan.identity.clone(),
        source_outputs: outputs,
        definitions: frozen,
        dependencies: dependencies(&edges),
        edges,
    };
    validate_object(&object)?;
    Ok(object)
}
fn all_outputs(object: &QueryObject) -> Result<Vec<SourceOutput>, QueryError> {
    let mut outputs = object.source_outputs.clone();
    for node in &object.definitions {
        for output in &node.source_outputs {
            if let Some(previous) = outputs.iter().find(|old| old.source == output.source) {
                if previous.schema != output.schema {
                    return Err(invalid("conflicting frozen output schemas"));
                }
            } else {
                outputs.push(output.clone());
            }
        }
    }
    Ok(outputs)
}
fn replay_bindings(object: &QueryObject, native: &Catalog) -> Result<Bindings, QueryError> {
    let mut bindings = Bindings::default();
    for node in &object.definitions {
        let text =
            query_language::format::format_query(&node.identity_ast, native).map_err(language)?;
        let id = node.question.to_string();
        bindings.saved.insert(
            id.clone(),
            vec![crate::SavedQuery {
                text,
                revision: id.clone(),
                first_note: id,
            }],
        );
    }
    Ok(bindings)
}
fn expand(query: &Query, native: &Catalog, bindings: &Bindings) -> Result<Query, QueryError> {
    let text = query_language::format::format_query(query, native).map_err(language)?;
    let (query, _) = crate::compile::source::expand(&text, bindings)?;
    Ok(query_language::bind::bind(&query, native)
        .map_err(language)?
        .into_query())
}
fn check_node(
    id: QuestionId,
    form: &serde_json::Value,
    parameters: &BTreeMap<String, String>,
    query: &Query,
    outputs: &[SourceOutput],
) -> Result<(), QueryError> {
    let native = catalog(outputs);
    let actual = identity::question_id_from_normal_form(form, parameters).map_err(language)?;
    if actual != id || identity::normal_form(query, &native).map_err(language)? != *form {
        return Err(invalid(
            "question identity, normal form and typed syntax disagree",
        ));
    }
    let bound = query_language::bind::bind(query, &native).map_err(language)?;
    if identity::normal_form(bound.query(), &native).map_err(language)? != *form {
        return Err(invalid(
            "canonical question changes when bound to native declarations",
        ));
    }
    if used_outputs(query, &native)? != outputs {
        return Err(invalid(
            "question contains unused or unordered output schemas",
        ));
    }
    Ok(())
}
/// Validate current executable semantics using owned native declarations and
/// the complete immutable definition closure. This function performs no reads.
pub fn validate_object(object: &QueryObject) -> Result<(), QueryError> {
    check_node(
        object.id,
        &object.normal_form,
        &object.parameters,
        &object.identity_ast,
        &object.source_outputs,
    )?;
    let native = catalog(&all_outputs(object)?);
    let actual_edges = edges(&object.identity_ast, &native)?;
    if actual_edges != object.edges || dependencies(&actual_edges) != object.dependencies {
        return Err(invalid(
            "question composition edges do not match its syntax",
        ));
    }
    let mut by_id = BTreeMap::new();
    for node in &object.definitions {
        check_node(
            node.question,
            &node.normal_form,
            &node.parameters,
            &node.identity_ast,
            &node.source_outputs,
        )?;
        if by_id.insert(node.question.to_string(), node).is_some() {
            return Err(invalid("duplicate frozen question body"));
        }
    }
    let mut seen = BTreeSet::new();
    let mut queue = VecDeque::from(object.dependencies.clone());
    while let Some(id) = queue.pop_front() {
        let key = id.to_string();
        if !seen.insert(key.clone()) {
            continue;
        }
        let node = by_id
            .get(&key)
            .ok_or_else(|| invalid(format!("missing frozen question {id}")))?;
        queue.extend(dependencies(&edges(&node.identity_ast, &native)?));
    }
    if seen.len() != by_id.len() {
        return Err(invalid("question contains an unrelated frozen definition"));
    }
    let bindings = replay_bindings(object, &native)?;
    // A supplied schema cannot promote a field that its frozen definition does
    // not actually produce. Kept outputs remain identified immutable inputs.
    for output in &native.source_outputs {
        if let OutputSource::Saved { name, arguments } = &output.source {
            let source = Query {
                source: Source::Saved {
                    name: name.clone(),
                    arguments: arguments.clone(),
                    condition: None,
                    output: OutputBinding::Known,
                },
                attach: vec![],
                stages: vec![],
            };
            let expanded = expand(&source, &native, &bindings)?;
            let actual: OutputSchema = crate::compile::schema::infer(&expanded, &native)?;
            if actual != output.schema {
                return Err(invalid(
                    "saved output schema differs from its frozen definition",
                ));
            }
        }
    }
    let _ = expand(&object.identity_ast, &native, &bindings)?;
    Ok(())
}
/// Reconstruct a pure plan without resolving current aliases or reading data.
/// The caller obtains the expected version from its authenticated run context.
pub fn replay(object: &QueryObject, expected_catalog: &str) -> Result<Plan, QueryError> {
    if expected_catalog != catalog_fingerprint() {
        return Err(QueryError::NotBuilt {
            what: "the captured query catalog version".into(),
        });
    }
    validate_object(object)?;
    let native = catalog(&all_outputs(object)?);
    let original = query_language::bind::bind(&object.identity_ast, &native).map_err(language)?;
    let expanded = expand(
        original.query(),
        &native,
        &replay_bindings(object, &native)?,
    )?;
    let expanded = query_language::bind::bind(&expanded, &native).map_err(language)?;
    let text = query_language::format::format_query(original.query(), &native).map_err(language)?;
    let bindings = replay_bindings(object, &native)?;
    Ok(Plan {
        original,
        expanded,
        text,
        identity: object.identity_ast.clone(),
        bindings,
        catalog: native.clone(),
        definitions: vec![],
        frozen: object.definitions.clone(),
        identity_catalog: native,
        question_id: object.id,
    })
}
