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
    for output in &mut catalog.source_outputs {
        output.schema.match_patterns.clear();
        output.relation_scope = None;
    }
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
/// Parse the canonical reference carried by saved figure denominators.
pub fn question_id_from_reference(reference: &str) -> Result<QuestionId, QueryError> {
    polylogue_model::Reference::parse(reference)
        .ok()
        .and_then(|reference| reference.question_id())
        .ok_or_else(|| invalid("expected an exact lowercase query:<hash> reference"))
}
/// Saved names used by typed source and condition nodes, including nested
/// queries. Payload strings are not names. This inspects syntax only and does
/// not resolve definitions or establish executable object custody.
pub fn referenced_saved_names(query: &Query) -> Vec<String> {
    let names = std::cell::RefCell::new(BTreeSet::new());
    let mut query = query.clone();
    walk(
        &mut query,
        &mut |query| {
            if let Source::Saved { name, .. } = &query.source {
                names.borrow_mut().insert(name.clone());
            }
            Ok(())
        },
        &mut |condition| {
            if let ConditionKind::Saved { name, .. } = &condition.kind {
                names.borrow_mut().insert(name.clone());
            }
            Ok(())
        },
    )
    .expect("collecting typed saved names is infallible");
    names.into_inner().into_iter().collect()
}

/// Exact semantic object sources requested by the parsed query. Ordinary text,
/// field values and saved names are not executable object lookup requests.
pub fn referenced_question_ids(query: &Query) -> Vec<QuestionId> {
    referenced_saved_names(query)
        .iter()
        .filter_map(|name| parse_id(name).ok())
        .collect()
}

fn sources(query: &Query) -> Result<Vec<(Source, OutputSource)>, QueryError> {
    let mut sources = Vec::new();
    let mut query = query.clone();
    walk(
        &mut query,
        &mut |query| {
            let key = match &query.source {
                Source::Saved {
                    name, arguments, ..
                } => OutputSource::Saved {
                    name: name.clone(),
                    arguments: arguments.clone(),
                },
                Source::Result {
                    reference, part, ..
                } => OutputSource::Result {
                    reference: reference.clone(),
                    part: *part,
                },
                _ => return Ok(()),
            };
            sources.push((query.source.clone(), key));
            Ok(())
        },
        &mut |_| Ok(()),
    )?;
    Ok(sources)
}
fn used_outputs(query: &Query, catalog: &Catalog) -> Result<Vec<SourceOutput>, QueryError> {
    let mut outputs: Vec<SourceOutput> = Vec::new();
    for (source, key) in sources(query)? {
        // Shared binding owns semantic argument matching, including lexical
        // quoting and argument order. Its validated catalog rejects duplicates.
        let mut schema = catalog
            .source_output(&source)
            .cloned()
            .ok_or_else(|| invalid("canonical source needs exactly one frozen output schema"))?;
        schema.match_patterns.clear();
        let output = SourceOutput {
            source: key,
            schema,
            relation_scope: None,
        };
        if !outputs.contains(&output) {
            outputs.push(output);
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
    let native = &plan.identity_catalog;
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
    let edges = edges(&plan.identity, native)?;
    let object = QueryObject {
        id: plan.question_id,
        normal_form: identity::normal_form(&plan.identity, native).map_err(language)?,
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
    let mut outputs: Vec<SourceOutput> = Vec::new();
    for output in object.source_outputs.iter().chain(
        object
            .definitions
            .iter()
            .flat_map(|node| node.source_outputs.iter()),
    ) {
        let mut output = output.clone();
        output.schema.match_patterns.clear();
        output.relation_scope = None;
        if let Some(previous) = outputs.iter().find(|old| old.source == output.source) {
            if previous.schema != output.schema {
                return Err(invalid("conflicting frozen output schemas"));
            }
        } else {
            outputs.push(output);
        }
    }
    Ok(outputs)
}
/// Frozen hash-keyed definitions for a retained semantic object. The root is
/// included so a saved name can point at it without resolving mutable aliases.
pub fn saved_query_bindings(
    object: &QueryObject,
) -> Result<BTreeMap<String, Vec<crate::SavedQuery>>, QueryError> {
    validate_object(object)?;
    let native = runtime_catalog(object)?;
    let mut saved = replay_bindings(object, &native)?.saved;
    let id = object.id.to_string();
    saved.insert(
        id.clone(),
        vec![crate::SavedQuery {
            text: query_language::format::format_query(&object.identity_ast, &native)
                .map_err(language)?,
            revision: id.clone(),
            root: id,
            hash: object.id,
            denominator: None,
            conflicts: vec![],
            body: crate::SavedQueryBody::Frozen {
                syntax: query_language::format::format_query(&object.identity_ast, &native)
                    .map_err(language)?,
            },
        }],
    );
    Ok(saved)
}

/// Runtime qualifier/projection metadata is rebuilt exclusively from the
/// retained frozen closure. No current saved-name lookup or clock is involved.
fn runtime_catalog(object: &QueryObject) -> Result<Catalog, QueryError> {
    let mut native = catalog(&all_outputs(object)?);
    let mut nodes = BTreeMap::new();
    for node in &object.definitions {
        if nodes.insert(node.question.to_string(), node).is_some() {
            return Err(invalid("duplicate frozen question body"));
        }
    }
    fn hydrate(
        index: usize,
        native: &mut Catalog,
        nodes: &BTreeMap<String, &FrozenQueryDefinition>,
        done: &mut BTreeSet<usize>,
        visiting: &mut BTreeSet<usize>,
    ) -> Result<(), QueryError> {
        if done.contains(&index) {
            return Ok(());
        }
        if !visiting.insert(index) {
            return Err(invalid("recursive frozen output metadata"));
        }
        let OutputSource::Saved { name, .. } = native.source_outputs[index].source.clone() else {
            visiting.remove(&index);
            done.insert(index);
            return Ok(());
        };
        let node = nodes
            .get(&name)
            .ok_or_else(|| invalid(format!("missing frozen question {name}")))?;
        for output in &node.source_outputs {
            let dependency = native
                .source_outputs
                .iter()
                .position(|stored| stored.source == output.source)
                .ok_or_else(|| invalid("missing frozen output schema"))?;
            hydrate(dependency, native, nodes, done, visiting)?;
        }
        crate::compile::schema::supply_matches(&node.identity_ast, native)?;
        let bound = query_language::bind::bind(&node.identity_ast, native).map_err(language)?;
        let text = query_language::format::format_query(bound.query(), native).map_err(language)?;
        let patterns = query_language::parse::relation_patterns(&text, native)
            .map_err(language)?
            .ok_or_else(|| invalid("frozen relation qualifier metadata is unresolved"))?;
        native.source_outputs[index].schema.match_patterns = patterns;
        native.source_outputs[index].relation_scope = Some(crate::compile::schema::relation_scope(
            &bound,
            &native.source_outputs[index].schema,
        )?);
        visiting.remove(&index);
        done.insert(index);
        Ok(())
    }
    let mut done = BTreeSet::new();
    let mut visiting = BTreeSet::new();
    for index in 0..native.source_outputs.len() {
        hydrate(index, &mut native, &nodes, &mut done, &mut visiting)?;
    }
    for node in &object.definitions {
        crate::compile::schema::supply_matches(&node.identity_ast, &mut native)?;
    }
    crate::compile::schema::supply_matches(&object.identity_ast, &mut native)?;
    Ok(native)
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
                text: text.clone(),
                revision: id.clone(),
                root: id,
                hash: node.question,
                denominator: None,
                conflicts: vec![],
                body: crate::SavedQueryBody::Frozen {
                    syntax: text.clone(),
                },
            }],
        );
    }
    for output in &native.source_outputs {
        if let OutputSource::Result { reference, part } = &output.source {
            let id = reference
                .strip_prefix("result:")
                .ok_or_else(|| invalid("kept output has no canonical result identity"))?;
            let unit = output
                .schema
                .unit
                .clone()
                .ok_or_else(|| invalid("kept output has no row domain"))?;
            let columns = output
                .schema
                .fields
                .iter()
                .map(|field| {
                    use query_language::catalog::FieldKind;
                    let kind = match field.kind {
                        FieldKind::Integer | FieldKind::Number => crate::ColKind::Number,
                        FieldKind::Money => crate::ColKind::Money,
                        FieldKind::Time => crate::ColKind::Time,
                        FieldKind::Duration => crate::ColKind::Duration,
                        FieldKind::Bool => crate::ColKind::Bool,
                        FieldKind::Json => crate::ColKind::Json,
                        FieldKind::List(_) => crate::ColKind::Ids,
                        _ => crate::ColKind::Text,
                    };
                    crate::Column {
                        name: field.name.clone(),
                        kind,
                        reference: None,
                    }
                })
                .collect();
            let kept = bindings
                .kept
                .entry(id.to_owned())
                .or_insert_with(|| crate::Kept {
                    unit: unit.clone(),
                    columns: vec![],
                    match_columns: vec![],
                });
            if kept.unit != unit {
                return Err(invalid("kept output parts disagree about their row domain"));
            }
            match part {
                ResultPart::Rows => kept.columns = columns,
                ResultPart::Matches => kept.match_columns = columns,
            }
        }
    }
    Ok(bindings)
}
fn expand(query: &Query, native: &Catalog, bindings: &Bindings) -> Result<Query, QueryError> {
    let text = query_language::format::format_query(query, native).map_err(language)?;
    let (query, _) = crate::compile::source::expand(&text, bindings)?;
    let mut native = native.clone();
    crate::compile::schema::supply_matches(&query, &mut native)?;
    Ok(query_language::bind::bind(&query, &native)
        .map_err(language)?
        .into_query())
}
fn check_node(
    id: QuestionId,
    form: &serde_json::Value,
    parameters: &BTreeMap<String, String>,
    query: &Query,
    outputs: &[SourceOutput],
    runtime: &Catalog,
) -> Result<(), QueryError> {
    let mut native = catalog(outputs);
    native.match_outputs = runtime.match_outputs.clone();
    for output in &mut native.source_outputs {
        let stored = runtime
            .source_outputs
            .iter()
            .find(|stored| stored.source == output.source)
            .ok_or_else(|| invalid("missing runtime output schema"))?;
        output.schema.match_patterns = stored.schema.match_patterns.clone();
        output.relation_scope = stored.relation_scope.clone();
    }
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
    let mut recorded = outputs.to_vec();
    for output in &mut recorded {
        output.schema.match_patterns.clear();
        output.relation_scope = None;
    }
    if used_outputs(query, &native)? != recorded {
        return Err(invalid(
            "question contains unused or unordered output schemas",
        ));
    }
    Ok(())
}
/// Validate current executable semantics using owned native declarations and
/// the complete immutable definition closure. This function performs no reads.
pub fn validate_object(object: &QueryObject) -> Result<(), QueryError> {
    let native = runtime_catalog(object)?;
    check_node(
        object.id,
        &object.normal_form,
        &object.parameters,
        &object.identity_ast,
        &object.source_outputs,
        &native,
    )?;
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
            &native,
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
    let mut native = runtime_catalog(object)?;
    let original = query_language::bind::bind(&object.identity_ast, &native).map_err(language)?;
    let expanded = expand(
        original.query(),
        &native,
        &replay_bindings(object, &native)?,
    )?;
    crate::compile::schema::supply_matches(&expanded, &mut native)?;
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
