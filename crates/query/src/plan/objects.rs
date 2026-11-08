//! Canonical object admission and replay use the product's native declarations.
use super::*;
use polylogue_model::query_objects::{FrozenQueryDefinition, QueryEdge, QueryObject, SourceOutput};
use query_language::identity;
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
    crate::catalog::catalog(&[])
}

/// A supported native declaration version, independent of transport vocabulary.
/// This belongs in execution bindings, never in semantic question identity.
/// The native catalog is fixed for the life of the process, so it is hashed once.
pub fn catalog_fingerprint() -> &'static str {
    static FINGERPRINT: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
        // Catalog has no arbitrary JSON maps; derived struct serialization has stable
        // field order even when serde_json's preserve_order feature is enabled.
        let bytes = serde_json::to_vec(&native_catalog()).expect("native catalog is serializable");
        blake3::hash(&bytes).to_hex().to_string()
    });
    &FINGERPRINT
}
fn catalog(outputs: &[SourceOutput]) -> Catalog {
    let mut catalog = native_catalog();
    catalog.source_outputs = outputs.iter().map(SourceOutput::catalog_output).collect();
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
fn used_outputs(
    query: &Query,
    catalog: &Catalog,
) -> Result<Vec<query_language::catalog::SourceOutput>, QueryError> {
    let mut outputs: Vec<query_language::catalog::SourceOutput> = Vec::new();
    for (source, key) in sources(query)? {
        // Shared binding owns semantic argument matching, including lexical
        // quoting and argument order. Its validated catalog rejects duplicates.
        let schema = catalog
            .source_output(&source)
            .cloned()
            .ok_or_else(|| invalid("canonical source needs exactly one frozen output schema"))?;
        let output = query_language::catalog::SourceOutput {
            source: key,
            schema,
        };
        if !outputs.contains(&output) {
            outputs.push(output);
        }
    }
    Ok(outputs)
}
fn capture_outputs(
    query: &Query,
    catalog: &Catalog,
) -> Result<Vec<SourceOutput>, QueryError> {
    used_outputs(query, catalog)?
        .into_iter()
        .map(|output| {
            Ok(SourceOutput {
                source: output.source,
                schema: output.schema,
            })
        })
        .collect()
}
fn edge_key(edge: &QueryEdge) -> String {
    match edge {
        QueryEdge::Saved { query } => format!("saved:{query}"),
        QueryEdge::Result { result } => format!("result:{result}"),
        QueryEdge::Combine { query } => format!("combine:{query}"),
        QueryEdge::In { query } => format!("in:{query}"),
    }
}
fn inline_identity(query: &Query, native: &Catalog) -> Result<QuestionId, QueryError> {
    identity::question_id(query, native, &BTreeMap::new()).map_err(language)
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
            for stage in &query.transforms {
                if let TransformKind::Combine { query, .. }
                | TransformKind::Join { query, .. }
                | TransformKind::Diff { query } = &stage.kind
                {
                    output.borrow_mut().push(QueryEdge::Combine {
                        query: inline_identity(query, native)?,
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
                    query: inline_identity(query, native)?,
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
    if let Some(object) = &plan.retained_object {
        return Ok(object.clone());
    }
    let outputs = capture_outputs(&plan.identity, &plan.identity_catalog)?;
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
                    query: query.clone(),
                    source_outputs: capture_outputs(query, &plan.identity_catalog)?,
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
        query: plan.identity.clone(),
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
        let output = output.clone();
        if let Some(previous) = outputs.iter().find(|old| old.source == output.source) {
            if previous != &output {
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
    let syntax = query_language::format::format_query(&object.query, &native).map_err(language)?;
    let (text, body) = (
        syntax.clone(),
        crate::SavedQueryBody::Frozen {
            object: Some(std::sync::Arc::new(object.clone())),
            syntax,
        },
    );
    saved.insert(
        id.clone(),
        vec![crate::SavedQuery {
            text,
            revision: id.clone(),
            root: id,
            hash: object.id,
            denominator: None,
            conflicts: vec![],
            body,
        }],
    );
    Ok(saved)
}

pub(super) fn frozen_root(object: &QueryObject) -> FrozenQueryDefinition {
    FrozenQueryDefinition {
        question: object.id,
        normal_form: object.normal_form.clone(),
        parameters: object.parameters.clone(),
        query: object.query.clone(),
        source_outputs: object.source_outputs.clone(),
    }
}

/// Select a saved node and exactly its dependency closure.
pub(super) fn retained_definition(
    object: &QueryObject,
    question: QuestionId,
) -> Result<(FrozenQueryDefinition, Vec<FrozenQueryDefinition>), QueryError> {
    validate_object(object)?;
    let node = if question == object.id {
        frozen_root(object)
    } else {
        object
            .definitions
            .iter()
            .find(|node| node.question == question)
            .cloned()
            .ok_or_else(|| invalid("saved query differs from its declared semantic hash"))?
    };
    let native = runtime_catalog(object)?;
    let mut queue = VecDeque::from(dependencies(&edges(&node.query, &native)?));
    let mut seen = BTreeSet::new();
    let mut closure = Vec::new();
    while let Some(id) = queue.pop_front() {
        if !seen.insert(id.to_string()) {
            continue;
        }
        let dependency = object
            .definitions
            .iter()
            .find(|node| node.question == id)
            .ok_or_else(|| invalid(format!("missing retained query:{id}")))?;
        queue.extend(dependencies(&edges(&dependency.query, &native)?));
        closure.push(dependency.clone());
    }
    Ok((node, closure))
}

/// Bind only the declarations retained with this immutable definition closure.
fn runtime_catalog(object: &QueryObject) -> Result<Catalog, QueryError> {
    Ok(catalog(&all_outputs(object)?))
}
fn replay_bindings(object: &QueryObject, native: &Catalog) -> Result<Bindings, QueryError> {
    let mut bindings = Bindings::default();
    let captured = std::sync::Arc::new(object.clone());
    for node in &object.definitions {
        let id = node.question.to_string();
        let syntax =
            query_language::format::format_query(&node.query, native).map_err(language)?;
        let (text, body) = (
            syntax.clone(),
            crate::SavedQueryBody::Frozen {
                object: Some(captured.clone()),
                syntax,
            },
        );
        bindings.saved.insert(
            id.clone(),
            vec![crate::SavedQuery {
                text,
                revision: id.clone(),
                root: id,
                hash: node.question,
                denominator: None,
                conflicts: vec![],
                body,
            }],
        );
    }
    for output in all_outputs(object)? {
        if let OutputSource::Result { reference, part } = &output.source {
            let id = reference
                .strip_prefix("result:")
                .ok_or_else(|| invalid("kept output has no canonical result identity"))?;
            let unit = output.schema.unit.clone().unwrap_or_default();
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
                        FieldKind::List(ref inner) if **inner == FieldKind::Reference => {
                            crate::ColKind::Ids
                        }
                        FieldKind::List(_) => crate::ColKind::Json,
                        _ => crate::ColKind::Text,
                    };
                    crate::Column {
                        name: field.name.clone(),
                        kind,
                        reference: (field.kind == FieldKind::Reference)
                            .then(|| {
                                field
                                    .unit
                                    .as_deref()
                                    .and_then(crate::ColumnReference::from_unit)
                            })
                            .flatten(),
                    }
                })
                .collect();
            let kept = bindings
                .kept
                .entry(id.to_owned())
                .or_insert_with(|| crate::Kept {
                    output_metadata: Default::default(),
                    shape: None,
                    native_identity: false,
                    unit: unit.clone(),
                    columns: vec![],
                    match_columns: vec![],
                });
            if !kept.unit.is_empty() && !unit.is_empty() && kept.unit != unit {
                return Err(invalid("kept output parts disagree about their row domain"));
            }
            if kept.unit.is_empty() {
                kept.unit = unit;
            }
            match part {
                ResultPart::Rows => {
                    kept.columns = columns;
                    kept.output_metadata.rows = Some(output.schema.clone());
                }
                ResultPart::Matches => {
                    kept.match_columns = columns;
                    kept.output_metadata.matches = Some(output.schema.clone());
                }
            }
        }
    }
    Ok(bindings)
}
fn expand(query: &Query, native: &Catalog, bindings: &Bindings) -> Result<Query, QueryError> {
    let text = query_language::format::format_query(query, native).map_err(language)?;
    let (query, _) = crate::compile::source::expand_ast(query.clone(), &text, bindings)?;
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
    let recorded: Vec<_> = outputs.iter().map(SourceOutput::catalog_output).collect();
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
    let actual = identity::question_id_from_normal_form(&object.normal_form, &object.parameters)
        .map_err(language)?;
    if actual != object.id {
        return Err(invalid(
            "retained question hash differs from its normal form",
        ));
    }
    for node in &object.definitions {
        if identity::question_id_from_normal_form(&node.normal_form, &node.parameters)
            .map_err(language)?
            != node.question
        {
            return Err(invalid(
                "retained dependency hash differs from its normal form",
            ));
        }
    }
    let native = runtime_catalog(object)?;
    check_node(
        object.id,
        &object.normal_form,
        &object.parameters,
        &object.query,
        &object.source_outputs,
    )?;
    let actual_edges = edges(&object.query, &native)?;
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
            &node.query,
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
        queue.extend(dependencies(&edges(&node.query, &native)?));
    }
    if seen.len() != by_id.len() {
        return Err(invalid("question contains an unrelated frozen definition"));
    }
    let bindings = replay_bindings(object, &native)?;
    // A supplied schema cannot promote a field outside the semantic relation
    // declared by its frozen definition. Kept outputs remain identified immutable inputs.
    for output in &native.source_outputs {
        if let OutputSource::Saved { name, arguments } = &output.source {
            let source = Query {
                source: Source::Saved {
                    name: name.clone(),
                    arguments: arguments.clone(),
                    condition: None,
                    output: SourceColumns::Known,
                },
                attach: vec![],
                transforms: vec![],
            };
            let expanded = expand(&source, &native, &bindings)?;
            let (_, actual) = crate::compile::schema::output_metadata(&expanded, &native)?;
            if actual != output.schema {
                return Err(invalid(
                    "saved output schema differs from its frozen definition",
                ));
            }
        }
    }
    let _ = expand(&object.query, &native, &bindings)?;
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
    let native = runtime_catalog(object)?;
    let original = query_language::bind::bind(&object.query, &native).map_err(language)?;
    let expanded = expand(
        original.query(),
        &native,
        &replay_bindings(object, &native)?,
    )?;

    let expanded = query_language::bind::bind(&expanded, &native).map_err(language)?;
    let text = query_language::format::format_query(original.query(), &native).map_err(language)?;
    let bindings = replay_bindings(object, &native)?;
    Ok(Plan {
        retained_object: Some(object.clone()),
        original,
        expanded,
        text,
        identity: object.query.clone(),
        bindings,
        catalog: native.clone(),
        definitions: vec![],
        frozen: object.definitions.clone(),
        identity_catalog: native,
        question_id: object.id,
    })
}
