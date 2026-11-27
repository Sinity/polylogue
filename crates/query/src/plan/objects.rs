//! Saving a query object and replaying it use Polylogue's built-in declarations.
use super::*;
use polylogue_model::query_objects::{QueryEdge, QueryObject, SavedQueryDefinition, SourceOutput};
use query_language::identity;
use std::collections::{BTreeSet, VecDeque};
use std::sync::Arc;

fn language(error: query_language::Error) -> QueryError {
    QueryError::language(error, "")
}
fn built_in_catalog() -> Arc<Catalog> {
    crate::catalog::catalog(&[])
}

/// The version of the built-in declarations, independent of transport vocabulary.
/// This belongs in execution bindings, never in question identity.
/// The built-in catalog is fixed for the life of the process, so it is hashed once.
pub fn catalog_fingerprint() -> &'static str {
    static FINGERPRINT: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
        // Catalog has no arbitrary JSON maps; derived struct serialization has stable
        // field order even when serde_json's preserve_order feature is enabled.
        let bytes =
            serde_json::to_vec(&*built_in_catalog()).expect("built-in catalog is serializable");
        blake3::hash(&bytes).to_hex().to_string()
    });
    &FINGERPRINT
}
fn catalog(outputs: &[SourceOutput]) -> Arc<Catalog> {
    let mut catalog = built_in_catalog();
    if !outputs.is_empty() {
        Arc::make_mut(&mut catalog).source_outputs =
            outputs.iter().map(SourceOutput::catalog_output).collect();
    }
    catalog
}
fn parse_id(name: &str) -> Result<QuestionId, QueryError> {
    if name.len() != 64
        || !name
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
    {
        return Err(QueryError::invalid(format!(
            "saved source {name:?} has no resolved question identity"
        )));
    }
    polylogue_model::BlobHash::parse(name)
        .map(|hash| QuestionId(hash.0))
        .map_err(|_| QueryError::invalid("invalid question identity"))
}
/// Parse the `query:<hash>` reference carried by saved figure denominators.
pub fn question_id_from_reference(reference: &str) -> Result<QuestionId, QueryError> {
    polylogue_model::Reference::parse(reference)
        .ok()
        .and_then(|reference| reference.question_id())
        .ok_or_else(|| QueryError::invalid("expected an exact lowercase query:<hash> reference"))
}
/// Saved names used by typed source and condition nodes, including nested
/// queries. Payload strings are not names. This inspects syntax only and does
/// not resolve definitions or load the objects they name.
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

/// Kept results (`result:<id>`, by id) read as sources anywhere in the query,
/// nested queries included. Syntax only, like [`referenced_saved_names`].
pub fn referenced_kept_results(query: &Query) -> Vec<String> {
    let ids = std::cell::RefCell::new(BTreeSet::new());
    let mut query = query.clone();
    walk(
        &mut query,
        &mut |query| {
            if let Source::Result { reference, .. } = &query.source {
                ids.borrow_mut().insert(
                    reference
                        .strip_prefix("result:")
                        .unwrap_or(reference)
                        .to_owned(),
                );
            }
            Ok(())
        },
        &mut |_| Ok(()),
    )
    .expect("collecting typed kept references is infallible");
    ids.into_inner().into_iter().collect()
}

/// Saved query objects the parsed query names by hash. Other text, field
/// values and saved names are not object lookups.
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
        // Shared binding matches arguments, including lexical
        // quoting and argument order. Its validated catalog rejects duplicates.
        let schema = catalog
            .source_output(&source)
            .cloned()
            .ok_or_else(|| QueryError::invalid("source needs exactly one saved output schema"))?;
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
fn capture_outputs(query: &Query, catalog: &Catalog) -> Result<Vec<SourceOutput>, QueryError> {
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
fn inline_identity(query: &Query, catalog: &Catalog) -> Result<QuestionId, QueryError> {
    identity::question_id(query, catalog, &BTreeMap::new()).map_err(language)
}
fn edges(query: &Query, catalog: &Catalog) -> Result<Vec<QueryEdge>, QueryError> {
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
                        query: inline_identity(query, catalog)?,
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
                    query: inline_identity(query, catalog)?,
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
    if let Some(object) = &plan.saved_object {
        return Ok(object.clone());
    }
    let outputs = capture_outputs(&plan.identity, &plan.identity_catalog)?;
    let catalog = &plan.identity_catalog;
    let mut saved_definitions = plan.saved_definitions.clone();
    // Inline composition bodies are saved as definitions, like named inputs.
    let mut query = plan.identity.clone();
    let mut first = true;
    walk(
        &mut query,
        &mut |query| {
            if first {
                first = false;
                return Ok(());
            }
            let question = inline_identity(query, &plan.identity_catalog)?;
            if !saved_definitions
                .iter()
                .any(|node| node.query_id == question)
            {
                saved_definitions.push(SavedQueryDefinition {
                    query_id: question,
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
    saved_definitions.sort_by_key(|node| node.query_id.to_string());
    saved_definitions.dedup_by_key(|node| node.query_id);
    let edges = edges(&plan.identity, catalog)?;
    let object = QueryObject {
        id: plan.question_id,
        normal_form: identity::normal_form(&plan.identity, catalog).map_err(language)?,
        parameters: BTreeMap::new(),
        query: plan.identity.clone(),
        source_outputs: outputs,
        definitions: saved_definitions,
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
                return Err(QueryError::invalid("conflicting saved output schemas"));
            }
        } else {
            outputs.push(output);
        }
    }
    Ok(outputs)
}
/// Saved hash-keyed definitions for a saved query object. The root is
/// included so a saved name can point at it without resolving mutable aliases.
pub fn saved_query_bindings(
    object: &QueryObject,
) -> Result<BTreeMap<String, Vec<crate::SavedQuery>>, QueryError> {
    validate_object(object)?;
    let catalog = runtime_catalog(object)?;
    let mut saved = replay_bindings(object, &catalog)?.saved;
    let id = object.id.to_string();
    let syntax = query_language::format::format_query(&object.query, &catalog).map_err(language)?;
    let (text, body) = (
        syntax.clone(),
        crate::SavedQueryBody::Available {
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

pub(super) fn root_definition(object: &QueryObject) -> SavedQueryDefinition {
    SavedQueryDefinition {
        query_id: object.id,
        normal_form: object.normal_form.clone(),
        parameters: object.parameters.clone(),
        query: object.query.clone(),
        source_outputs: object.source_outputs.clone(),
    }
}

/// Select a saved node and exactly its dependencies.
pub(super) fn definition_with_dependencies(
    object: &QueryObject,
    question: QuestionId,
) -> Result<(SavedQueryDefinition, Vec<SavedQueryDefinition>), QueryError> {
    validate_object(object)?;
    let node = if question == object.id {
        root_definition(object)
    } else {
        object
            .definitions
            .iter()
            .find(|node| node.query_id == question)
            .cloned()
            .ok_or_else(|| QueryError::invalid("saved query differs from its declared hash"))?
    };
    let catalog = runtime_catalog(object)?;
    let mut queue = VecDeque::from(dependencies(&edges(&node.query, &catalog)?));
    let mut seen = BTreeSet::new();
    let mut dependency_nodes = Vec::new();
    while let Some(id) = queue.pop_front() {
        if !seen.insert(id.to_string()) {
            continue;
        }
        let dependency = object
            .definitions
            .iter()
            .find(|node| node.query_id == id)
            .ok_or_else(|| QueryError::invalid(format!("missing saved query:{id}")))?;
        queue.extend(dependencies(&edges(&dependency.query, &catalog)?));
        dependency_nodes.push(dependency.clone());
    }
    Ok((node, dependency_nodes))
}

/// Bind only the declarations kept with this immutable set of definitions.
fn runtime_catalog(object: &QueryObject) -> Result<Arc<Catalog>, QueryError> {
    Ok(catalog(&all_outputs(object)?))
}
fn replay_bindings(object: &QueryObject, catalog: &Catalog) -> Result<Bindings, QueryError> {
    let mut bindings = Bindings::default();
    let captured = std::sync::Arc::new(object.clone());
    for node in &object.definitions {
        let id = node.query_id.to_string();
        let syntax =
            query_language::format::format_query(&node.query, catalog).map_err(language)?;
        let (text, body) = (
            syntax.clone(),
            crate::SavedQueryBody::Available {
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
                hash: node.query_id,
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
                .ok_or_else(|| QueryError::invalid("kept output has no stored result identity"))?;
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
                    row_identity: false,
                    unit: unit.clone(),
                    columns: vec![],
                    match_columns: vec![],
                });
            if !kept.unit.is_empty() && !unit.is_empty() && kept.unit != unit {
                return Err(QueryError::invalid(
                    "kept output parts disagree about their row domain",
                ));
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
fn expand(query: &Query, catalog: &Catalog, bindings: &Bindings) -> Result<Query, QueryError> {
    let text = query_language::format::format_query(query, catalog).map_err(language)?;
    let (query, _) = crate::compile::source::expand_ast(query.clone(), &text, bindings)?;
    Ok(query_language::bind::bind(&query, catalog)
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
    let catalog = catalog(outputs);
    let actual = identity::question_id_from_normal_form(form, parameters).map_err(language)?;
    if actual != id || identity::normal_form(query, &catalog).map_err(language)? != *form {
        return Err(QueryError::invalid(
            "question identity, normal form and typed syntax disagree",
        ));
    }
    let bound = query_language::bind::bind(query, &catalog).map_err(language)?;
    if identity::normal_form(bound.query(), &catalog).map_err(language)? != *form {
        return Err(QueryError::invalid(
            "saved question changes when bound to built-in declarations",
        ));
    }
    let recorded: Vec<_> = outputs.iter().map(SourceOutput::catalog_output).collect();
    if used_outputs(query, &catalog)? != recorded {
        return Err(QueryError::invalid(
            "question contains unused or unordered output schemas",
        ));
    }
    Ok(())
}
/// Check the object against the built-in declarations and the complete
/// immutable set of definitions. This function performs no reads.
pub fn validate_object(object: &QueryObject) -> Result<(), QueryError> {
    let actual = identity::question_id_from_normal_form(&object.normal_form, &object.parameters)
        .map_err(language)?;
    if actual != object.id {
        return Err(QueryError::invalid(
            "saved question hash differs from its normal form",
        ));
    }
    for node in &object.definitions {
        if identity::question_id_from_normal_form(&node.normal_form, &node.parameters)
            .map_err(language)?
            != node.query_id
        {
            return Err(QueryError::invalid(
                "saved dependency hash differs from its normal form",
            ));
        }
    }
    let catalog = runtime_catalog(object)?;
    check_node(
        object.id,
        &object.normal_form,
        &object.parameters,
        &object.query,
        &object.source_outputs,
    )?;
    let actual_edges = edges(&object.query, &catalog)?;
    if actual_edges != object.edges || dependencies(&actual_edges) != object.dependencies {
        return Err(QueryError::invalid(
            "question composition edges do not match its syntax",
        ));
    }
    let mut by_id = BTreeMap::new();
    for node in &object.definitions {
        check_node(
            node.query_id,
            &node.normal_form,
            &node.parameters,
            &node.query,
            &node.source_outputs,
        )?;
        if by_id.insert(node.query_id.to_string(), node).is_some() {
            return Err(QueryError::invalid("duplicate saved question body"));
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
            .ok_or_else(|| QueryError::invalid(format!("missing saved question {id}")))?;
        queue.extend(dependencies(&edges(&node.query, &catalog)?));
    }
    if seen.len() != by_id.len() {
        return Err(QueryError::invalid(
            "question contains an unrelated saved definition",
        ));
    }
    let bindings = replay_bindings(object, &catalog)?;
    // A supplied schema cannot add a field outside the relation its saved
    // definition declares. Kept outputs remain identified immutable inputs.
    for output in &catalog.source_outputs {
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
            let expanded = expand(&source, &catalog, &bindings)?;
            let (_, actual) = crate::compile::schema::output_metadata(&expanded, &catalog)?;
            if actual != output.schema {
                return Err(QueryError::invalid(
                    "saved output schema differs from its saved definition",
                ));
            }
        }
    }
    let _ = expand(&object.query, &catalog, &bindings)?;
    Ok(())
}
/// Rebuild the same question from accepted execution inputs, without resolving
/// saved names against a newer catalog snapshot.
pub fn replay_binding(
    binding: &polylogue_model::query_objects::ReplayBinding,
) -> Result<Plan, QueryError> {
    let outputs: Vec<SourceOutput> = binding
        .source_outputs
        .iter()
        .map(|output| SourceOutput {
            source: output.source.clone(),
            schema: output.schema.clone(),
        })
        .collect();
    let catalog = catalog(&outputs);
    let mut definitions = Vec::new();
    let mut queries = vec![binding.bound_ast.clone()];
    queries.extend(
        binding
            .definitions
            .iter()
            .map(|definition| definition.ast.clone()),
    );
    for mut query in queries {
        walk(
            &mut query,
            &mut |query| {
                let question = inline_identity(query, &catalog)?;
                if question != binding.context.question
                    && !definitions
                        .iter()
                        .any(|definition: &SavedQueryDefinition| definition.query_id == question)
                {
                    definitions.push(SavedQueryDefinition {
                        query_id: question,
                        normal_form: identity::normal_form(query, &catalog).map_err(language)?,
                        parameters: BTreeMap::new(),
                        query: query.clone(),
                        source_outputs: capture_outputs(query, &catalog)?,
                    });
                }
                Ok(())
            },
            &mut |_| Ok(()),
        )?;
    }
    let edges = edges(&binding.bound_ast, &catalog)?;
    let object = QueryObject {
        id: binding.context.question,
        normal_form: identity::normal_form(&binding.bound_ast, &catalog).map_err(language)?,
        parameters: BTreeMap::new(),
        query: binding.bound_ast.clone(),
        source_outputs: capture_outputs(&binding.bound_ast, &catalog)?,
        definitions,
        dependencies: dependencies(&edges),
        edges,
    };
    let expected = binding
        .context
        .versions
        .get("catalog")
        .map(String::as_str)
        .ok_or_else(|| QueryError::invalid("accepted execution has no catalog identity"))?;
    let mut plan = replay(&object, expected)?;
    plan.text = binding.original_text.clone();
    plan.definitions = binding
        .definitions
        .iter()
        .map(|definition| ResolvedDefinition {
            name: definition.name.clone(),
            revision: definition.revision.clone(),
            root: definition.root.clone(),
            question_id: definition.query_id,
        })
        .collect();
    Ok(plan)
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
    let catalog = runtime_catalog(object)?;
    let original = query_language::bind::bind(&object.query, &catalog).map_err(language)?;
    let bindings = replay_bindings(object, &catalog)?;
    let expanded = expand(original.query(), &catalog, &bindings)?;

    let (expanded, expanded_text) = bind_spelled(&expanded, &catalog).map_err(language)?;
    let text =
        query_language::format::format_query(original.query(), &catalog).map_err(language)?;
    Ok(Plan {
        saved_object: Some(object.clone()),
        original,
        expanded,
        expanded_text,
        text,
        identity: object.query.clone(),
        bindings,
        catalog: catalog.clone(),
        definitions: vec![],
        saved_definitions: object.definitions.clone(),
        identity_catalog: catalog,
        question_id: object.id,
    })
}
