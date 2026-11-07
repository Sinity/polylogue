//! The deterministic format-1 representation upgrade. Original JSON is retained
//! separately as identity and execution-custody evidence by the numbered upgrade.
use query_language::catalog::Catalog;
use serde_json::{Value, json};

type Result<T> = std::result::Result<T, String>;

fn current<T: serde::de::DeserializeOwned + serde::Serialize>(value: Value) -> Result<Value> {
    let typed = serde_json::from_value::<T>(value)
        .map_err(|e| format!("upgraded model representation: {e}"))?;
    serde_json::to_value(typed).map_err(|e| format!("serialize upgraded model representation: {e}"))
}

fn string<'a>(v: &'a Value, key: &str) -> Result<&'a str> {
    v.get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("format-1 {key} is not a string"))
}
fn array_mut<'a>(v: &'a mut Value, key: &str) -> Result<&'a mut Vec<Value>> {
    v.get_mut(key)
        .and_then(Value::as_array_mut)
        .ok_or_else(|| format!("format-1 {key} is not an array"))
}
fn take(v: &mut Value, key: &str) -> Result<Value> {
    v.as_object_mut()
        .and_then(|v| v.remove(key))
        .ok_or_else(|| format!("format-1 missing {key}"))
}
fn current_unit(unit: &str) -> &str {
    match unit {
        "notes" | "note" | "pl.notes" | "pl.note" => "imported_notes",
        unit => unit,
    }
}

fn field(name: &str) -> Value {
    json!({"kind":"field","name":name})
}
fn literal(value: Value) -> Value {
    json!({"kind":"value","value":value})
}
fn column(expression: Value, name: String) -> Value {
    json!({"expression":expression,"name":name})
}
fn operand_name(v: &Value) -> Result<String> {
    Ok(match string(v, "kind")? {
        "field" => string(v, "name")?.replace(['.', '-'], "_"),
        "function" => string(v, "name")?.into(),
        kind => return Err(format!("unknown format-1 operand {kind}")),
    })
}
fn operand(v: &mut Value) -> Result<()> {
    match string(v, "kind")? {
        "field" => {
            string(v, "name")?;
        }
        "function" => {
            for arg in array_mut(v, "arguments")? {
                operand(arg)?;
            }
        }
        kind => return Err(format!("unknown format-1 operand {kind}")),
    }
    Ok(())
}
fn extension(v: &mut Value, outputs: &[Value], definitions: &[Value]) -> Result<()> {
    for a in array_mut(v, "arguments")? {
        if let Some(c) = a.get_mut("condition") {
            condition(c, outputs, definitions)?;
        }
        if let Some(q) = a.get_mut("query") {
            query(q, outputs, definitions)?;
        }
        if let Some(args) = a.get_mut("optional") {
            let mut holder = json!({"arguments":args.take()});
            extension(&mut holder, outputs, definitions)?;
            *args = take(&mut holder, "arguments")?;
        }
    }
    Ok(())
}
fn notes_condition(v: &mut Value, unit: &str) -> Result<()> {
    let span = v["span"].clone();
    let kind = v.get_mut("kind").ok_or("format-1 condition missing kind")?;
    match kind["kind"]
        .as_str()
        .ok_or("format-1 condition missing tag")?
    {
        "reference" => {
            let reference = string(kind, "reference")?;
            let Some(value) = reference.strip_prefix("note:") else {
                return Ok(());
            };
            if current_unit(unit) != "imported_notes" {
                return Err("format-1 note references require the notes row domain".into());
            }
            let (root, revision) = value
                .rsplit_once('@')
                .map_or((value, None), |(root, revision)| (root, Some(revision)));
            if root.is_empty() || revision.is_some_and(str::is_empty) {
                return Err("invalid format-1 note reference".into());
            }
            let scalar = |value: &str| json!({"kind":"scalar","text":value,"quoted":true});
            let atom = |kind: Value| json!({"span":span,"kind":kind});
            let mut conditions = vec![atom(
                json!({"kind":"compare","operand":field("first_note"),"op":"eq","value":scalar(root)}),
            )];
            conditions.push(if let Some(revision) = revision {
                atom(json!({"kind":"match","field":"id","value":scalar(revision)}))
            } else {
                atom(json!({"kind":"match","field":"current","value":scalar("true")}))
            });
            for field in ["kind", "current", "withdrawn"] {
                conditions.push(atom(
                    json!({"kind":"match","field":field,"value":{"kind":"any"}}),
                ));
            }
            *kind = json!({"kind":"and","conditions":conditions});
        }
        "and" | "or" => {
            for condition in array_mut(kind, "conditions")? {
                notes_condition(condition, unit)?;
            }
        }
        "not" => notes_condition(
            kind.get_mut("condition")
                .ok_or("missing format-1 negation")?,
            unit,
        )?,
        "exists" | "count" => {
            let nested = string(kind, "unit")?.to_owned();
            if let Some(condition) = kind.get_mut("condition").filter(|value| !value.is_null()) {
                notes_condition(condition, &nested)?;
            }
        }
        "sequence" => {
            let nested = string(kind, "unit")?.to_owned();
            for step in array_mut(kind, "steps")? {
                notes_condition(step, &nested)?;
            }
        }
        "extension" => notes_arguments(array_mut(kind, "arguments")?, unit)?,
        _ => (),
    }
    Ok(())
}
fn notes_arguments(arguments: &mut [Value], unit: &str) -> Result<()> {
    for argument in arguments {
        if let Some(condition) = argument.get_mut("condition") {
            notes_condition(condition, unit)?;
        }
        if let Some(optional) = argument.get_mut("optional") {
            notes_arguments(
                optional
                    .as_array_mut()
                    .ok_or("invalid historical optional arguments")?,
                unit,
            )?;
        }
    }
    Ok(())
}

fn condition(v: &mut Value, outputs: &[Value], definitions: &[Value]) -> Result<()> {
    let k = v.get_mut("kind").ok_or("format-1 condition missing kind")?;
    match string(k, "kind")?.to_owned().as_str() {
        "and" | "or" => {
            for c in array_mut(k, "conditions")? {
                condition(c, outputs, definitions)?;
            }
        }
        "not" => condition(
            k.get_mut("condition").ok_or("missing condition")?,
            outputs,
            definitions,
        )?,
        "compare" => {
            let mut left = take(k, "operand")?;
            operand(&mut left)?;
            let right = literal(take(k, "value")?);
            k["left"] = left;
            k["right"] = right;
        }
        "since" | "until" => {
            let p = take(k, "operand")?;
            if string(&p, "kind")? != "field" {
                return Err("format-1 computed since/until has no field-bound equivalent".into());
            }
            k["field"] = json!(string(&p, "name")?);
        }
        "between" | "time_range" | "in" => {
            let mut left = take(k, "operand")?;
            operand(&mut left)?;
            k["left"] = left;
            if let Some(q) = k.get_mut("query") {
                query(q, outputs, definitions)?;
            }
        }
        "similar" => {
            k["kind"] = json!("rank");
            k["list"] = json!("similar");
        }
        "exists" | "count" => {
            k["unit"] = json!(current_unit(string(k, "unit")?));
            if let Some(c) = k.get_mut("condition").filter(|v| !v.is_null()) {
                condition(c, outputs, definitions)?;
            }
        }
        "field_exists" => {
            let p = take(k, "operand")?;
            if string(&p, "kind")? != "field" {
                return Err(
                    "format-1 computed field_exists needs an expression-known operation".into(),
                );
            }
            *k = json!({"kind":"match","field":string(&p,"name")?,"value":{"kind":"known"}});
        }
        "time_difference" => {
            let mut left = take(k, "left")?;
            let mut right = take(k, "right")?;
            operand(&mut left)?;
            operand(&mut right)?;
            let value = take(k, "value")?;
            k["kind"] = json!("compare");
            k["left"] =
                json!({"kind":"arithmetic","operator":"subtract","left":left,"right":right});
            k["right"] = literal(value);
        }
        "sequence" => {
            k["unit"] = json!(current_unit(string(k, "unit")?));
            for c in array_mut(k, "steps")? {
                condition(c, outputs, definitions)?;
            }
            for g in array_mut(k, "gaps")? {
                *g = if let Some(s) = g.as_str() {
                    json!({"kind":s})
                } else if let Some(w) = g.get("within") {
                    json!({"kind":"within","distance":{"kind":"time","duration":w["duration"]}})
                } else {
                    return Err("invalid format-1 gap".into());
                };
            }
        }
        "extension" => extension(k, outputs, definitions)?,
        "overlaps" => return Err("format-1 overlaps requires the owning unit span relation".into()),
        "match" | "text" | "reference" | "saved" => (),
        kind => return Err(format!("unknown format-1 condition {kind}")),
    }
    Ok(())
}
fn aggregate_name(a: &Value) -> Result<String> {
    let name = if let Some(s) = a["function"].as_str() {
        match s {
            "correlation" => "corr",
            s => s,
        }
        .to_string()
    } else if let Some(p) = a["function"].get("percentile") {
        format!("p{p}")
    } else {
        return Err("invalid format-1 aggregate function".into());
    };
    let fields = a["fields"].as_array().ok_or("invalid aggregate fields")?;
    if fields.is_empty() {
        Ok(name)
    } else {
        let fields = fields
            .iter()
            .map(|f| f.as_str().ok_or("invalid aggregate field"))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(format!(
            "{name}_{}",
            fields.join("_").replace(['.', '-'], "_")
        ))
    }
}
fn query(v: &mut Value, outputs: &[Value], definitions: &[Value]) -> Result<()> {
    let (mut parent, mut available) = query_pattern(v, outputs, definitions, 0)?;
    let source = v.get_mut("source").ok_or("format-1 query missing source")?;
    match string(source, "kind")? {
        "unit" => source["unit"] = json!(current_unit(string(source, "unit")?)),
        "saved" | "result" => {
            if source["output"] == "deferred" {
                source["output"] = json!("unknown");
            }
        }
        kind => {
            return Err(format!(
                "format-1 {kind} source has no Polylogue relation equivalent"
            ));
        }
    }
    if let Some(c) = source.get_mut("condition").filter(|v| !v.is_null()) {
        notes_condition(c, &parent)?;
        condition(c, outputs, definitions)?;
    }
    for a in array_mut(v, "attach")? {
        let unit = string(a, "unit")?.to_owned();
        a["unit"] = json!(current_unit(&unit));
        for w in array_mut(a, "windows")? {
            if let Some(c) = w.get_mut("where") {
                notes_condition(c, &unit)?;
                condition(c, outputs, definitions)?;
            }
        }
    }
    let mut stages = take(v, "stages")?;
    let mut transformed = Vec::new();
    for s in stages.as_array_mut().ok_or("invalid format-1 stages")? {
        let k = s.get_mut("kind").ok_or("missing stage kind")?;
        match string(k, "kind")?.to_owned().as_str() {
            "where" => {
                let c = k.get_mut("condition").ok_or("missing condition")?;
                notes_condition(c, &parent)?;
                condition(c, outputs, definitions)?;
            }
            "scope" => {
                parent = string(k, "unit")?.to_owned();
                k["unit"] = json!(current_unit(&parent));
                available.clear();
                patterns(&k["condition"], &mut available);
                if let Some(c) = k.get_mut("condition").filter(|v| !v.is_null()) {
                    notes_condition(c, &parent)?;
                    condition(c, outputs, definitions)?;
                }
            }
            "sort" => {
                for key in array_mut(k, "keys")? {
                    let mut p = take(key, "operand")?;
                    operand(&mut p)?;
                    key["expression"] = p;
                }
            }
            "group" => {
                available.clear();
                for p in array_mut(k, "keys")? {
                    let name = operand_name(p)?;
                    operand(p)?;
                    *p = column(p.take(), name);
                }
            }
            "fields" => {
                available.clear();
                let names = take(k, "fields")?;
                let mut columns = Vec::new();
                for name in names.as_array().ok_or("invalid fields")? {
                    let name = name.as_str().ok_or("invalid field")?;
                    columns.push(column(field(name), name.replace(['.', '-'], "_")));
                }
                k["columns"] = json!(columns);
            }
            "aggregate" => {
                available.clear();
                for a in array_mut(k, "aggregates")? {
                    let name = aggregate_name(a)?;
                    let names = take(a, "fields")?;
                    a["arguments"] = Value::Array(
                        names
                            .as_array()
                            .ok_or("invalid fields")?
                            .iter()
                            .map(|n| {
                                n.as_str()
                                    .map(field)
                                    .ok_or_else(|| "invalid aggregate field".into())
                            })
                            .collect::<Result<Vec<_>>>()?,
                    );
                    a["name"] = json!(name);
                }
            }
            "combine" => query(
                k.get_mut("query").ok_or("missing query")?,
                outputs,
                definitions,
            )?,
            "measure" => {
                let name = take(k, "name")?;
                let mut by = take(k, "by")?;
                for p in by.as_array_mut().ok_or("invalid measure by")? {
                    operand(p)?;
                }
                let args = by
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|p| json!({"expression":p}))
                    .collect::<Vec<_>>();
                *k = json!({"kind":"extension","name":"measure","arguments":[{"value":{"kind":"scalar","text":name,"quoted":true}},{"optional":if args.is_empty(){vec![]}else{vec![json!({"keyword":"by"}),json!({"list":args})]}}]});
            }
            "extension" => extension(k, outputs, definitions)?,
            "matches" => {
                transformed.extend(matches_projection(&parent, &available, &s["span"])?);
                available.clear();
                continue;
            }
            "temporal" => {
                return Err(
                    "format-1 temporal needs nearest-match and output-shape preserving lowering"
                        .into(),
                );
            }
            "limit" | "offset" | "first" | "distinct" | "fill" | "as_of" => (),
            kind => return Err(format!("unknown format-1 stage {kind}")),
        }
        transformed.push(s.clone());
    }
    v["transforms"] = json!(transformed);
    // Validate the new syntax here so failed upgrades roll back as one batch.
    serde_json::from_value::<query_language::ast::Query>(v.clone())
        .map_err(|e| format!("upgraded query syntax: {e}"))?;
    Ok(())
}
fn outputs(v: &mut Value, definitions: &[Value]) -> Result<()> {
    let captured = v
        .as_array()
        .ok_or("invalid format-1 source outputs")?
        .clone();
    for output in v.as_array_mut().ok_or("invalid format-1 source outputs")? {
        if output["source"]["kind"] == "saved"
            && let Some(definition) = find_definition(&output["source"], definitions)
        {
            let ast = definition
                .get("identity_ast")
                .or_else(|| definition.get("ast"))
                .ok_or("missing captured saved AST")?;
            let nested = definition["source_outputs"]
                .as_array()
                .map_or(captured.as_slice(), Vec::as_slice);
            let domain = output_domain(ast, nested, definitions, 0)?;
            if domain.is_none() {
                output["schema"]["main_time"] = Value::Null;
            }
            output["schema"]["unit"] = json!(domain);
        }
        if let Some(unit) = output["schema"]["unit"].as_str() {
            output["schema"]["unit"] = json!(current_unit(unit));
        }
        for f in array_mut(&mut output["schema"], "fields")? {
            field_kind(&mut f["kind"])?;
            if let Some(unit) = f["unit"].as_str() {
                f["unit"] = json!(current_unit(unit));
            }
        }
    }
    Ok(())
}
fn field_kind(v: &mut Value) -> Result<()> {
    if v == "match" {
        *v = json!("exact");
    } else if let Some(k) = v.get_mut("list") {
        field_kind(k)?;
    }
    Ok(())
}
fn unavailable_condition(
    condition: &Value,
    outputs: &[Value],
    definitions: &[Value],
    depth: usize,
) -> Result<Option<super::QueryUnavailable>> {
    use super::QueryUnavailable as U;
    let kind = condition
        .get("kind")
        .ok_or("missing original condition kind")?;
    Ok(
        match kind["kind"]
            .as_str()
            .ok_or("missing original condition tag")?
        {
            "since" | "until" if kind["operand"]["kind"] != "field" => Some(U::ComputedTimeBound),
            "field_exists" if kind["operand"]["kind"] != "field" => Some(U::ComputedExistence),
            "overlaps" => Some(U::SpanOverlap),
            "and" | "or" => {
                let mut reason = None;
                for condition in kind["conditions"]
                    .as_array()
                    .ok_or("invalid original conditions")?
                {
                    if let Some(found) =
                        unavailable_condition(condition, outputs, definitions, depth)?
                    {
                        reason = Some(found);
                        break;
                    }
                }
                reason
            }
            "not" => unavailable_condition(&kind["condition"], outputs, definitions, depth)?,
            "exists" | "count" => {
                if kind["condition"].is_null() {
                    None
                } else {
                    unavailable_condition(&kind["condition"], outputs, definitions, depth)?
                }
            }
            "sequence" => {
                let mut reason = None;
                for step in kind["steps"]
                    .as_array()
                    .ok_or("invalid original sequence")?
                {
                    if let Some(found) = unavailable_condition(step, outputs, definitions, depth)? {
                        reason = Some(found);
                        break;
                    }
                }
                reason
            }
            "in" => unavailability(&kind["query"], outputs, definitions, depth)?,
            "saved" => {
                if let Some(definition) = find_definition(kind, definitions) {
                    let ast = definition
                        .get("identity_ast")
                        .or_else(|| definition.get("ast"))
                        .ok_or("missing captured saved condition AST")?;
                    let nested = definition["source_outputs"]
                        .as_array()
                        .map_or(outputs, Vec::as_slice);
                    if unavailability(ast, nested, definitions, depth + 1)?.is_some() {
                        Some(U::Dependency {
                            question: serde_json::from_value(definition["question"].clone())
                                .map_err(|error| {
                                    format!("unavailable condition identity: {error}")
                                })?,
                        })
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            "extension" => unavailable_arguments(
                kind["arguments"]
                    .as_array()
                    .ok_or("invalid original arguments")?,
                outputs,
                definitions,
                depth,
            )?,
            _ => None,
        },
    )
}
fn unavailable_arguments(
    arguments: &[Value],
    outputs: &[Value],
    definitions: &[Value],
    depth: usize,
) -> Result<Option<super::QueryUnavailable>> {
    for argument in arguments {
        let reason = if let Some(condition) = argument.get("condition") {
            unavailable_condition(condition, outputs, definitions, depth)?
        } else if let Some(query) = argument.get("query") {
            unavailability(query, outputs, definitions, depth)?
        } else if let Some(optional) = argument.get("optional") {
            unavailable_arguments(
                optional
                    .as_array()
                    .ok_or("invalid original optional arguments")?,
                outputs,
                definitions,
                depth,
            )?
        } else {
            None
        };
        if reason.is_some() {
            return Ok(reason);
        }
    }
    Ok(None)
}
/// These cases are selected from the frozen producer representation and the
/// original backend's refusal paths, never from failures of the current compiler.
fn unavailability(
    query: &Value,
    outputs: &[Value],
    definitions: &[Value],
    depth: usize,
) -> Result<Option<super::QueryUnavailable>> {
    use super::{HistoricalNativeSource as S, QueryUnavailable as U};
    if depth > definitions.len() + 1 {
        return Err("cyclic original query closure".into());
    }
    serde_json::from_value::<historical_query_format_v1::ast::Query>(query.clone())
        .map_err(|error| format!("format-1 AST: {error}"))?;
    let source = &query["source"];
    let source_kind = match source["kind"].as_str() {
        Some("walk") => Some(S::Walk),
        Some("recall") => Some(S::Recall),
        Some("recall_last") => Some(S::RecallLast),
        Some("recall_mark") => Some(S::RecallMark),
        _ => None,
    };
    if let Some(source_kind) = source_kind {
        return Ok(Some(U::NativeSource { source_kind }));
    }
    if source["kind"] == "saved"
        && let Some(definition) = find_definition(source, definitions)
    {
        let ast = definition
            .get("identity_ast")
            .or_else(|| definition.get("ast"))
            .ok_or("missing captured definition AST")?;
        let nested = definition["source_outputs"]
            .as_array()
            .map_or(outputs, Vec::as_slice);
        if unavailability(ast, nested, definitions, depth + 1)?.is_some() {
            let question = serde_json::from_value(definition["question"].clone())
                .map_err(|error| format!("unavailable dependency identity: {error}"))?;
            return Ok(Some(U::Dependency { question }));
        }
    }
    if !source["condition"].is_null()
        && let Some(reason) =
            unavailable_condition(&source["condition"], outputs, definitions, depth)?
    {
        return Ok(Some(reason));
    }
    for attachment in query["attach"]
        .as_array()
        .ok_or("missing original attachments")?
    {
        for window in attachment["windows"]
            .as_array()
            .ok_or("missing original attachment windows")?
        {
            if let Some(condition) = window.get("where")
                && let Some(reason) = unavailable_condition(condition, outputs, definitions, depth)?
            {
                return Ok(Some(reason));
            }
        }
    }
    let (mut parent, mut patterns_here) = query_pattern(query, outputs, definitions, 0)?;
    for stage in query["stages"]
        .as_array()
        .ok_or("missing original stages")?
    {
        let kind = &stage["kind"];
        match kind["kind"].as_str() {
            Some("temporal") => return Ok(Some(U::TemporalStage)),
            Some("where") => {
                if let Some(reason) =
                    unavailable_condition(&kind["condition"], outputs, definitions, depth)?
                {
                    return Ok(Some(reason));
                }
            }
            Some("scope") => {
                parent = string(kind, "unit")?.into();
                patterns_here.clear();
                patterns(&kind["condition"], &mut patterns_here);
                if !kind["condition"].is_null()
                    && let Some(reason) =
                        unavailable_condition(&kind["condition"], outputs, definitions, depth)?
                {
                    return Ok(Some(reason));
                }
            }
            Some("group" | "aggregate" | "fields") => patterns_here.clear(),
            Some("combine") => {
                if let Some(reason) = unavailability(&kind["query"], outputs, definitions, depth)? {
                    return Ok(Some(reason));
                }
            }
            Some("extension") => {
                if let Some(reason) = unavailable_arguments(
                    kind["arguments"]
                        .as_array()
                        .ok_or("invalid original extension arguments")?,
                    outputs,
                    definitions,
                    depth,
                )? {
                    return Ok(Some(reason));
                }
            }
            Some("matches") => {
                if let [pattern] = patterns_here.as_slice() {
                    match pattern {
                        Pattern::Exists(unit)
                            if parent != "sessions"
                                || !native_columns(unit)?.contains(&"session") =>
                        {
                            if !native_columns(&parent)?.contains(&"id") {
                                return Ok(Some(U::RetiredStorageKey { unit: parent }));
                            }
                        }
                        Pattern::Sequence(unit, _) if unit != "tool_calls" => {
                            return Ok(Some(U::MatchUnit { unit: unit.clone() }));
                        }
                        _ => (),
                    }
                }
                patterns_here.clear();
            }
            _ => (),
        }
    }
    Ok(None)
}

fn definition(v: &mut Value, definitions: &[Value]) -> Result<()> {
    let captured = v["source_outputs"]
        .as_array()
        .ok_or("missing definition source_outputs")?
        .clone();
    let original = take(v, "identity_ast")?;
    v["body"] = if let Some(reason) = unavailability(&original, &captured, definitions, 0)? {
        json!({"kind":"unavailable","reason":reason})
    } else {
        let mut current = original;
        query(&mut current, &captured, definitions)?;
        json!({"kind":"executable","query":current})
    };
    outputs(
        v.get_mut("source_outputs")
            .ok_or("missing definition source_outputs")?,
        definitions,
    )
}

/// Upgrade one captured semantic node using the complete retained format-1
/// closure to recover transparent saved-query match provenance.
pub fn frozen_definition(source: &Value, closure: &[Value]) -> Result<Value> {
    let mut v = source.clone();
    definition(&mut v, closure)?;
    current::<super::FrozenQueryDefinition>(v)
}
/// Convert the executable body while retaining all original identities and edges.
pub fn query_object(source: &Value) -> Result<Value> {
    let mut v = source.clone();
    let definitions = v["definitions"]
        .as_array()
        .ok_or("missing definitions")?
        .clone();
    definition(&mut v, &definitions)?;
    for d in array_mut(&mut v, "definitions")? {
        definition(d, &definitions)?;
    }
    current::<super::QueryObject>(v)
}
/// Rebind representation-dependent executable metadata. Original sealed context
/// and request/run hashes remain in the caller's retained format-1 record.
pub fn query_run(source: &Value, current_catalog: &str) -> Result<Value> {
    let mut v = source.clone();
    match v["binding"]["kind"].as_str() {
        Some("original") => {
            let r = &mut v["binding"]["execution"]["replay"];
            let definitions = r["definitions"]
                .as_array()
                .ok_or("missing replay definitions")?
                .clone();
            let captured = r["source_outputs"]
                .as_array()
                .ok_or("missing replay source_outputs")?
                .clone();
            if let Some(reason) = unavailability(&r["bound_ast"], &captured, &definitions, 0)? {
                v["binding"] = json!({"kind":"unavailable","reason":reason});
                return current::<super::QueryRun>(v);
            }
            query(
                r.get_mut("bound_ast").ok_or("missing replay bound_ast")?,
                &captured,
                &definitions,
            )?;
            outputs(
                r.get_mut("source_outputs")
                    .ok_or("missing replay source_outputs")?,
                &definitions,
            )?;
            for d in array_mut(r, "definitions")? {
                query(
                    d.get_mut("ast").ok_or("missing frozen ast")?,
                    &captured,
                    &definitions,
                )?;
            }
            let versions = r["context"]["versions"]
                .as_object_mut()
                .ok_or("missing replay versions")?;
            versions.insert("catalog".into(), json!(current_catalog));
        }
        Some("original_object") => (),
        _ => return Err("invalid format-1 run binding".into()),
    }
    current::<super::QueryRun>(v)
}

/// Retired content-position invocations retain exact source custody, with no
/// attempt to reinterpret their old positions as current page cursors.
pub fn retired_read_run(source: &Value) -> Result<Value> {
    if source.get("historical_source").is_some() {
        return Err("retired read source cannot contain nested evidence".into());
    }
    let mut v = source.clone();
    let binding = &source["binding"];
    let invocation = match binding["kind"].as_str() {
        Some("original") => {
            let hash =
                super::metadata_hash(b"polylogue-execution-binding-v1\0", &binding["execution"])
                    .map_err(|error| error.to_string())?;
            if !binding["execution"].is_object()
                || source["question"].is_null()
                || binding["execution"]["presentation"]["kind"] != "read"
            {
                return Err(
                    "retired Run2 original requires its execution and actual question".into(),
                );
            }
            json!({"kind":"original","binding_hash":hash})
        }
        Some("continuation") => {
            let root = serde_json::from_value::<crate::RunId>(binding["root"].clone())
                .map_err(|error| format!("retired read root: {error}"))?;
            if serde_json::to_value(&root).map_err(|e| e.to_string())? == source["id"] {
                return Err("retired continuation cannot name itself as its root".into());
            }
            let hash = serde_json::from_value::<crate::BlobHash>(binding["binding_hash"].clone())
                .map_err(|error| format!("retired root binding hash: {error}"))?;
            if !binding["seek"].is_object() || source["question"].is_null() {
                return Err(
                    "retired Run2 continuation requires its original position and question".into(),
                );
            }
            json!({"kind":"continuation","root":root,"binding_hash":hash,"seek":binding["seek"]})
        }
        _ => return Err("retired Run2 source has no original or continuation invocation".into()),
    };
    v["binding"] = json!({"kind":"retired_read","invocation":invocation});
    current::<super::QueryRun>(v)
}

fn question_key(value: &Value) -> Option<String> {
    let bytes = value.as_array()?;
    if bytes.len() != 32 {
        return None;
    }
    bytes
        .iter()
        .map(|b| u8::try_from(b.as_u64()?).ok().map(|b| format!("{b:02x}")))
        .collect()
}
fn argument_values_equal(a: &Value, b: &Value) -> bool {
    match (a["kind"].as_str(), b["kind"].as_str()) {
        (Some("scalar"), Some("scalar")) => a["text"] == b["text"],
        (Some("set"), Some("set")) => match (a["values"].as_array(), b["values"].as_array()) {
            (Some(a), Some(b)) => {
                a.len() == b.len() && a.iter().zip(b).all(|(a, b)| argument_values_equal(a, b))
            }
            _ => false,
        },
        _ => a == b,
    }
}
fn arguments_equal(a: &Value, b: &Value) -> bool {
    match (a.as_array(), b.as_array()) {
        (Some(a), Some(b)) => {
            a.len() == b.len()
                && a.iter().all(|a| {
                    b.iter().any(|b| {
                        a["parameter"] == b["parameter"]
                            && argument_values_equal(&a["value"], &b["value"])
                    })
                })
        }
        _ => false,
    }
}

fn source_output<'a>(source: &Value, outputs: &'a [Value]) -> Option<&'a Value> {
    outputs
        .iter()
        .find(|o| {
            let other = &o["source"];
            match source["kind"].as_str() {
                Some("saved") => {
                    other["kind"] == "saved"
                        && other["name"] == source["name"]
                        && arguments_equal(&other["arguments"], &source["arguments"])
                }
                Some("result") => {
                    other["kind"] == "result"
                        && other["reference"] == source["reference"]
                        && other["part"] == source["part"]
                }
                _ => false,
            }
        })
        .map(|o| &o["schema"])
}
fn find_definition<'a>(source: &Value, definitions: &'a [Value]) -> Option<&'a Value> {
    let name = source["name"].as_str()?;
    definitions.iter().find(|definition| {
        definition["name"].as_str() == Some(name)
            || question_key(&definition["question"]).as_deref() == Some(name)
    })
}
fn output_domain(
    query: &Value,
    outputs: &[Value],
    definitions: &[Value],
    depth: usize,
) -> Result<Option<String>> {
    if depth > definitions.len() + 1 {
        return Err("cyclic format-1 saved output closure".into());
    }
    let source = &query["source"];
    let mut domain = if source["kind"] == "unit" {
        Some(current_unit(string(source, "unit")?).to_owned())
    } else if source["kind"] == "saved" {
        if let Some(definition) = find_definition(source, definitions) {
            let ast = definition
                .get("identity_ast")
                .or_else(|| definition.get("ast"))
                .ok_or("missing captured saved AST")?;
            let nested = definition["source_outputs"]
                .as_array()
                .map_or(outputs, Vec::as_slice);
            output_domain(ast, nested, definitions, depth + 1)?
        } else {
            source_output(source, outputs)
                .and_then(|schema| schema["unit"].as_str())
                .map(str::to_owned)
        }
    } else {
        source_output(source, outputs)
            .and_then(|schema| schema["unit"].as_str())
            .map(str::to_owned)
    };
    let mut grouped = None;
    for stage in query["stages"]
        .as_array()
        .ok_or("invalid captured stages")?
    {
        let kind = stage["kind"]["kind"]
            .as_str()
            .ok_or("invalid captured stage kind")?;
        match kind {
            "scope" => domain = Some(current_unit(string(&stage["kind"], "unit")?).to_owned()),
            "group" => {
                grouped = Some(domain.take());
            }
            "aggregate" | "matches" => domain = None,
            "first" => {
                if let Some(input) = grouped.take() {
                    domain = input;
                }
            }
            _ => (),
        }
        if !matches!(kind, "group") {
            grouped = None;
        }
    }
    Ok(domain)
}

fn query_pattern(
    v: &Value,
    outputs: &[Value],
    definitions: &[Value],
    depth: usize,
) -> Result<(String, Vec<Pattern>)> {
    if depth > definitions.len() + 1 {
        return Err("cyclic format-1 saved query closure".into());
    }
    let source = &v["source"];
    let mut available = Vec::new();
    let mut parent = source["unit"].as_str().unwrap_or("").to_owned();
    if matches!(source["kind"].as_str(), Some("saved" | "result")) {
        parent = source_output(source, outputs)
            .and_then(|o| o["unit"].as_str())
            .unwrap_or("")
            .to_owned();
        if source["kind"] == "saved" {
            let name = string(source, "name")?;
            if let Some(def) = definitions.iter().find(|d| {
                d["name"].as_str() == Some(name)
                    || question_key(&d["question"]).as_deref() == Some(name)
            }) {
                let ast = def
                    .get("identity_ast")
                    .or_else(|| def.get("ast"))
                    .ok_or("missing captured saved AST")?;
                let nested_outputs = def["source_outputs"]
                    .as_array()
                    .map_or(outputs, Vec::as_slice);
                let (_, mut inherited) =
                    query_pattern(ast, nested_outputs, definitions, depth + 1)?;
                for stage in ast["stages"].as_array().ok_or("invalid captured stages")? {
                    match stage["kind"]["kind"].as_str() {
                        Some("scope") => {
                            inherited.clear();
                            patterns(&stage["kind"]["condition"], &mut inherited);
                        }
                        Some("group" | "aggregate" | "fields" | "matches") => inherited.clear(),
                        _ => (),
                    }
                }
                available = inherited;
            }
        }
    }
    patterns(&source["condition"], &mut available);
    Ok((parent, available))
}
fn arithmetic(operator: &str, left: Value, right: Value) -> Value {
    json!({"kind":"arithmetic","operator":operator,"left":left,"right":right})
}
fn numeric(n: usize) -> Value {
    literal(json!({"kind":"scalar","text":n.to_string(),"quoted":false}))
}
fn matches_projection(parent: &str, available: &[Pattern], span: &Value) -> Result<Vec<Value>> {
    let [pattern] = available else {
        return Err("format-1 matches needs one captured qualifier".into());
    };
    let stage = |kind: Value| json!({"span":span,"kind":kind});
    let matched = stage(json!({"kind":"matches"}));
    match pattern {
        Pattern::Exists(unit) => {
            let names = native_columns(unit)?;
            let mut columns = Vec::new();
            if parent != "sessions" || !names.contains(&"session") {
                if !native_columns(parent)?.contains(&"id") {
                    return Err("retired private parent storage key".into());
                }
                columns.push(column(
                    json!({"kind":"function","name":"text","arguments":[field("parent.id")]}),
                    "parent_id".into(),
                ));
            }
            columns.push(column(field("match"), "match".into()));
            columns.extend(
                names
                    .iter()
                    .map(|n| column(field(&format!("step.{n}")), (*n).into())),
            );
            Ok(vec![
                matched,
                stage(json!({"kind":"fields","columns":columns})),
            ])
        }
        Pattern::Sequence(unit, n) if unit == "tool_calls" && *n > 0 => {
            let mut aggregates = Vec::new();
            let mut columns = vec![
                column(field("session"), "session".into()),
                column(field("match"), "match".into()),
            ];
            for step in 1..=*n {
                for suffix in ["message", "command", "outcome", "at", "position"] {
                    let name = if suffix == "position" {
                        format!("__format1_step{step}_position")
                    } else {
                        format!("step{step}_{suffix}")
                    };
                    aggregates.push(json!({"function":"min","arguments":[{"kind":"function","name":"case_eq","arguments":[field("position"),numeric(step),field(&format!("step.{suffix}"))]}],"name":name}));
                    if suffix != "position" {
                        columns.push(column(field(&name), name));
                    }
                }
            }
            for gap in 1..*n {
                let prefix = if *n == 2 {
                    "gap".into()
                } else {
                    format!("gap{gap}")
                };
                let distance = arithmetic(
                    "subtract",
                    field(&format!("__format1_step{}_position", gap + 1)),
                    field(&format!("__format1_step{gap}_position")),
                );
                columns.push(column(
                    arithmetic("subtract", distance, numeric(1)),
                    format!("{prefix}_steps"),
                ));
                columns.push(column(
                    arithmetic(
                        "subtract",
                        field(&format!("step{}_at", gap + 1)),
                        field(&format!("step{gap}_at")),
                    ),
                    format!("{prefix}_time"),
                ));
            }
            Ok(vec![
                matched,
                stage(
                    json!({"kind":"group","keys":[column(field("step.session"),"session".into()),column(field("match"),"match".into())]}),
                ),
                stage(json!({"kind":"aggregate","aggregates":aggregates})),
                stage(json!({"kind":"fields","columns":columns})),
                stage(
                    json!({"kind":"sort","keys":[{"expression":field("session"),"descending":false},{"expression":field("match"),"descending":false}]}),
                ),
            ])
        }
        _ => Err(
            "format-1 sequence matches outside tool_calls is not a predecessor executable relation"
                .into(),
        ),
    }
}

// Native emitted columns frozen from the format-1 Polylogue unit registry.
// Filter-only catalog fields were never produced by these relations.
fn native_columns(unit: &str) -> Result<&'static [&'static str]> {
    Ok(match unit {
        "sessions" | "session" => &[
            "id",
            "origin",
            "kind",
            "title",
            "started",
            "last_active",
            "cwd",
            "repo",
        ],
        "documents" | "document" => &[
            "id",
            "origin",
            "purpose",
            "title",
            "source_time",
            "authorship",
        ],
        "messages" | "message" => &["id", "session", "role", "written_by", "at", "model"],
        "blocks" | "block" => &["session", "message", "kind", "idx", "at"],
        "tool_calls" | "tool_call" => &[
            "session",
            "message",
            "tool",
            "action",
            "command",
            "path",
            "outcome",
            "exit_code",
            "at",
        ],
        "files" | "file" => &[
            "session", "path", "reads", "writes", "edits", "first_at", "last_at",
        ],
        "events" | "event" => &["id", "session", "input", "source", "kind", "at", "data"],
        "notes" | "note" => &[
            "id",
            "kind",
            "key",
            "value",
            "author",
            "created_at",
            "first_note",
            "target",
        ],
        "assertions" | "assertion" => &[
            "reference",
            "id",
            "kind",
            "key",
            "value",
            "author",
            "created_at",
            "root",
            "about",
        ],
        "links" | "link" => &["parent", "child", "kind", "how", "evidence", "pending"],
        "turns" | "turn" => &[
            "id",
            "session",
            "start",
            "end",
            "burst",
            "prompts",
            "slash_command",
            "answer",
            "attempts",
            "path",
            "latency",
            "tool_calls",
            "errors",
            "input_tokens",
            "output_tokens",
            "cost",
            "abandoned",
        ],
        "deliveries" | "delivery" => &["id", "recipient", "event", "outcome", "at", "bytes"],
        "live_intervals" | "live_interval" => &[
            "session",
            "aspect",
            "value",
            "start",
            "end",
            "duration_ms",
            "closed_by",
        ],
        "processes" | "process" => &[
            "session",
            "boot_id",
            "pid",
            "start",
            "parent_pid",
            "comm",
            "argv",
            "cwd",
            "started_at",
            "ended_at",
            "role",
            "tool_call",
        ],
        "changes" | "change" => &["kind", "ref", "seq", "live_changed", "assertion_kind"],
        _ => return Err(format!("unknown format-1 native unit {unit}")),
    })
}

#[derive(Clone)]
enum Pattern {
    Exists(String),
    Sequence(String, usize),
}
fn patterns(v: &Value, out: &mut Vec<Pattern>) {
    let k = &v["kind"];
    match k["kind"].as_str() {
        Some("and") => {
            if let Some(cs) = k["conditions"].as_array() {
                for c in cs {
                    patterns(c, out);
                }
            }
        }
        Some("exists") => {
            if let Some(u) = k["unit"].as_str() {
                out.push(Pattern::Exists(u.into()));
            }
        }
        Some("sequence") => {
            if let (Some(u), Some(cs)) = (k["unit"].as_str(), k["steps"].as_array()) {
                out.push(Pattern::Sequence(u.into(), cs.len()));
            }
        }
        _ => (),
    }
}

fn historical_field(field: &Value) -> Result<historical_query_format_v1::catalog::Field> {
    let mut f = field.clone();
    fn kind(k: &mut Value) {
        if k == "exact" {
            *k = json!("match");
        } else if let Some(inner) = k.get_mut("list") {
            kind(inner);
        }
    }
    kind(&mut f["kind"]);
    if f.get("read_as").is_none() {
        f["read_as"] = json!("");
    }
    if f.get("unit").is_none() {
        f["unit"] = Value::Null;
    }
    serde_json::from_value(f).map_err(|e| format!("format-1 field projection: {e}"))
}
fn historical_catalog(catalog: &Catalog) -> Result<historical_query_format_v1::catalog::Catalog> {
    use historical_query_format_v1::catalog as old;
    let mut units = Vec::new();
    for unit in &catalog.units {
        units.push(old::Unit {
            name: unit.name.clone(),
            sequence_unit: unit.sequence_unit.clone(),
            aliases: unit
                .singular
                .iter()
                .cloned()
                .chain(std::iter::once(format!("pl.{}", unit.name)))
                .chain(unit.singular.iter().map(|name| format!("pl.{name}")))
                .collect(),
            fields: unit
                .fields
                .iter()
                .map(|f| historical_field(&serde_json::to_value(f).map_err(|e| e.to_string())?))
                .collect::<Result<_>>()?,
            defaults: Vec::new(),
            default_order: unit.default_order.clone(),
            identity: unit.identity.clone(),
            main_time: unit.main_time.clone(),
            help: String::new(),
        });
    }
    if let Some(imported) = units
        .iter()
        .find(|unit| unit.name == "imported_notes")
        .cloned()
    {
        let mut notes = imported;
        notes.name = "notes".into();
        notes.aliases = vec!["note".into(), "pl.notes".into(), "pl.note".into()];
        notes.fields.retain(|field| {
            [
                "id",
                "kind",
                "key",
                "value",
                "author",
                "agent",
                "judged",
                "target",
                "write_counter",
                "created_at",
                "first_note",
                "current",
                "withdrawn",
                "since",
                "evidence_until",
                "until",
            ]
            .contains(&field.name.as_str())
        });
        for field in &mut notes.fields {
            if field.unit.as_deref() == Some("imported_notes") {
                field.unit = Some("notes".into());
            }
        }
        units.push(notes);
    }
    let mut source_outputs = Vec::new();
    for output in &catalog.source_outputs {
        let source = serde_json::from_value(
            serde_json::to_value(&output.source).map_err(|e| e.to_string())?,
        )
        .map_err(|e| format!("format-1 output source: {e}"))?;
        source_outputs.push(old::SourceOutput {
            source,
            relation_scope: None,
            schema: old::OutputSchema {
                unit: output.schema.unit.clone(),
                main_time: output.schema.main_time.clone(),
                match_patterns: Vec::new(),
                fields: output
                    .schema
                    .fields
                    .iter()
                    .map(|f| historical_field(&serde_json::to_value(f).map_err(|e| e.to_string())?))
                    .collect::<Result<_>>()?,
            },
        });
    }
    Ok(old::Catalog {
        default_unit: catalog.default_unit.clone(),
        recall_last_unit: None,
        units,
        commands: Vec::new(),
        extensions: Vec::new(),
        references: Vec::new(),
        source_outputs,
        match_outputs: Vec::new(),
        unpiped_stages: false,
    })
}
fn historical_matches(
    ast: &Value,
    catalog: &mut historical_query_format_v1::catalog::Catalog,
    outputs: &[Value],
    closure: &[Value],
) -> Result<()> {
    use historical_query_format_v1::catalog as old;
    fn collect(v: &Value, queries: &mut Vec<Value>) {
        match v {
            Value::Object(fs) => {
                if fs.contains_key("source") && fs.contains_key("stages") {
                    queries.push(v.clone());
                }
                for v in fs.values() {
                    collect(v, queries);
                }
            }
            Value::Array(vs) => {
                for v in vs {
                    collect(v, queries);
                }
            }
            _ => (),
        }
    }
    let mut queries = Vec::new();
    collect(ast, &mut queries);
    for query in queries {
        let (mut parent, mut available) = query_pattern(&query, outputs, closure, 0)?;
        for stage in query["stages"]
            .as_array()
            .ok_or("invalid format-1 stages")?
        {
            let k = &stage["kind"];
            match k["kind"].as_str() {
                Some("scope") => {
                    parent = string(k, "unit")?.into();
                    available.clear();
                    patterns(&k["condition"], &mut available);
                }
                Some("group" | "aggregate" | "fields") => available.clear(),
                Some("matches") => {
                    let [pattern] = available.as_slice() else {
                        continue;
                    };
                    let mut fields = Vec::new();
                    let field = |name: &str, kind| old::Field {
                        name: name.into(),
                        kind,
                        ..old::Field::default()
                    };
                    let pattern = match pattern {
                        Pattern::Exists(unit) => {
                            let names = native_columns(unit)?;
                            if parent != "sessions" || !names.contains(&"session") {
                                fields.push(field("parent_id", old::FieldKind::Text));
                            }
                            fields.push(field("match", old::FieldKind::Number));
                            for name in names {
                                if let Some(f) = catalog
                                    .units
                                    .iter()
                                    .find(|u| u.name == *unit || u.aliases.contains(unit))
                                    .and_then(|u| u.fields.iter().find(|f| f.name == *name))
                                {
                                    fields.push(f.clone());
                                }
                            }
                            old::MatchPattern::Exists { unit: unit.clone() }
                        }
                        Pattern::Sequence(unit, n) => {
                            fields.push(field("session", old::FieldKind::Reference));
                            fields.push(field("match", old::FieldKind::Number));
                            if unit == "tool_calls" {
                                for step in 1..=*n {
                                    for name in ["message", "command", "outcome", "at"] {
                                        if let Some(native) = catalog
                                            .units
                                            .iter()
                                            .find(|u| u.name == "tool_calls")
                                            .and_then(|u| u.fields.iter().find(|f| f.name == name))
                                        {
                                            let mut f = native.clone();
                                            f.name = format!("step{step}_{name}");
                                            fields.push(f);
                                        }
                                    }
                                }
                                for gap in 1..*n {
                                    let prefix = if *n == 2 {
                                        "gap".into()
                                    } else {
                                        format!("gap{gap}")
                                    };
                                    fields.push(field(
                                        &format!("{prefix}_steps"),
                                        old::FieldKind::Number,
                                    ));
                                    fields.push(field(
                                        &format!("{prefix}_time"),
                                        old::FieldKind::Duration,
                                    ));
                                }
                            }
                            old::MatchPattern::Sequence {
                                unit: unit.clone(),
                                steps: *n,
                            }
                        }
                    };
                    catalog.match_outputs.push(old::MatchOutput {
                        source: old::MatchSource {
                            parent_unit: parent.clone(),
                            pattern,
                        },
                        schema: old::OutputSchema {
                            unit: Some(parent.clone()),
                            fields,
                            main_time: None,
                            match_patterns: Vec::new(),
                        },
                    });
                    available.clear();
                }
                _ => (),
            }
        }
    }
    Ok(())
}
/// Reproduce the exact frozen format-1 identity encoder on its original AST.
pub fn legacy_normal_form(ast: &Value, catalog: &Catalog) -> Result<Value> {
    legacy_normal_form_with_closure(ast, catalog, &[])
}
/// The closure supplies the original qualifier provenance of saved sources.
pub fn legacy_normal_form_with_closure(
    ast: &Value,
    catalog: &Catalog,
    closure: &[Value],
) -> Result<Value> {
    let query = serde_json::from_value::<historical_query_format_v1::ast::Query>(ast.clone())
        .map_err(|e| format!("format-1 AST: {e}"))?;
    let outputs = serde_json::to_value(&catalog.source_outputs).map_err(|e| e.to_string())?;
    let mut catalog = historical_catalog(catalog)?;
    historical_matches(
        ast,
        &mut catalog,
        outputs.as_array().ok_or("invalid catalog outputs")?,
        closure,
    )?;
    historical_query_format_v1::identity::normal_form(&query, &catalog).map_err(|e| e.to_string())
}

/// Verify original identity using its original captured output declarations,
/// independently of the executable relation's evolved row domain.
pub fn legacy_definition_normal_form(
    source: &Value,
    catalog: &Catalog,
    closure: &[Value],
) -> Result<Value> {
    let ast = &source["identity_ast"];
    let query = serde_json::from_value::<historical_query_format_v1::ast::Query>(ast.clone())
        .map_err(|error| format!("format-1 AST: {error}"))?;
    let outputs = source["source_outputs"]
        .as_array()
        .ok_or("missing original captured source outputs")?;
    let mut catalog = historical_catalog(catalog)?;
    catalog.source_outputs = serde_json::from_value(json!(outputs))
        .map_err(|error| format!("format-1 output declarations: {error}"))?;
    historical_matches(ast, &mut catalog, outputs, closure)?;
    historical_query_format_v1::identity::normal_form(&query, &catalog)
        .map_err(|error| error.to_string())
}

#[cfg(test)]
mod tests;
