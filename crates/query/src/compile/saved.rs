//! Saved definition lookup and typed parameter substitution.
use super::*;
pub fn saved_definition<'o>(
    text: &str,
    position: usize,
    name: &str,
    opts: &'o impl Definitions,
) -> Result<&'o SavedQuery, QueryError> {
    match opts.saved_queries().get(name).map(Vec::as_slice) {
        Some([one]) if one.conflicts.is_empty() => {
            one.executable_text()?;
            Ok(one)
        }
        Some([one]) => Err(QueryError::Ambiguous {
            what: format!("query:{name}"),
            candidates: one.conflicts.iter().map(ToString::to_string).collect(),
        }),
        Some(several @ [_, _, ..]) => Err(QueryError::Ambiguous {
            what: format!("query:{name}"),
            candidates: several
                .iter()
                .map(|saved| {
                    polylogue_model::Reference::Assertion(
                        polylogue_model::RevisionReference::exact(
                            saved.root.clone(),
                            saved.revision.clone(),
                        ),
                    )
                    .to_string()
                })
                .collect(),
        }),
        _ => {
            let known: Vec<&str> = opts.saved_queries().keys().map(String::as_str).collect();
            Err(QueryError::syntax(
                text,
                position,
                format!(
                    "no saved query named {name:?}{}",
                    if known.is_empty() {
                        String::new()
                    } else {
                        format!(" (saved: {})", known.join(", "))
                    }
                ),
                None,
            ))
        }
    }
}

/// A saved query's parameters: each `$word` standing for a value in its
/// text is replaced by the value the call gives, in the parsed query, so a
/// value never changes the query's shape.
pub struct Parameters {
    name: String,
    arguments: Vec<Argument>,
    collect_only: bool,
}

impl Parameters {
    pub fn new(name: &str, arguments: &[Argument]) -> Self {
        Self {
            name: name.to_owned(),
            arguments: arguments.to_vec(),
            collect_only: false,
        }
    }

    pub fn names(query: &mut Query) -> Result<Vec<String>, QueryError> {
        let mut collector = Self::new("", &[]);
        collector.collect_only = true;
        let mut names = vec![];
        collector.query(query, &mut names, 0)?;
        Ok(names)
    }

    pub fn bind(&self, query: &mut Query, position: usize) -> Result<(), QueryError> {
        let mut used = vec![];
        self.query(query, &mut used, position)?;
        if let Some(unused) = self.arguments.iter().find(|a| !used.contains(&a.parameter)) {
            return Err(QueryError::Invalid {
                message: format!(
                    "query:{} has no parameter {}{}",
                    self.name,
                    unused.parameter,
                    if used.is_empty() {
                        String::new()
                    } else {
                        format!(" (its parameters: {})", used.join(", "))
                    }
                ),
                position,
            });
        }
        Ok(())
    }

    fn query(
        &self,
        query: &mut Query,
        used: &mut Vec<String>,
        position: usize,
    ) -> Result<(), QueryError> {
        query_language::depth::deeper(|| {
            match &mut query.source {
                Source::Unit { condition, .. } | Source::Result { condition, .. } => {
                    if let Some(condition) = condition {
                        self.cond(condition, used, position)?;
                    }
                }
                Source::Saved {
                    arguments,
                    condition,
                    ..
                } => {
                    for argument in arguments {
                        self.value(&mut argument.value, used, position)?;
                    }
                    if let Some(condition) = condition {
                        self.cond(condition, used, position)?;
                    }
                }
            }
            for child in &mut query.attach {
                for window in &mut child.windows {
                    if let Window::Where(condition) = window {
                        self.cond(condition, used, position)?;
                    }
                }
            }
            for transform in &mut query.transforms {
                match &mut transform.kind {
                    TransformKind::Where { condition }
                    | TransformKind::Scope {
                        condition: Some(condition),
                        ..
                    } => self.cond(condition, used, position)?,
                    TransformKind::Combine { query, .. } | TransformKind::Diff { query } => {
                        self.query(query, used, position)?
                    }
                    TransformKind::Join { query, on } => {
                        self.query(query, used, position)?;
                        if let JoinOn::Order {
                            within: Some(within),
                            ..
                        } = on
                        {
                            self.distance(within, used, position)?;
                        }
                    }
                    TransformKind::Group { keys: columns } | TransformKind::Fields { columns } => {
                        for column in columns {
                            self.expression(&mut column.expression, used, position)?;
                        }
                    }
                    TransformKind::Sort { keys } => {
                        for key in keys {
                            self.expression(&mut key.expression, used, position)?;
                        }
                    }
                    TransformKind::Aggregate { aggregates } => {
                        for aggregate in aggregates {
                            for expression in &mut aggregate.arguments {
                                self.expression(expression, used, position)?;
                            }
                        }
                    }
                    TransformKind::AsOf { value, .. } => self.value(value, used, position)?,
                    TransformKind::Extension { arguments, .. } => {
                        self.extension(arguments, used, position)?
                    }
                    _ => {}
                }
            }
            Ok(())
        })
    }

    fn cond(
        &self,
        c: &mut Condition,
        used: &mut Vec<String>,
        position: usize,
    ) -> Result<(), QueryError> {
        query_language::depth::deeper(|| {
            match &mut c.kind {
                ConditionKind::And { conditions } | ConditionKind::Or { conditions } => {
                    for condition in conditions {
                        self.cond(condition, used, position)?;
                    }
                }
                ConditionKind::Sequence { steps, gaps, .. } => {
                    for condition in steps {
                        self.cond(condition, used, position)?;
                    }
                    for gap in gaps {
                        if let Gap::Within { distance } = gap {
                            self.distance(distance, used, position)?;
                        }
                    }
                }
                ConditionKind::Not { condition }
                | ConditionKind::Exists {
                    condition: Some(condition),
                    ..
                }
                | ConditionKind::Count {
                    condition: Some(condition),
                    ..
                } => self.cond(condition, used, position)?,
                ConditionKind::In { left, query } => {
                    self.expression(left, used, position)?;
                    self.query(query, used, position)?;
                }
                ConditionKind::Order {
                    query,
                    within,
                    pick,
                    ..
                } => {
                    self.query(query, used, position)?;
                    if let Some(within) = within {
                        self.distance(within, used, position)?;
                    }
                    if let MatchChoice::FirstBy { keys } = pick {
                        for key in keys {
                            self.expression(&mut key.expression, used, position)?;
                        }
                    }
                }
                ConditionKind::Compare { left, right, .. } => {
                    self.expression(left, used, position)?;
                    self.expression(right, used, position)?;
                }
                ConditionKind::Match { value, .. }
                | ConditionKind::Since { value, .. }
                | ConditionKind::Until { value, .. }
                | ConditionKind::Rank { value, .. } => self.value(value, used, position)?,
                ConditionKind::TimeRange { left, value } => {
                    self.expression(left, used, position)?;
                    self.value(value, used, position)?;
                }
                ConditionKind::Between { left, from, to } => {
                    self.expression(left, used, position)?;
                    self.value(from, used, position)?;
                    self.value(to, used, position)?;
                }
                ConditionKind::Text { text, .. } => {
                    if let Some(argument) = self.argument(text, used, position)? {
                        *text = value_text(&argument.value)?;
                    }
                }
                ConditionKind::Saved { arguments, .. } => {
                    for argument in arguments {
                        self.value(&mut argument.value, used, position)?;
                    }
                }
                ConditionKind::Extension { arguments, .. } => {
                    self.extension(arguments, used, position)?
                }
                _ => {}
            }
            Ok(())
        })
    }

    fn expression(
        &self,
        expression: &mut Expression,
        used: &mut Vec<String>,
        position: usize,
    ) -> Result<(), QueryError> {
        query_language::depth::deeper(|| {
            match expression {
                Expression::Value { value } => self.value(value, used, position)?,
                Expression::Function { arguments, .. } => {
                    for argument in arguments {
                        self.expression(argument, used, position)?;
                    }
                }
                Expression::Arithmetic { left, right, .. } => {
                    self.expression(left, used, position)?;
                    self.expression(right, used, position)?;
                }
                Expression::Field { .. } => {}
            }
            Ok(())
        })
    }

    fn distance(
        &self,
        distance: &mut Distance,
        used: &mut Vec<String>,
        position: usize,
    ) -> Result<(), QueryError> {
        if let Distance::Time { duration } = distance {
            self.value(duration, used, position)?;
        }
        Ok(())
    }

    fn extension(
        &self,
        arguments: &mut [ExtensionArgument],
        used: &mut Vec<String>,
        position: usize,
    ) -> Result<(), QueryError> {
        for argument in arguments {
            match argument {
                ExtensionArgument::Value(value) => self.value(value, used, position)?,
                ExtensionArgument::Expression(expression) => {
                    self.expression(expression, used, position)?
                }
                ExtensionArgument::Query(query) => self.query(query, used, position)?,
                ExtensionArgument::Condition(condition) => self.cond(condition, used, position)?,
                ExtensionArgument::Optional(arguments) | ExtensionArgument::List(arguments) => {
                    self.extension(arguments, used, position)?
                }
                _ => {}
            }
        }
        Ok(())
    }
    fn value(
        &self,
        value: &mut Value,
        used: &mut Vec<String>,
        position: usize,
    ) -> Result<(), QueryError> {
        query_language::depth::deeper(|| {
            match value {
                Value::Parameter { name } => {
                    let marker = format!("${name}");
                    if let Some(argument) = self.argument(&marker, used, position)? {
                        *value = argument.value.clone();
                    }
                }
                Value::Scalar {
                    text,
                    quoted: false,
                } => {
                    if let Some(argument) = self.argument(text, used, position)? {
                        *value = argument.value.clone();
                    }
                }
                Value::Set { values } => {
                    for value in values {
                        self.value(value, used, position)?;
                    }
                }
                _ => {}
            }
            Ok(())
        })
    }

    /// The value given for `$word`, or `None` when `text` is no parameter.
    fn argument(
        &self,
        text: &str,
        used: &mut Vec<String>,
        position: usize,
    ) -> Result<Option<&Argument>, QueryError> {
        let Some(parameter) = parameter_name(text) else {
            return Ok(None);
        };
        if !used.iter().any(|u| u == parameter) {
            used.push(parameter.to_owned());
        }
        if self.collect_only {
            return Ok(None);
        }
        let argument = self
            .arguments
            .iter()
            .find(|a| a.parameter == parameter)
            .ok_or_else(|| QueryError::Invalid {
                message: format!(
                    "query:{} needs a value for {parameter}: `query:{}({parameter}=…)`",
                    self.name, self.name
                ),
                position,
            })?;
        Ok(Some(argument))
    }
}

/// `$name`: a parameter of a saved query.
fn parameter_name(text: &str) -> Option<&str> {
    let name = text.strip_prefix('$')?;
    let mut chars = name.chars();
    (chars.next()?.is_ascii_alphabetic() && chars.all(|c| c.is_ascii_alphanumeric() || c == '_'))
        .then_some(name)
}
