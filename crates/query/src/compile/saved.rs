//! Saved definition lookup and typed parameter substitution.
use super::*;
pub fn saved_definition<'o>(
    text: &str,
    position: usize,
    name: &str,
    opts: &'o impl Definitions,
) -> Result<&'o SavedQuery, QueryError> {
    match opts.saved_queries().get(name).map(Vec::as_slice) {
        Some([one]) => Ok(one),
        Some(several @ [_, _, ..]) => Err(QueryError::Invalid {
            message: format!(
                "query:{name} names {} saved queries ({}); withdraw or rename all but one",
                several.len(),
                several
                    .iter()
                    .map(|s| format!("note:{}", s.first_note))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            position,
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
        match &mut query.source {
            Source::Unit {
                condition: Some(condition),
                ..
            }
            | Source::RecallLast {
                condition: Some(condition),
                ..
            }
            | Source::Result {
                condition: Some(condition),
                ..
            } => self.cond(condition, used, position)?,
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
            _ => {}
        }
        for child in &mut query.attach {
            for window in &mut child.windows {
                if let Window::Where(cond) = window {
                    self.cond(cond, used, position)?;
                }
            }
        }
        for stage in &mut query.stages {
            match &mut stage.kind {
                StageKind::Where { condition: cond }
                | StageKind::Scope {
                    condition: Some(cond),
                    ..
                } => self.cond(cond, used, position)?,
                StageKind::Combine { query, .. } => self.query(query, used, position)?,
                _ => {}
            }
        }
        Ok(())
    }

    fn cond(
        &self,
        c: &mut Condition,
        used: &mut Vec<String>,
        position: usize,
    ) -> Result<(), QueryError> {
        match &mut c.kind {
            ConditionKind::And { conditions }
            | ConditionKind::Or { conditions }
            | ConditionKind::Sequence {
                steps: conditions, ..
            } => {
                for condition in conditions {
                    self.cond(condition, used, position)?;
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
            ConditionKind::In { query, .. } => self.query(query, used, position)?,
            ConditionKind::Match { value, .. }
            | ConditionKind::Compare { value, .. }
            | ConditionKind::Since { value, .. }
            | ConditionKind::Until { value, .. }
            | ConditionKind::Similar { value }
            | ConditionKind::TimeRange { value, .. } => self.value(value, used, position)?,
            ConditionKind::Between { from, to, .. } => {
                self.value(from, used, position)?;
                self.value(to, used, position)?;
            }
            ConditionKind::Text { text, mode } => {
                if let Some(argument) = self.argument(text, used, position)? {
                    *text = value_text(&argument.value)?.into();
                    *mode = if matches!(argument.value, Value::Scalar { quoted: true, .. }) {
                        TextMode::Phrase
                    } else {
                        TextMode::Word
                    };
                }
            }
            ConditionKind::Saved { arguments, .. } => {
                for argument in arguments {
                    self.value(&mut argument.value, used, position)?;
                }
            }
            _ => {}
        }
        Ok(())
    }
    fn value(
        &self,
        value: &mut Value,
        used: &mut Vec<String>,
        position: usize,
    ) -> Result<(), QueryError> {
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
