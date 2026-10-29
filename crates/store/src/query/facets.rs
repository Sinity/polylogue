//! Per-session observed values from the same field expressions as query lowering.
use super::{compile, fields, units};
use crate::{Reader, SqlValue, StoreError};
use polylogue_model::SessionId;
use polylogue_query::{ColKind, QueryError, ast::Unit};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct FacetKey {
    pub unit: Unit,
    pub field: String,
}
impl Ord for FacetKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (self.unit.name(), &self.field).cmp(&(other.unit.name(), &other.field))
    }
}
impl PartialOrd for FacetKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl FacetKey {
    pub fn new(unit: Unit, field: impl Into<String>) -> Self {
        Self {
            unit,
            field: field.into(),
        }
    }
}
pub type FacetCounts = BTreeMap<FacetKey, BTreeMap<String, u64>>;
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FacetCount {
    pub key: FacetKey,
    pub value: String,
    pub count: u64,
}
#[derive(Debug, thiserror::Error)]
pub enum FacetError {
    #[error(transparent)]
    Store(#[from] StoreError),
    #[error(transparent)]
    Query(#[from] QueryError),
}
fn invalid(message: impl Into<String>) -> FacetError {
    QueryError::Invalid {
        message: message.into(),
        position: 0,
    }
    .into()
}

impl Reader<'_> {
    /// Native rows are counted within one physical session. Model/tool
    /// collections retain their child grain; cwd ancestors count the session
    /// once. An instant is supplied explicitly; no day policy is needed.
    pub fn visit_session_facets<E>(
        &self,
        session: &SessionId,
        keys: &[FacetKey],
        now: jiff::Timestamp,
        mut visit: impl FnMut(FacetCount) -> Result<(), E>,
    ) -> Result<(), E>
    where
        E: From<FacetError>,
    {
        self.check().map_err(FacetError::from).map_err(E::from)?;
        let mut distinct = BTreeSet::new();
        for key in keys {
            if !distinct.insert(key) {
                return Err(E::from(invalid("duplicate facet field")));
            }
        }
        // This is a transfer batch below SQLite's compound-SELECT bound;
        // every requested field is visited, with no answer-size limit.
        for keys in keys.chunks(100) {
            let branches = keys
                .iter()
                .enumerate()
                .map(|(index, key)| branch(index, key, now))
                .collect::<Result<Vec<_>, _>>()
                .map_err(E::from)?;
            let result = self.query_each(
                &branches.join(" UNION ALL "),
                &[SqlValue::Text(session.to_string())],
                |row| -> Result<(), FacetVisitError<E>> {
                    let key = row["field"]
                        .as_u64()
                        .and_then(|i| usize::try_from(i).ok())
                        .and_then(|i| keys.get(i))
                        .ok_or_else(|| {
                            FacetVisitError::Facet(invalid("facet returned an invalid field"))
                        })?;
                    let value = row["value"].as_str().ok_or_else(|| {
                        FacetVisitError::Facet(invalid("facet returned a non-text value"))
                    })?;
                    let count = row["n"].as_u64().ok_or_else(|| {
                        FacetVisitError::Facet(invalid("facet returned an invalid count"))
                    })?;
                    visit(FacetCount {
                        key: key.clone(),
                        value: value.into(),
                        count,
                    })
                    .map_err(FacetVisitError::Consumer)
                },
            );
            match result {
                Ok(()) => {}
                Err(FacetVisitError::Consumer(error)) => return Err(error),
                Err(FacetVisitError::Facet(error)) => return Err(E::from(error)),
            }
        }
        Ok(())
    }

    /// Collect values when the caller needs a map. Publishers use the
    /// visitor so a session with many distinct values does not form one buffer.
    pub fn session_facets(
        &self,
        session: &SessionId,
        keys: &[FacetKey],
        now: jiff::Timestamp,
    ) -> Result<FacetCounts, FacetError> {
        let mut out = keys
            .iter()
            .cloned()
            .map(|key| (key, BTreeMap::new()))
            .collect::<FacetCounts>();
        self.visit_session_facets::<FacetError>(session, keys, now, |row| {
            out.get_mut(&row.key)
                .expect("requested facet")
                .insert(row.value, row.count);
            Ok(())
        })?;
        Ok(out)
    }
}

fn branch(index: usize, key: &FacetKey, now: jiff::Timestamp) -> Result<String, FacetError> {
    let (unit, name) = match (key.unit, key.field.as_str()) {
        (Unit::Sessions, "model") => (Unit::Messages, "model"),
        (Unit::Sessions, "tool") => (Unit::ToolCalls, "tool"),
        _ => (key.unit, key.field.as_str()),
    };
    let spec = units::spec(unit);
    let aliases = fields::Aliases::plain();
    if unit != Unit::Sessions && !spec.in_session {
        return Err(invalid(format!(
            "{} has no per-session row domain",
            unit.name()
        )));
    }
    if unit == Unit::Sessions && name == "cwd" {
        // Prefixes are derived from native slash components. SQL owns the
        // distinct set and can spill it; Rust keeps one path at a time.
        let directories = fields::working_directories("s");
        return Ok(format!(
            "SELECT {index} AS field,value,1 AS n FROM (WITH RECURSIVE native(value) AS (SELECT DISTINCT cj.value FROM session s,{directories} cj WHERE s.id=?1 AND cj.value IS NOT NULL), prefixes(full,n,value) AS (SELECT value,0,'' FROM native UNION ALL SELECT full,n+instr(substr(full,n+1),'/'),substr(full,1,n+instr(substr(full,n+1),'/')-1) FROM prefixes WHERE instr(substr(full,n+1),'/')>0) SELECT value FROM native UNION SELECT value FROM prefixes WHERE value<>'' UNION SELECT '/' WHERE EXISTS(SELECT 1 FROM native WHERE substr(value,1,1)='/'))"
        ));
    }
    let projection = compile::field_expression(unit, name, now)?;
    if matches!(projection.kind, ColKind::Json | ColKind::Ids) {
        return Err(invalid(format!(
            "{}.{name} is not a scalar facet",
            unit.name()
        )));
    }
    let own = if matches!(unit, Unit::Messages | Unit::Blocks | Unit::ToolCalls) {
        format!(
            " AND {}",
            crate::session_sql::own_messages(&aliases.m, None)
        )
    } else {
        String::new()
    };
    Ok(format!(
        "SELECT {index} AS field,CAST({expr} AS TEXT) AS value,count(*) AS n FROM {from} WHERE s.id=?1{own} AND ({expr}) IS NOT NULL GROUP BY ({expr})",
        expr = projection.sql,
        from = spec.from(&aliases)
    ))
}

enum FacetVisitError<E> {
    Consumer(E),
    Facet(FacetError),
}
impl<E> From<StoreError> for FacetVisitError<E> {
    fn from(error: StoreError) -> Self {
        Self::Facet(error.into())
    }
}
