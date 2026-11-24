//! The default view each unit applies unless a query lifts it, declared once and
//! bound from a source's own top-level conditions (docs/query.md, Defaults).
//! The store lowers each identity to a predicate or a population fold.
use crate::ast::{Condition, ConditionKind, Expression, TextMode, Unit};
use std::collections::HashSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DefaultId {
    /// Continuations, subagents and forks fold into their logical session.
    LogicalSessions,
    /// Stub sessions stay out.
    NotStub,
    /// Hidden sessions stay out of a sessions list.
    SessionsVisible,
    /// Duplicate copies of a message stay out.
    NotDuplicate,
    /// Messages off the active path stay out; an unknown path stays in.
    OnActivePath,
    /// Wrapper calls whose children are counted stay out.
    CountedCalls,
    /// Rows of hidden sessions stay out.
    Visible,
    /// Withdrawals and read marks stay out of annotations.
    NotWithdrawalOrSeen,
    /// Only current revisions.
    CurrentAnnotation,
    /// Only the current nonwithdrawn alternate revisions.
    CurrentAlternate,
    /// Read marks are absent from the ordinary feed.
    NotSeen,
    /// Withdrawn annotations stay out.
    NotWithdrawn,
    /// Deliveries of hidden sessions stay out; deliveries without a session
    /// stay in.
    VisibleIfSession,
}

impl DefaultId {
    /// The top-level fields that lift this default.
    pub fn lifted_by(self) -> &'static [&'static str] {
        match self {
            Self::LogicalSessions => &[
                "physical",
                "kind",
                "parent",
                "root",
                "depth",
                "connected",
                "link",
                "id",
            ],
            Self::NotStub => &["source"],
            Self::SessionsVisible => &["hidden", "id"],
            Self::NotDuplicate => &["duplicate"],
            Self::OnActivePath => &["active"],
            Self::CountedCalls => &["action"],
            Self::Visible => &["hidden"],
            Self::NotWithdrawalOrSeen => &["kind", "note", "annotation"],
            Self::NotSeen => &["annotation_kind"],
            Self::CurrentAlternate => &["current", "root", "withdrawn"],
            Self::CurrentAnnotation => &[
                "current",
                "root",
                "withdrawn",
                "note",
                "annotation",
                "seen_by",
            ],
            Self::NotWithdrawn => &["withdrawn", "note", "annotation", "seen_by"],
            Self::VisibleIfSession => &[],
        }
    }

    /// A reference to the query's own rows lifts this default for its match.
    pub fn lifted_by_reference(self) -> bool {
        matches!(
            self,
            Self::LogicalSessions
                | Self::SessionsVisible
                | Self::NotDuplicate
                | Self::OnActivePath
                | Self::Visible
        )
    }

    /// Whether this default keeps hidden sessions out, which a measured
    /// population never does.
    fn hides(self) -> bool {
        self.lifted_by().contains(&"hidden")
    }

    /// What the default keeps out and what brings those rows back, for
    /// `--why`.
    pub fn describe(self) -> String {
        if self == Self::LogicalSessions {
            let fields: Vec<String> = self.lifted_by().iter().map(|f| format!("{f}:")).collect();
            return format!(
                "default: continuations, subagents and forks fold into their logical session (a top-level {} or a reference lifts it)",
                fields.join(", ")
            );
        }
        let mut back: Vec<String> = self.lifted_by().iter().map(|f| format!("{f}:")).collect();
        if self.lifted_by_reference() {
            back.push("a reference".to_owned());
        }
        if back.is_empty() {
            "default".to_owned()
        } else {
            format!("default (a top-level {} lifts it)", back.join(", "))
        }
    }
}

/// The defaults a unit declares, in the order they apply.
pub fn declared(unit: Unit) -> &'static [DefaultId] {
    use DefaultId::*;
    match unit {
        Unit::Sessions => &[LogicalSessions, NotStub, SessionsVisible],
        Unit::Messages | Unit::Blocks => &[NotDuplicate, OnActivePath, Visible],
        Unit::ToolCalls => &[NotDuplicate, OnActivePath, CountedCalls, Visible],
        Unit::Files | Unit::Events | Unit::Turns | Unit::LiveIntervals | Unit::Processes => {
            &[Visible]
        }
        Unit::Annotations => &[NotWithdrawalOrSeen, CurrentAnnotation, NotWithdrawn],
        Unit::Alternates => &[CurrentAlternate],
        Unit::Feed => &[NotSeen],
        Unit::Deliveries => &[VisibleIfSession],
        Unit::Documents | Unit::SessionLinks | Unit::Queries | Unit::Runs | Unit::Results => &[],
    }
}

/// The defaults a selection binds by its own top-level conditions, for a
/// population as listed and for one a `| measure` reads, which includes
/// hidden sessions.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Defaults {
    pub plain: Vec<DefaultId>,
    pub measured: Vec<DefaultId>,
}

impl Defaults {
    pub fn bind(unit: Unit, top: &[&Condition], origins: &[String]) -> Self {
        Self {
            plain: bind(unit, top, origins, false),
            measured: bind(unit, top, origins, true),
        }
    }

    /// The defaults that apply where a measured population is read or not.
    pub fn get(&self, measured: bool) -> &[DefaultId] {
        if measured {
            &self.measured
        } else {
            &self.plain
        }
    }
}

/// The defaults a source binds by its own top-level conditions `top`. Only
/// a top-level AND condition naming a default's field lifts it; a saved
/// relation as the source stands in for the unit's defaults; a measured
/// population includes hidden sessions.
fn bind(unit: Unit, top: &[&Condition], origins: &[String], measuring: bool) -> Vec<DefaultId> {
    if top
        .iter()
        .any(|condition| matches!(condition.kind, ConditionKind::Saved { .. }))
    {
        return vec![];
    }
    let named = named_fields(top, unit);
    let row_reference = top
        .iter()
        .any(|c| is_reference(c, unit, origins) && !names_a_session(c, unit, origins));
    let session_reference = top.iter().any(|c| names_a_session(c, unit, origins));
    declared(unit)
        .iter()
        .copied()
        .filter(|d| !(d.lifted_by_reference() && row_reference))
        .filter(|d| !(d.lifted_by_reference() && session_reference && d.hides()))
        .filter(|d| !d.lifted_by().iter().any(|f| named.contains(*f)))
        .filter(|d| !(measuring && d.hides()))
        .collect()
}

/// Field names the top-level conditions name, `session.` prefixes removed
/// for row units.
fn named_fields(top: &[&Condition], unit: Unit) -> HashSet<String> {
    let mut out = HashSet::new();
    for c in top {
        let f = match &c.kind {
            ConditionKind::Match { field, .. } => field.as_str(),
            ConditionKind::Reference { reference } => match reference.split_once(':') {
                Some((field, _)) => field,
                None => continue,
            },
            ConditionKind::Since { field, .. } | ConditionKind::Until { field, .. } => field,
            ConditionKind::Compare { left, .. } | ConditionKind::Between { left, .. } => match left
            {
                Expression::Field { name } => name,
                _ => continue,
            },
            _ => continue,
        };
        let f = f.to_ascii_lowercase();
        let f = if unit != Unit::Sessions {
            f.strip_prefix("session.").map(str::to_owned).unwrap_or(f)
        } else {
            f
        };
        out.insert(f);
    }
    out
}

/// A reference that names a session rather than this unit's own rows:
/// `session:X`, `origin:native`, `latest`, `self`. Over another
/// unit it lifts only the session's visibility default; the unit's own
/// row defaults (duplicate copies, the active path) still apply, the
/// same as reading `X | unit`.
pub fn names_a_session(c: &Condition, unit: Unit, origins: &[String]) -> bool {
    if unit == Unit::Sessions {
        return false;
    }
    match &c.kind {
        ConditionKind::Match { field, .. } => {
            let f = field.to_ascii_lowercase();
            f == "session" || origins.contains(&f)
        }
        ConditionKind::Text {
            text,
            mode: TextMode::Word,
        } => matches!(text.as_str(), "latest" | "self"),
        _ => false,
    }
}

/// A condition that names rows by reference rather than describing them.
pub fn is_reference(c: &Condition, unit: Unit, origins: &[String]) -> bool {
    match &c.kind {
        ConditionKind::Reference { .. } => true,
        ConditionKind::Match { field, .. } => {
            let f = field.to_ascii_lowercase();
            origins.contains(&f)
                || (f == "event" && unit != Unit::Deliveries)
                || matches!(
                    f.as_str(),
                    "session"
                        | "message"
                        | "block"
                        | "run"
                        | "request"
                        | "document"
                        | "span"
                        | "annotation"
                        | "delivery"
                )
        }
        ConditionKind::Text {
            text,
            mode: TextMode::Word,
        } => crate::catalog::REFERENCE_WORDS.contains(&text.as_str()) || id_shaped(text),
        _ => false,
    }
}

/// At least 8 characters of hex digits and dashes, containing a digit or
/// a dash: the shape of a pasted native ID.
pub fn id_shaped(word: &str) -> bool {
    word.len() >= 8
        && word.chars().all(|c| c.is_ascii_hexdigit() || c == '-')
        && word.chars().any(|c| c.is_ascii_digit() || c == '-')
}
#[cfg(test)]
mod tests;
