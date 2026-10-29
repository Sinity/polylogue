//! Authored statements and their one set of typed links. Currency and target
//! state are computed from these facts, rather than stored on a statement.
use crate::{Author, BlobHash, Json, Reference, Timestamp, authorship::Channel};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum AssertionKind {
    Note,
    Tag,
    Highlight,
    Title,
    Outcome,
    Claim,
    Finding,
    Decision,
    Lesson,
    Question,
    Goal,
    Problem,
    Blocker,
    Handoff,
    Correction,
    Forecast,
    Judgment,
    Comparison,
    Rubric,
    Pass,
    Annotation,
    Memory,
    Method,
    SavedQuery,
    Signal,
    Page,
    Slot,
    Member,
    Role,
    Ping,
    Withdrawal,
    Setting,
}
stored_names!(AssertionKind {
    Note=>"note",Tag=>"tag",Highlight=>"highlight",Title=>"title",Outcome=>"outcome",Claim=>"claim",Finding=>"finding",Decision=>"decision",Lesson=>"lesson",Question=>"question",Goal=>"goal",Problem=>"problem",Blocker=>"blocker",Handoff=>"handoff",Correction=>"correction",Forecast=>"forecast",Judgment=>"judgment",Comparison=>"comparison",Rubric=>"rubric",Pass=>"pass",Annotation=>"annotation",Memory=>"memory",Method=>"method",SavedQuery=>"saved_query",Signal=>"signal",Page=>"page",Slot=>"slot",Member=>"member",Role=>"role",Ping=>"ping",Withdrawal=>"withdrawal",Setting=>"setting"
});

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cardinality {
    Many,
    OnePerAbout,
    OnePerName,
    Tag,
    Annotation,
    OnePerScopeKey,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueShape {
    Text,
    OptionalText,
    Tag,
    Outcome,
    Claim,
    Decision,
    Forecast,
    Judgment,
    Comparison,
    Rubric,
    Pass,
    Annotation,
    Method,
    SavedQuery,
    Signal,
    Page,
    Slot,
    Member,
    Role,
    Ping,
    Setting,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum About {
    Anything,
    Span,
    Session,
    ExactAssertion,
    Several,
    None,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KindSpec {
    pub value: ValueShape,
    pub cardinality: Cardinality,
    pub about: About,
    pub key_required: bool,
    pub markup: bool,
    pub briefs: bool,
}
impl AssertionKind {
    pub fn spec(self) -> KindSpec {
        use Cardinality::*;
        let (value, cardinality, about, key_required, markup, briefs) = match self {
            Self::Note => (ValueShape::Text, Many, About::Anything, false, true, true),
            Self::Tag => (ValueShape::Tag, Tag, About::Anything, true, true, true),
            Self::Highlight => (
                ValueShape::OptionalText,
                Many,
                About::Span,
                true,
                true,
                false,
            ),
            Self::Title => (
                ValueShape::Text,
                OnePerAbout,
                About::Session,
                false,
                true,
                false,
            ),
            Self::Outcome => (
                ValueShape::Outcome,
                OnePerAbout,
                About::Session,
                false,
                true,
                true,
            ),
            Self::Claim => (ValueShape::Claim, Many, About::Anything, false, true, true),
            Self::Decision => (
                ValueShape::Decision,
                Many,
                About::Anything,
                false,
                true,
                true,
            ),
            Self::Finding
            | Self::Lesson
            | Self::Question
            | Self::Goal
            | Self::Problem
            | Self::Blocker
            | Self::Handoff
            | Self::Correction => (ValueShape::Text, Many, About::Anything, false, true, true),
            Self::Forecast => (
                ValueShape::Forecast,
                Many,
                About::Anything,
                true,
                true,
                true,
            ),
            Self::Judgment => (
                ValueShape::Judgment,
                Many,
                About::ExactAssertion,
                true,
                false,
                false,
            ),
            Self::Comparison => (
                ValueShape::Comparison,
                Many,
                About::Several,
                true,
                false,
                false,
            ),
            Self::Rubric => (
                ValueShape::Rubric,
                OnePerName,
                About::None,
                true,
                false,
                false,
            ),
            Self::Pass => (ValueShape::Pass, Many, About::None, false, false, false),
            Self::Annotation => (
                ValueShape::Annotation,
                Annotation,
                About::Anything,
                true,
                true,
                false,
            ),
            Self::Memory => (
                ValueShape::Text,
                OnePerScopeKey,
                About::None,
                true,
                false,
                true,
            ),
            Self::Method => (
                ValueShape::Method,
                OnePerName,
                About::None,
                true,
                false,
                true,
            ),
            Self::SavedQuery => (
                ValueShape::SavedQuery,
                OnePerName,
                About::None,
                true,
                false,
                false,
            ),
            Self::Signal => (
                ValueShape::Signal,
                OnePerName,
                About::None,
                true,
                false,
                false,
            ),
            Self::Page => (
                ValueShape::Page,
                OnePerName,
                About::None,
                true,
                false,
                false,
            ),
            Self::Slot => (ValueShape::Slot, Many, About::Anything, false, false, false),
            Self::Member => (
                ValueShape::Member,
                Many,
                About::Anything,
                false,
                false,
                false,
            ),
            Self::Role => (ValueShape::Role, Many, About::Anything, false, false, false),
            Self::Ping => (ValueShape::Ping, Many, About::Anything, false, false, false),
            Self::Withdrawal => (
                ValueShape::OptionalText,
                Many,
                About::None,
                false,
                false,
                false,
            ),
            Self::Setting => (
                ValueShape::Setting,
                OnePerName,
                About::None,
                true,
                false,
                false,
            ),
        };
        KindSpec {
            value,
            cardinality,
            about,
            key_required,
            markup,
            briefs,
        }
    }
}

#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    PartialOrd,
    Ord,
    Serialize,
    Deserialize,
    schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum LinkRole {
    About,
    Evidence,
    Scope,
    Origin,
    Revises,
    Withdraws,
    Answers,
    Challenges,
    Supports,
    Qualifies,
    Corrects,
    DerivedFrom,
    Blocks,
    SlotOf,
    MemberOf,
    RoleOf,
}
stored_names!(LinkRole {About=>"about",Evidence=>"evidence",Scope=>"scope",Origin=>"origin",Revises=>"revises",Withdraws=>"withdraws",Answers=>"answers",Challenges=>"challenges",Supports=>"supports",Qualifies=>"qualifies",Corrects=>"corrects",DerivedFrom=>"derived_from",Blocks=>"blocks",SlotOf=>"slot_of",MemberOf=>"member_of",RoleOf=>"role_of"});
impl LinkRole {
    pub fn requires_exact_assertion(self) -> bool {
        matches!(
            self,
            Self::Revises
                | Self::Answers
                | Self::Challenges
                | Self::Supports
                | Self::Qualifies
                | Self::Corrects
                | Self::DerivedFrom
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AssertionLink {
    pub role: LinkRole,
    pub reference: Reference,
    pub seen_hash: Option<BlobHash>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Assertion {
    pub id: String,
    pub root: String,
    pub kind: AssertionKind,
    pub key: Option<String>,
    pub value: Option<Json>,
    pub author: Author,
    pub via: Channel,
    pub machine: String,
    pub created_at: Timestamp,
    pub valid_until: Option<Timestamp>,
    pub commit_seq: u64,
    pub links: Vec<AssertionLink>,
}
impl Assertion {
    pub fn expired_at(&self, now: Timestamp) -> bool {
        self.valid_until.is_some_and(|until| until < now)
    }
    pub fn about(&self) -> impl Iterator<Item = &Reference> {
        self.links
            .iter()
            .filter(|link| link.role == LinkRole::About)
            .map(|link| &link.reference)
    }
    pub fn text(&self) -> Option<&str> {
        self.value.as_ref().and_then(|value| {
            value
                .as_str()
                .or_else(|| value.get("text").and_then(Json::as_str))
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct NewAssertion {
    pub kind: AssertionKind,
    pub key: Option<String>,
    pub value: Option<Json>,
    pub valid_until: Option<Timestamp>,
    pub links: Vec<NewAssertionLink>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum TargetState {
    Current,
    ChangedSince,
    Waiting,
    Missing,
}

/// A writer resolves text hashes against its accepting read snapshot.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct NewAssertionLink {
    pub role: LinkRole,
    pub reference: Reference,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "change", content = "value", rename_all = "snake_case")]
pub enum Change<T> {
    Keep,
    Set(T),
    Clear,
}
impl<T> Default for Change<T> {
    fn default() -> Self {
        Self::Keep
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(default, deny_unknown_fields)]
pub struct AssertionPatch {
    pub key: Change<String>,
    pub value: Change<Json>,
    pub valid_until: Change<Timestamp>,
    pub links: Option<Vec<NewAssertionLink>>,
}

#[cfg(test)]
mod tests;
