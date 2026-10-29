//! Authored statements and their one set of typed links. Currency and target
//! state are computed from these facts, rather than stored on a statement.
use crate::{Author, BlobHash, Json, Reference, Timestamp, authorship::Channel};
use serde::{Deserialize, Serialize};

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
macro_rules! assertion_kinds {
    ($($kind:ident => ($name:literal,$value:ident,$cardinality:ident,$about:ident,$key:literal,$markup:literal,$briefs:literal)),+ $(,)?) => {
        #[derive(Debug,Clone,Copy,PartialEq,Eq,Hash,Serialize,Deserialize,schemars::JsonSchema)]
        #[serde(rename_all="snake_case")]
        pub enum AssertionKind {$($kind),+}
        stored_names!(AssertionKind {$($kind=>$name),+});
        impl AssertionKind {
            pub fn spec(self)->KindSpec {match self {$(
                Self::$kind=>KindSpec {value:ValueShape::$value,cardinality:Cardinality::$cardinality,about:About::$about,key_required:$key,markup:$markup,briefs:$briefs}
            ),+}}
        }
    };
}
assertion_kinds! {
    Note => ("note",Text,Many,Anything,false,true,true),
    Tag => ("tag",Tag,Tag,Anything,true,true,true),
    Highlight => ("highlight",OptionalText,Many,Span,true,true,false),
    Title => ("title",Text,OnePerAbout,Session,false,true,false),
    Outcome => ("outcome",Outcome,OnePerAbout,Session,false,true,true),
    Claim => ("claim",Claim,Many,Anything,false,true,true),
    Finding => ("finding",Claim,Many,Anything,false,true,true),
    Decision => ("decision",Decision,Many,Anything,false,true,true),
    Lesson => ("lesson",Text,Many,Anything,false,true,true),
    Question => ("question",Claim,Many,Anything,false,true,true),
    Goal => ("goal",Text,Many,Anything,false,true,true),
    Problem => ("problem",Text,Many,Anything,false,true,true),
    Blocker => ("blocker",Text,Many,Anything,false,true,true),
    Handoff => ("handoff",Text,Many,Anything,false,true,true),
    Correction => ("correction",Text,Many,Anything,false,true,true),
    Forecast => ("forecast",Forecast,Many,Anything,true,true,true),
    Judgment => ("judgment",Judgment,Many,ExactAssertion,true,false,false),
    Comparison => ("comparison",Comparison,Many,Several,true,false,false),
    Rubric => ("rubric",Rubric,OnePerName,None,true,false,false),
    Pass => ("pass",Pass,Many,None,false,false,false),
    Annotation => ("annotation",Annotation,Annotation,Anything,true,true,false),
    Memory => ("memory",Text,OnePerScopeKey,None,true,false,true),
    Method => ("method",Method,OnePerName,None,true,false,true),
    SavedQuery => ("saved_query",SavedQuery,OnePerName,None,true,false,false),
    Signal => ("signal",Signal,OnePerName,None,true,false,false),
    Page => ("page",Page,OnePerName,None,true,false,false),
    Slot => ("slot",Slot,Many,Anything,false,false,false),
    Member => ("member",Member,Many,Anything,true,false,false),
    Role => ("role",Role,OnePerName,None,true,false,false),
    Ping => ("ping",Ping,Many,Session,false,false,false),
    Withdrawal => ("withdrawal",OptionalText,Many,None,false,false,false),
    Setting => ("setting",Setting,OnePerName,None,true,false,false),
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

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "change", content = "value", rename_all = "snake_case")]
pub enum Change<T> {
    #[default]
    Keep,
    Set(T),
    Clear,
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
