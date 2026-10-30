//! Authored statements and their one set of typed links. Currency and target
//! state are computed from these facts, rather than stored on a statement.
pub mod cardinality;
pub mod currency;
pub mod policy;
pub mod values;

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
    Setting,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyShape {
    None,
    Optional,
    Required,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueShape {
    Text,
    OptionalText,
    Tag,
    Outcome,
    Claim,
    Finding,
    Question,
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
    One,
    OptionalOne,
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
    pub key: KeyShape,
    pub markup: bool,
    pub briefs: bool,
}
macro_rules! assertion_kinds {
    ($($kind:ident => ($name:literal,$value:ident,$cardinality:ident,$about:ident,$key:ident,$markup:literal,$briefs:literal)),+ $(,)?) => {
        #[derive(Debug,Clone,Copy,PartialEq,Eq,Hash,Serialize,Deserialize,schemars::JsonSchema)]
        #[serde(rename_all="snake_case")]
        pub enum AssertionKind {$($kind),+}
        stored_names!(AssertionKind {$($kind=>$name),+});
        impl AssertionKind {
            pub fn spec(self)->KindSpec {match self {$(
                Self::$kind=>KindSpec {value:ValueShape::$value,cardinality:Cardinality::$cardinality,about:About::$about,key:KeyShape::$key,markup:$markup,briefs:$briefs}
            ),+}}
        }
    };
}
assertion_kinds! {
    Note => ("note",Text,Many,Anything,None,true,true),
    Tag => ("tag",Tag,Tag,Anything,Required,true,true),
    Highlight => ("highlight",OptionalText,Many,Span,Required,true,false),
    Title => ("title",Text,OnePerAbout,Session,None,true,false),
    Outcome => ("outcome",Outcome,OnePerAbout,Session,None,true,true),
    Claim => ("claim",Claim,Many,Anything,Optional,true,true),
    Finding => ("finding",Finding,Many,Anything,Optional,true,true),
    Decision => ("decision",Decision,Many,Anything,Optional,true,true),
    Lesson => ("lesson",Text,Many,Anything,Optional,true,true),
    Question => ("question",Question,Many,Anything,Optional,true,true),
    Goal => ("goal",Text,Many,Anything,Optional,true,true),
    Problem => ("problem",Text,Many,Anything,Optional,true,true),
    Blocker => ("blocker",Text,Many,Anything,Optional,true,true),
    Handoff => ("handoff",Text,Many,Anything,Optional,true,true),
    Correction => ("correction",Text,Many,Anything,Optional,true,true),
    Forecast => ("forecast",Forecast,Many,Anything,Required,true,true),
    Judgment => ("judgment",Judgment,Many,ExactAssertion,Required,false,false),
    Comparison => ("comparison",Comparison,Many,Several,Required,false,false),
    Rubric => ("rubric",Rubric,OnePerName,None,Required,false,false),
    Pass => ("pass",Pass,Many,None,None,false,false),
    Annotation => ("annotation",Annotation,Annotation,One,Required,true,false),
    Memory => ("memory",Text,OnePerScopeKey,None,Required,false,true),
    Method => ("method",Method,OnePerName,None,Required,false,true),
    SavedQuery => ("saved_query",SavedQuery,OnePerName,None,Required,false,false),
    Signal => ("signal",Signal,OnePerName,None,Required,false,false),
    Page => ("page",Page,OnePerName,None,Required,false,false),
    Slot => ("slot",Slot,Many,OptionalOne,None,false,false),
    Member => ("member",Member,Many,Session,Required,false,false),
    Role => ("role",Role,OnePerName,None,Required,false,false),
    Ping => ("ping",Ping,Many,Session,None,false,false),
    Withdrawal => ("withdrawal",OptionalText,Many,None,None,false,false),
    Setting => ("setting",Setting,Setting,None,Required,false,false),
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
    /// Identity-only graph operations do not bind a text reading. Other roles
    /// record a hash when the addressed source supplies recorded text.
    pub fn uses_text_hash(self) -> bool {
        self != Self::Withdraws
    }
    pub fn requires_exact_assertion(self) -> bool {
        matches!(
            self,
            Self::Revises
                | Self::Answers
                | Self::Challenges
                | Self::Supports
                | Self::Qualifies
                | Self::Corrects
                | Self::Blocks
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
#[schemars(transform=values::schema::canonical)]
pub struct AssertionFact {
    pub id: String,
    pub root: String,
    pub kind: AssertionKind,
    pub key: Option<String>,
    pub value: Option<Json>,
    pub author: Author,
    pub via: Option<Channel>,
    pub machine: Option<String>,
    pub created_at: Option<Timestamp>,
    pub valid_until: Option<Timestamp>,
    pub links: Vec<AssertionLink>,
}
/// An independently restored historical fact need not have an original native
/// request. Import claims do not certify native authorship or grant privilege.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AssertionRecord {
    #[serde(rename = "ref")]
    pub reference: String,
    pub assertion: AssertionFact,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub imports: Vec<crate::authored::ImportAttribution>,
}

#[derive(Debug, thiserror::Error)]
pub enum AssertionError {
    #[error("assertion ID and root must be nonempty")]
    Identity,
    #[error(transparent)]
    Structure(#[from] policy::InvalidAssertion),
    #[error(transparent)]
    Value(#[from] values::ShapeError),
    #[error(transparent)]
    Setting(#[from] crate::settings::Error),
    #[error("withdraws is an identity-only link and cannot carry a text hash")]
    IdentityTextHash,
}

impl AssertionFact {
    /// Link order does not change the identity of an immutable assertion.
    pub fn canonicalized(&self) -> Result<Self, AssertionError> {
        self.validate()?;
        let mut fact = self.clone();
        fact.links
            .sort_by_cached_key(|link| (link.role.as_str(), link.reference.to_string()));
        Ok(fact)
    }
    /// Copies one captured accepting request without resampling any clock or
    /// deriving provenance from the value. Publication authority is separate.
    pub fn from_request(
        id: String,
        root: String,
        statement: NewAssertion,
        links: Vec<AssertionLink>,
        request: &crate::user::UserRequest,
    ) -> Self {
        Self {
            id,
            root,
            kind: statement.kind,
            key: statement.key,
            value: statement.value,
            author: request.authorship.author.clone(),
            via: Some(request.authorship.via),
            machine: Some(request.machine.clone()),
            created_at: Some(request.at),
            valid_until: statement.valid_until,
            links,
        }
    }
    /// Checks immutable shape only. The accepting store separately validates
    /// current bases, configured allowances and resolved source evidence.
    pub fn validate(&self) -> Result<values::Value, AssertionError> {
        if self.id.is_empty() || self.root.is_empty() {
            return Err(AssertionError::Identity);
        }
        policy::validate_links(
            self.kind,
            self.key.as_deref(),
            self.links.iter().map(|link| (link.role, &link.reference)),
        )?;
        if self
            .links
            .iter()
            .any(|link| !link.role.uses_text_hash() && link.seen_hash.is_some())
        {
            return Err(AssertionError::IdentityTextHash);
        }
        let value = values::decode(self.kind, self.key.as_deref(), self.value.as_ref())?;
        if let values::Value::Setting(setting) = &value {
            let key = self.key.as_deref().expect("setting key checked by policy");
            setting
                .fields
                .value
                .validate_for(key, crate::settings::Owner::Standalone)?;
            // A canonical host target is already resolved. Historical missing
            // accepting provenance never invents its applicability to a host.
            crate::settings::target_machine(key, setting.fields.machine.as_deref(), "")?;
        }
        Ok(value)
    }
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
        let value = self.value.as_ref()?;
        match self.kind.spec().value {
            ValueShape::Rubric | ValueShape::Role => {
                value.get("instructions").and_then(crate::Json::as_str)
            }
            ValueShape::Judgment => value
                .as_str()
                .or_else(|| value.get("reason").and_then(crate::Json::as_str)),
            _ => values::text(value),
        }
    }
}

/// A local read projection. Its counter is never part of frozen semantic fact
/// identity and is never invented for a historical record with no receipt.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Assertion {
    pub fact: AssertionFact,
    pub commit_seq: u64,
    pub imports: crate::authored::ImportHistory,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
#[schemars(transform=values::schema::live)]
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
