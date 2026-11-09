//! Authored statements and their one set of typed links. Currency follows
//! the shared descendant rule of the `authored` crate over these links; the
//! store keeps it in an index written with each record.
pub mod policy;
pub mod uniqueness;
pub mod values;

use crate::{Author, BlobHash, Json, Reference, Timestamp, authorship::Channel};
use authored::{Uniqueness, WhenHeld};
use serde::{Deserialize, Serialize};

/// Kinds with no uniqueness key: a target may hold many.
const MANY: Option<Uniqueness> = None;
/// One current value per target: `title`, `outcome`, `hidden`.
const ONE_PER_ABOUT: Option<Uniqueness> = Some(Uniqueness::new(&["about"], WhenHeld::Revise));
/// One tag per target, name and value; writing it again changes nothing.
const TAG: Option<Uniqueness> = Some(Uniqueness::new(
    &["about", "key", "value"],
    WhenHeld::Already,
));
/// One record per name: rubrics, saved queries, signals, methods, roles, pages.
const ONE_PER_NAME: Option<Uniqueness> = Some(Uniqueness::new(&["key"], WhenHeld::Revise));
/// One answer per row, rubric revision, field and author.
const ANNOTATION: Option<Uniqueness> = Some(Uniqueness::new(
    &["about", "rubric", "key", "author"],
    WhenHeld::Revise,
));
/// One memory per path and scope.
const MEMORY: Option<Uniqueness> = Some(Uniqueness::new(&["key", "scope"], WhenHeld::Revise));
/// One setting per name and target host.
const SETTING: Option<Uniqueness> = Some(Uniqueness::new(&["key", "machine"], WhenHeld::Revise));
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
    /// Which components define "the same slot", so that a second write
    /// revises instead of adding; `None` means a target may hold many.
    pub uniqueness: Option<Uniqueness>,
    pub about: About,
    pub key: KeyShape,
    pub markup: bool,
    pub briefs: bool,
}
macro_rules! assertion_kinds {
    ($($kind:ident => ($name:literal,$value:ident,$uniqueness:ident,$about:ident,$key:ident,$markup:literal,$briefs:literal)),+ $(,)?) => {
        #[derive(Debug,Clone,Copy,PartialEq,Eq,Hash,Serialize,Deserialize,schemars::JsonSchema)]
        #[serde(rename_all="snake_case")]
        pub enum AssertionKind {$($kind),+}
        stored_names!(AssertionKind {$($kind=>$name),+});
        impl AssertionKind {
            pub fn spec(self)->KindSpec {match self {$(
                Self::$kind=>KindSpec {value:ValueShape::$value,uniqueness:$uniqueness,about:About::$about,key:KeyShape::$key,markup:$markup,briefs:$briefs}
            ),+}}
        }
    };
}
assertion_kinds! {
    Note => ("note",Text,MANY,Anything,None,true,true),
    Tag => ("tag",Tag,TAG,Anything,Required,true,true),
    Highlight => ("highlight",OptionalText,MANY,Span,Required,true,false),
    Title => ("title",Text,ONE_PER_ABOUT,Session,None,true,false),
    Outcome => ("outcome",Outcome,ONE_PER_ABOUT,Session,None,true,true),
    Hidden => ("hidden",OptionalText,ONE_PER_ABOUT,Session,None,false,false),
    Claim => ("claim",Claim,MANY,Anything,Optional,true,true),
    Finding => ("finding",Finding,MANY,Anything,Optional,true,true),
    Decision => ("decision",Decision,MANY,Anything,Optional,true,true),
    Lesson => ("lesson",Text,MANY,Anything,Optional,true,true),
    Question => ("question",Question,MANY,Anything,Optional,true,true),
    Goal => ("goal",Text,MANY,Anything,Optional,true,true),
    Problem => ("problem",Text,MANY,Anything,Optional,true,true),
    Handoff => ("handoff",Text,MANY,Anything,Optional,true,true),
    Correction => ("correction",Text,MANY,Anything,Optional,true,true),
    Forecast => ("forecast",Forecast,MANY,Anything,Required,true,true),
    Judgment => ("judgment",Judgment,MANY,ExactAssertion,Required,false,false),
    Comparison => ("comparison",Comparison,MANY,Several,Required,false,false),
    Rubric => ("rubric",Rubric,ONE_PER_NAME,None,Required,false,false),
    Pass => ("pass",Pass,MANY,None,None,false,false),
    Annotation => ("annotation",Annotation,ANNOTATION,One,Required,true,false),
    Memory => ("memory",Text,MEMORY,None,Required,false,true),
    Method => ("method",Method,ONE_PER_NAME,None,Required,false,true),
    SavedQuery => ("saved_query",SavedQuery,ONE_PER_NAME,None,Required,false,false),
    Signal => ("signal",Signal,ONE_PER_NAME,None,Required,false,false),
    Page => ("page",Page,ONE_PER_NAME,None,Required,false,false),
    Slot => ("slot",Slot,MANY,OptionalOne,None,false,false),
    Member => ("member",Member,MANY,Session,Required,false,false),
    Role => ("role",Role,ONE_PER_NAME,None,Required,false,false),
    Ping => ("ping",Ping,MANY,Session,None,false,false),
    Withdrawal => ("withdrawal",OptionalText,MANY,None,None,false,false),
    Setting => ("setting",Setting,SETTING,None,Required,false,false),
}
impl KindSpec {
    /// A single-valued mark (`title`, `outcome`, `hidden`): one value per
    /// target. Bulk writes over more than one row never write one.
    pub fn single_valued_mark(&self) -> bool {
        self.uniqueness
            .is_some_and(|uniqueness| uniqueness.components == ["about"])
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
    /// Identity-only graph operations do not bind a text reading. Other roles
    /// record a hash when the addressed source supplies recorded text.
    pub fn uses_text_hash(self) -> bool {
        self != Self::Withdraws
    }
    /// A typed relation between assertions: the relating assertion is also
    /// `about` the revision it relates to.
    pub fn is_relation(self) -> bool {
        matches!(
            self,
            Self::Answers
                | Self::Challenges
                | Self::Supports
                | Self::Qualifies
                | Self::Corrects
                | Self::Blocks
        )
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
    /// For a link to a span: the exact text it covered when written, so a
    /// reader can find the text again after a re-parse moves its offsets.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quote: Option<Quote>,
}

/// A text quote selector (W3C Web Annotation `TextQuoteSelector`): the exact
/// text and up to [`Quote::CONTEXT`] characters on either side of it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Quote {
    pub exact: String,
    pub prefix: String,
    pub suffix: String,
}

impl Quote {
    /// Characters of context kept before and after the quoted text.
    pub const CONTEXT: usize = 32;

    /// The quote of the bytes `start..end` of `text`, which must fall on
    /// character boundaries.
    pub fn of(text: &str, start: usize, end: usize) -> Option<Self> {
        let exact = text.get(start..end)?;
        let before = &text[..start];
        let after = &text[end..];
        let prefix_start = before
            .char_indices()
            .rev()
            .nth(Self::CONTEXT - 1)
            .map_or(0, |(index, _)| index);
        let suffix_end = after
            .char_indices()
            .nth(Self::CONTEXT)
            .map_or(after.len(), |(index, _)| index);
        Some(Self {
            exact: exact.into(),
            prefix: before[prefix_start..].into(),
            suffix: after[..suffix_end].into(),
        })
    }

    /// Where the quoted text is in `text` now, as a byte range: the one place
    /// the exact text occurs with its context, else the one place it occurs
    /// at all. Two or more places, or none, find nothing; a quote is never
    /// guessed.
    pub fn find(&self, text: &str) -> Option<(usize, usize)> {
        if self.exact.is_empty() {
            return None;
        }
        let places: Vec<usize> = text
            .match_indices(self.exact.as_str())
            .map(|(start, _)| start)
            .collect();
        let with_context: Vec<usize> = places
            .iter()
            .copied()
            .filter(|start| {
                text[..*start].ends_with(self.prefix.as_str())
                    && text[start + self.exact.len()..].starts_with(self.suffix.as_str())
            })
            .collect();
        let chosen = if with_context.len() == 1 {
            with_context
        } else {
            places
        };
        match chosen.as_slice() {
            [start] => Some((*start, start + self.exact.len())),
            _ => None,
        }
    }
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
    pub via: Channel,
    pub machine: String,
    pub created_at: Timestamp,
    pub valid_until: Option<Timestamp>,
    pub links: Vec<AssertionLink>,
}
/// One assertion as a bundle carries it: its stored reference and its fact.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AssertionRecord {
    #[serde(rename = "ref")]
    pub reference: String,
    pub assertion: AssertionFact,
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
    #[error("only a link to a span carries a quote")]
    QuoteWithoutSpan,
    #[error("an assertion time is outside the range a millisecond count can hold")]
    Time,
}

impl AssertionFact {
    /// Link order does not change the identity of an immutable assertion.
    /// Times are kept to the millisecond, as they are stored.
    pub fn canonicalized(&self) -> Result<Self, AssertionError> {
        self.validate()?;
        let mut fact = self.clone();
        fact.links
            .sort_by_cached_key(|link| (link.role.as_str(), link.reference.to_string()));
        let millisecond =
            |time: Timestamp| Timestamp::from_unix_ms(time.unix_ms()).ok_or(AssertionError::Time);
        fact.created_at = millisecond(fact.created_at)?;
        fact.valid_until = fact.valid_until.map(millisecond).transpose()?;
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
            via: request.authorship.via,
            machine: request.machine.clone(),
            created_at: request.at,
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
        if self
            .links
            .iter()
            .any(|link| link.quote.is_some() && !matches!(link.reference, Reference::Span { .. }))
        {
            return Err(AssertionError::QuoteWithoutSpan);
        }
        let value = values::decode(self.kind, self.key.as_deref(), self.value.as_ref())?;
        if let values::Value::Setting(setting) = &value {
            let key = self.key.as_deref().expect("setting key checked by policy");
            setting
                .fields
                .value
                .validate_for(key, crate::settings::Owner::Standalone)?;
            // A canonical host target is already resolved; the writing
            // machine never stands in for it.
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

/// A stored assertion with this archive's write counter, which is never part
/// of the fact's identity.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Assertion {
    pub fact: AssertionFact,
    pub commit_seq: u64,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TargetState {
    Current,
    /// The text changed, but the quoted text was found once in the new text.
    Moved,
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
