//! The assertion operations take the domain's closed write shapes.
use polylogue_model::{
    Reference,
    assertion::{AssertionPatch, NewAssertion},
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Assert {
    /// JSON array of assertions
    #[schemars(length(min = 1), extend("argument_form" = "positional"))]
    pub assertions: Vec<NewAssertion>,
}
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AssertQuery {
    /// Query selecting the assertion targets
    #[schemars(length(min = 1), extend("argument_form" = "positional"))]
    pub query: String,
    /// Assertion JSON without about links
    pub assertion: NewAssertion,
}
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Revise {
    /// assertion:<root>@<revision> references
    #[serde(with = "polylogue_model::assertion_revisions")]
    #[schemars(with = "Vec<Reference>", length(min = 1), extend("argument_form" = "words"))]
    pub bases: Vec<polylogue_model::RevisionReference>,
    /// Assertion patch JSON
    pub patch: AssertionPatch,
}
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Withdraw {
    /// Natural assertion or variant references
    #[schemars(length(min = 1), extend("argument_form" = "words"))]
    pub targets: Vec<Reference>,
    /// Why
    pub reason: Option<String>,
}
/// The operator accepts or rejects one revision of an agent's note.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Judge {
    /// The note's natural assertion reference
    #[schemars(extend("argument_form" = "positional"))]
    pub note: String,
    /// Accept the revision; without it, reject it
    pub accepted: bool,
    /// Why
    pub reason: Option<String>,
}
