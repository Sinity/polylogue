//! Canonical assertion operations retain the domain's closed write shapes.
use polylogue_model::{
    Reference,
    assertion::{AssertionPatch, NewAssertion},
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Assert {
    #[schemars(length(min = 1))]
    pub assertions: Vec<NewAssertion>,
}
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AssertQuery {
    #[schemars(length(min = 1))]
    pub query: String,
    pub assertion: NewAssertion,
}
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Revise {
    #[serde(with = "polylogue_model::assertion_revisions")]
    #[schemars(with = "Vec<Reference>", length(min = 1))]
    pub bases: Vec<polylogue_model::RevisionReference>,
    pub patch: AssertionPatch,
}
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Withdraw {
    #[schemars(length(min = 1))]
    pub targets: Vec<Reference>,
    pub reason: Option<String>,
}
