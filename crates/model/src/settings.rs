//! The declared runtime setting catalog. Deployment configuration and Sinex's
//! primary attention context have their own owners and are not authored keys.
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Scope {
    Archive,
    Host,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    Boolean,
    Variants,
    QuietHours,
    InterruptBudget,
    EmbeddingProvider,
    Text,
    Endpoint,
    NonnegativeNumber,
    Plan,
    Price,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct Declaration {
    pub key: &'static str,
    pub kind: Kind,
    pub default: Option<&'static str>,
    pub scope: Scope,
    /// Primary mode receives this value only from Sinex's public context.
    pub standalone_only: bool,
}
pub const CATALOG: &[Declaration] = &[
    Declaration {
        key: "browser.actions",
        kind: Kind::Boolean,
        default: Some("off"),
        scope: Scope::Archive,
        standalone_only: false,
    },
    Declaration {
        key: "reading.variants",
        kind: Kind::Variants,
        default: Some("[]"),
        scope: Scope::Archive,
        standalone_only: false,
    },
    Declaration {
        key: "fetch-links",
        kind: Kind::Boolean,
        default: Some("off"),
        scope: Scope::Archive,
        standalone_only: false,
    },
    Declaration {
        key: "signal.quiet_hours",
        kind: Kind::QuietHours,
        default: Some("null"),
        scope: Scope::Host,
        standalone_only: true,
    },
    Declaration {
        key: "signal.interrupts_per_hour",
        kind: Kind::InterruptBudget,
        default: Some("4"),
        scope: Scope::Host,
        standalone_only: true,
    },
    Declaration {
        key: "embed.provider",
        kind: Kind::EmbeddingProvider,
        default: Some("off"),
        scope: Scope::Archive,
        standalone_only: false,
    },
    Declaration {
        key: "embed.endpoint",
        kind: Kind::Endpoint,
        default: None,
        scope: Scope::Archive,
        standalone_only: false,
    },
    Declaration {
        key: "embed.model",
        kind: Kind::Text,
        default: None,
        scope: Scope::Archive,
        standalone_only: false,
    },
    Declaration {
        key: "embed.price_per_mtok",
        kind: Kind::NonnegativeNumber,
        default: None,
        scope: Scope::Archive,
        standalone_only: false,
    },
    Declaration {
        key: "plan.*",
        kind: Kind::Plan,
        default: None,
        scope: Scope::Archive,
        standalone_only: false,
    },
    Declaration {
        key: "price.*",
        kind: Kind::Price,
        default: None,
        scope: Scope::Archive,
        standalone_only: false,
    },
];
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Owner {
    Standalone,
    Primary,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "type", content = "value", rename_all = "snake_case")]
pub enum Value {
    Boolean(bool),
    Variants(Vec<String>),
    QuietHours(Option<[u16; 2]>),
    InterruptBudget(usize),
    EmbeddingProvider(EmbeddingProvider),
    Text(String),
    Endpoint(String),
    NonnegativeNumber(f64),
    Plan(Plan),
    Price(crate::price::Price),
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum EmbeddingProvider {
    Off,
    Local,
    Voyage,
}
/// One typed setting answer. A missing primary attention value names its
/// Sinex owner instead of exposing a Polylogue default.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Current {
    pub key: String,
    pub kind: Kind,
    pub default: Option<Value>,
    pub scope: Scope,
    pub owner: String,
    pub value: Option<Value>,
    pub authored: Option<Authored>,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Authored {
    pub id: String,
    pub machine: String,
    pub author: crate::Author,
    pub at: crate::Timestamp,
}
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum Error {
    #[error("unknown runtime setting {0:?}")]
    UnknownKey(String),
    #[error("runtime setting {0:?} cannot be blank")]
    Blank(String),
    #[error("invalid value for runtime setting {key:?}: expected {expected}")]
    Invalid { key: String, expected: &'static str },
    #[error("runtime setting {0:?} is owned by Sinex in primary mode")]
    SinexOwned(String),
}
pub fn declaration(key: &str) -> Result<&'static Declaration, Error> {
    CATALOG
        .iter()
        .find(|declaration| {
            if let Some(prefix) = declaration.key.strip_suffix('*') {
                key.strip_prefix(prefix).is_some_and(|name| {
                    !name.is_empty()
                        && !name
                            .chars()
                            .any(|c| c.is_control() || c.is_whitespace() || c == '*')
                })
            } else {
                declaration.key == key
            }
        })
        .ok_or_else(|| Error::UnknownKey(key.to_string()))
}
/// Resolve the authored destination from the declared scope. The accepting
/// daemon supplies its captured machine; request text cannot change that fact.
pub fn target_machine(
    key: &str,
    explicit: Option<&str>,
    accepting_machine: &str,
) -> Result<Option<String>, Error> {
    let spec = declaration(key)?;
    let invalid = |expected| Error::Invalid {
        key: key.into(),
        expected,
    };
    match spec.scope {
        Scope::Archive => {
            if explicit.is_some() {
                return Err(invalid("archive scope without a machine override"));
            }
            Ok(None)
        }
        Scope::Host => {
            let machine = explicit.unwrap_or(accepting_machine);
            if machine.trim().is_empty() || machine == "*" || machine.chars().any(char::is_control)
            {
                return Err(invalid("a nonempty host machine identity"));
            }
            Ok(Some(machine.into()))
        }
    }
}
pub fn parse(key: &str, text: &str, owner: Owner) -> Result<Value, Error> {
    let spec = declaration(key)?;
    if owner == Owner::Primary && spec.standalone_only {
        return Err(Error::SinexOwned(key.to_string()));
    }
    let text = text.trim();
    if text.is_empty() {
        return Err(Error::Blank(key.to_string()));
    }
    let invalid = |expected| Error::Invalid {
        key: key.to_string(),
        expected,
    };
    Ok(match spec.kind {
        Kind::Boolean => Value::Boolean(match text {
            "on" => true,
            "off" => false,
            _ => return Err(invalid("on or off")),
        }),
        Kind::Variants => {
            let values: Vec<String> = serde_json::from_str(text)
                .map_err(|_| invalid("an ordered JSON list of variant keys"))?;
            if values.iter().any(|value| {
                value.trim().is_empty()
                    || value.chars().any(char::is_control)
                    || !matches!(
                        value.split(':').next(),
                        Some("translation" | "transliteration" | "simplification" | "summary")
                    )
            }) {
                return Err(invalid("an ordered JSON list of variant keys"));
            }
            Value::Variants(values)
        }
        Kind::QuietHours => {
            #[derive(Deserialize)]
            #[serde(deny_unknown_fields)]
            struct Minutes {
                from: u16,
                to: u16,
            }
            let value: Option<Minutes> = serde_json::from_str(text)
                .map_err(|_| invalid("null or {\"from\":MINUTE,\"to\":MINUTE}"))?;
            if value
                .as_ref()
                .is_some_and(|value| value.from >= 1440 || value.to >= 1440)
            {
                return Err(invalid("civil minutes from 0 through 1439"));
            }
            Value::QuietHours(value.map(|value| [value.from, value.to]))
        }
        Kind::InterruptBudget => Value::InterruptBudget(
            text.parse()
                .map_err(|_| invalid("a nonnegative whole interrupt count"))?,
        ),
        Kind::EmbeddingProvider => Value::EmbeddingProvider(match text {
            "off" => EmbeddingProvider::Off,
            "local" => EmbeddingProvider::Local,
            "voyage" => EmbeddingProvider::Voyage,
            _ => return Err(invalid("off, local or voyage")),
        }),
        Kind::Text => Value::Text(text.to_string()),
        Kind::Endpoint => {
            if !(text.starts_with("http://") || text.starts_with("https://"))
                || text.chars().any(char::is_whitespace)
            {
                return Err(invalid("an HTTP or HTTPS endpoint"));
            }
            Value::Endpoint(text.to_string())
        }
        Kind::NonnegativeNumber => {
            let value: f64 = text
                .parse()
                .map_err(|_| invalid("a finite nonnegative number"))?;
            if !value.is_finite() || value < 0.0 {
                return Err(invalid("a finite nonnegative number"));
            }
            Value::NonnegativeNumber(value)
        }
        Kind::Plan => {
            let value: Plan =
                serde_json::from_str(text).map_err(|_| invalid("a typed subscription plan"))?;
            value.validate().map_err(|_| {
                invalid("an unambiguous plan with valid anchor day, lane and nonnegative amounts")
            })?;
            Value::Plan(value)
        }
        Kind::Price => {
            let value: crate::price::Price =
                serde_json::from_str(text).map_err(|_| invalid("a typed model price"))?;
            let mut rates = vec![value.input, value.output];
            rates.extend(
                [value.cached_input, value.cache_write, value.cache_write_1h]
                    .into_iter()
                    .flatten(),
            );
            rates.extend(value.tiers.values().copied());
            if let Some(long) = &value.long {
                rates.extend([long.input, long.output]);
                rates.extend(
                    [long.cached_input, long.cache_write, long.cache_write_1h]
                        .into_iter()
                        .flatten(),
                );
            }
            if rates.iter().any(|rate| !rate.is_finite() || *rate < 0.0) {
                return Err(invalid("finite nonnegative model prices"));
            }
            Value::Price(value)
        }
    })
}
pub fn current(key: &str, stored: Option<&str>, owner: Owner) -> Result<Option<Value>, Error> {
    let spec = declaration(key)?;
    // A primary consumer must never turn a missing owner value into a local default.
    if owner == Owner::Primary && spec.standalone_only {
        return Err(Error::SinexOwned(key.to_string()));
    }
    stored
        .or(spec.default)
        .map(|text| parse(key, text, owner))
        .transpose()
}

#[derive(Debug, Clone)]
pub struct Snapshot {
    values: std::collections::BTreeMap<String, Value>,
}
impl Snapshot {
    pub fn new(entries: impl IntoIterator<Item = Current>) -> Self {
        Self {
            values: entries
                .into_iter()
                .filter_map(|entry| entry.value.map(|value| (entry.key, value)))
                .collect(),
        }
    }
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.values.get(key)
    }
    pub fn browser_actions(&self) -> Result<bool, Error> {
        match self.get("browser.actions") {
            Some(Value::Boolean(value)) => Ok(*value),
            _ => Err(Error::Invalid {
                key: "browser.actions".into(),
                expected: "a typed boolean snapshot entry",
            }),
        }
    }
    pub fn reading_variants(&self) -> Result<&[String], Error> {
        match self.get("reading.variants") {
            Some(Value::Variants(value)) => Ok(value),
            _ => Err(Error::Invalid {
                key: "reading.variants".into(),
                expected: "a typed ordered variant list",
            }),
        }
    }
    pub fn standalone_attention(&self) -> Result<(Option<[u16; 2]>, usize), Error> {
        match (
            self.get("signal.quiet_hours"),
            self.get("signal.interrupts_per_hour"),
        ) {
            (Some(Value::QuietHours(quiet)), Some(Value::InterruptBudget(limit))) => {
                Ok((*quiet, *limit))
            }
            _ => Err(Error::SinexOwned("attention".into())),
        }
    }
}

pub const PLAN_UNITS: &[&str] = &[
    "api_nusd",
    "reported_nusd",
    "input_tokens",
    "cached_input_tokens",
    "cache_write_tokens",
    "output_tokens",
    "reasoning_tokens",
];
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Plan {
    pub name: String,
    pub origin: String,
    pub cycle_anchor_day: Option<i8>,
    pub price_nusd: Option<i64>,
    pub budget: Option<Budget>,
}
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Budget {
    pub population: Population,
    pub unit: String,
    pub amount: i64,
    pub overage_rate_nusd_per_unit: Option<i64>,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Population {
    Responses,
    RecordedReports,
}
impl Plan {
    pub fn validate(&self) -> Result<(), Error> {
        if self.name.trim().is_empty()
            || self.name.eq_ignore_ascii_case("max")
            || self.origin.trim().is_empty()
            || self
                .cycle_anchor_day
                .is_some_and(|day| !(1..=31).contains(&day))
            || self.price_nusd.is_some_and(|price| price < 0)
            || self.budget.as_ref().is_some_and(|budget| {
                !PLAN_UNITS.contains(&budget.unit.as_str())
                    || budget.amount < 0
                    || budget
                        .overage_rate_nusd_per_unit
                        .is_some_and(|rate| rate < 0)
            })
        {
            return Err(Error::Invalid {
                key: "plan.*".into(),
                expected: "an unambiguous plan with valid anchor day, lane and nonnegative amounts",
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
