//! Structural extraction precedes the retired request decoder. Raw values keep
//! the available encoding; no current UserWrite or Product is deserialized here.
use crate::{Error, Result};
use polylogue_model::authored::ImportAttribution;
use serde::{
    Deserialize,
    de::{SeqAccess, Visitor},
};
use serde_json::value::RawValue;
use std::{
    fmt,
    io::{BufReader, Read},
};

#[derive(Deserialize)]
struct Payload<'a> {
    #[serde(borrow)]
    record: &'a RawValue,
}

#[derive(Deserialize)]
struct Record<'a> {
    product: &'a str,
    #[serde(default, borrow)]
    request: Option<&'a RawValue>,
    #[serde(default, borrow)]
    outcome: Option<&'a RawValue>,
    #[serde(default, borrow)]
    effect: Option<&'a RawValue>,
    #[serde(default, borrow)]
    prepared: Option<&'a RawValue>,
    #[serde(default, borrow)]
    claims: Option<&'a RawValue>,
    #[serde(default, borrow)]
    current: Option<&'a RawValue>,
}

#[derive(Deserialize)]
struct Prepared<'a> {
    #[serde(borrow)]
    request: &'a RawValue,
    #[serde(borrow)]
    outcome: &'a RawValue,
    #[serde(borrow)]
    effect: &'a RawValue,
}

pub(super) struct Fields<'a> {
    pub request: &'a [u8],
    pub outcome: &'a [u8],
    pub effect: &'a [u8],
    /// Carried import observations are claims here, not new local observations.
    pub claims: Vec<ImportAttribution>,
}

pub(super) fn extract<'a>(event_type: &str, encoded_payload: &'a [u8]) -> Result<Fields<'a>> {
    let payload: Payload<'a> = parse(encoded_payload)?;
    record(event_type, payload.record.get().as_bytes())
}

fn record<'a>(event_type: &str, bytes: &'a [u8]) -> Result<Fields<'a>> {
    let record: Record<'a> = parse(bytes)?;
    if event_type == "polylogue.request_restore" {
        if record.product != "restored_request" {
            return invalid("historical restore kind carries a different record variant");
        }
        let prepared: Prepared<'a> = parse(required(record.prepared)?.get().as_bytes())?;
        let mut claims: Vec<ImportAttribution> =
            crate::spool::decode_json(required(record.claims)?.get().as_bytes())?;
        let current: ImportAttribution =
            crate::spool::decode_json(required(record.current)?.get().as_bytes())?;
        if !claims.contains(&current) {
            claims.push(current);
        }
        Ok(Fields {
            request: prepared.request.get().as_bytes(),
            outcome: prepared.outcome.get().as_bytes(),
            effect: prepared.effect.get().as_bytes(),
            claims,
        })
    } else {
        if record.product != "request" {
            return invalid("historical request kind carries a different record variant");
        }
        Ok(Fields {
            request: required(record.request)?.get().as_bytes(),
            outcome: required(record.outcome)?.get().as_bytes(),
            effect: required(record.effect)?.get().as_bytes(),
            claims: Vec::new(),
        })
    }
}

fn required(value: Option<&RawValue>) -> Result<&RawValue> {
    value.ok_or_else(|| {
        Error::Invalid("historical request omitted its retained accepted body".into())
    })
}
fn invalid<T>(message: &str) -> Result<T> {
    Err(Error::Invalid(message.into()))
}

pub(crate) fn parse<'a, T: Deserialize<'a>>(bytes: &'a [u8]) -> Result<T> {
    let mut decoder = serde_json::Deserializer::from_slice(bytes);
    decoder.disable_recursion_limit();
    let value = T::deserialize(&mut decoder)?;
    decoder.end()?;
    Ok(value)
}

/// One retained request body is a bare PreparedRequest, not a native event.
pub(super) fn prepared(bytes: &[u8]) -> Result<Fields<'_>> {
    let prepared: Prepared<'_> = parse(bytes)?;
    Ok(Fields {
        request: prepared.request.get().as_bytes(),
        outcome: prepared.outcome.get().as_bytes(),
        effect: prepared.effect.get().as_bytes(),
        claims: Vec::new(),
    })
}

#[derive(Deserialize)]
struct Write<'a> {
    #[serde(borrow)]
    product: &'a RawValue,
}

/// A restore material may contain many requests. Keep one original product
/// encoding in memory and preserve carried imports as declared claims.
pub(super) fn visit_restored(
    reader: impl Read,
    mut visit: impl for<'a> FnMut(Fields<'a>) -> Result<()>,
) -> Result<()> {
    visit_writes(reader, |encoded| {
        let write: Write<'_> = parse(encoded.get().as_bytes())?;
        visit(record(
            "polylogue.request_restore",
            write.product.get().as_bytes(),
        )?)
    })
}

pub(crate) fn visit_writes(
    reader: impl Read,
    mut visit: impl FnMut(&RawValue) -> Result<()>,
) -> Result<()> {
    struct Writes<'a, F> {
        visit: &'a mut F,
        failure: &'a mut Option<Error>,
    }
    impl<'de, F: FnMut(&RawValue) -> Result<()>> Visitor<'de> for Writes<'_, F> {
        type Value = ();
        fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("retained restoration product array")
        }
        fn visit_seq<A: SeqAccess<'de>>(
            self,
            mut sequence: A,
        ) -> std::result::Result<(), A::Error> {
            while let Some(encoded) = sequence.next_element::<Box<RawValue>>()? {
                let result = (self.visit)(&encoded);
                if let Err(error) = result {
                    *self.failure = Some(error);
                    return Err(serde::de::Error::custom(
                        "retained restoration could not be projected",
                    ));
                }
            }
            Ok(())
        }
    }
    let mut decoder = serde_json::Deserializer::from_reader(BufReader::new(reader));
    decoder.disable_recursion_limit();
    let mut failure = None;
    let result = serde::de::Deserializer::deserialize_seq(
        &mut decoder,
        Writes {
            visit: &mut visit,
            failure: &mut failure,
        },
    );
    if let Some(error) = failure {
        return Err(error);
    }
    result?;
    decoder.end()?;
    Ok(())
}
