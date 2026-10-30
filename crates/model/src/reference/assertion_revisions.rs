//! Field codec for revision bases. The internal identity pair also serves
//! variants and pages; a wire assertion base always names its complete grain.
use super::{Reference, RevisionReference};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

pub fn serialize<S: Serializer>(
    bases: &[RevisionReference],
    serializer: S,
) -> Result<S::Ok, S::Error> {
    let references = bases
        .iter()
        .map(|base| {
            if base.revision.is_none() {
                return Err(serde::ser::Error::custom(
                    "an assertion revision base must be exact",
                ));
            }
            Ok(Reference::Assertion(base.clone()))
        })
        .collect::<Result<Vec<_>, S::Error>>()?;
    references.serialize(serializer)
}

pub fn deserialize<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Vec<RevisionReference>, D::Error> {
    Vec::<Reference>::deserialize(deserializer)?
        .into_iter()
        .map(|reference| match reference {
            Reference::Assertion(base) if base.revision.is_some() => Ok(base),
            _ => Err(serde::de::Error::custom(
                "an assertion revision base must be an exact assertion reference",
            )),
        })
        .collect()
}
