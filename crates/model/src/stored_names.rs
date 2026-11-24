//! The names a fieldless enum is stored, queried and serialized by. One
//! declaration gives the enum its variants and their names, `ALL`, `NAMES`,
//! the name lookup methods, and its serde and JSON Schema forms. The default
//! methods are `as_str` and `from_name`; a leading `name_fn, parse_fn;` names
//! them otherwise and makes `ALL` an array. The query language offers `NAMES`
//! as a field's closed set, so a value the model never stores is never
//! offered and every stored value is accepted; the schema text takes its
//! `CHECK` lists from the same `NAMES`.

/// A fieldless enum stored and queried by name, so a store can decode a
/// column into it and refuse a name the enum does not have.
pub trait StoredName: Copy + 'static {
    fn as_str(self) -> &'static str;
    fn from_name(name: &str) -> Option<Self>;
}

/// Declares a fieldless enum and the name of each variant:
///
/// ```ignore
/// stored_names! {
///     /// What a delivery answered.
///     #[derive(Debug, Clone, Copy, PartialEq, Eq)]
///     pub enum DeliveryEvent {
///         SessionStart => "session_start",
///         /// A hint just before a tool runs.
///         Advice => "advice",
///     }
/// }
/// ```
///
/// Serialize, Deserialize and JsonSchema are derived from the declared names,
/// so the crate using it depends on `serde` and `schemars`.
#[macro_export]
macro_rules! stored_names {
    (
        $(#[$meta:meta])*
        $vis:vis enum $ty:ident { $($(#[$variant_meta:meta])* $variant:ident => $name:literal),+ $(,)? }
    ) => {
        $crate::stored_names!(@enum [$(#[$meta])*] $vis $ty { $($(#[$variant_meta])* $variant => $name),+ });
        $crate::stored_names!(@impl $ty, as_str, from_name, &'static [$ty], &[$($ty::$variant),+]; { $($variant => $name),+ });
    };
    (
        $name_fn:ident, $parse_fn:ident;
        $(#[$meta:meta])*
        $vis:vis enum $ty:ident { $($(#[$variant_meta:meta])* $variant:ident => $name:literal),+ $(,)? }
    ) => {
        $crate::stored_names!(@enum [$(#[$meta])*] $vis $ty { $($(#[$variant_meta])* $variant => $name),+ });
        $crate::stored_names!(@impl $ty, $name_fn, $parse_fn, [$ty; [$($name),+].len()], [$($ty::$variant),+]; { $($variant => $name),+ });
    };
    (@enum [$(#[$meta:meta])*] $vis:vis $ty:ident { $($(#[$variant_meta:meta])* $variant:ident => $name:literal),+ }) => {
        $(#[$meta])*
        #[derive(::serde::Serialize, ::serde::Deserialize, ::schemars::JsonSchema)]
        $vis enum $ty {
            $($(#[$variant_meta])* #[serde(rename = $name)] $variant),+
        }
    };
    (@impl $ty:ident, $name_fn:ident, $parse_fn:ident, $all_type:ty, $all:expr; { $($variant:ident => $name:literal),+ }) => {
        impl $crate::StoredName for $ty {
            fn as_str(self) -> &'static str {
                $ty::$name_fn(self)
            }
            fn from_name(name: &str) -> Option<Self> {
                $ty::$parse_fn(name)
            }
        }

        impl $ty {
            pub const ALL: $all_type = $all;
            pub const NAMES: &'static [&'static str] = &[$($name),+];

            pub const fn $name_fn(self) -> &'static str {
                match self {
                    $($ty::$variant => $name),+
                }
            }

            pub fn $parse_fn(name: &str) -> Option<Self> {
                match name {
                    $($name => Some($ty::$variant),)+
                    _ => None,
                }
            }
        }
    };
}

#[cfg(test)]
mod tests;
