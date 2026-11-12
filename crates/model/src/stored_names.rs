//! The names a fieldless enum is stored and queried by. One declaration
//! gives the enum `ALL`, `NAMES`, `as_str` and `from_name`; the query
//! language offers `NAMES` as a field's closed set, so a value the model
//! never stores is never offered and every stored value is accepted. The
//! names are the enum's serde names; the model's tests check that.

/// A fieldless enum stored and queried by name, so a store can decode a
/// column into it and refuse a name the enum does not have.
pub trait StoredName: Copy + 'static {
    const NAMES: &'static [&'static str];
    fn as_str(self) -> &'static str;
    fn from_name(name: &str) -> Option<Self>;
}

macro_rules! stored_names {
    ($ty:ident { $($variant:ident => $name:literal),+ $(,)? }) => {
        impl $crate::StoredName for $ty {
            const NAMES: &'static [&'static str] = $ty::NAMES;
            fn as_str(self) -> &'static str {
                $ty::as_str(self)
            }
            fn from_name(name: &str) -> Option<Self> {
                $ty::from_name(name)
            }
        }

        impl $ty {
            pub const ALL: &'static [$ty] = &[$($ty::$variant),+];
            pub const NAMES: &'static [&'static str] = &[$($name),+];

            pub const fn as_str(self) -> &'static str {
                match self {
                    $($ty::$variant => $name),+
                }
            }

            pub fn from_name(name: &str) -> Option<Self> {
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
