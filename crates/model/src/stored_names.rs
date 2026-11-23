//! The names a fieldless enum is stored and queried by. One declaration
//! gives the enum `ALL`, `NAMES` and its name lookup methods. The default
//! methods are `as_str` and `from_name`; the named form uses the declared
//! method names and an `ALL` array. The query language offers `NAMES` as a
//! field's closed set, so a value the model never stores is never offered
//! and every stored value is accepted. The
//! names are the enum's serde names; the model's tests check that.

/// A fieldless enum stored and queried by name, so a store can decode a
/// column into it and refuse a name the enum does not have.
pub trait StoredName: Copy + 'static {
    fn as_str(self) -> &'static str;
    fn from_name(name: &str) -> Option<Self>;
}

#[macro_export]
macro_rules! stored_names {
    ($ty:ident { $($variant:ident => $name:literal),+ $(,)? }) => {
        $crate::stored_names!(@impl $ty, as_str, from_name, &'static [$ty], &[$($ty::$variant),+]; { $($variant => $name),+ });
    };
    ($ty:ident, $name_fn:ident, $parse_fn:ident { $($variant:ident => $name:literal),+ $(,)? }) => {
        $crate::stored_names!(@impl $ty, $name_fn, $parse_fn, [$ty; [$($name),+].len()], [$($ty::$variant),+]; { $($variant => $name),+ });
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
