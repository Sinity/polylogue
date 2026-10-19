//! The names a fieldless enum is stored and queried by. One declaration
//! gives the enum `ALL`, `NAMES`, `as_str` and `from_name`; the query
//! language offers `NAMES` as a field's closed set, so a value the model
//! never stores is never offered and every stored value is accepted. The
//! names are the enum's serde names; the model's tests check that.

macro_rules! stored_names {
    ($ty:ident { $($variant:ident => $name:literal),+ $(,)? }) => {
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

/// The distinct strings of `xs`, in first-seen order, padded with `""` to
/// `N`; the count is the second value. Lets a table derive a closed set
/// from a mapping at compile time.
pub const fn distinct<const N: usize>(xs: [&'static str; N]) -> ([&'static str; N], usize) {
    let mut out = [""; N];
    let mut n = 0;
    let mut i = 0;
    while i < N {
        let mut j = 0;
        let mut seen = false;
        while j < n {
            if str_eq(out[j], xs[i]) {
                seen = true;
            }
            j += 1;
        }
        if !seen {
            out[n] = xs[i];
            n += 1;
        }
        i += 1;
    }
    (out, n)
}

const fn str_eq(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    if a.len() != b.len() {
        return false;
    }
    let mut i = 0;
    while i < a.len() {
        if a[i] != b[i] {
            return false;
        }
        i += 1;
    }
    true
}

#[cfg(test)]
mod tests;
