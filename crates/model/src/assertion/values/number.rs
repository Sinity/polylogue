//! Exact JSON-number comparison without expanding exponents or rounding.
use serde_json::Number;
use std::cmp::Ordering;

#[derive(Debug, PartialEq, Eq)]
struct Integer {
    negative: bool,
    digits: String,
}
impl Integer {
    fn parse(text: &str) -> Self {
        let negative = text.starts_with('-');
        let digits = text.trim_start_matches(['+', '-']).trim_start_matches('0');
        Self {
            negative: negative && !digits.is_empty(),
            digits: if digits.is_empty() {
                "0".into()
            } else {
                digits.into()
            },
        }
    }
    fn cmp(&self, other: &Self) -> Ordering {
        match (self.negative, other.negative) {
            (true, false) => Ordering::Less,
            (false, true) => Ordering::Greater,
            _ => {
                let order = magnitude(&self.digits, &other.digits);
                if self.negative {
                    order.reverse()
                } else {
                    order
                }
            }
        }
    }
    fn add(&self, delta: i128) -> Self {
        let other = Self::parse(&delta.to_string());
        if self.negative == other.negative {
            Self {
                negative: self.negative,
                digits: add(&self.digits, &other.digits),
            }
        } else {
            let order = magnitude(&self.digits, &other.digits);
            if order == Ordering::Equal {
                Self::parse("0")
            } else if order == Ordering::Greater {
                Self {
                    negative: self.negative,
                    digits: subtract(&self.digits, &other.digits),
                }
            } else {
                Self {
                    negative: other.negative,
                    digits: subtract(&other.digits, &self.digits),
                }
            }
        }
    }
}
fn magnitude(left: &str, right: &str) -> Ordering {
    left.len().cmp(&right.len()).then_with(|| left.cmp(right))
}
fn add(left: &str, right: &str) -> String {
    let mut left = left.bytes().rev();
    let mut right = right.bytes().rev();
    let mut carry = 0;
    let mut digits = Vec::new();
    loop {
        let (a, b) = (left.next(), right.next());
        if a.is_none() && b.is_none() && carry == 0 {
            break;
        }
        let total = a.map_or(0, |v| v - b'0') + b.map_or(0, |v| v - b'0') + carry;
        digits.push(b'0' + total % 10);
        carry = total / 10;
    }
    digits.reverse();
    String::from_utf8(digits).expect("decimal digits")
}
// left >= right, both nonnegative decimal integers.
fn subtract(left: &str, right: &str) -> String {
    let mut right = right.bytes().rev();
    let mut borrow = 0;
    let mut digits = Vec::new();
    for a in left.bytes().rev() {
        let b = right.next().map_or(0, |v| v - b'0') + borrow;
        let a = a - b'0';
        digits.push(
            b'0' + if a < b {
                borrow = 1;
                a + 10 - b
            } else {
                borrow = 0;
                a - b
            },
        );
    }
    while digits.len() > 1 && digits.last() == Some(&b'0') {
        digits.pop();
    }
    digits.reverse();
    String::from_utf8(digits).expect("decimal digits")
}
struct Decimal {
    negative: bool,
    order: Integer,
    digits: String,
}
impl Decimal {
    fn new(number: &Number) -> Self {
        let text = number.to_string();
        let (mantissa, exponent) = text.split_once(['e', 'E']).unwrap_or((&text, "0"));
        let negative = mantissa.starts_with('-');
        let mantissa = mantissa.trim_start_matches('-');
        let before = mantissa.find('.').unwrap_or(mantissa.len());
        let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
        let significant = digits.trim_start_matches('0');
        if significant.is_empty() {
            return Self {
                negative: false,
                order: Integer::parse("0"),
                digits: String::new(),
            };
        }
        let leading = digits.len() - significant.len();
        let order = Integer::parse(exponent).add(before as i128 - leading as i128 - 1);
        Self {
            negative,
            order,
            digits: significant.trim_end_matches('0').into(),
        }
    }
    fn cmp(&self, other: &Self) -> Ordering {
        match (self.digits.is_empty(), other.digits.is_empty()) {
            (true, true) => return Ordering::Equal,
            (true, false) => {
                return if other.negative {
                    Ordering::Greater
                } else {
                    Ordering::Less
                };
            }
            (false, true) => {
                return if self.negative {
                    Ordering::Less
                } else {
                    Ordering::Greater
                };
            }
            _ => {}
        }
        match (self.negative, other.negative) {
            (true, false) => Ordering::Less,
            (false, true) => Ordering::Greater,
            _ => {
                let order = self.order.cmp(&other.order).then_with(|| {
                    let length = self.digits.len().max(other.digits.len());
                    self.digits
                        .bytes()
                        .chain(std::iter::repeat(b'0'))
                        .take(length)
                        .cmp(
                            other
                                .digits
                                .bytes()
                                .chain(std::iter::repeat(b'0'))
                                .take(length),
                        )
                });
                if self.negative {
                    order.reverse()
                } else {
                    order
                }
            }
        }
    }
}
pub(super) fn compare(left: &Number, right: &Number) -> Ordering {
    Decimal::new(left).cmp(&Decimal::new(right))
}
pub(super) fn integer(number: &Number) -> bool {
    let decimal = Decimal::new(number);
    decimal.digits.is_empty()
        || decimal
            .order
            .cmp(&Integer::parse(&(decimal.digits.len() - 1).to_string()))
            != Ordering::Less
}
/// One text for every spelling of a number: `1`, `1.0` and `10e-1` give the
/// same text, and numbers that compare unequal give different texts.
pub(super) fn canonical(number: &Number) -> String {
    let decimal = Decimal::new(number);
    if decimal.digits.is_empty() {
        return "0".into();
    }
    format!(
        "{}0.{}e{}{}",
        if decimal.negative { "-" } else { "" },
        decimal.digits,
        if decimal.order.negative { "-" } else { "" },
        decimal.order.digits
    )
}
