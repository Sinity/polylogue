use super::*;

#[test]
fn dollars_parse_exactly_and_refuse_out_of_range() {
    assert_eq!(nusd_from_usd("2.90"), Some(2_900_000_000));
    assert_eq!(
        nusd_from_usd("12345678.123456789"),
        Some(12_345_678_123_456_789)
    );
    assert_eq!(nusd_from_usd("0.0000000005"), Some(1));
    assert_eq!(nusd_from_usd("9223372036.854775807"), Some(i64::MAX));
    assert_eq!(nusd_from_usd("9223372036.854775808"), None);
    assert_eq!(nusd_from_usd("1e300"), None);
    assert_eq!(nusd_from_usd("NaN"), None);
}

#[test]
fn exact_dollars_round_trip_through_text_and_json() {
    for (nusd, text) in [
        (0, "0"),
        (1, "0.000000001"),
        (-5, "-0.000000005"),
        (1_234_500_000, "1.2345"),
        (2_000_000_000, "2"),
        (i64::MAX, "9223372036.854775807"),
        (i64::MIN, "-9223372036.854775808"),
    ] {
        assert_eq!(usd_text(nusd), text);
        assert_eq!(usd_number(nusd).to_string(), text);
        assert_eq!(nusd_from_usd(text), Some(nusd));
    }
}

#[test]
fn display_rounds_cents_and_keeps_sub_cent_amounts_visible() {
    assert_eq!(usd_display(0), "$0");
    assert_eq!(usd_display(12_345_000_000), "$12.35");
    assert_eq!(usd_display(10_000_000), "$0.01");
    assert_eq!(usd_display(-12_344_999_999), "-$12.34");
    assert_eq!(usd_display(1_234_567), "$0.00123");
    assert_eq!(usd_display(9_995_000), "$0.01");
    assert_eq!(usd_display(42), "$0.000000042");
}

#[test]
fn quotients_round_half_away_from_zero() {
    assert_eq!(rounded_quotient(5, 2), Some(3));
    assert_eq!(rounded_quotient(-5, 2), Some(-3));
    assert_eq!(rounded_quotient(4, 3), Some(1));
    assert_eq!(rounded_quotient(-4, 3), Some(-1));
    assert_eq!(rounded_quotient(5, -2), Some(-3));
    assert_eq!(rounded_quotient(1, 0), None);
}

#[test]
fn an_amount_carries_nano_usd_and_exact_dollars() {
    assert_eq!(
        serde_json::to_string(&Amount(1_234_500_001)).unwrap(),
        r#"{"nusd":1234500001,"usd":1.234500001}"#
    );
}
