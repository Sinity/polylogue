use super::*;
use std::io::Cursor;

#[test]
fn the_head_starts_after_a_byte_order_mark_and_keeps_its_length() {
    let mut marked = BYTE_ORDER_MARK.to_vec();
    marked.extend(br#"[{"uuid":"x"}]"#);
    marked.extend(std::iter::repeat_n(b' ', 100));
    let head = read_head(&mut Cursor::new(marked)).unwrap();
    assert_eq!(head.len(), HEAD_LEN);
    assert_eq!(first_byte(&head), Some(b'['));
    assert_eq!(first_key(&head), Some("uuid"));
}

#[test]
fn the_head_of_an_unmarked_input_is_unchanged() {
    let head = read_head(&mut Cursor::new(b"\xEF\xBBx".to_vec())).unwrap();
    assert_eq!(head, b"\xEF\xBBx");
    let head = read_head(&mut Cursor::new(br#"{"a":1}"#.to_vec())).unwrap();
    assert_eq!(head, br#"{"a":1}"#);
}

#[test]
fn rewinding_skips_only_a_whole_mark() {
    let mut marked = Cursor::new(b"\xEF\xBB\xBF{}".to_vec());
    rewind_past_byte_order_mark(&mut marked).unwrap();
    assert_eq!(marked.position(), 3);
    let mut plain = Cursor::new(b"\xEF\xBBx".to_vec());
    plain.set_position(2);
    rewind_past_byte_order_mark(&mut plain).unwrap();
    assert_eq!(plain.position(), 0);
}
