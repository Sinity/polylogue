use super::*;

const KEY: &[u8] = b"-----BEGIN RSA PRIVATE KEY-----";
const END: &[u8] = b"-----END RSA PRIVATE KEY-----";

#[test]
fn a_private_key_header_opens_and_its_end_in_text_closes() {
    let open = after_block(None, KEY, None);
    assert_eq!(open, Some(Some(END.to_vec())));
    assert_eq!(
        after_block(
            open.clone(),
            b"",
            Some(b"abc -----END RSA PRIVATE KEY----- x")
        ),
        None
    );
    // An END elsewhere in the stored block is not read as text: it stays open.
    assert_eq!(after_block(open.clone(), END, None), open);
    // Another kind's END does not close it.
    assert_eq!(
        after_block(open.clone(), b"", Some(b"-----END PRIVATE KEY-----")),
        open
    );
}

#[test]
fn only_private_key_headers_open_and_a_header_and_end_in_one_text_close() {
    assert_eq!(
        after_block(None, b"-----BEGIN CERTIFICATE-----", None),
        None
    );
    assert_eq!(after_block(None, b"-----BEGIN PRIVATE KEY", None), None);
    assert_eq!(
        after_block(
            None,
            b"",
            Some(b"-----BEGIN PRIVATE KEY-----body-----END PRIVATE KEY-----")
        ),
        None
    );
    assert_eq!(
        after_block(None, b"-----BEGIN PRIVATE KEY-----", None),
        Some(Some(b"-----END PRIVATE KEY-----".to_vec()))
    );
}

#[test]
fn a_label_too_long_to_keep_never_closes() {
    let mut header = b"-----BEGIN ".to_vec();
    header.extend(std::iter::repeat_n(b'A', 600));
    header.extend(b" PRIVATE KEY-----");
    assert_eq!(after_block(None, &header, None), Some(None));
}
