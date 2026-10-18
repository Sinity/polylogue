use super::*;
use polylogue_store::parser_state::{Accounting, Head};
fn fixture() -> (Receipt, Head) {
    let token = Token {
        index_generation: "synthetic-index".into(),
        input_pk: 1,
        chain: 1,
        generation: 2,
        step: 3,
        basis: [4; 32],
    };
    let basis = ParsedFrom {
        input: "synthetic-input".into(),
        original_hash: Some(BlobHash::of(b"raw")),
        chain: 1,
        revision: 1,
        length: 10,
        parser_version: 1,
        interpretation: BlobHash::of(b"interpretation"),
    };
    let head = Head {
        token: token.clone(),
        parser_version: 1,
        acquired_upto: 10,
        inspected_upto: 10,
        parsed_upto: 10,
        status: Phase::Done,
        finalization_pending: false,
        counts: Accounting::default(),
    };
    (
        Receipt {
            predecessor: token,
            dispositions: [0; 5],
            unsupported_effects: false,
            rows: SourceRows {
                scope: SourceRowScope::ImportedSourceRows,
                parsed_from: basis,
                state: SourceRowState::Complete,
            },
        },
        head,
    )
}
#[test]
fn receipt_rejects_any_changed_parser_state_or_input_version() {
    let (receipt, head) = fixture();
    receipt
        .validate_published(&head, &receipt.rows.parsed_from)
        .unwrap();
    for alter in [
        |t: &mut Token| t.index_generation.push('x'),
        |t: &mut Token| t.input_pk += 1,
        |t: &mut Token| t.chain += 1,
        |t: &mut Token| t.generation += 1,
        |t: &mut Token| t.step += 1,
        |t: &mut Token| t.basis[0] ^= 1,
    ] {
        let mut wrong = head.clone();
        alter(&mut wrong.token);
        assert!(
            receipt
                .validate_published(&wrong, &receipt.rows.parsed_from)
                .is_err()
        );
    }
    let mut wrong = receipt.rows.parsed_from.clone();
    wrong.interpretation = BlobHash::of(b"new recipe");
    assert!(receipt.validate_published(&head, &wrong).is_err());
    let mut wrong = receipt.rows.parsed_from.clone();
    wrong.original_hash = Some(BlobHash::of(b"different raw"));
    assert!(receipt.validate_published(&head, &wrong).is_err());
}
#[test]
fn damaged_status_counts_and_missing_effects_stay_unfinished() {
    let (receipt, head) = fixture();
    let mut wrong = head.clone();
    wrong.finalization_pending = true;
    assert!(
        receipt
            .validate_published(&wrong, &receipt.rows.parsed_from)
            .is_err()
    );
    let mut wrong = head.clone();
    wrong.counts.framed = 1;
    wrong.counts.undecodable = 1;
    assert!(
        receipt
            .validate_published(&wrong, &receipt.rows.parsed_from)
            .is_err()
    );
    let mut wrong = receipt.clone();
    wrong.unsupported_effects = true;
    assert!(
        wrong
            .validate_published(&head, &wrong.rows.parsed_from)
            .is_err()
    );
    let mut wrong = head.clone();
    wrong.parsed_upto = 9;
    assert!(
        receipt
            .validate_published(&wrong, &receipt.rows.parsed_from)
            .is_err()
    );
    let encoded = serde_json::to_vec(&receipt).unwrap();
    let decoded: Receipt = serde_json::from_slice(&encoded).unwrap();
    decoded
        .validate_published(&head, &decoded.rows.parsed_from)
        .unwrap();
}
