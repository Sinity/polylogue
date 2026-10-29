use super::*;
use std::convert::Infallible;

#[derive(Default)]
struct Graph {
    revisions: Vec<Reference>,
    revises: Vec<(Reference, Reference)>,
    withdraws: Vec<(Reference, Reference)>,
}
fn r(value: &str) -> Reference {
    Reference::parse(value).unwrap()
}
impl RevisionLinks for Graph {
    type Error = Infallible;
    fn revisions(&mut self, wanted: &Reference) -> Result<Vec<Reference>, Self::Error> {
        Ok(self
            .revisions
            .iter()
            .filter(|revision| root(revision).as_ref() == Some(wanted))
            .cloned()
            .collect())
    }
    fn revisers(&mut self, wanted: &Reference) -> Result<Vec<Reference>, Self::Error> {
        Ok(self
            .revises
            .iter()
            .filter(|(_, target)| target == wanted)
            .map(|(source, _)| source.clone())
            .collect())
    }
    fn withdrawers(&mut self, wanted: &Reference) -> Result<Vec<Reference>, Self::Error> {
        Ok(self
            .withdraws
            .iter()
            .filter(|(_, target)| target == wanted)
            .map(|(source, _)| source.clone())
            .collect())
    }
}
#[test]
fn graph_currency_exposes_forks_and_withdrawal_restoration() {
    let (first, left, right, withdrawal, restore) = (
        r("assertion:root@first"),
        r("assertion:root@left"),
        r("assertion:root@right"),
        r("assertion:withdraw@withdraw"),
        r("assertion:restore@restore"),
    );
    let mut graph = Graph {
        revisions: vec![
            first.clone(),
            left.clone(),
            right.clone(),
            withdrawal.clone(),
            restore.clone(),
        ],
        revises: vec![
            (left.clone(), first.clone()),
            (right.clone(), first.clone()),
        ],
        withdraws: vec![],
    };
    assert_eq!(
        Currency::new(&mut graph)
            .heads(&r("assertion:root"))
            .unwrap(),
        vec![left.clone(), right.clone()]
    );
    graph.withdraws = vec![
        (withdrawal.clone(), left.clone()),
        (withdrawal.clone(), right.clone()),
    ];
    assert_eq!(
        Currency::new(&mut graph)
            .heads(&r("assertion:root"))
            .unwrap(),
        vec![first.clone()]
    );
    graph.withdraws.push((restore, withdrawal));
    assert_eq!(
        Currency::new(&mut graph)
            .heads(&r("assertion:root"))
            .unwrap(),
        vec![left, right]
    );
}
#[test]
fn graph_currency_is_identical_for_every_input_order() {
    let nodes = [
        r("variant:root@first"),
        r("variant:root@second"),
        r("assertion:withdraw@w"),
    ];
    for order in [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ] {
        let mut graph = Graph {
            revisions: order.iter().map(|n| nodes[*n].clone()).collect(),
            revises: vec![(nodes[1].clone(), nodes[0].clone())],
            withdraws: vec![(nodes[2].clone(), nodes[1].clone())],
        };
        assert_eq!(
            Currency::new(&mut graph).heads(&r("variant:root")).unwrap(),
            vec![nodes[0].clone()]
        );
        graph.withdraws.push((nodes[2].clone(), r("variant:root")));
        assert!(
            Currency::new(&mut graph)
                .heads(&r("variant:root"))
                .unwrap()
                .is_empty()
        );
    }
}
#[test]
fn withdrawing_the_root_of_a_withdrawal_restores_its_targets() {
    let target = r("assertion:target@t");
    let withdrawal = r("assertion:withdraw@w");
    let restore = r("assertion:restore@r");
    let mut graph = Graph {
        revisions: vec![target.clone(), withdrawal.clone(), restore.clone()],
        withdraws: vec![
            (withdrawal, target.clone()),
            (restore, r("assertion:withdraw")),
        ],
        ..Graph::default()
    };
    assert!(Currency::new(&mut graph).current(&target).unwrap());
}
#[test]
fn cyclic_withdrawal_has_a_typed_error_instead_of_an_order_winner() {
    let first = r("assertion:first@1");
    let second = r("assertion:second@2");
    let mut graph = Graph {
        revisions: vec![first.clone(), second.clone()],
        withdraws: vec![(first.clone(), second.clone()), (second, first.clone())],
        ..Graph::default()
    };
    assert!(matches!(
        Currency::new(&mut graph).withdrawn(&first),
        Err(CurrencyError::Cycle(_))
    ));
}
