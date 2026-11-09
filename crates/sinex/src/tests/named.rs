use super::*;
use polylogue_model::{
    ChangePosition, RunId, SessionId, Timestamp,
    authorship::{CallerEvidence, Channel, compute_authorship},
    query_objects::{
        ExecutionBinding, ExecutionPresentation, QueryExecutionOptions, QueryObject, QueryRun,
        QueryRunOutcome, ReplayBinding, RunBinding, RunOutcome,
    },
};
use polylogue_store::primary::NamedProductSelector;
use query_language::{
    cursor::CursorContext,
    identity,
    time::{Clock, DaySetting},
};
use std::{collections::BTreeMap, io::Read};

pub(crate) fn phases() -> Vec<(String, Product)> {
    let normal_form = json!({"source":{"kind":"unit","unit":"sessions","condition":null},"attach":[],"transforms":[]});
    let parameters = BTreeMap::new();
    let id = identity::question_id_from_normal_form(&normal_form, &parameters).unwrap();
    let mut syntax = normal_form.clone();
    syntax["source"]["implicit"] = false.into();
    let object = QueryObject {
        id,
        normal_form,
        parameters,
        query: serde_json::from_value(syntax).unwrap(),
        source_outputs: vec![],
        definitions: vec![],
        dependencies: vec![],
        edges: vec![],
    };
    let clock = Clock::new(
        jiff::Timestamp::from_millisecond(1000).unwrap(),
        DaySetting::new(jiff::tz::TimeZone::UTC, jiff::civil::time(4, 0, 0, 0)).unwrap(),
    );
    let run = QueryRun {
        id: RunId::parse("00000000-0000-7000-8000-000000000001").unwrap(),
        question: Some(id),
        at: Timestamp::from_unix_ms(1000).unwrap(),
        read_at: ChangePosition {
            generation: "original-scope-generation".into(),
            index: 7,
            archive: 3,
        },
        machine: "synthetic-machine".into(),
        authorship: compute_authorship(
            Channel::Mcp,
            &CallerEvidence {
                session: Some(SessionId::parse("codex:synthetic-agent").unwrap()),
                ..Default::default()
            },
            None,
        ),
        binding: RunBinding::Original {
            execution: Box::new(ExecutionBinding {
                replay: ReplayBinding {
                    bound_ast: object.query.clone(),
                    source_outputs: vec![],
                    definitions: vec![],
                    context: CursorContext {
                        question: id,
                        clock: clock.wire(),
                        resolved_times: vec![],
                        random_seed: 7,
                        read_positions: BTreeMap::from([("index".into(), "7".into())]),
                        versions: BTreeMap::from([("scope".into(), "original-scope".into())]),
                    },
                    scope_fingerprint: Some("original-scope".into()),
                    original_text: "sessions".into(),
                },
                query_options: QueryExecutionOptions {
                    direct_reference: None,
                    fields: vec![],
                    self_session: None,
                    embeddings: false,
                    why: false,
                    ranking_identity: None,
                },
                presentation: ExecutionPresentation::Query,
                accepted_request: Some(
                    json!({"operation":"query","id":"synthetic-caller".repeat(5000)}),
                ),
            }),
        },
    };
    let outcome = QueryRunOutcome {
        run: run.id.clone(),
        binding_hash: run.binding_hash().unwrap(),
        run_hash: run.record_hash().unwrap(),
        at: run.at,
        outcome: RunOutcome::Ok,
        count: Some(5),
        exact: true,
        next: None,
    };
    vec![
        (
            format!("query:{id}"),
            Product::Query {
                object: Box::new(object),
            },
        ),
        (
            format!("query-run:{}", run.id),
            Product::QueryRun { run: Box::new(run) },
        ),
        (
            format!("query-run-outcome:{}", outcome.run),
            Product::QueryRunOutcome {
                outcome: Box::new(outcome),
            },
        ),
    ]
}

fn retained(sinex: &Sinex, selector: NamedProductSelector) -> Vec<Product> {
    let mut found = vec![];
    sinex
        .visit_named_products(&selector, &mut |product| {
            found.push(product.clone());
            Ok(())
        })
        .unwrap();
    found
}

#[test]
fn named_query_phases_survive_restart_and_publication_until_confirmed_application() {
    let root = tempfile::tempdir().unwrap();
    let expected = phases();
    let sinex = client(root.path());
    for (name, product) in &expected {
        let write = ProductWrite {
            product: product.clone(),
            location: None,
            parents: vec![],
            machine: Some("synthetic-machine".into()),
            output_index: 0,
        };
        assert!(
            sinex
                .prepare_named_products(name, std::slice::from_ref(&write))
                .unwrap()
                .pending
        );
        sinex.prepare_named_products(name, &[write]).unwrap();
        let payload = crate::products::payload(product, 0).unwrap();
        let kind = crate::products::kind(product);
        let declaration = crate::kinds::declarations("polylogue")
            .into_iter()
            .find(|d| d.kind.name == kind)
            .unwrap();
        let version = match kind {
            "polylogue.query" => 2,
            "polylogue.query_run" => 4,
            "polylogue.query_run_outcome" => 1,
            other => panic!("unexpected named query phase kind: {other}"),
        };
        assert_eq!(declaration.kind.version, version, "{kind}");
        assert_eq!(
            declaration.replace_rule,
            Some(sinex_model::ReplaceRule::SuppressRepeat)
        );
        jsonschema::validator_for(declaration.payload_schema.as_ref().unwrap())
            .unwrap()
            .validate(&payload)
            .unwrap();
        assert_eq!(crate::products::decode(kind, payload).unwrap(), *product);
    }
    assert_ne!(
        crate::products::key(&expected[1].1, 0).unwrap(),
        crate::products::key(&expected[2].1, 0).unwrap()
    );
    drop(sinex);
    let sinex = client(root.path());
    let prefix = format!("query:{}", &expected[0].0[6..18]);
    assert_eq!(
        retained(&sinex, NamedProductSelector::Prefix(prefix)),
        vec![expected[0].1.clone()]
    );
    assert!(retained(&sinex, NamedProductSelector::Exact("query:missing".into())).is_empty());
    for (ordinal, (name, product)) in expected.iter().enumerate() {
        assert_eq!(
            retained(&sinex, NamedProductSelector::Exact(name.clone())),
            vec![product.clone()]
        );
        let observation = format!("{name}:observation");
        let bytes = {
            let queue = sinex.queue().unwrap();
            let mut bytes = vec![];
            queue
                .body(&observation)
                .unwrap()
                .read_to_end(&mut bytes)
                .unwrap();
            bytes
        };
        let physical_root = root.path().join(format!("receipt-{ordinal}"));
        std::fs::create_dir_all(&physical_root).unwrap();
        let physical = extent(&physical_root, &bytes);
        sinex.queue().unwrap().acknowledge(serde_json::from_value(json!({"number":ordinal+1,"frame":observation,"material":crate::material_id("polylogue-test",&observation),"material_extents":[physical.encode()],"admitted":[],"content_hash":null,"result":"admitted","counts":crate::tests::zero_counts(),"commit_seq":null})).unwrap()).unwrap();
    }
    sinex.drain_observations().unwrap();
    for (ordinal, (name, _)) in expected.iter().enumerate() {
        sinex.queue().unwrap().acknowledge(serde_json::from_value(json!({"number":ordinal+10,"frame":name,"material":null,"admitted":[],"content_hash":null,"result":"admitted","counts":crate::tests::zero_counts(),"commit_seq":10})).unwrap()).unwrap();
    }
    sinex
        .queue()
        .unwrap()
        .retire(
            &crate::tests::applied(9),
            &crate::tests::every_event_applied,
        )
        .unwrap();
    drop(sinex);
    let sinex = client(root.path());
    for (name, product) in &expected {
        assert_eq!(
            retained(&sinex, NamedProductSelector::Exact(name.clone())),
            vec![product.clone()]
        );
    }
    sinex
        .queue()
        .unwrap()
        .retire(
            &crate::tests::applied(10),
            &crate::tests::every_event_applied,
        )
        .unwrap();
    for (name, _) in expected {
        assert!(retained(&sinex, NamedProductSelector::Exact(name)).is_empty());
    }
}

#[test]
fn source_material_identity_preserves_distinct_receiving_producers() {
    let name = "query:synthetic-question:observation";
    assert_ne!(
        crate::material_id("receiver-one", name),
        crate::material_id("receiver-two", name)
    );
    assert_eq!(
        crate::material_id("receiver-one", name),
        crate::material_id("receiver-one", name)
    );
}

#[test]
fn an_attachment_binding_is_an_ordinary_named_product() {
    let root = tempfile::tempdir().unwrap();
    let sinex = client(root.path());
    let write = ProductWrite {
        product: Product::AttachmentBinding {
            record: Box::new(super::custody_rebuild::witness()),
        },
        location: None,
        parents: vec![],
        machine: None,
        output_index: 0,
    };
    let prepared = sinex
        .prepare_named_products("ordinary-name", std::slice::from_ref(&write))
        .unwrap();
    assert_eq!(prepared.identity, "ordinary-name");
    assert!(prepared.pending);
    assert_eq!(sinex.status().unwrap().prepared_materials, 1);
}
