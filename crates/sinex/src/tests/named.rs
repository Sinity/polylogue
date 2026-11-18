use super::*;
use polylogue_model::{
    ChangePosition, RunId, SessionId, Timestamp,
    authorship::{Author, Authorship, Channel},
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
        authorship: Authorship {
            author: Author::Agent(Some(SessionId::parse("codex:synthetic-agent").unwrap())),
            via: Channel::Mcp,
        },
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
                    references: None,
                    fields: vec![],
                    self_session: None,
                    here: None,
                    seen_marks_of: None,
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

fn kept(sinex: &Sinex, selector: NamedProductSelector) -> Vec<Product> {
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
fn named_query_phases_survive_restart_until_applied() {
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
            "polylogue.query_run" => 6,
            "polylogue.query_run_outcome" => 2,
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
        assert_eq!(
            crate::products::decode(&crate::kinds::kind_reference(kind).unwrap(), payload).unwrap(),
            *product
        );
    }
    assert_ne!(
        crate::products::key(&expected[1].1, 0).unwrap(),
        crate::products::key(&expected[2].1, 0).unwrap()
    );
    drop(sinex);
    let sinex = client(root.path());
    let prefix = format!("query:{}", &expected[0].0[6..18]);
    assert_eq!(
        kept(&sinex, NamedProductSelector::Prefix(prefix)),
        vec![expected[0].1.clone()]
    );
    assert!(kept(&sinex, NamedProductSelector::Exact("query:missing".into())).is_empty());
    for (ordinal, (name, product)) in expected.iter().enumerate() {
        assert_eq!(
            kept(&sinex, NamedProductSelector::Exact(name.clone())),
            vec![product.clone()]
        );
        let observation = format!("{}:observation", crate::products::named_frame(name));
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
        let extent_root = root.path().join(format!("acknowledgment-{ordinal}"));
        std::fs::create_dir_all(&extent_root).unwrap();
        let stored = extent(&extent_root, &bytes);
        sinex.queue().unwrap().acknowledge(serde_json::from_value(json!({"number":ordinal+1,"frame":observation,"material":crate::material_id("polylogue-test",&observation),"material_extents":[stored.encode()],"admitted":[],"content_hash":null,"result":"admitted","counts":crate::tests::zero_counts(),"commit_seq":null})).unwrap()).unwrap();
    }
    sinex.drain_observations().unwrap();
    for (ordinal, (name, _)) in expected.iter().enumerate() {
        let name = crate::products::named_frame(name);
        sinex.queue().unwrap().acknowledge(serde_json::from_value(json!({"number":ordinal+10,"frame":name,"material":null,"admitted":[],"content_hash":null,"result":"admitted","counts":crate::tests::zero_counts(),"commit_seq":10})).unwrap()).unwrap();
    }
    sinex
        .queue()
        .unwrap()
        .remove_confirmed(
            &crate::tests::applied(9),
            &crate::tests::every_event_applied,
        )
        .unwrap();
    drop(sinex);
    let sinex = client(root.path());
    for (name, product) in &expected {
        assert_eq!(
            kept(&sinex, NamedProductSelector::Exact(name.clone())),
            vec![product.clone()]
        );
    }
    sinex
        .queue()
        .unwrap()
        .remove_confirmed(
            &crate::tests::applied(10),
            &crate::tests::every_event_applied,
        )
        .unwrap();
    for (name, _) in expected {
        assert!(kept(&sinex, NamedProductSelector::Exact(name)).is_empty());
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
fn attachment_binding_is_a_named_product() {
    let root = tempfile::tempdir().unwrap();
    let sinex = client(root.path());
    let write = ProductWrite {
        product: Product::AttachmentBinding {
            record: Box::new(super::binding_rebuild::binding_record()),
        },
        location: None,
        parents: vec![],
        machine: None,
        output_index: 0,
    };
    let prepared = sinex
        .prepare_named_products("plain-name", std::slice::from_ref(&write))
        .unwrap();
    assert_eq!(prepared.identity, "plain-name");
    assert!(prepared.pending);
    assert_eq!(sinex.status().unwrap().prepared_materials, 1);
}

#[test]
fn old_incomplete_query_run_and_current_completion_use_one_publisher_identity() {
    let root = tempfile::tempdir().unwrap();
    let sinex = client(root.path());
    let phases = phases();
    let (name, begin) = &phases[1];
    let (_, outcome) = &phases[2];
    // Preserved predecessor bytes remain in their immutable original frame.
    let mut old_payload = crate::products::payload(begin, 0).unwrap();
    let options = old_payload["record"]["run"]["binding"]["execution"]["query_options"]
        .as_object_mut()
        .unwrap();
    options.remove("references");
    options.insert("direct_reference".into(), serde_json::Value::Null);
    let old_run = &old_payload["record"]["run"];
    let converted =
        polylogue_store::query_objects::line_population::convert_run(old_run.clone()).unwrap();
    assert_eq!(
        Product::QueryRun {
            run: Box::new(converted.run.clone())
        },
        *begin
    );
    old_payload["binding_hash"] = serde_json::to_value(converted.old_binding).unwrap();
    old_payload["run_hash"] = serde_json::to_value(converted.old_run).unwrap();
    let old: sinex_protocol::Ingress = serde_json::from_value(json!({
        "kind":"events","source":{"name":"polylogue","version":8},"frame":name,
        "events":[{"kind":{"name":"polylogue.query_run","version":4},"payload":old_payload,"class":"open"}],"failures":[]
    })).unwrap();
    sinex.queue().unwrap().prepare(old.clone()).unwrap();
    let write = |product: &Product| ProductWrite {
        product: product.clone(),
        location: None,
        parents: vec![],
        machine: None,
        output_index: 0,
    };
    let current = sinex.prepare_named_products(name, &[write(begin)]).unwrap();
    assert_eq!(
        current.identity,
        format!("polylogue-source-{}:{name}", crate::SOURCE_VERSION)
    );
    assert_eq!(sinex.queue().unwrap().decode_frame(name).unwrap(), old);
    let completed = sinex
        .prepare_named_products(&phases[2].0, &[write(outcome)])
        .unwrap();
    assert_eq!(
        completed.identity,
        format!("polylogue-source-{}:{}", crate::SOURCE_VERSION, phases[2].0)
    );
    let queued = sinex.status().unwrap();
    for _ in 0..2 {
        assert_eq!(
            sinex
                .prepare_named_products(name, &[write(begin)])
                .unwrap()
                .identity,
            current.identity
        );
        assert_eq!(
            sinex
                .prepare_named_products(&phases[2].0, &[write(outcome)])
                .unwrap()
                .identity,
            completed.identity
        );
    }
    let repeated = sinex.status().unwrap();
    assert_eq!(queued.prepared_materials, repeated.prepared_materials);
    assert_eq!(queued.prepared_writes, repeated.prepared_writes);
    assert_eq!(
        kept(&sinex, NamedProductSelector::Exact(name.clone())),
        vec![begin.clone()]
    );
    assert_eq!(
        kept(&sinex, NamedProductSelector::Exact(phases[2].0.clone())),
        vec![outcome.clone()]
    );
}

#[test]
fn unchanged_named_prior_source_observation_keeps_original_material_on_retry() {
    for source_version in [8, 9] {
        let source = tempfile::tempdir().unwrap();
        let target = tempfile::tempdir().unwrap();
        let previous = client(source.path());
        let write = ProductWrite {
            product: Product::AttachmentBinding {
                record: Box::new(super::binding_rebuild::binding_record()),
            },
            location: None,
            parents: vec![],
            machine: None,
            output_index: 0,
        };
        previous
            .prepare_named_products("unchanged-binding", std::slice::from_ref(&write))
            .unwrap();
        let observation = "unchanged-binding:observation";
        let queue = previous.queue().unwrap();
        let mut original = queue.decode_frame(observation).unwrap();
        let sinex_protocol::Ingress::Material { material, .. } = &mut original else {
            panic!("expected observation material")
        };
        material.source.version = source_version;
        let mut bytes = vec![];
        queue
            .body(observation)
            .unwrap()
            .read_to_end(&mut bytes)
            .unwrap();
        drop(queue);
        let saved = crate::spool::Spool::open(target.path(), "polylogue-test").unwrap();
        saved
            .prepare_material(original.clone(), bytes.as_slice())
            .unwrap();
        drop(saved);
        let current = client(target.path());
        current
            .prepare_named_products("unchanged-binding", &[write])
            .unwrap();
        let queue = current.queue().unwrap();
        assert_eq!(queue.decode_frame(observation).unwrap(), original);
        let mut repeated = vec![];
        queue
            .body(observation)
            .unwrap()
            .read_to_end(&mut repeated)
            .unwrap();
        assert_eq!(repeated, bytes);
    }
}

#[test]
fn converted_product_file_uses_normal_query_names_and_repeats_without_effects() {
    use std::io::Write;
    let root = tempfile::tempdir().unwrap();
    let file = root.path().join("current-products.jsonl");
    let expected = phases();
    let mut output = std::fs::File::create(&file).unwrap();
    for (_, product) in &expected {
        serde_json::to_writer(&mut output, product).unwrap();
        output.write_all(b"\n").unwrap();
    }
    drop(output);
    let sdk = client(&root.path().join("producer"));
    assert_eq!(
        crate::line_population::prepare_current(&file, &sdk).unwrap(),
        3
    );
    for (name, product) in &expected {
        assert_eq!(
            kept(&sdk, NamedProductSelector::Exact(name.clone())),
            vec![product.clone()]
        );
    }
    let before = sdk.status().unwrap();
    assert_eq!(
        crate::line_population::prepare_current(&file, &sdk).unwrap(),
        3
    );
    let after = sdk.status().unwrap();
    assert_eq!(before.prepared_materials, after.prepared_materials);
    assert_eq!(before.prepared_writes, after.prepared_writes);
    drop(sdk);
    let sdk = client(&root.path().join("producer"));
    for (name, product) in expected {
        assert_eq!(kept(&sdk, NamedProductSelector::Exact(name)), vec![product]);
    }
}

#[test]
fn unchanged_prior_source_restoration_retries_keep_original_named_frames() {
    for source_version in [8, 9] {
        use polylogue_model::authored::{AuthoredEffect, PreparedRequest};
        use polylogue_model::user::{UserOutcome, UserWrite};
        let prepared = PreparedRequest {
            request: polylogue_store::fixture::authored::operator_request(
                "preserved-authored".into(),
                UserWrite::Assert { assertions: vec![] },
            ),
            outcome: UserOutcome {
                id: "preserved-authored".into(),
                repeat: false,
                pending: false,
            },
            effect: AuthoredEffect::default(),
        };
        let name = format!(
            "authored-restore:{}",
            blake3::hash(&serde_json::to_vec(&prepared).unwrap()).to_hex()
        );
        let write = ProductWrite {
            product: Product::RestoredRequest { prepared },
            location: None,
            parents: vec![],
            machine: None,
            output_index: 0,
        };
        for material in [false, true] {
            let root = tempfile::tempdir().unwrap();
            let identity = if material {
                format!("{name}:observation")
            } else {
                name.clone()
            };
            let bytes = serde_json::to_vec(std::slice::from_ref(&write)).unwrap();
            let original = if material {
                let source = tempfile::tempdir().unwrap();
                let previous = client(source.path());
                previous
                    .prepare_named_products(&name, std::slice::from_ref(&write))
                    .unwrap();
                let mut frame = previous.queue().unwrap().decode_frame(&identity).unwrap();
                let sinex_protocol::Ingress::Material { material, .. } = &mut frame else {
                    panic!("expected material")
                };
                material.source.version = source_version;
                frame
            } else {
                let mut source = crate::source_reference();
                source.version = source_version;
                sinex_protocol::Ingress::Events {
                    source: Some(source),
                    frame: identity.clone(),
                    events: crate::products::events(std::slice::from_ref(&write)).unwrap(),
                    failures: vec![],
                }
            };
            let queue = crate::spool::Spool::open(root.path(), "polylogue-test").unwrap();
            if material {
                queue
                    .prepare_material(original.clone(), bytes.as_slice())
                    .unwrap();
            } else {
                queue.prepare(original.clone()).unwrap();
            }
            drop(queue);
            let current = client(root.path());
            for _ in 0..2 {
                current
                    .prepare_named_products(&name, std::slice::from_ref(&write))
                    .unwrap();
                let queue = current.queue().unwrap();
                assert_eq!(queue.decode_frame(&identity).unwrap(), original);
                if material {
                    let mut actual = vec![];
                    queue
                        .body(&identity)
                        .unwrap()
                        .read_to_end(&mut actual)
                        .unwrap();
                    assert_eq!(actual, bytes);
                }
            }
        }
    }
}
