//! The common decoder has no provider attribution policy or descriptor wire.
use polylogue_model::RawJson as Value;
use polylogue_sources::{
    codex,
    selection::{
        self, Operation, PathPart, Recipe, RecordRange, ScalarError, SelectedRaw, ValueError,
    },
};
use std::io::{self, Write};
#[derive(Debug)]
struct Refusal(&'static str);
struct ExampleRecipe {
    inputs: Vec<SelectedRaw>,
    operation: Operation,
}
impl Recipe for ExampleRecipe {
    type Error = Refusal;
    fn provider(&self) -> &'static str {
        "synthetic"
    }
    fn validate(&self) -> Result<(), Refusal> {
        selection::validate_inputs(&self.inputs, self.operation).map_err(Refusal)
    }
    fn inputs(&self) -> &[SelectedRaw] {
        &self.inputs
    }
    fn operation(&self) -> Operation {
        self.operation
    }
    fn invalid(&self, reason: &'static str) -> Refusal {
        Refusal(reason)
    }
    fn write_scalar<W: Write>(
        &self,
        value: &Value,
        sink: &mut W,
    ) -> Result<(), ScalarError<Refusal>> {
        // A synthetic adapter's closed scalar rule; common selection performs
        // no provider-specific author classification or source-path discovery.
        let flag = value.as_bool().ok_or(ScalarError::Type)?;
        if !flag {
            return Err(ScalarError::Invalid(Refusal(
                "false is not a supported supplied fact",
            )));
        }
        serde_json::to_writer(sink, "synthetic-selected").map_err(ScalarError::Json)
    }
}
fn make_recipe(raw: &[u8], operation: Operation) -> ExampleRecipe {
    ExampleRecipe {
        inputs: vec![SelectedRaw {
            record: RecordRange::new(0, raw.len() as u64).unwrap(),
            path: vec![PathPart::Key("selected".into())],
        }],
        operation,
    }
}
#[test]
fn second_adapter_uses_same_path_decode_core_and_owns_its_scalar_rule() {
    let raw = br#"{"selected":true,"unrelated":"do not interpret"}"#;
    let recipe = make_recipe(raw, Operation::ProviderScalar);
    assert_eq!(recipe.provider(), "synthetic");
    let mut output = Vec::new();
    selection::write_value(
        &recipe,
        |_| Ok::<_, ()>(raw.to_vec()),
        raw.len(),
        &mut output,
    )
    .unwrap();
    assert_eq!(output, br#""synthetic-selected""#);
    let raw = br#"{"selected":false}"#;
    let recipe = make_recipe(raw, Operation::ProviderScalar);
    assert!(matches!(
        selection::write_value(
            &recipe,
            |_| Ok::<_, ()>(raw.to_vec()),
            raw.len(),
            &mut Vec::new()
        ),
        Err(ValueError::Selection(Refusal(
            "false is not a supported supplied fact"
        )))
    ));
}
#[test]
fn invalid_time_and_resolver_errors_are_reported_unchanged() {
    let bad = br#"{"selected":"yesterday"}"#;
    assert!(matches!(
        selection::write_value(
            &make_recipe(bad, Operation::ProviderTime),
            |_| Ok::<_, ()>(bad.to_vec()),
            bad.len(),
            &mut Vec::new()
        ),
        Err(ValueError::Time)
    ));
    let raw = br#"{"selected":"1970-01-01T00:00:00Z"}"#;
    let recipe = make_recipe(raw, Operation::ProviderTime);
    let mut written = Vec::new();
    selection::write_value(
        &recipe,
        |_| Ok::<_, ()>(raw.to_vec()),
        raw.len(),
        &mut written,
    )
    .unwrap();
    assert!(!written.is_empty());
    let marker = Box::new(42);
    let ptr = &*marker as *const i32;
    let mut marker = Some(marker);
    let error = selection::write_value(
        &recipe,
        |_| Err::<Vec<u8>, _>(marker.take().unwrap()),
        raw.len(),
        &mut Vec::new(),
    )
    .unwrap_err();
    let ValueError::Resolve(marker) = error else {
        panic!("lost resolver error")
    };
    assert_eq!(&*marker as *const i32, ptr);
}
#[test]
fn provider_scalar_keeps_sink_error_object_and_never_falls_back_to_json() {
    struct Fail;
    impl Write for Fail {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> {
            Err(io::Error::from_raw_os_error(28))
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    let raw = br#"{"selected":true}"#;
    let recipe = make_recipe(raw, Operation::ProviderScalar);
    let error =
        selection::write_value(&recipe, |_| Ok::<_, ()>(raw.to_vec()), raw.len(), &mut Fail)
            .unwrap_err();
    assert!(matches!(error,ValueError::Io(error) if error.raw_os_error()==Some(28)));
}
#[test]
fn codex_promoted_recipe_keeps_typed_paths_and_refuses_historical_interpretation() {
    use codex::selection::{Binding, FactSelection, Transform};
    let range: codex::origins::RecordRange = RecordRange::new(3, 15).unwrap();
    let selected = FactSelection {
        codec_version: 1,
        adapter_version: 2,
        transform: Transform::JsonStringV1,
        inputs: vec![SelectedRaw {
            record: range,
            path: vec![PathPart::Key("payload".into()), PathPart::Index(0)],
        }],
    };
    let mut historical = selected.clone();
    historical.adapter_version = 1;
    assert!(historical.validate().is_err());
    let mut bytes = Vec::new();
    selected
        .write_bound(
            Binding {
                input: "retained-input",
                chain: 1,
            },
            &mut bytes,
        )
        .unwrap();
    assert_eq!(bytes,br#"{"format":"codex-field-origin","binding":{"input":"retained-input","chain":1},"selection":{"codec_version":1,"adapter_version":2,"transform":"JsonStringV1","inputs":[{"record":{"start":3,"end":15},"path":[{"Key":"payload"},{"Index":0}]}]}}"#);
    let mut cursor = selected.cursor();
    assert_eq!(selected.advance(&mut cursor).unwrap(), Some(range));
    assert_eq!(serde_json::to_vec(&cursor).unwrap(), br#"{"next":1}"#);
    let mut cursor = serde_json::from_slice(&serde_json::to_vec(&cursor).unwrap()).unwrap();
    assert_eq!(selected.advance(&mut cursor).unwrap(), None);
    assert_eq!(selected.provider(), "codex");
}

#[test]
fn shared_envelope_writer_preserves_provider_owned_format_and_typed_io_failure() {
    let envelope = serde_json::json!({"format":"synthetic-field-origin","version":1,"locator":{"record":{"start":1,"end":5}}});
    let mut bytes = Vec::new();
    selection::write_json(&envelope, &mut bytes).unwrap();
    assert_eq!(bytes, serde_json::to_vec(&envelope).unwrap());
    assert_eq!(
        serde_json::from_slice::<Value>(&bytes).unwrap()["format"].as_str(),
        Some("synthetic-field-origin")
    );
    struct Fail;
    impl Write for Fail {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> {
            Err(io::Error::from_raw_os_error(28))
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    assert!(
        matches!(selection::write_json(&envelope,&mut Fail),Err(selection::JsonWriteError::Io(error)) if error.raw_os_error()==Some(28))
    );
}
