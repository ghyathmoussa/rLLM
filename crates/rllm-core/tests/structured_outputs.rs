use rllm_core::request::{SamplingParams, StructuredOutputParams};

fn json_constraint() -> StructuredOutputParams {
    StructuredOutputParams {
        json_schema: Some(serde_json::json!({"type": "object"})),
        json_object: None,
        xml: None,
        regex: None,
        grammar: None,
        choice: None,
    }
}

#[test]
fn accepts_one_json_constraint() {
    let params =
        SamplingParams { structured_outputs: Some(json_constraint()), ..SamplingParams::default() };
    assert!(params.validate().is_ok());
}

#[test]
fn rejects_multiple_constraint_kinds() {
    let mut structured = json_constraint();
    structured.xml = Some("root ::= \"<ok/>\"".into());
    let params =
        SamplingParams { structured_outputs: Some(structured), ..SamplingParams::default() };
    assert!(params.validate().unwrap_err().to_string().contains("exactly one"));
}

#[test]
fn rejects_stop_conditions_that_can_truncate_structure() {
    let params = SamplingParams {
        structured_outputs: Some(json_constraint()),
        stop: vec!["}".into()],
        ..SamplingParams::default()
    };
    assert!(params.validate().unwrap_err().to_string().contains("stop conditions"));
}
