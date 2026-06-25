use rllm_core::config::{QuantizationConfig, QuantizationKind};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuantSchema {
    pub quant_method: Option<String>,
    pub format: Option<String>,
    pub weight_num_bits: Option<usize>,
    pub weight_strategy: Option<String>,
    pub weight_symmetric: Option<bool>,
    pub ignore: Vec<String>,
}

impl QuantSchema {
    pub fn from_hf_value(value: &Value) -> Option<Self> {
        let quant_method = value.get("quant_method").and_then(Value::as_str).map(ToOwned::to_owned);
        let format = value.get("format").and_then(Value::as_str).map(ToOwned::to_owned);
        let ignore = value
            .get("ignore")
            .and_then(Value::as_array)
            .map(|values| {
                values.iter().filter_map(Value::as_str).map(ToOwned::to_owned).collect::<Vec<_>>()
            })
            .unwrap_or_default();

        let mut schema = Self { quant_method, format, ignore, ..Self::default() };
        if let Some(weights) = first_weights_config(value) {
            schema.weight_num_bits =
                weights.get("num_bits").and_then(Value::as_u64).map(|bits| bits as usize);
            schema.weight_strategy =
                weights.get("strategy").and_then(Value::as_str).map(ToOwned::to_owned);
            schema.weight_symmetric = weights.get("symmetric").and_then(Value::as_bool);
        }

        Some(schema)
    }

    pub fn is_int8_weight_only(&self) -> bool {
        self.is_compressed_tensors_int8()
            && self.weight_num_bits.unwrap_or(8) == 8
            && self.weight_strategy.as_deref().map(|strategy| strategy == "channel" || strategy == "tensor").unwrap_or(true)
    }

    pub fn is_compressed_tensors_int8(&self) -> bool {
        self.quant_method
            .as_deref()
            .is_some_and(|method| method.eq_ignore_ascii_case("compressed-tensors"))
            || self.format.as_deref().is_some_and(|format| format == "int-quantized")
    }

    pub fn unsupported_int8_strategy(&self) -> Option<&str> {
        if !self.is_compressed_tensors_int8() || self.weight_num_bits.unwrap_or(8) != 8 {
            return None;
        }
        self.weight_strategy.as_deref().filter(|strategy| *strategy != "channel" && *strategy != "tensor")
    }

    pub fn to_core_config(&self) -> Option<QuantizationConfig> {
        if self.is_int8_weight_only() {
            return Some(QuantizationConfig {
                kind: QuantizationKind::Int8,
                group_size: None,
                bits: Some(8),
            });
        }
        None
    }
}

fn first_weights_config(value: &Value) -> Option<&Value> {
    value
        .get("config_groups")
        .and_then(Value::as_object)
        .and_then(|groups| groups.values().find_map(|group| group.get("weights")))
        .or_else(|| value.get("weights"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_compressed_tensors_int_quantized() {
        let value = serde_json::json!({
            "quant_method": "compressed-tensors",
            "format": "int-quantized",
            "ignore": ["lm_head"],
            "config_groups": {
                "group_0": {
                    "weights": {
                        "num_bits": 8,
                        "strategy": "channel",
                        "symmetric": true
                    }
                }
            }
        });
        let schema = QuantSchema::from_hf_value(&value).unwrap();
        assert!(schema.is_int8_weight_only());
        assert_eq!(schema.ignore, vec!["lm_head"]);
        assert_eq!(schema.to_core_config().unwrap().kind, QuantizationKind::Int8);
    }

    #[test]
    fn accepts_asymmetric_int8() {
        let value = serde_json::json!({
            "quant_method": "compressed-tensors",
            "format": "int-quantized",
            "weights": {
                "num_bits": 8,
                "strategy": "channel",
                "symmetric": false
            }
        });
        let schema = QuantSchema::from_hf_value(&value).unwrap();
        assert!(schema.is_int8_weight_only());
    }

    #[test]
    fn accepts_tensor_strategy_int8() {
        let value = serde_json::json!({
            "quant_method": "compressed-tensors",
            "format": "int-quantized",
            "weights": {
                "num_bits": 8,
                "strategy": "tensor",
                "symmetric": true
            }
        });
        let schema = QuantSchema::from_hf_value(&value).unwrap();
        assert!(schema.is_int8_weight_only());
    }
}
