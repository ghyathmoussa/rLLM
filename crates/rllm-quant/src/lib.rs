pub mod schema;

#[cfg(feature = "candle-backend")]
pub mod int8;
#[cfg(feature = "candle-backend")]
pub mod method;
#[cfg(feature = "candle-backend")]
pub mod qtensor;
#[cfg(feature = "candle-backend")]
pub mod unquant;

#[cfg(feature = "candle-backend")]
pub use int8::{Int8Linear, Int8WeightOnlyFactory};
#[cfg(feature = "candle-backend")]
pub use method::{LinearMethod, QuantMethodFactory, WeightSource, factory_from_config};
#[cfg(feature = "candle-backend")]
pub use qtensor::QuantTensor;
pub use schema::QuantSchema;
#[cfg(feature = "candle-backend")]
pub use unquant::{UnquantizedFactory, UnquantizedLinear};
