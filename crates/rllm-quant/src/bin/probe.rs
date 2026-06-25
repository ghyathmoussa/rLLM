use candle_core::quantized::QMatMul;
use candle_core::{Module, Tensor};

#[allow(invalid_value)]
fn main() {
    let q: &QMatMul = unsafe { std::mem::zeroed() };
    let x: &Tensor = unsafe { std::mem::zeroed() };
    let _y = q.forward(x);
}
