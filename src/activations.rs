pub fn relu(x: &f64) -> f64 {
    return if *x > 0.0 { *x } else { 0.0 };
}

pub fn relu_derivate(x: &f64) -> f64 {
    return if *x <= 0.0 { 0.0 } else { 1.0 };
}

pub fn sigmoid(x: &f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

pub fn sigmoid_derivate(x: &f64) -> f64 {
    let y = *x;
    return y * (1.0 - y);
}

pub fn linear(x: &f64) -> f64 {
    return *x;
}

#[allow(unused_variables)]
pub fn linear_derivate(x: &f64) -> f64 {
    return 1.0;
}

use serde::{ Serialize, Deserialize };

#[derive(Clone)]
#[derive(Serialize, Deserialize)]
#[allow(dead_code)]
pub enum ActivationType {
    ReLu,
    Sigmoid,
    Default,
    Linear,
}

pub struct Activation {
    pub t: ActivationType,
    pub f: fn(x: &f64) -> f64,
    pub d: fn(x: &f64) -> f64,
}

impl Activation {
    pub fn create(_type: ActivationType) -> Activation {
        match _type {
            ActivationType::ReLu => Activation {
                t: _type,
                f: relu,
                d: relu_derivate,
            },
            ActivationType::Sigmoid => Activation {
                t: _type,
                f: sigmoid,
                d: sigmoid_derivate,
            },
            ActivationType::Linear | _ => Activation {
                t: _type,
                f: linear,
                d: linear_derivate,
            },
        }
    }
}
