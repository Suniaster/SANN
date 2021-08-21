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
    let y = sigmoid(x);
    return y * (1.0 - y);
}

pub enum ActivationType {
    ReLu,
    Sigmoid,
    Default,
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
            _ => Activation {
                t: _type,
                f: sigmoid,
                d: sigmoid_derivate,
            },
        }
    }
}
