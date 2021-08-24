use super::super::lib::activations::*;
use super::helper::*;
use rand;

pub struct Axon {
    weight: f64,
    value: f64,
    src: Container<Neuron>,
    dest: Container<Neuron>,
}

pub struct Neuron {
    outputs: Vec<Container<Axon>>,
    pub inputs: Vec<Container<Axon>>,
    pub activation: Activation,
    pub bias: f64,
    output_val: f64,
    delta_error: f64,
}

impl Axon {
    fn update(&mut self) {
        self.value = self.src.borrow().output_val * self.weight;
    }
}

impl Neuron {
    pub fn new() -> Container<Neuron> {
        return Neuron::new_activation(ActivationType::ReLu);
    }

    pub fn new_activation(_type: ActivationType) -> Container<Neuron> {
        return new_container(Neuron {
            bias: (rand::random::<f64>() * 2.0) - 1.0,
            output_val: 0.0,
            activation: Activation::create(_type),
            inputs: vec![],
            outputs: vec![],
            delta_error: 0.0,
        });
    }

    pub fn project(src: &Container<Neuron>, dest: &Container<Neuron>) {
        let new_axon = new_container(Axon {
            weight: (rand::random::<f64>() * 2.0) - 1.0,
            src: src.clone(),
            dest: dest.clone(),
            value: 0.0,
        });

        src.borrow_mut().outputs.push(new_axon.clone());
        dest.borrow_mut().inputs.push(new_axon.clone());
    }

    pub fn set_output_error(&mut self, expected_val: f64) {
        let error = expected_val - self.output_val;
        self.delta_error = error * (self.activation.d)(&self.output_val);
    }

    pub fn set_backpropag_error(&mut self) {
        let mut total_error = 0.0;
        for axon in self.outputs.iter() {
            let _axon = axon.borrow();
            let delta = _axon
                        .dest.borrow()
                        .delta_error;
            total_error += delta * _axon.weight;
        }
        self.delta_error = total_error * (self.activation.d)(&self.output_val);
    }

    pub fn update_weights(&mut self, l_rate: f64) {
        for axon in self.inputs.iter() {
            let mut _axon = axon.borrow_mut();
            _axon.weight += self.delta_error * _axon.src.get_out() * l_rate;
        }
        self.bias += self.delta_error * l_rate;
    }

    pub fn change_activation(&mut self, _type: ActivationType) {
        self.activation = Activation::create(_type);
    }

    pub fn activate(&mut self) -> f64 {
        let mut out = 0.0;
        for axon in self.inputs.iter() {
            let _axon = axon.borrow();
            out += _axon.weight * _axon.src.borrow().output_val;
        }
        out += self.bias;
        self.output_val = (self.activation.f)(&out);
        return self.output_val;
    }

    pub fn get_weights(&self) -> Vec<f64> {
        let mut out = vec![];
        for axon in self.inputs.iter() {
            out.push(axon.borrow().weight);
        }
        return out;
    }

    pub fn set_out(&mut self, val: f64) {
        self.output_val = val;
    }
}

macro_rules! copy_f_to_container {
    ($f_name: ident) => {
        fn $f_name(&self) {
            self.borrow_mut().$f_name();
        }
    };
    ($f_name: ident, $param_type:ty) => {
        fn $f_name(&self, aux: $param_type) {
            self.borrow_mut().$f_name(aux);
        }
    };
    ($f_name: ident, $r_name:ident, $r_type: ty) => {
        fn $f_name(&self) -> $r_type {
            return self.borrow().$r_name;
        }
    };
}

pub trait NeuronContainer {
    fn change_activation(&self, _type: ActivationType);
    fn activate(&self) -> f64;
    fn set_out(&self, val: f64);
    fn get_out(&self) -> f64;
    fn get_delta(&self) -> f64;
    fn set_output_error(&self, expected_val: f64);
    fn set_backpropag_error(&self);
    fn update_weights(&self, learning_rate: f64);
    fn get_weights(&self) -> Vec<f64>;
}

impl NeuronContainer for Container<Neuron> {
    fn activate(&self) -> f64 {
        return self.borrow_mut().activate();
    }

    fn get_weights(&self) -> Vec<f64> {
        return  self.borrow().get_weights();
    }

    // Getters
    copy_f_to_container!(get_out, output_val, f64);
    copy_f_to_container!(get_delta, delta_error, f64);

    // Call function
    copy_f_to_container!(set_backpropag_error);

    // Call function with param
    copy_f_to_container!(change_activation, ActivationType);
    copy_f_to_container!(update_weights, f64);
    copy_f_to_container!(set_output_error, f64);
    copy_f_to_container!(set_out, f64);
}
