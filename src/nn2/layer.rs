use super::super::lib::activations::*;
use super::helper::*;
use super::node::*;

pub struct Layer {
    neurons: Vec<Container<Neuron>>,
}

macro_rules! create_neuron_mapper {
    ($f_name: ident, $mapped_f: ident) => {
        pub fn $f_name(&self) {
            for neuron in self.neurons.iter() {
                neuron.$mapped_f();
            }
        }
    };
    ($f_name: ident, $mapped_f: ident, $param_t: ty) => {
        pub fn $f_name(&self, p1: $param_t) {
            for neuron in self.neurons.iter() {
                neuron.$mapped_f(p1);
            }
        }
    };
}

impl Layer {
    pub fn new(size: u16) -> Layer {
        return Layer::new_activation(size, ActivationType::ReLu);
    }

    pub fn new_activation(size: u16, _type: ActivationType) -> Layer {
        let mut new_layer = Layer { neurons: vec![] };
        for _ in 0..size {
            new_layer
                .neurons
                .push(Neuron::new_activation(_type.clone()));
        }
        return new_layer;
    }

    pub fn project(&self, other_layer: &Layer) {
        for neuron in self.neurons.iter() {
            for other_neuron in other_layer.neurons.iter() {
                Neuron::project(neuron, other_neuron);
            }
        }
    }

    pub fn set_output_error(&self, error: &[f64]) {
        for i in 0..self.neurons.len() {
            self.neurons[i].set_output_error(error[i]);
        }
    }

    create_neuron_mapper!(update_error, set_backpropag_error);
    create_neuron_mapper!(activate, activate);
    create_neuron_mapper!(update, update_weights, f64);

    pub fn change_activation(&mut self, _type: ActivationType) {
        for neuron in self.neurons.iter_mut() {
            neuron.change_activation(_type.clone());
        }
    }

    pub fn get_state(&self) -> Vec<f64> {
        let mut ret = vec![];
        for neuron in self.neurons.iter() {
            ret.push(neuron.get_out());
        }
        return ret;
    }

    pub fn set_state(&mut self, input: Vec<f64>) {
        for neuron_i in 0..self.neurons.len() {
            let neuron = &self.neurons[neuron_i];
            neuron.set_out(input[neuron_i]);
        }
    }
}
