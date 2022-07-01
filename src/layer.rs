use super::activations::*;
use super::helper::*;
use super::node::*;

use ndarray::arr0;
use ndarray::array;
use ndarray::Array1;
use ndarray::Array2;

pub struct Layer {
    pub neurons: Vec<Container<Neuron>>,
    pub weights: Array2<f64>,
    pub biases: Array1<f64>
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
        return Layer::new_activation(size, ActivationType::Default);
    }

    pub fn new_activation(size: u16, _type: ActivationType) -> Layer {
        let mut new_layer = Layer {
            neurons: vec![],
            weights: Array2::from_shape_vec((0, size as usize), vec![]).unwrap(),
            biases: array![],
        };
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
    create_neuron_mapper!(activate_neurons, activate);
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

    pub fn set_state(&mut self, input: &[f64]) {
        for neuron_i in 0..self.neurons.len() {
            let neuron = &self.neurons[neuron_i];
            neuron.set_out(input[neuron_i]);
        }
    }

    pub fn activate(&self, input: Vec<f64>) -> Vec<f64> {
        let result = self.weights.dot(&Array1::from(input));
        let activation = &self.neurons[0].borrow().activation;
        let sum = &result + &self.biases;
        return sum.map(activation.f).to_vec();
    }

    pub fn update_weights(&mut self) {
        let projected_size = self.neurons[0].borrow().inputs.len();
        let mut weights = Array2::from_shape_vec((0, projected_size), vec![]).unwrap();

        self.biases = array![];
        for neuron in self.neurons.iter() {
            weights
                .push_row(Array1::from_vec(neuron.get_weights()).view())
                .unwrap();

            self.biases
                .push(ndarray::Axis(0), arr0(neuron.borrow().bias).view())
                .expect("Nao salvo os Bias");
        }
        self.weights = weights;
    }
}
