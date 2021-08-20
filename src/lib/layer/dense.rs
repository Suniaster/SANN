use ndarray::arr0;
use ndarray::array;
use ndarray::Array1;
use ndarray::Array2;
use std::vec::Vec;

use super::super::perceptron::*;

pub struct LayerDense {
    weights_matrix: Array2<f64>,
    biases: Array1<f64>,
}

impl Forward for LayerDense {
    fn foward(&self, input: &Array1<f64>) -> Array1<f64> {
        let result = self.weights_matrix.dot(input);
        return &result + &self.biases;
    }
}

impl LayerDense {
    pub fn from_rand(input_shape: usize, size: usize) -> LayerDense {
        let mut vec: Vec<Perceptron> = Vec::new();
        for _ in 0..size {
            vec.push(Perceptron::from_rand(input_shape));
        }

        return LayerDense::from_neurons(Array1::from_vec(vec));
    }

    pub fn from_neurons(neurons: Array1<Perceptron>) -> LayerDense {
        let shape: usize = neurons[0].shape;

        let mut new_layer = LayerDense {
            biases: array![],
            weights_matrix: Array2::from_shape_vec((0, shape), vec![]).unwrap(),
        };

        for neuron in neurons.iter() {
            new_layer.push_neuron(&neuron);
        }

        return new_layer;
    }

    pub fn push_neuron(&mut self, neuron: &Perceptron) {
        self.weights_matrix
            .push_row(neuron.input_weigths.view())
            .expect("Neuronio adicionado com quantidade de input diferente de Layer");
        self.biases
            .push(ndarray::Axis(0), arr0(neuron.bias).view())
            .expect("Nao salvo os Bias");
    }
}
