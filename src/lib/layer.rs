use ndarray::arr0;
use ndarray::array;
use ndarray::Array1;
use ndarray::Array2;
use std::vec::Vec;

use super::activations::*;
use super::perceptron::Perceptron;

pub struct Layer {
    pub weights_matrix: Array2<f64>,
    pub biases: Array1<f64>,
    pub activation: Activation,
    pub last_output: Array1<f64>,
    pub deltas: Array1<f64>,
}

impl Layer {
    pub fn calculate_error(output: &Array1<f64>, expected: &Array1<f64>) -> f64 {
        return output
            .iter()
            .zip(expected.iter())
            .fold(0.0, |sum: f64, (x, y)| -> f64 {
                return sum + (x - y).powi(2);
            });
    }

    pub fn create_empty(input_shape: usize) -> Layer {
        return Layer {
            biases: array![],
            last_output: array![],
            deltas: array![],
            weights_matrix: Array2::from_shape_vec((0, input_shape), vec![]).unwrap(),
            activation: Activation::create(ActivationType::Sigmoid),
        };
    }

    pub fn from_neurons(neurons: Array1<Perceptron>) -> Layer {
        let shape: usize = neurons[0].shape;
        let mut new_layer = Layer::create_empty(shape);
        for neuron in neurons.iter() {
            new_layer.push_neuron(&neuron);
        }
        return new_layer;
    }

    pub fn from_rand(input_shape: usize, size: usize) -> Layer {
        let mut vec: Vec<Perceptron> = Vec::new();
        for _ in 0..size {
            vec.push(Perceptron::from_rand(input_shape));
        }

        return Layer::from_neurons(Array1::from_vec(vec));
    }

    pub fn activate(&self, input: &Array1<f64>) -> Array1<f64> {
        let result = self.weights_matrix.dot(input);
        return (&result + &self.biases).map(self.activation.f);
    }

    pub fn forward(&mut self, input: &Array1<f64>) -> &Array1<f64> {
        self.last_output = self.activate(input);
        return &self.last_output;
    }

    pub fn learn(
        &mut self,
        last_output: &Array1<f64>,
        expected: &Array1<f64>,
        learning_rate: f64,
    ) -> f64 {
        let loss = Layer::calculate_error(last_output, expected);
        println!("Layer Loss: {}", loss);
        println!("Layer Loss: {}", learning_rate);

        return loss;
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
