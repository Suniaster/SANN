
pub mod activations;
pub mod helper;
pub mod layer;
pub mod network;
pub mod node;
pub mod io;
pub mod examples;


use ndarray::{Array1, Array2};

pub trait NetLayer {
    fn activate(&self, input: &Array1<f64>) -> Array1<f64>;

    fn get_output_error(&self, output: &Array1<f64>, expected: &Array1<f64>) -> Array1<f64>;
    fn get_backpropag_error(&self, this_layer_out: &Array1<f64>, next_layer_deltas: &Array1<f64>, next_layer_ws: &Array2<f64>) -> Array1<f64>;
    
    fn update_params(&mut self, deltas: &Array1<f64>, previous_layer_output: &Array1<f64>, learning_rate: f64);
    fn set_activation(&mut self, _type: activations::ActivationType);

    fn get_weights(&self) -> &Array2<f64>;
    fn get_format(&self) -> (usize, usize);
}

pub struct Ann {
    layers: Vec<Box<dyn NetLayer>>,
    layers_output: Vec<Array1<f64>>,
    layers_deltas: Vec<Array1<f64>>,
}

impl Ann {
    pub fn new() -> Ann {
        return Ann { 
            layers: vec![], 
            layers_output: vec![], 
            layers_deltas: vec![],
        };
    }

    pub fn set_activations(&mut self, activations: &[activations::ActivationType]) {
        for i in 0..activations.len() {
            self.layers[i].set_activation(activations[i].clone());
        }
    }

    pub fn add_layer(&mut self, layer: Box<dyn NetLayer>) {
        self.layers_output.push(Array1::zeros(layer.get_format().1));
        self.layers_deltas.push(Array1::zeros(layer.get_format().1));
        self.layers.push(layer);
    }

    pub fn activate(&self, input: &Array1<f64>) -> Array1<f64> {
        let mut output = input.clone();
        for layer in self.layers.iter() {
            output = layer.activate(&output);
        }
        return output;
    }

    fn activate_saving_output(&mut self, input: &Array1<f64>){
        let mut output = input.clone();
        for (i, layer) in self.layers.iter().enumerate() {
            output = layer.activate(&output);
            self.layers_output[i] = output.clone();
        }
    }

    fn update_deltas(&mut self, expected: &Array1<f64>) {
        let last_layer = &self.layers.last().expect("Empty network");
        let last_layer_out = self.layers_output.last().expect("Empty network");

        let mut deltas = last_layer.get_output_error(last_layer_out, expected);
        self.layers_deltas[self.layers.len() -1] = deltas.clone();

        for (i, layer) in self.layers.iter().enumerate().rev().skip(1) {
            let this_layer_out = &self.layers_output[i];
            let next_layer_ws = self.layers[i+1].get_weights();

            deltas = layer.get_backpropag_error(this_layer_out, &deltas, next_layer_ws);
            self.layers_deltas[i] = deltas.clone();
        }
    }

    pub fn learn(&mut self, input: &Array1<f64>, expected: &Array1<f64>, learning_rate: f64) -> f64 {
        self.activate_saving_output(input);
        self.update_deltas(expected);

        for (i, layer) in self.layers.iter_mut().enumerate().skip(1) {
            let pv_layer_out = &self.layers_output[i - 1];
            let deltas = &self.layers_deltas[i];
            layer.update_params(deltas, pv_layer_out, learning_rate);
        }
        
        let pos_learn_output = self.activate(input);
        let errors_squared = (&pos_learn_output - expected) * (pos_learn_output - expected);
        return errors_squared.sum();
    }

    pub fn train(&mut self, inputs: &Vec<Array1<f64>>, expecteds: &Vec<Array1<f64>>, iterations:usize, learning_rate: f64) -> f64 {
        for _ in 0..iterations {
            for (i, input) in inputs.iter().enumerate() {
                self.learn(input, &expecteds[i], learning_rate);
            }
        }
        return self.learn(&inputs[0], &expecteds[0], learning_rate);
    }
}

pub struct DenseLayer {
    weights: Array2<f64>,
    biases: Array1<f64>,
    activation: activations::Activation,
}

impl DenseLayer {
    pub fn new(input_size: usize, output_size: usize) -> DenseLayer {
        let weights = Array2::ones((output_size, input_size));
        let biases = Array1::zeros(output_size);
        return DenseLayer {
            weights,
            biases,
            activation: activations::Activation::create(activations::ActivationType::Default),
        };
    }
}

impl NetLayer for DenseLayer {
    fn activate(&self, input: &Array1<f64>) -> Array1<f64> {
        let output = self.weights.dot(input) + &self.biases;
        return output.map(self.activation.f);
    }

    fn get_output_error(&self, output: &Array1<f64>, expected: &Array1<f64>) -> Array1<f64> {
        let derivatives = output.mapv(|x| (self.activation.d)(&x));
        let mut errors = (output - expected) * (output - expected);
        errors = derivatives * errors;
        return errors;
    }

    fn get_backpropag_error(&self, this_layer_out: &Array1<f64>, next_layer_deltas: &Array1<f64>, next_layer_ws: &Array2<f64>) -> Array1<f64> {
        let derivatives = this_layer_out.mapv(|x| (self.activation.d)(&x));
        let errors = next_layer_ws.t().dot(next_layer_deltas) * derivatives;
        return errors;
    }

    fn update_params(&mut self, deltas: &Array1<f64>, previous_layer_output: &Array1<f64>, learning_rate: f64) {
        for i in 0..self.weights.nrows() {
            let j_l_w = previous_layer_output * deltas[i];

            for j in 0..self.weights.ncols() {
                self.weights[(i, j)] -= j_l_w[j] * learning_rate;
            }
        }

        self.biases -= &(deltas * learning_rate);
    }

    fn set_activation(&mut self, _type: activations::ActivationType) {
        self.activation = activations::Activation::create(_type);
    }

    fn get_weights(&self) -> &Array2<f64> {
        return &self.weights;
    }

    fn get_format(&self) -> (usize, usize) {
        return (self.weights.shape()[0], self.weights.shape()[1]);
    }
}