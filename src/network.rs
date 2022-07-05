use ndarray::{Array1};
use super::activations;
use super::layer::NetLayer;

pub struct Ann {
    pub layers: Vec<Box<dyn NetLayer>>,
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

    pub fn get_loss(&self, input: &Array1<f64>, expected: &Array1<f64>) -> f64 {
        let out = self.activate(input);
        let sub = out - expected;
        let sqrd = sub.map(|x| x * x);
        return sqrd.sum();
    }

    pub fn get_loss_batch(&self, inputs: &Vec<Array1<f64>>, expecteds: &Vec<Array1<f64>>) -> f64 {
        let mut loss = 0.0;
        for i in 0..inputs.len() {
            loss += self.get_loss(&inputs[i], &expecteds[i]);
        }
        return loss / inputs.len() as f64;
    }

    pub fn randomize(&mut self) {
        for layer in self.layers.iter_mut().skip(1) {
            layer.randomize_params();
        }
    }

    fn activate_saving_output(&mut self, input: &Array1<f64>){
        let output = input.clone();
        self.layers_output[0] = self.layers[0].activate(&output);
        for (i, layer) in self.layers.iter().enumerate().skip(1) {
            self.layers_output[i] = layer.activate(&self.layers_output[i-1]);
        }
    }

    fn last_layer_error(&self, expected: &Array1<f64>) -> Array1<f64> {
        // Considering the last function to have linear activation.
        return &self.layers_output[self.layers_output.len()-1] - expected;
    }

    fn update_deltas(&mut self, expected: &Array1<f64>) {
        let mut deltas = self.last_layer_error(expected);
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
        return self.get_loss(input, expected);
    }

    pub fn train(&mut self, inputs: &Vec<Array1<f64>>, expecteds: &Vec<Array1<f64>>, iterations:usize, learning_rate: f64) -> f64 {
        for _ in 0..iterations {
            for (i, input) in inputs.iter().enumerate() {
                self.learn(input, &expecteds[i], learning_rate);
            }
        }
        return self.get_loss_batch(inputs, expecteds);
    }
}