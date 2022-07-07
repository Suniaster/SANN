use ndarray::{Array1};

use crate::{layer::NetLayer, network::Ann};

pub trait NetworkBackPropagation {
    fn last_layer_errors(&self, expected: &Array1<f64>) -> Array1<f64>;

    fn get_layers(&mut self) -> &mut Vec<Box<dyn NetLayer>>;
    fn get_layers_deltas(&mut self) -> &mut Vec<Array1<f64>>;
    fn get_layers_output(&self) -> &Vec<Array1<f64>>;
    
    fn activate_saving_output(&mut self, input: &Array1<f64>);
    fn get_loss_batch(&self, inputs: &Vec<Array1<f64>>, targets: &Vec<Array1<f64>>) -> f64;

    fn update_deltas(&mut self, expected: &Array1<f64>) {
        let mut deltas = self.last_layer_errors(expected);
        let layers_len = self.get_layers_deltas().len();
        
        self.get_layers_deltas()[layers_len - 1] = deltas.clone();

        for i in (0..layers_len).rev().skip(1) {
            let next_layer_deltas = self.get_layers_deltas()[i + 1].clone();
            let next_layer_ws = self.get_layers()[i].get_weights().clone();
            let this_layer_out = self.get_layers_output()[i].clone();

            deltas = self.get_layers()[i].get_backpropag_error(
                &this_layer_out, 
                &next_layer_deltas, 
                &next_layer_ws
            );
            self.get_layers_deltas()[i] = deltas.clone();
        }
    }

    fn learn(&mut self, input: &Array1<f64>, target: &Array1<f64>, learning_rate: f64){
        self.activate_saving_output(input);
        self.update_deltas(target);
        for i in 1..self.get_layers().len() {
            let prev_layer_out = self.get_layers_output()[i - 1].clone();
            let deltas = self.get_layers_deltas()[i].clone();
            self.get_layers()[i].update_params(
                &deltas, 
                &prev_layer_out, 
                learning_rate
            );
        }
    }

    fn train(&mut self, inputs: &Vec<Array1<f64>>, targets: &Vec<Array1<f64>>, iterations: usize, learning_rate: f64) -> f64 {
        for _ in 0..iterations {
            for (i, input) in inputs.iter().enumerate() {
                self.learn(input, &targets[i], learning_rate);
            }
        }
        return self.get_loss_batch(inputs, targets);
    }
}

impl NetworkBackPropagation for Ann {
    fn last_layer_errors(&self, expected: &Array1<f64>) -> Array1<f64> {
        // Considering the last function to have linear activation.
        return &self.layers_output[self.layers_output.len()-1] - expected;
    }

    fn get_layers(&mut self) -> &mut Vec<Box<dyn NetLayer>> {
        return &mut self.layers;
    }

    fn get_layers_deltas(&mut self) -> &mut Vec<Array1<f64>> {
        return &mut self.layers_deltas;
    }

    fn get_layers_output(&self) -> &Vec<Array1<f64>> {
        return &self.layers_output;
    }
    
    fn activate_saving_output(&mut self, input: &Array1<f64>){
        let output = input.clone();
        self.layers_output[0] = self.layers[0].activate(&output);
        for (i, layer) in self.layers.iter().enumerate().skip(1) {
            self.layers_output[i] = layer.activate(&self.layers_output[i-1]);
        }
    }

    fn get_loss_batch(&self, inputs: &Vec<Array1<f64>>, expecteds: &Vec<Array1<f64>>) -> f64 {
        let mut loss = 0.0;
        for i in 0..inputs.len() {
            loss += self.get_error(&inputs[i], &expecteds[i]);
        }
        return loss / inputs.len() as f64;
    }
}