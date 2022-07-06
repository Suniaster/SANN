use ndarray::{Array1};

use crate::layer::NetLayer;

pub trait NetworkBackPropagation {
    fn last_layer_errors(&self, expected: &Array1<f64>) -> Array1<f64>;

    fn update_deltas(&mut self, expected: &Array1<f64>) {
        let mut deltas = self.last_layer_errors(expected);
        let all_layers_deltas = self.get_layers_deltas();
        let layers_len = all_layers_deltas.len();
        
        all_layers_deltas[layers_len - 1] = deltas.clone();
        let layers = self.get_layers();
        let outputs = self.get_outputs();

        for i in (0..layers_len).rev().skip(1) {
            let next_layer_deltas = all_layers_deltas[i + 1];
            let next_layer_ws = layers[i].get_weights();

            deltas = layers[i].get_backpropag_error(
                &outputs[i], 
                &next_layer_deltas, 
                &next_layer_ws
            );
            all_layers_deltas[i] = deltas.clone();
        }
    }
    
    fn get_layers_deltas(&mut self) -> &mut Vec<Array1<f64>>;
    fn get_outputs(&self) -> &Vec<Array1<f64>>;
    fn get_layers(&mut self) -> &mut Vec<Box<dyn NetLayer>>;

    fn learn(&mut self, input: &Array1<f64>, target: &Array1<f64>, learning_rate: f64);
    fn get_loss_batch(&self, inputs: &Vec<Array1<f64>>, targets: &Vec<Array1<f64>>) -> f64;

    fn train(&mut self, inputs: &Vec<Array1<f64>>, targets: &Vec<Array1<f64>>, iterations: usize, learning_rate: f64) -> f64 {
        for _ in 0..iterations {
            for (i, input) in inputs.iter().enumerate() {
                self.learn(input, &targets[i], learning_rate);
            }
        }
        return self.get_loss_batch(inputs, targets);
    }
}