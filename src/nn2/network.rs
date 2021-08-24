use super::super::lib::activations::*;
use super::layer::*;
use super::node::*;

pub struct Network {
    layers: Vec<Layer>,
}

impl Network {
    pub fn new(layers_format: &[u16]) -> Network {
        let mut layers: Vec<Layer> = vec![];
        for i in 0..layers_format.len() {
            let new_layer = Layer::new(layers_format[i]);
            layers.push(new_layer);
            if i != 0 {
                layers[i - 1].project(&layers[i]);
            }
        }
        return Network { layers };
    }

    pub fn format(&mut self, activations: &[ActivationType]) {
        for i in 0..activations.len() {
            self.layers[i].change_activation(activations[i].clone());
        }
    }

    pub fn get_loss(&self, expected: &[f64]) -> f64{
        let mut loss = 0.0;
        for i in 0..self.layers.last().expect("Empty network").neurons.len() {
            let neuron = &self.layers.last().expect("Empty network").neurons[i];
            loss += (neuron.get_out() - expected[i]).powi(2);
        }
        return loss;
    }

    pub fn train(&mut self,
        input: &Vec<Vec<f64>>,
        expected: &Vec<Vec<f64>>,
        learning_rate: f64,
        iterations: usize,){
        for iteration in 0..iterations{
            print!("\rIteration {} ######", iteration + 1);
            let mut loss = 0.0;
            for i in 0..input.len(){
                self.activate(&input[i]);
                loss += self.update_errors(&expected[i]);
                self.learn(learning_rate);
            }            
            loss = loss / input.len() as f64;
            print!(" Loss: {} ", loss);
        }
        println!();
    }
    pub fn update_errors(&self, expected: &[f64])-> f64 {
        let output_layer = self.layers.last().expect("Empty network");
        output_layer.set_output_error(expected);

        for layer in self.layers.iter().rev().skip(1) {
            layer.update_error();
        }
        return self.get_loss(expected);
    }

    pub fn learn(&self, learning_rate: f64) {
        for layer in self.layers.iter() {
            layer.update(learning_rate);
        }
    }

    pub fn set_input(&mut self, input: &[f64]) {
        self.layers[0].set_state(input);
    }

    pub fn get_output(&self) -> Vec<f64> {
        return self.layers.last().expect("No Output layer").get_state();
    }

    pub fn activate(&mut self, input: &[f64]) -> Vec<f64> {
        self.set_input(input);
        for layer in self.layers.iter_mut().skip(1) {
            layer.activate();
        }
        return self.get_output();
    }
}
