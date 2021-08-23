use super::super::lib::activations::*;
use super::layer::*;

pub struct Network {
    layers: Vec<Layer>,
}

impl Network {
    pub fn new(layers_format: Vec<u16>) -> Network {
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

    pub fn set_input(&mut self, input: Vec<f64>) {
        self.layers[0].set_state(input);
    }

    pub fn get_output(&self) -> Vec<f64> {
        return self.layers.last().expect("No Output layer").get_state();
    }

    pub fn activate(&mut self, input: Vec<f64>) -> Vec<f64> {
        self.set_input(input);
        for layer in self.layers.iter_mut().skip(1) {
            layer.activate()
        }
        return self.get_output();
    }
}
