use ndarray::{Array1};
use crate::layer::NetLayerSerialize;
use super::layer::NetLayer;

pub struct Ann {
    pub layers: Vec<Box<dyn NetLayer>>,
    pub input_format: usize
}

impl Ann {

    pub fn new_empty() -> Ann {
        return Ann::new(0);
    }

    pub fn new(input_format: usize) -> Ann {
        return Ann { 
            layers: vec![],
            input_format
        };
    }

    pub fn push<L: NetLayer + NetLayerSerialize>(&mut self, out_format: usize) -> &mut Box<dyn NetLayer> {
        let input_format:usize;
        if self.layers.len() > 0 {
            input_format = self.layers[self.layers.len() - 1].get_format().1;
        } else {
            input_format = self.input_format;
        }
        self.add_layer(L::create((input_format, out_format)));
        return self.layers.last_mut().unwrap();
    }

    pub fn add_layer(&mut self, layer: Box<dyn NetLayer>) {
        if self.layers.len() == 0 {
            self.input_format = layer.get_format().0;
        }
        self.layers.push(layer);
    }

    pub fn activate(&self, input: &Array1<f64>) -> Array1<f64> {
        let mut output = input.clone();
        for layer in self.layers.iter() {
            output = layer.activate(&output);
        }
        return output;
    }

    pub fn get_error(&self, input: &Array1<f64>, expected: &Array1<f64>) -> f64 {
        let out = self.activate(input);
        let sub = out - expected;
        let sqrd = sub.map(|x| x * x);
        return sqrd.sum();
    }

    pub fn randomize(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.randomize_params();
        }
    }
}