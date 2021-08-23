#![allow(dead_code)]

use ndarray::array;
use ndarray::Array1;
use rand;
use std::vec::Vec;

#[derive(Debug)]
pub struct Perceptron {
    pub input_weigths: Array1<f64>,
    pub bias: f64,
    pub shape: usize,
}

impl Perceptron {
    pub fn from_rand(input_shape: usize) -> Perceptron {
        let mut weights: Vec<f64> = Vec::new();
        for _ in 0..input_shape {
            weights.push(rand::random::<f64>() * 2.0 - 1.0);
        }
        return Perceptron {
            input_weigths: Array1::from_vec(weights),
            bias: rand::random::<f64>() * 2.0 - 1.0,
            shape: input_shape,
        };
    }

    pub fn foward(&self, input: &Array1<f64>) -> Array1<f64> {
        let propagation = self.input_weigths.dot(input);
        return array![propagation + self.bias];
    }

    fn print(&self) {
        println!("{:?}", self);
    }
}
