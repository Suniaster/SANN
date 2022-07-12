use super::{activations, NetLayerSerialize};
use ndarray::{Array1, Array2};

use super::NetLayer;
use rand;
use rand::Rng;

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

    fn update_params(
        &mut self,
        this_layer_deltas: &Array1<f64>,
        previous_layer_output: &Array1<f64>,
        learning_rate: f64,
    ) {
        for i in 0..self.weights.nrows() {
            let j_l_w = previous_layer_output * this_layer_deltas[i];

            for j in 0..self.weights.ncols() {
                self.weights[(i, j)] -= j_l_w[j] * learning_rate;
            }
        }

        self.biases -= &(this_layer_deltas * learning_rate);
    }

    fn set_activation(&mut self, _type: activations::ActivationType) {
        self.activation = activations::Activation::create(_type);
    }

    fn get_activation(&self) -> activations::Activation {
        return activations::Activation::create(self.activation.t.clone());
    }

    fn get_weights(&self) -> Array2<f64> {
        return self.weights.clone();
    }

    fn set_weights(&mut self, weights: Array2<f64>) {
        self.weights = weights;
    }

    fn get_format(&self) -> (usize, usize) {
        return (self.weights.shape()[1], self.weights.shape()[0]);
    }

    fn randomize_params(&mut self) {
        let mut rng = rand::thread_rng();
        self.weights.mapv_inplace(|_| rng.gen::<f64>() * 2. - 1.);
        self.biases.mapv_inplace(|_| rng.gen::<f64>() * 2. - 1.);
    }

    fn get_biases(&self) -> Array1<f64> {
        return self.biases.clone();
    }

    fn set_biases(&mut self, biases: Array1<f64>) {
        self.biases = biases;
    }

    fn get_type_name(&self) -> String {
        return String::from("Dense");
    }
}

impl NetLayerSerialize for DenseLayer {
    fn create(format: (usize, usize)) -> Box<dyn NetLayer> {
        return Box::new(DenseLayer::new(format.0, format.1));
    }
}
