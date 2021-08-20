use ndarray::Array1;
use super::perceptron;

pub mod dense;
pub mod activations;
pub enum Layer {
    Dense(dense::LayerDense),
    Sigmoid,
    ReLu,
}

impl perceptron::Forward for Layer {
    fn foward(&self, input: &Array1<f64>) -> Array1<f64> {
        match self {
            Layer::Dense(layer) => layer.foward(input),
            Layer::ReLu => activations::forward_relu(input),
            Layer::Sigmoid => activations::foward_sigmoid(input)
        }
    }
}

impl Layer {
    pub fn new_dense(input_shape: usize, output_shape: usize) -> Layer {
        return Layer::Dense(dense::LayerDense::from_rand(input_shape, output_shape));
    }
}
