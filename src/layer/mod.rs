use super::activations;
use ndarray::{Array1, Array2};

pub mod dense;

pub trait NetLayer {
    fn activate(&self, input: &Array1<f64>) -> Array1<f64>;

    fn update_params(
        &mut self,
        this_layer_deltas: &Array1<f64>,
        previous_layer_output: &Array1<f64>,
        learning_rate: f64,
    );

    fn randomize_params(&mut self) {}

    fn get_weights(&self) -> &Array2<f64>;
    fn set_weights(&mut self, weights: Array2<f64>);

    fn get_biases(&self) -> &Array1<f64>;
    fn set_biases(&mut self, biases: Array1<f64>);

    fn get_format(&self) -> (usize, usize);

    fn set_activation(&mut self, _type: activations::ActivationType);
    fn get_activation(&self) -> &activations::Activation;

    fn get_type_name(&self) -> String;
}

pub trait NetLayerSerialize {
    fn create(format: (usize, usize)) -> Box<dyn NetLayer>;
}
