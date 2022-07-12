use super::activations;
use ndarray::{Array1, Array2};

pub mod dense;
pub mod recurrent;
pub mod dropout;

pub trait NetLayer {
    fn activate(&self, input: &Array1<f64>) -> Array1<f64>;

    fn update_params(
        &mut self,
        _this_layer_deltas: &Array1<f64>,
        _previous_layer_output: &Array1<f64>,
        _learning_rate: f64,
    ){}

    fn randomize_params(&mut self) {}

    fn get_weights(&self) -> Array2<f64>{
        return Array2::zeros(self.get_format());
    }

    fn set_weights(&mut self, _weights: Array2<f64>){}

    fn get_biases(&self) -> Array1<f64>{
        return Array1::zeros(self.get_format().1);
    }
    fn set_biases(&mut self, _biases: Array1<f64>){}

    fn get_format(&self) -> (usize, usize);

    fn set_activation(&mut self, _type: activations::ActivationType){
    }
    fn get_activation(&self) -> activations::Activation {
        return activations::Activation::create(activations::ActivationType::Default);
    }

    fn get_type_name(&self) -> String;

    fn set_training_mode(&mut self, _mode: bool){}
}

pub trait NetLayerSerialize {
    fn create(format: (usize, usize)) -> Box<dyn NetLayer>;
}
