use ndarray::Array1;
use super::NetLayer;

pub struct DropoutLayer{
    pub dropout_probability: f64,
    pub dropout_mask: Array1<f64>,
    pub input_format: usize,
    pub active: bool,
}

impl DropoutLayer {
    pub fn new(input_format: usize, dropout_probability: f64) -> DropoutLayer {
        let dropout_mask = Array1::ones(input_format);
        let dropout_mask = dropout_mask.map(|_:&f64| {
            if rand::random::<f64>() < dropout_probability {
                return 0.0;
            } else {
                return 1.0/(1. - dropout_probability);
            }
        });

        return DropoutLayer {
            active: false,
            dropout_probability,
            dropout_mask,
            input_format,
        };
    }


}

impl NetLayer for DropoutLayer  {
    fn activate(&self, input: &Array1<f64>) -> Array1<f64> {
        if self.active {
            return input.clone();
        } else {
            return input * &self.dropout_mask;
        }
    }

    fn get_type_name(&self) -> String {
        return String::from("DropoutLayer");
    }

    fn get_format(&self) -> (usize, usize) {
        return (self.input_format, self.input_format);
    }

    fn update_params(
            &mut self,
            _this_layer_deltas: &Array1<f64>,
            _previous_layer_output: &Array1<f64>,
            _learning_rate: f64,
        ) {
        self.dropout_mask = self.dropout_mask.map(|_:&f64| {
            if rand::random::<f64>() < self.dropout_probability {
                return 0.0;
            } else {
                return 1.0/(1. - self.dropout_probability);
            }
        });
    }

    fn set_training_mode(&mut self, _mode: bool) {
        self.active = _mode;
    }
}