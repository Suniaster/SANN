use ndarray::{Array1, Array2};

use crate::{layer::NetLayer, network::Ann};

struct BackPropagationData {
    layers_output: Vec<Array1<f64>>,
    layers_deltas: Vec<Array1<f64>>,
}

impl BackPropagationData {
    fn new<T: NetworkBackPropagation>(net: &mut T) -> BackPropagationData {
        return BackPropagationData {
            layers_output: net
                .get_layers()
                .iter()
                .map(|layer| Array1::zeros(layer.get_format().1))
                .collect(),
            layers_deltas: net
                .get_layers()
                .iter()
                .map(|layer| Array1::zeros(layer.get_format().1))
                .collect(),
        };
    }

    fn activate_net(&mut self, net: &mut dyn NetworkBackPropagation, input: &Array1<f64>) {
        self.layers_output[0] = net.get_layers()[0].activate(input);
        for i in 1..net.get_layers().len() {
            self.layers_output[i] = net.get_layers()[i].activate(&self.layers_output[i - 1]);
        }
    }

    fn update_deltas(&mut self, net: &mut dyn NetworkBackPropagation, expected: &Array1<f64>) {
        let l_len = net.get_layers().len();
        self.layers_deltas[l_len - 1] =
            net.last_layer_errors(&self.layers_output[l_len - 1], expected);

        for i in (0..l_len - 1).rev() {
            let next_layer_deltas = &self.layers_deltas[i + 1];
            let next_layer_ws = net.get_layers()[i + 1].get_weights();
            let this_layer_out = &self.layers_output[i];

            self.layers_deltas[i] = net.get_layers()[i].get_backpropag_error(
                this_layer_out,
                next_layer_deltas,
                next_layer_ws,
            );
        }
    }
}

pub trait NetworkBackPropagation {
    fn last_layer_errors(&self, output: &Array1<f64>, expected: &Array1<f64>) -> Array1<f64>;

    fn get_layers_mut(&mut self) -> &mut Vec<Box<dyn NetLayer>>;
    fn get_layers(&self) -> &Vec<Box<dyn NetLayer>>;

    fn get_loss_batch(&self, inputs: &Vec<Array1<f64>>, targets: &Vec<Array1<f64>>) -> f64;

    fn backpropagate(&mut self, input: &Array1<f64>, target: &Array1<f64>, learning_rate: f64)
    where
        Self: Sized,
    {
        let mut bp_data = BackPropagationData::new(self);
        bp_data.activate_net(self, input);
        bp_data.update_deltas(self, target);

        for i in 1..self.get_layers().len() {
            let prev_layer_out = &bp_data.layers_output[i - 1];
            let this_layer_deltas = &bp_data.layers_deltas[i];
            self.get_layers_mut()[i].update_params(
                this_layer_deltas,
                prev_layer_out,
                learning_rate,
            );
        }
    }

    fn train(
        &mut self,
        inputs: &Vec<Array1<f64>>,
        targets: &Vec<Array1<f64>>,
        iterations: usize,
        learning_rate: f64,
    ) -> f64
    where
        Self: Sized,
    {
        for _ in 0..iterations {
            for (i, input) in inputs.iter().enumerate() {
                self.backpropagate(input, &targets[i], learning_rate);
            }
        }
        return self.get_loss_batch(inputs, targets);
    }
}

impl NetworkBackPropagation for Ann {
    fn last_layer_errors(&self, output: &Array1<f64>, expected: &Array1<f64>) -> Array1<f64> {
        // Considering the last function to have linear activation.
        return output - expected;
    }

    fn get_layers(&self) -> &Vec<Box<dyn NetLayer>> {
        return &self.layers;
    }

    fn get_layers_mut(&mut self) -> &mut Vec<Box<dyn NetLayer>> {
        return &mut self.layers;
    }

    fn get_loss_batch(&self, inputs: &Vec<Array1<f64>>, expecteds: &Vec<Array1<f64>>) -> f64 {
        let mut loss = 0.0;
        for i in 0..inputs.len() {
            loss += self.get_error(&inputs[i], &expecteds[i]);
        }
        return loss / inputs.len() as f64;
    }
}

impl dyn NetLayer {
    fn get_backpropag_error(
        &self,
        this_layer_out: &Array1<f64>,
        next_layer_deltas: &Array1<f64>,
        next_layer_ws: &Array2<f64>,
    ) -> Array1<f64> {
        let derivative_f = self.get_activation().d;
        let derivatives = this_layer_out.mapv(|x| (derivative_f)(&x));
        let errors = next_layer_ws.t().dot(next_layer_deltas) * derivatives;
        return errors;
    }
}