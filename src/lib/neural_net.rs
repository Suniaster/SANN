use super::layer;
use ndarray::Array1;
use super::activations::ActivationType;
use super::activations::Activation;

pub struct NeuralNet {
    layers: Vec<layer::Layer>,
}

impl NeuralNet {
    pub fn from_format(format: &[i32]) -> NeuralNet {
        let mut layers: Vec<layer::Layer> = Vec::new();
        let mut input_shape: i32 = 1;
        for i in 0..format.len() {
            if i == 0 {
                input_shape = format[0];
                continue;
            }
            layers.push(layer::Layer::from_rand(
                input_shape as usize,
                format[i] as usize,
            ));
            input_shape = format[i];
        }
        return NeuralNet { layers };
    }

    pub fn format(&mut self, acts: &[ActivationType]){
        for i in 0..acts.len() {
            self.layers[i].activation = Activation::create(acts[i].clone());
        }
    }

    pub fn calculate_loss(output: &f64, expected: &f64) -> f64 {
        return (output - expected).powi(2);
    }

    pub fn calculate_loss_derivative(output: &f64, expected: &f64) -> f64 {
        return 2.0 * (output - expected);
    }

    pub fn calculate_loss_vec(input: &Array1<f64>, expected: &Array1<f64>) -> f64 {
        return input
            .iter()
            .zip(expected.iter())
            .fold(0.0, |sum, (x, y)| -> f64 {
                return sum + NeuralNet::calculate_loss(x, y);
            });
    }

    pub fn train_batch(
        &mut self,
        input: &Vec<Array1<f64>>,
        expected: &Vec<Array1<f64>>,
        learning_rate: f64,
        iterations: usize,
    ) {
        for i in 0..iterations {
            print!("\rIteration {} ######", i + 1);
            let mut loss = 0.0;
            for x in 0..input.len() {
                let result = self.train(&input[x], &expected[x], learning_rate);
                loss += NeuralNet::calculate_loss_vec(result, &expected[x]);
            }
            loss = loss / input.len() as f64;
            print!(" Loss: {} ", loss);
        }
        println!();
    }

    pub fn train(
        &mut self,
        input: &Array1<f64>,
        expected: &Array1<f64>,
        learning_rate: f64,
    ) -> &Array1<f64> {
        self.foward(input);
        self.backward(expected);
        return self.learn(input, learning_rate);
    }

    pub fn learn(&mut self, input: &Array1<f64>, learning_rate: f64) -> &Array1<f64> {
        let mut last_layer_out = input;

        for layer in self.layers.iter_mut() {
            for neuron_i in 0..layer.weights_matrix.shape()[0] {
                for neuron_weight_index in 0..layer.weights_matrix.shape()[1] {
                    layer.biases[neuron_i] += learning_rate * layer.deltas[neuron_i];
                    layer.weights_matrix[[neuron_i, neuron_weight_index]] += learning_rate
                        * layer.deltas[neuron_i]
                        * last_layer_out[neuron_weight_index];
                }
            }

            last_layer_out = &layer.last_output;
        }
        return &self
            .layers
            .iter()
            .rev()
            .next()
            .expect("No layer")
            .last_output;
    }

    pub fn backward(&mut self, expected: &Array1<f64>) {
        let mut iterator = self.layers.iter_mut().rev();

        let output_layer = iterator.next().expect("NN with no input layer");

        let errors: Array1<f64> = output_layer
            .last_output
            .iter()
            .zip(expected.iter())
            .map(|(w, expected)| -> f64 { NeuralNet::calculate_loss_derivative(expected, w) })
            .collect();

        output_layer.deltas = errors
            .iter()
            .zip(output_layer.last_output.iter())
            .map(|(err, out)| -> f64 { (*err) * (output_layer.activation.d)(out) })
            .collect();

        let mut prev_layer = output_layer;
        for layer in iterator {
            let mut errors: Vec<f64> = vec![];

            for neuron_i in 0..layer.weights_matrix.shape()[0] {
                let mut error = 0.0;
                for prev_layer_neuron_index in 0..prev_layer.weights_matrix.shape()[0] {
                    let weight = prev_layer
                        .weights_matrix
                        .get((prev_layer_neuron_index, neuron_i))
                        .expect("Incorrect acess on prev-layer backpropagation");
                    let delta = prev_layer
                        .deltas
                        .get(prev_layer_neuron_index)
                        .expect("Incorrect access in delta vector");
                    error += *delta * *weight
                }
                errors.push(error);
            }

            layer.deltas = errors
                .iter()
                .zip(layer.last_output.iter())
                .map(|(err, out)| -> f64 { (*err) * (layer.activation.d)(out) })
                .collect();
            prev_layer = layer;
            // println!("delta v: {:?}", prev_layer.deltas);
        }
    }

    pub fn activate(&self, input: &Array1<f64>) -> Array1<f64> {
        let mut iterator = self.layers.iter();
        let input_layer = iterator.next().expect("NN with no input layer");

        let mut layer_input = input_layer.activate(input);

        for hidden_layer in iterator {
            layer_input = hidden_layer.activate(&layer_input);
        }

        return layer_input;
    }

    pub fn foward(&mut self, input: &Array1<f64>) -> &Array1<f64> {
        let mut iterator = self.layers.iter_mut();
        let input_layer = iterator.next().expect("NN with no input layer");

        let mut layer_input = input_layer.forward(input);

        for hidden_layer in iterator {
            layer_input = hidden_layer.forward(&layer_input);
        }

        return layer_input;
    }
}
