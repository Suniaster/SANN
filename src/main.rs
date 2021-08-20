use ndarray::array;

// use lib::neural_net::perceptron::Neuron;
mod lib;
use lib::layer::*;
use lib::neural_net::*;

fn main() {
    let input1 = array![1.0, 1.0, 1.0];

    let nn = NeuralNet::from_format(&[3, 2, 1]);
    println!("{:?}", nn.foward(&input1));

    let nn2 = NeuralNet::create(
        vec![
            Layer::new_dense(3, 2),
            Layer::ReLu,
            Layer::new_dense(2, 1),
            Layer::ReLu,
        ]
    );

    println!("{:?}", nn2.foward(&input1));
}
