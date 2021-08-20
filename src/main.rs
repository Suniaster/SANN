use ndarray::array;

// use lib::neural_net::perceptron::Neuron;
mod lib;
// use lib::layer::*;
// use lib::neural_net::*;

fn main() {
    let input1 = array![1.0, 1.0, 1.0];

    let nn = lib::neural_net::NeuralNet::from_format(&[3, 2, 1]);
    println!("{:?}", nn.activate(&input1));

    let nn2 = lib::neural_net::NeuralNet::from_format(&[3, 1]);

    // let loss = lib::layer::dense::calculate_error(
    //     &[0.0, 0.0],
    //     &[1.0, 30.0]
    // );
    // println!("Loss iss {}", loss);
    println!("{:?}", nn2.activate(&input1));

    println!("###########");
    // println!("{:?}", nn2.train(&input1, &array![1.0]));
    // let mut perceptron = lib::perceptron::Perceptron::from_rand(2);
    // println!("{:?}", nn2.train(&perceptron, &array![1.0]));
}
