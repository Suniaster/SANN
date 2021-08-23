#![allow(dead_code)]

pub mod activations;
pub mod layer;
pub mod neural_net;
pub mod perceptron;


#[test]
fn activate_million_times(){  
    time_test!();
    use ndarray::array;
    let input1 = array![1.0, 0.0];
    let nn2 = neural_net::NeuralNet::from_format(&[2, 3, 2, 2]);
    for _ in 0..1_000_000{
        nn2.activate(&input1);
    }
}