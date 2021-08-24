#![allow(dead_code)]
pub mod activations;
pub mod helper;
pub mod layer;
pub mod network;
pub mod node;

#[test]
fn activate_million_times() {
    time_test!();
    use super::lib::activations::ActivationType;

    let mut xor_net = network::Network::new(&[2, 3, 100, 50, 2, 2]);
    xor_net.format(&[
        ActivationType::Linear,
        ActivationType::Sigmoid,
        ActivationType::Sigmoid,
        ActivationType::Sigmoid,
    ]);

    let input = &[1.0, 0.0];
    for _ in 0..50_000 {
        xor_net.activate(input);
    }
}

pub fn train() {
    time_test!();
    use super::lib::activations::ActivationType;

    let mut xor_net = network::Network::new(&[2, 3, 2, 1]);
    xor_net.format(&[
        ActivationType::Linear,
        ActivationType::ReLu,
        ActivationType::ReLu,
        ActivationType::Sigmoid,
    ]);

    let input = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];
    let expected = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];

    xor_net.train(&input, &expected, 0.15, 100_000);

    println!("Result: {:?}", xor_net.activate(&input[0]));
    println!("Result: {:?}", xor_net.activate(&input[1]));
    println!("Result: {:?}", xor_net.activate(&input[2]));
    println!("Result: {:?}", xor_net.activate(&input[3]));
}
