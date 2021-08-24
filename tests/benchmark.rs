use sann::network;
use sann::activations::*;

#[macro_use]
extern crate time_test;


#[test]
#[ignore]
fn activate_million_times() {
    time_test!();

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

#[test]
#[ignore]
pub fn train() {
    time_test!();

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