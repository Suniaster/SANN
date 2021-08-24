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