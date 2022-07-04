use sann::network;
use sann::activations::*;

#[macro_use]
extern crate time_test;


#[test]
#[ignore]
fn activate_million_times() {
    time_test!();

    let mut xor_net = network::Network::new(&[2, 3, 100, 50, 50, 2, 2]);

    let input = &[1.0, 0.0];
    for _ in 0..1_000_000 {
        xor_net.activate(input);
    }
}

#[test]
pub fn creation_time_test2() {
    time_test!();
    let mut xor_net = network::Network::new(&[2, 3, 100, 50, 50, 2, 2]);
    println!("Result {:?}", xor_net.activate(&[1.0, 0.0]));
}