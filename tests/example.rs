use sann::network;
use sann::activations::*;
#[macro_use]
extern crate time_test;

#[test]
pub fn save_net(){
    time_test!();

    let mut xnet = network::Network::new(&[2, 3, 3, 2]);
    xnet.format(&[
        ActivationType::Linear,
        ActivationType::ReLu,
        ActivationType::ReLu,
        ActivationType::Sigmoid,
    ]);

    xnet.save(&String::from("teste.json"));
}