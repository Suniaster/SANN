use sann::network;
use sann::activations::*;
use sann::io;
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
    
    let first_file = String::from("original.json");
    io::save_net(&xnet, &first_file);

    let net2 = io::load_net(&first_file);
    
    io::save_net(&net2, &String::from("loaded.json"));
}
