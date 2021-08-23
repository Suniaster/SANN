#![allow(dead_code)]
pub mod helper;
pub mod layer;
pub mod network;
pub mod node;



#[test]
fn activate_million_times(){
    time_test!();
    use super::lib::activations::ActivationType;

    let mut xor_net = network::Network::new(&[2, 3, 2, 2]);
    xor_net.format(&[
        ActivationType::Linear,
        ActivationType::Sigmoid,
        ActivationType::Sigmoid,
        ActivationType::Sigmoid,
    ]);

    for _ in 0..1_000_000 {
        xor_net.activate(vec![1.0, 0.0]);
    }
}
