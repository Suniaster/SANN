use sann::activations::ActivationType;
use sann::network::Ann;
use sann::layer::dense::DenseLayer;

use sann::io;
use ndarray::Array1;

#[test]
pub fn test_saving_network(){
    let mut ann = Ann::new(2);

    ann.push::<DenseLayer>(3)
        .set_activation(ActivationType::Sigmoid);
    ann.push::<DenseLayer>(2)
        .set_activation(ActivationType::Sigmoid);
    ann.push::<DenseLayer>(3)
        .set_activation(ActivationType::ReLu);
    ann.push::<DenseLayer>(1)
        .set_activation(ActivationType::Linear);

    ann.randomize();

    io::save_net(&ann, &String::from("test.json"));

    let net2 = io::load_net(&String::from("test.json"));
    
    let input = Array1::from_vec(vec![1.0, 1.0]);

    assert_eq!(ann.activate(&input), net2.activate(&input));
}