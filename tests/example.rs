use sann::network::Ann;
use sann::layer::dense::DenseLayer;

use sann::io;
use ndarray::Array1;


#[test]
pub fn test_saving_network(){
    let mut ann = Ann::new();
    ann.add_layer(Box::new(DenseLayer::new(2, 3)));
    ann.add_layer(Box::new(DenseLayer::new(3, 2)));
    ann.randomize();

    io::save_net(&ann, &String::from("test.json"));

    let net2 = io::load_net(&String::from("test.json"));
    
    let input = Array1::from_vec(vec![1.0, 0.0]);

    assert_eq!(ann.activate(&input), net2.activate(&input));
}