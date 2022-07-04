use sann::*;
use ndarray::Array1;


#[test]
pub fn test_net(){
    let mut ann = Ann::new();
    ann.add_layer(Box::new(DenseLayer::new(2, 3)));
    ann.add_layer(Box::new(DenseLayer::new(3, 2)));

    let input = Array1::from_vec(vec![1.0, 1.0]);
    let result = ann.activate(&input);

    println!("Result {:?}", result);
}

#[test]
pub fn creation_time_test(){
    let mut ann = Ann::new();
    //[2, 3, 100, 50, 2, 2]
    ann.add_layer(Box::new(DenseLayer::new(2, 3)));
    ann.add_layer(Box::new(DenseLayer::new(3, 100)));
    ann.add_layer(Box::new(DenseLayer::new(100, 50)));
    ann.add_layer(Box::new(DenseLayer::new(50, 50)));
    ann.add_layer(Box::new(DenseLayer::new(50, 2)));
    ann.add_layer(Box::new(DenseLayer::new(2, 2)));
    let input = Array1::from_vec(vec![1.0, 0.0]);
    println!("Result {:?}", ann.activate(&input));
}