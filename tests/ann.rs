use sann::*;
use sann::activations::ActivationType;
use ndarray::Array1;

#[macro_use]
extern crate time_test;

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
pub fn train_test(){
    let mut ann = Ann::new();
    ann.add_layer(Box::new(DenseLayer::new(2, 3)));
    ann.add_layer(Box::new(DenseLayer::new(3, 3)));
    ann.add_layer(Box::new(DenseLayer::new(3, 2)));
    ann.add_layer(Box::new(DenseLayer::new(2, 1)));

    ann.set_activations(&vec![
        ActivationType::Linear,
        ActivationType::Sigmoid,
        ActivationType::ReLu,
        ActivationType::Sigmoid,
    ]);
    
    println!("-----");
    for i in 0..100_000 {
        let input = Array1::from_vec(vec![1.0, 1.0]);
        let loss = ann.learn(&input, &Array1::from_vec(vec![0.0]), 0.1);
        println!("\rIteration {}:\t\tResult {:?}", i, loss);
    }
    println!("------");
}

#[test]
pub fn test_time(){
    time_test!();
    let mut ann = Ann::new();
    //[2, 3, 100, 50, 2, 2]
    ann.add_layer(Box::new(DenseLayer::new(2, 3)));
    ann.add_layer(Box::new(DenseLayer::new(3, 100)));
    ann.add_layer(Box::new(DenseLayer::new(100, 50)));
    ann.add_layer(Box::new(DenseLayer::new(50, 50)));
    ann.add_layer(Box::new(DenseLayer::new(50, 2)));
    ann.add_layer(Box::new(DenseLayer::new(2, 2)));

    let input = Array1::from_vec(vec![1.0, 0.0]);

    for _ in 0..1_000_000 {
        ann.activate(&input);
    }
}

#[test]
pub fn creation_time_test(){
    time_test!();
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