use sann::{*, activations::ActivationType};
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

#[test]
pub fn learn_test(){
    let mut ann = Ann::new();
    //[2, 3, 3, 1]
    ann.add_layer(Box::new(DenseLayer::new(2, 3)));
    ann.add_layer(Box::new(DenseLayer::new(3, 3)));
    ann.add_layer(Box::new(DenseLayer::new(3, 1)));
    let input = Array1::from_vec(vec![1.0, 1.0]);
    let target = Array1::from_vec(vec![1.0]);

    ann.learn(&input, &target, 0.1);
}

#[test]
pub fn train_test(){
    let mut ann = Ann::new();
    //[2, 3, 3, 1]
    ann.add_layer(Box::new(DenseLayer::new(2, 2)));
    ann.add_layer(Box::new(DenseLayer::new(2, 2)));
    ann.add_layer(Box::new(DenseLayer::new(2, 2)));
    ann.add_layer(Box::new(DenseLayer::new(2, 1)));
    ann.randomize();

    ann.set_activations(&[
        ActivationType::Linear,
        ActivationType::Sigmoid,
        ActivationType::Sigmoid,
        ActivationType::Linear,
    ]);

    let input =  vec![
        Array1::from_vec(vec![1.0, 1.0]),
        Array1::from_vec(vec![1.0, 0.0]),
        Array1::from_vec(vec![0.0, 1.0]),
        Array1::from_vec(vec![0.0, 0.0]),
    ];

    let expected = vec![
        Array1::from_vec(vec![0.0]),
        Array1::from_vec(vec![1.0]),
        Array1::from_vec(vec![1.0]),
        Array1::from_vec(vec![0.0]),
    ];

    let result = ann.activate(&input[0]);
    println!("Result {:?}", result);
    let loss = ann.get_loss_batch(&input, &expected);
    println!("Loss before training {:?}", loss);


    let loss = ann.train(&input, &expected, 10_000,  0.1);
    println!("Loss after training: {}", loss);

    // Result after training:
    let result = ann.activate(&input[0]);
    println!("Result 0: {:?}", result);
    let result = ann.activate(&input[1]);
    println!("Result 1: {:?}", result);
    let result = ann.activate(&input[2]);
    println!("Result 2: {:?}", result);
    let result = ann.activate(&input[3]);
    println!("Result 3: {:?}", result);
}