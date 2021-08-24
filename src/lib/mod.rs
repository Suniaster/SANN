#![allow(dead_code)]

pub mod activations;
pub mod layer;
pub mod neural_net;
pub mod perceptron;


#[test]
fn activate_million_times(){  
    time_test!();
    use ndarray::array;
    let input1 = array![1.0, 0.0];
    let nn2 = neural_net::NeuralNet::from_format(&[2, 3, 100, 50, 2, 2]);
    
    for _ in 0..50_000{
        nn2.activate(&input1);
    }
}


pub fn train(){  
    time_test!();
    use ndarray::array;
    use super::lib::activations::ActivationType;
    let input1 = array![1.0, 0.0];
    let input3 = array![0.0, 0.0];

    let batch = vec![
        array![0.0, 0.0],
        array![0.0, 1.0],
        array![1.0, 0.0],
        array![1.0, 1.0],
    ];
    let y = vec![
        array![0.0],
        array![1.0],
        array![1.0],
        array![0.0],
    ];

    let mut nn2 = neural_net::NeuralNet::from_format(&[2, 3, 2, 1]);
    nn2.format(&[
        ActivationType::ReLu,
        ActivationType::ReLu,
        ActivationType::Sigmoid,
    ]);

    nn2.train_batch(&batch, &y, 0.15, 100_000);
    
    println!("Predict 1: {:?}", nn2.activate(&input1));
    println!("Predict 0: {:?}", nn2.activate(&input3));
}