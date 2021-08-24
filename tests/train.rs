use sann::network;
use sann::activations::*;
use sann::io;

#[test]
#[ignore]
pub fn train_xor() {
    let mut xor_net = network::Network::new(&[2, 3, 2, 1]);
    xor_net.format(&[
        ActivationType::Linear,
        ActivationType::ReLu,
        ActivationType::ReLu,
        ActivationType::Sigmoid,
    ]);

    let input = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];
    let expected = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];

    xor_net.train(&input, &expected, 0.15, 100_000);

    println!("Result: {:?}", xor_net.activate(&input[0]));
    println!("Result: {:?}", xor_net.activate(&input[1]));
    println!("Result: {:?}", xor_net.activate(&input[2]));
    println!("Result: {:?}", xor_net.activate(&input[3]));

    io::save_net(&xor_net, &String::from("xor.json"));
}

#[test]
#[ignore]
fn load_xor() { 
    // train_xor();
    let input = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];

    let mut net = io::load_net(&String::from("xor.json"));

    println!("Result: {:?}", net.activate(&input[0]));
    println!("Result: {:?}", net.activate(&input[1]));
    println!("Result: {:?}", net.activate(&input[2]));
    println!("Result: {:?}", net.activate(&input[3]));
}