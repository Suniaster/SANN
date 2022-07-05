# Simple Artificial Neural Network

A crate that implements simple usage of dense neural networks.

## Instalation

Add this to your dependencies on `Cargo.toml`:

```toml
sann = { git = "https://github.com/Suniaster/SANN.git" }
```

## Usage

Create dense network with format and activations:
```rust
use sann::network;
use sann::activations::*;

let mut net = network::Network::new(&[2, 3, 5, 2, 1]);
net.format(&[
    ActivationType::Linear,
    ActivationType::ReLu,
    ActivationType::Sigmoid,
    ActivationType::ReLu,
    ActivationType::Sigmoid,
]);

```

Train a XOR network:

```rust
let mut xor_net = # ... initialize network
let input = vec![
    vec![0.0, 0.0],
    vec![0.0, 1.0],
    vec![1.0, 0.0],
    vec![1.0, 1.0],
];
let expected = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];

xor_net.train(&input, &expected, 0.15, 100_000);

println!("Result should be almost 0: {:?}", xor_net.activate(&input[0]));
println!("Result should be almost 1: {:?}", xor_net.activate(&input[1]));
println!("Result should be almost 1: {:?}", xor_net.activate(&input[2]));
println!("Result should be almost 0: {:?}", xor_net.activate(&input[3]));
```

Save and load network params:

```rust
let net1 = # ... initialize network

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
    
# ... loaded.json content is the same as original.json
```