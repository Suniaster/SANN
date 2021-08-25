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

let out_file = String::from("original.json");
io::save_net(&net1, &out_file);

let net2 = io::load_net(&out_file);
io::save_net(&net2, &String::from("loaded.json"));
# ... loaded.json content is the same as original.json
```