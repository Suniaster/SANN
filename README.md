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

let mut ann = Ann::new(2);
//[2, 3, 3, 1]
ann.push::<DenseLayer>(2)
    .set_activation(ActivationType::Linear);
ann.push::<DenseLayer>(30)
    .set_activation(ActivationType::Relu);
ann.push::<DenseLayer>(20)
    .set_activation(ActivationType::Sigmoid);
ann.push::<DenseLayer>(3)
    .set_activation(ActivationType::Linear);
```

Train a XOR network:

```rust
let mut ann = Ann::new(2);
//[2, 3, 3, 1]
ann.push::<DenseLayer>(2)
    .set_activation(ActivationType::Linear);
ann.push::<DenseLayer>(2)
    .set_activation(ActivationType::Sigmoid);
ann.push::<DenseLayer>(2)
    .set_activation(ActivationType::Sigmoid);
ann.push::<DenseLayer>(1)
    .set_activation(ActivationType::Linear);
ann.randomize();

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

let loss = ann.get_loss_batch(&input, &expected);
println!("Loss before training {:?}", loss);


let loss = ann.train(&input, &expected, 100_000,  0.1);
println!("Loss after training: {:?}", loss);

// Result after training:
let result = ann.activate(&input[0]);
println!("Result 0: {:?}", result);
let result = ann.activate(&input[1]);
println!("Result 1: {:?}", result);
let result = ann.activate(&input[2]);
println!("Result 2: {:?}", result);
let result = ann.activate(&input[3]);
println!("Result 3: {:?}", result);
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