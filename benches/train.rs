use sann::network::Ann;
use sann::layer::dense::DenseLayer;

use sann::activations::ActivationType;
use ndarray::Array1;

use sann::algorithms::NetworkBackPropagation;

use criterion::{criterion_group, criterion_main, Criterion};


fn performance_compare(c: &mut Criterion){
    let mut group = c.benchmark_group("Training");
    let mut ann = Ann::new(2);
    //[2, 3, 100, 50, 2, 2]
    ann.push::<DenseLayer>(3)
        .set_activation(ActivationType::Linear);
    ann.push::<DenseLayer>(4)
        .set_activation(ActivationType::ReLu);
    ann.push::<DenseLayer>(2)
        .set_activation(ActivationType::Sigmoid);
    ann.push::<DenseLayer>(1)
        .set_activation(ActivationType::ReLu);

    let input =  vec![
        Array1::from_vec(vec![1.0, 1.0]),
        Array1::from_vec(vec![1.0, 0.0]),
        Array1::from_vec(vec![0.0, 1.0]),
        Array1::from_vec(vec![0.0, 0.0]),
    ];

    let expected = vec![
        Array1::from_vec(vec![1.0]),
        Array1::from_vec(vec![0.0]),
        Array1::from_vec(vec![0.0]),
        Array1::from_vec(vec![1.0]),
    ];
    group.bench_function("new", |b| 
        b.iter(|| 
            ann.train(&input, &expected, 1000, 0.1)
        )
    );
    group.finish();
}
criterion_group!(benches, performance_compare);
criterion_main!(benches);
