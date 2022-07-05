use sann::network::Ann;
use sann::layer::dense::DenseLayer;

use ndarray::Array1;

use criterion::{criterion_group, criterion_main, Criterion};

fn criterion_benchmark_v2(c: &mut Criterion) {
    let mut ann = Ann::new_empty();
    //[2, 3, 100, 50, 2, 2]
    ann.add_layer(Box::new(DenseLayer::new(2, 3)));
    ann.add_layer(Box::new(DenseLayer::new(3, 100)));
    ann.add_layer(Box::new(DenseLayer::new(100, 50)));
    ann.add_layer(Box::new(DenseLayer::new(50, 50)));
    ann.add_layer(Box::new(DenseLayer::new(50, 2)));
    ann.add_layer(Box::new(DenseLayer::new(2, 2)));

    let input = Array1::from_vec(vec![1.0, 0.0]);

    c.bench_function("NewNetowrk Activation", |b| b.iter(|| ann.activate(&input)));
}

criterion_group!(benches, criterion_benchmark_v2);
criterion_main!(benches);
