use sann::network;
use sann::*;
use ndarray::Array1;

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};

fn criterion_benchmark(c: &mut Criterion) {
    let mut xor_net = network::Network::new(&[2, 3, 100, 50, 50, 2, 2]);
    let input = &[1.0, 0.0];

    c.bench_function("OldNetwork Activation", |b| b.iter(|| xor_net.activate(input)));
}

fn criterion_benchmark_v2(c: &mut Criterion) {
    let mut ann = Ann::new();
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

fn performance_compare(c: &mut Criterion){
    const LOOP_COUNT:i32 = 1_000;

    let mut group = c.benchmark_group("Activation");
    let mut xor_net = network::Network::new(&[2, 3, 100, 50, 50, 2, 2]);
    let input = &[1.0, 0.0];

    group.bench_with_input(BenchmarkId::new("old", 6),&LOOP_COUNT,  |b, _| b.iter(|| xor_net.activate(input)));

    let mut ann = Ann::new();
    //[2, 3, 100, 50, 2, 2]
    ann.add_layer(Box::new(DenseLayer::new(2, 3)));
    ann.add_layer(Box::new(DenseLayer::new(3, 100)));
    ann.add_layer(Box::new(DenseLayer::new(100, 50)));
    ann.add_layer(Box::new(DenseLayer::new(50, 50)));
    ann.add_layer(Box::new(DenseLayer::new(50, 2)));
    ann.add_layer(Box::new(DenseLayer::new(2, 2)));

    let input = Array1::from_vec(vec![1.0, 0.0]);

    group.bench_with_input(BenchmarkId::new("new", 6),&LOOP_COUNT, |b,_| b.iter(|| ann.activate(&input)));
    group.finish();
}
criterion_group!(benches, criterion_benchmark, criterion_benchmark_v2, performance_compare);
criterion_main!(benches);
