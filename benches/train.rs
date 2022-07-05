use sann::network::Ann;
use sann::layer::dense::DenseLayer;

use sann::activations::ActivationType;
use ndarray::Array1;

use criterion::{criterion_group, criterion_main, Criterion};


fn performance_compare(c: &mut Criterion){
    let mut group = c.benchmark_group("Training");
    let mut ann = Ann::new_empty();
    //[2, 3, 100, 50, 2, 2]
    ann.add_layer(Box::new(DenseLayer::new(2, 3)));
    ann.add_layer(Box::new(DenseLayer::new(3, 4)));
    ann.add_layer(Box::new(DenseLayer::new(4, 2)));
    ann.add_layer(Box::new(DenseLayer::new(2, 1)));
    
    ann.set_activations(&[
        ActivationType::Linear,
        ActivationType::ReLu,
        ActivationType::Sigmoid,
        ActivationType::ReLu,
    ]);

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
    let mut loss2 = 0.;
    group.bench_function("new", |b| 
        b.iter(|| 
            loss2 = ann.train(&input, &expected, 100, 0.1)
        )
    );

    println!("LOSS: {:?}", loss2);
    group.finish();
}
criterion_group!(benches, performance_compare);
criterion_main!(benches);
