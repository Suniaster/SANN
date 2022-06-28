use sann::v2::*;
use nalgebra::SVector;

#[test]
pub fn t1_test_neuron(){
    let n = Neuron::<4>::new(0.0);
    assert_eq!(n.activate(&SVector::from_vec(vec![0.0, 0.0, 0.0, 0.0])), 0.0);

    let mut n = Neuron::<4>::new(0.0);
    n.randomize();
    assert_eq!(n.weights.len(), 4);

    let input:SVector<f64, 4> = SVector::from_vec(vec![0.0, 0.0, 0.0, 0.0]);
    let output = n.activate(&input);
    assert_eq!(output, 0.0);

    let input:SVector<f64, 4> = SVector::from_vec(vec![1.0, 1.0, 1.0, 1.0]);
    let output = n.activate(&input);
    assert_ne!(output, 0.0);
}