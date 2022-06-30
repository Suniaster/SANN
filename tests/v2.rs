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
#[test]
pub fn t2_test_layer(){
    let mut layer = DenseLayer::<2, 3>::new();
    layer.randomize();

    let layer_fmt = layer.format();
    assert_eq!(layer_fmt.0, 2);

    let input:SVector<f64, 2> = SVector::from_vec(vec![0.0, 0.0]);
    let output = layer.activate(&input.data.0[0]);
    let output:SVector<f64, 3> = SVector::from_vec(output.to_vec());
    
    assert_eq!(output.len(), 3);
    assert_eq!(output.sum(), 0.0);

    let input:SVector<f64, 2> = SVector::from_vec(vec![1.0, 1.0]);
    let output = layer.activate(&input.data.0[0] );
    let output:SVector<f64, 3> = SVector::from_vec(output.to_vec());
    assert_ne!(output.sum(), 0.0);
}