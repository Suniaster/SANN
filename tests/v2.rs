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

    let input = vec![0.0, 0.0];
    let output = layer.activate(input);
    let output:SVector<f64, 3> = SVector::from_vec(output.to_vec());

    assert_eq!(output.len(), 3);
    assert_eq!(output.sum(), 0.0);

    let input = vec![1.0, 1.0];
    let output = layer.activate(input );
    let output:SVector<f64, 3> = SVector::from_vec(output.to_vec());
    assert_ne!(output.sum(), 0.0);
}

#[test]
pub fn test_network(){
    let mut network = ArtificialNetwork::new();

    let mut layer_1 = DenseLayer::<2, 3>::new();
    layer_1.normalize();
    network.add_layer(Box::new(layer_1));

    let mut layer_2 = DenseLayer::<3, 4>::new();
    layer_2.normalize();
    network.add_layer(Box::new(layer_2));

    let input = vec![1.0, 1.0];
    let output = network.activate(input);
    assert_eq!(output, vec![6., 6., 6., 6.]);
}


#[test]
#[should_panic]
pub fn test_error_projection(){
    let layer_1 = DenseLayer::<2, 3>::new();
    let layer_2 = DenseLayer::<4, 3>::new();

    let mut network = ArtificialNetwork::new();
    network.add_layer(Box::new(layer_1));
    network.add_layer(Box::new(layer_2));
}