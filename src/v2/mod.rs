#![feature(const_generics_defaults)]

use nalgebra::{SVector, SMatrix, DVector, Matrix, ArrayStorage, Const, DMatrix};
use rand::Rng;

pub struct Neuron<const D: usize>{
    pub weights: SVector<f64, D>,
    pub bias: f64,
}

impl<const D:usize> Neuron<D> {
    pub fn new(bias: f64) -> Neuron<D> {
        let zeros = vec![0.0; D];
        let weights = SVector::from_vec(zeros);
        Neuron {
            weights,
            bias: bias,
        }
    }

    pub fn randomize(&mut self) {
        let mut rng = rand::thread_rng();
        self.weights = SVector::from_fn(|_,_| rng.gen::<f64>());
    }

    pub fn normalize(&mut self) {
        let ones = vec![1.; D];
        self.weights = SVector::from_vec(ones);
        self.bias = 0.0;
    }

    pub fn activate(&self, inputs: &SVector<f64, D>) -> f64 {
        inputs.dot(&self.weights) + self.bias
    }
}

trait LayerFormat {
    const F: usize;
}  

pub trait NetLayer{
    fn activate(&self, inputs: Vec<f64>) -> Vec<f64>;
    fn chain_activation(&self, inputs: Vec<f64>) -> Vec<f64>;
    fn project(&mut self, next: Box<dyn NetLayer>);
    fn format(&self) -> (usize, usize);
}

pub struct DenseLayer<const IN_FMT: usize, const OUT_FMT: usize> {
    neurons: Vec<Neuron<IN_FMT>>,
    weights_mat: SMatrix<f64, OUT_FMT, IN_FMT>,
    bias_vec: SVector<f64, OUT_FMT>,
    next: Option<Box<dyn NetLayer>>,
}

impl<const IN_FMT:usize, const OUT_FMT:usize> DenseLayer<IN_FMT, OUT_FMT>{
    pub fn new() -> DenseLayer<IN_FMT, OUT_FMT> {
        let mut neurons = Vec::new();
        for _ in 0..OUT_FMT {
            neurons.push(Neuron::new(0.0));
        }
        DenseLayer {
            neurons,
            weights_mat: SMatrix::<f64, OUT_FMT, IN_FMT>::zeros(),
            bias_vec: SVector::<f64, OUT_FMT>::zeros(),
            next: None,
        }
    }

    pub fn randomize(&mut self) {
        for n in &mut self.neurons {
            n.randomize();
        }
        self.update_weights_mat();
    }

    pub fn normalize(&mut self) {
        for n in &mut self.neurons {
            n.normalize();
        }
        self.update_weights_mat();
    }

    pub fn update_weights_mat(&mut self) {
        for (i, n) in self.neurons.iter().enumerate() {
            for (j, w) in n.weights.iter().enumerate() {
                self.weights_mat[(i, j)] = *w;
            }
        }
    }

}

impl<const I:usize, const O:usize> NetLayer for DenseLayer<I,O> {
    fn activate(&self, inputs: Vec<f64>) -> Vec<f64> {
        let input_vec:SVector<f64, I> = SVector::from_vec(inputs);
        let out: SVector<f64, O> = self.weights_mat * input_vec + self.bias_vec;
        out.data.0[0].to_vec()
    }

    fn chain_activation(&self, inputs: Vec<f64>) -> Vec<f64> {
        let activation_result = self.activate(inputs);
        if let Some(ref next) = self.next {
            next.chain_activation(activation_result)
        } else {
            activation_result
        }
    }

    fn project(&mut self, next: Box<dyn NetLayer>) {
        self.next = Some(next);
    }

    fn format(&self) -> (usize, usize) {
        (I, O)
    }
}


/********** Network *********/

// pub struct ArtificialNetwork {
//     layers: Vec<Box<dyn NetLayer>>
// }