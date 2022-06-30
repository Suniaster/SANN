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

    pub fn activate(&self, inputs: &SVector<f64, D>) -> f64 {
        inputs.dot(&self.weights) + self.bias
    }
}

type GenerictMat<const D1:usize, const D2:usize> = Matrix<f64, Const<D1>, Const<D2>, ArrayStorage<f64, 2, 2>>;

trait LayerFormat {
    const F: usize;
}  

pub trait NetLayer<const I: usize, const O:usize>{
    fn activate(&self, inputs: &[f64;I]) -> [f64; O];
}

pub struct DenseLayer<const IN_FMT: usize, const OUT_FMT: usize> {
    neurons: Vec<Neuron<IN_FMT>>,
    weights_mat: SMatrix<f64, OUT_FMT, IN_FMT>,
    bias_vec: SVector<f64, OUT_FMT>,
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
            bias_vec: SVector::<f64, OUT_FMT>::zeros()
        }
    }

    pub fn randomize(&mut self) {
        for n in &mut self.neurons {
            n.randomize();
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

    pub fn format(&self) -> (usize, usize) {
        (IN_FMT, OUT_FMT)
    }
}

impl<const I:usize, const O:usize> NetLayer<I, O> for DenseLayer<I,O> {
    fn activate(&self, inputs: &[f64; I]) -> [f64; O] {
        let input_vec = SVector::from_vec(inputs.to_vec());
        let out = self.weights_mat * input_vec + self.bias_vec;
        out.data.0[0]
    }
}


/********** Network *********/

// pub struct ArtificialNetwork {
//     layers: Vec<Box<dyn NetLayer>>
// }