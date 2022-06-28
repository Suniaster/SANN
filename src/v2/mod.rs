use nalgebra::{SVector, SMatrix, DMatrix};
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

trait LayerActivation<const IN_D: usize, const OUT_D: usize> {
    fn activate(&self, inputs: &SVector<f64, IN_D>) -> &SVector<f64, IN_D>;
}

struct DefaultLayer<const IN_FMT: usize, const OUT_FMT: usize> {
    pub neurons: Vec<Neuron<IN_FMT>>,
    pub last_activation: SVector<f64, OUT_FMT>,
}

impl<const IN_FMT:usize, const OUT_FMT:usize> DefaultLayer<IN_FMT, OUT_FMT>{
    pub fn new(size:usize) -> DefaultLayer<IN_FMT, OUT_FMT> {
        let mut neurons = Vec::new();
        for _ in 0..size {
            neurons.push(Neuron::new(0.0));
        }
        DefaultLayer {
            neurons,
            last_activation: SVector::zeros(),
        }
    }

    pub fn get_layer_mat(&self) -> SMatrix<f64, IN_FMT, OUT_FMT> {
        let mut layer_mat: SMatrix<f64, IN_FMT, OUT_FMT> = SMatrix::zeros();
        for neuron in 0..OUT_FMT {
            for w in 0..IN_FMT {
                layer_mat[(neuron, w)] = self.neurons[neuron].weights[w];
            }
        }
        layer_mat
    }
}

impl<const I:usize, const O:usize> LayerActivation<I,O> for DefaultLayer<I,O> {
    
    fn activate(&self, inputs: &SVector<f64, I>) -> &SVector<f64, I> {
        for neuron in 0..O {
            outputs[neuron] = self.neurons[neuron].activate(inputs);
        }
        &outputs
    }

}