use nalgebra::{SVector, SMatrix};
use rand::Rng;
use super::activations::{Activation, ActivationType};

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
    fn format(&self) -> (usize, usize);
}

pub struct DenseLayer<const IN_FMT: usize, const OUT_FMT: usize> {
    neurons: Vec<Neuron<IN_FMT>>,
    weights_mat: SMatrix<f64, OUT_FMT, IN_FMT>,
    bias_vec: SVector<f64, OUT_FMT>,

    activation: Activation
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
            activation: Activation::create(ActivationType::Default)
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
        let out:Vec<f64> = out.data.0[0].to_vec();
        out.iter().map(|o| (self.activation.f)(o)).collect()
    }

    fn format(&self) -> (usize, usize) {
        (I, O)
    }
}


/********** Network *********/

pub struct ArtificialNetwork {
    layers: Vec<Box<dyn NetLayer>>
}

impl ArtificialNetwork {
    pub fn new() -> ArtificialNetwork {
        ArtificialNetwork {
            layers: Vec::new(),
        }
    }

    pub fn add_layer(&mut self, layer: Box<dyn NetLayer>) -> &mut Self {
        self.verify_new_layer(&layer);
        self.layers.push(layer);
        self
    }

    
    pub fn activate(&self, inputs: Vec<f64>) -> Vec<f64> {
        let mut inputs = inputs;
        for layer in &self.layers {
            inputs = layer.activate(inputs);
        }
        inputs
    }

    fn verify_new_layer(&self, new_layer: &Box<dyn NetLayer>){
        let layer_len = self.layers.len();
        if layer_len > 0 {
            let last_layer_format = self.layers[layer_len - 1].format();
            if last_layer_format.1 != new_layer.format().0 {
                panic!("Layer {:?} cannot project to layer {:?}", last_layer_format, new_layer.format().0);
            }
        }
    }
}