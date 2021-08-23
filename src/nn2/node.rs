use super::super::lib::activations::*;
use rand;
use super::helper::*;


pub struct Axon{
  weight: f64,
  value: f64,
  src: Container<Neuron>,
  dest: Container<Neuron>
}

pub struct Neuron{
  outputs: Vec<Container<Axon>>,
  inputs: Vec<Container<Axon>>,
  activation: Activation,
  bias: f64,
  pub output_val: f64
}

impl Axon {
  fn update(&mut self){
    self.value = self.src.borrow().output_val * self.weight;
  }
}

impl Neuron {

  pub fn new() -> Container<Neuron>{
      return Neuron::new_activation(ActivationType::ReLu);
  }

  pub fn new_activation(_type: ActivationType) -> Container<Neuron>{
    return new_container(Neuron {
      bias: (rand::random::<f64>() * 2.0) - 1.0,
      output_val: 0.0,
      activation: Activation::create(_type),
      inputs: vec![],
      outputs: vec![]
    })
  }

  pub fn project(src: &Container<Neuron>, dest: &Container<Neuron>){
    let new_axon = new_container(Axon {
      weight: (rand::random::<f64>() * 2.0) - 1.0,
      src:  src.clone(),
      dest: dest.clone(),
      value: 0.0
    });

    src.borrow_mut().outputs.push(new_axon.clone());
    dest.borrow_mut().inputs.push(new_axon.clone());
  }

  pub fn change_activation(&mut self, _type: ActivationType){
    self.activation = Activation::create(_type);
  }

  pub fn activate(&mut self) -> f64{
    self.output_val = 0.0;
    for axon in self.inputs.iter(){
      axon.borrow_mut().update();
      self.output_val += axon.borrow().value;
    }
    self.output_val = (self.activation.f)(&self.output_val);
    return self.output_val;
  }

  pub fn set_out(neuron: Container<Neuron>, val:f64){
    neuron.borrow_mut().output_val = val;
  }

} 