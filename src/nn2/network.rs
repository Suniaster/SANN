use super::helper::*;
use super::layer::*;
use super::super::lib::activations::*;


pub struct Network{
  input: Container<Layer>,
  output: Container<Layer>,
  hidden: Vec<Container<Layer>>,
}


impl Network {
  pub fn new(layers_format: Vec<u16>) -> Network {
    let mut net = Network {
      input: Layer::new(layers_format[0]), 
      output: Layer::new_activation(layers_format[ layers_format.len() - 1], ActivationType::Sigmoid), 
      hidden: vec![]
    };

    let mut last_layer = &net.input;
    for i in 1..(layers_format.len() - 1){
      let new_layer = Layer::new(layers_format[i]);
      last_layer.borrow_mut().project(&new_layer);
      net.hidden.push(new_layer);
      last_layer = &net.hidden.last().expect("");
    }

    last_layer.borrow_mut().project(&net.output);
    return net;
  }

  pub fn activate(&mut self, input: Vec<f64>) -> Vec<f64>{
    self.input.borrow_mut().set_state(input);
    
    for layer in self.hidden.iter() {
      layer.borrow_mut().activate()
    }

    self.output.borrow_mut().activate();
    return self.output.borrow_mut().get_state();
  }
}

