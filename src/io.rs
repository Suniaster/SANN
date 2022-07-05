use super::activations::ActivationType;

use std::fs;
use serde_json;

// https://doc.rust-lang.org/std/fs/fn.write.html
// https://stackoverflow.com/questions/64453932/how-to-use-serde-to-serialize-structs-containing-ndarray-fields
use ndarray::{Array2, Array1};

use serde::{ Serialize, Deserialize };


#[derive(Serialize, Deserialize)]
struct NetworkJsonInfo{
    layers: Vec<Array2<f64>>,
    biases: Vec<Array1<f64>>,
    activations: Vec<ActivationType>,
}

impl NetworkJsonInfo {
    // fn create_info(net: &Network) -> NetworkJsonInfo{
    //     let mut info = NetworkJsonInfo{
    //         layers: vec![], 
    //         activations: vec![],
    //         biases: vec![]
    //     };

    //     for layer in net.layers.iter() {
    //         info.layers.push(layer.weights.clone());
    //         info.activations.push(layer.get_act_type());
    //         info.biases.push(layer.biases.clone());
    //     }
    //     return info;
    // }

    // fn create_net(&self)-> Network{
    //     let mut format:Vec<u16> = vec![];
    //     // let mut types:Vec<ActivationType> = vec![];
    //     for i in 0..self.layers.len(){
    //         format.push(self.layers[i].shape()[0] as u16);
    //     }
        
    //     let mut new_net = Network::new(&format);
    //     new_net.format(&self.activations);

    //     for i in 0..new_net.layers.len(){
    //         new_net.layers[i].weights = self.layers[i].clone();
    //         new_net.layers[i].biases = self.biases[i].clone();
    //         new_net.layers[i].update_neurons_with_state();
    //     }

    //     return new_net;
    // }
}

// impl Neuron {
//     fn update_axon_weights(&mut self, ws: Array1<f64>){
//         for i in 0..self.inputs.len(){
//             self.inputs[i].borrow_mut().weight = ws[i];
//         }
//     }
// }

// impl Layer {
//     fn get_act_type(&self) -> ActivationType{
//         return self.neurons[0].borrow().activation.t.clone();
//     }
//     fn update_neurons_with_state(&mut self){
//         for i in 0..self.neurons.len(){
//             self.neurons[i].borrow_mut().bias = self.biases[i];
//             self.neurons[i].borrow_mut()
//                 .update_axon_weights(self.weights.row(i).to_owned());
//         }
//     }
// }

// pub fn save_net(net: &Network, f_name: &String){
//     let info = NetworkJsonInfo::create_info(net);
//     let serial = serde_json::to_string(&info).unwrap();
//     fs::write(f_name, serial).unwrap();
// }

// pub fn load_net(f_name: &String) -> Network {
//     let json_str:String = fs::read_to_string(f_name).unwrap();
//     let info: NetworkJsonInfo = serde_json::from_str(&json_str).unwrap();
//     return info.create_net();
// }