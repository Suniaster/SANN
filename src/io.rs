use super::activations::ActivationType;
use super::layer::*;
use super::network;

use std::fs;
use serde_json;

// https://doc.rust-lang.org/std/fs/fn.write.html
// https://stackoverflow.com/questions/64453932/how-to-use-serde-to-serialize-structs-containing-ndarray-fields
use ndarray::{Array2, Array1};

use serde::{ Serialize, Deserialize };


#[derive(Serialize, Deserialize)]
struct NetworkJsonInfo{
    weights: Vec<Array2<f64>>,
    biases: Vec<Array1<f64>>,
    activations: Vec<ActivationType>,
    types: Vec<String>,
    formats: Vec<(usize, usize)>,
}

impl NetworkJsonInfo {
    fn create_layer_from_type(layer_type: &String, format: (usize, usize)) -> Box<dyn NetLayer> {
        match layer_type.as_str() {
            "Dense" => dense::DenseLayer::from_format(format),
            "Input" => dense::DenseLayer::from_format(format),
            "Output" => dense::DenseLayer::from_format(format),
            _ => panic!("Error: Unknown layer type: {}", layer_type),
        }
    }


    fn create_info(net: &network::Ann) -> NetworkJsonInfo{
        let mut info = NetworkJsonInfo{
            weights: vec![],
            biases: vec![],
            activations: vec![],
            types: vec![],
            formats: vec![],
        };

        for layer in net.layers.iter() {
            info.weights.push(layer.get_weights().clone());
            info.biases.push(layer.get_biases().clone());
            info.activations.push(layer.get_activation().t.clone());
            info.types.push(layer.get_type_name());
            info.formats.push(layer.get_format());
        }
        return info;
    }

    fn create_net(&self)-> network::Ann{
        let mut new_net = network::Ann::new();
        for i in 0..self.formats.len(){
            new_net.add_layer(NetworkJsonInfo::create_layer_from_type(&self.types[i], self.formats[i]));
            new_net.layers[i].set_weights(self.weights[i].clone());
            new_net.layers[i].set_biases(self.biases[i].clone());
            new_net.layers[i].set_activation(self.activations[i].clone());
        }
        return new_net;
    }
}

pub fn save_net(net: &network::Ann, f_name: &String){
    let info = NetworkJsonInfo::create_info(net);
    let serial = serde_json::to_string(&info).unwrap();
    fs::write(f_name, serial).unwrap();
}

pub fn load_net(f_name: &String) -> network::Ann {
    let json_str:String = fs::read_to_string(f_name).unwrap();
    let info: NetworkJsonInfo = serde_json::from_str(&json_str).unwrap();
    return info.create_net();
}