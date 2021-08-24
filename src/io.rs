use super::layer::Layer;
use super::network::Network;
use std::fs::File;
use std::io::prelude::*;

extern crate rustc_serialize as serialize;

impl Layer {
    pub fn get_weights(&self) -> String{
        let weigths_s = self.weights.to_string();
        println!("{}", weigths_s);
        return weigths_s;
    }
}

impl Network {
    pub fn save(&self, f_name: &String){
        let file = File::create(f_name);
        println!("{:?}", file);
        for layer in self.layers.iter() {
            layer.get_weights();
        }
    }
}
