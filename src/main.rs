use ndarray::array;
mod lib;
use lib::*;

fn main() {
    let input1 = array![1.0, 0.0];

    let mut nn2 = lib::neural_net::NeuralNet::from_format(&[2, 2, 1]);

    println!("{:?}", nn2.activate(&input1));
    println!("###########");
    
    for i in 0..100 {
        println!("Iteration {} ###########", i);

        let output = nn2.train(&input1, &array![1.0], 0.15);
        println!("Output: {:?}", output);
    }

    // Proximos objetivos: Printar progresso, printar loss, rever algoritimos
    // Ver como implementar outros tipos de rede (i.e recursiva, com memoria)
}
