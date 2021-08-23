use ndarray::array;
mod lib;
mod nn2;
use nn2::node::Neuron;
use nn2::layer;
#[test]
fn test_nd_neural() {
  let input1 = array![1.0, 0.0];
  let input3 = array![0.0, 0.0];

  let batch = vec![
      array![1.0, 0.0],
      array![0.0, 1.0],
      array![0.0, 0.0],
      array![1.0, 1.0]
  ];
  let y = vec![
      array![0.0, 1.0],
      array![0.0, 1.0],
      array![1.0, 0.0],
      array![0.0, 1.0]
  ];

  let mut nn2 = lib::neural_net::NeuralNet::from_format(&[2, 3, 2, 2]);
  
  nn2.train_batch(&batch, &y, 0.8, 10000);

  println!("Predict 1: {:?}", nn2.activate(&input1));
  println!("Predict 0: {:?}", nn2.activate(&input3));

}

fn main() {
    
    let node1 = nn2::node::Neuron::new();
    let node2 = nn2::node::Neuron::new();

    Neuron::project(&node1, &node2);
    Neuron::set_out(node1, 1.0);

    println!("Output val: {}", node2.borrow_mut().activate());

    let l1 = layer::Layer::new(3);
    let l2 = layer::Layer::new(1);

    l1.borrow_mut().set_state(vec![4.0, 4.0, 4.0]);
    l1.borrow_mut().project(&l2);
    l2.borrow_mut().activate();
    println!("{:?}", l2.borrow_mut().get_state());

    let mut net = nn2::network::Network::new(vec![2,2, 1]);
    println!("net {:?}", net.activate(vec![1.0, 1.0]));
    // Proximos objetivos: Printar progresso, printar loss, rever algoritimos
    // Criar rede com ativacoes diferentes
    // 
    // Ver como implementar outros tipos de rede (i.e recursiva, com memoria)
}
