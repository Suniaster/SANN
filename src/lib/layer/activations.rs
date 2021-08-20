use ndarray::Array1;

pub fn forward_relu(input: &Array1<f64>) -> Array1<f64>{
  return input.map( |x| if *x > 0.0 {*x} else {0.0} )
}