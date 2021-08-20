use ndarray::Array1;
use std::f64::consts;

pub fn forward_relu(input: &Array1<f64>) -> Array1<f64>{
  return input.map( |x| if *x > 0.0 {*x} else {0.0} )
}

pub fn foward_sigmoid(input: &Array1<f64>) -> Array1<f64>{
  return input.map( |x| -> f64 {
    return 1.0/(1.0 +f64::powf(consts::E, -1.0*(*x)))
  })
}