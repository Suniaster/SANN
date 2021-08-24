mod lib;
mod nn2;

#[macro_use]
extern crate time_test;

fn main() {
    nn2::train();
    // lib::train();
    // Proximos objetivos: Printar progresso, printar loss, rever algoritimos
    // Criar rede com ativacoes diferentes
    //
    // Ver como implementar outros tipos de rede (i.e recursiva, com memoria)
}
