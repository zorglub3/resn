use esn::ESN;
use esn::generator::{generate_esn_matrix_f64, generate_random_dense_f64, generate_simple_f64};
use esn::offline::*;
use std::collections::VecDeque;

fn make_esn(
    internal_size: usize,
    input_size: usize,
    output_size: usize,
    connections: usize,
    radius: f64,
) -> ESN<f64> {
    // let internal_weights = generate_esn_matrix_f64(internal_size, connections, radius);
    let internal_weights = generate_simple_f64(internal_size, radius);
    let input_weights = generate_random_dense_f64(internal_size, input_size);
    let output_weights = generate_random_dense_f64(output_size, internal_size);
    let feedback_weights = generate_random_dense_f64(internal_size, output_size);
    let input_output_weights = generate_random_dense_f64(output_size, input_size);

    ESN::new(
        0.5,
        |x| x.tanh(),
        internal_weights,
        input_weights,
        output_weights,
        feedback_weights,
        input_output_weights,
    )
}

fn read_data(data_size: usize, include_bias: bool) -> Vec<f64> {
    let mut s = String::new();
    std::io::stdin().read_line(&mut s).expect("read_line error");

    let mut parts = s.split_whitespace().map(|s| s.parse::<f64>());

    let mut res = Vec::new();

    // bias input
    res.push(1.);

    let data_size = if include_bias {
        data_size - 1
    } else {
        data_size
    };

    for _i in 0..data_size {
        res.push(parts.next().unwrap().unwrap());
    }

    res
}

fn main() {
    let iterations = 10000;
    let input_size = 2;
    let output_size = 1;
    let internal_size = 200;
    let connections = 10;
    let radius = 0.9;
    let print_output = false;
    let learning_rate = 0.001;
    let force_teacher = true;
    let use_lms = false;
    let offline_training = true;
    let offline_skip: usize = 100;
    let include_bias = true;

    let mut history = VecDeque::new();

    let mut training_data = TrainingData::new();

    eprintln!("Building model");
    let mut model = make_esn(
        internal_size,
        input_size,
        output_size,
        connections,
        radius,
    );

    let mut prev_output = vec![0.; output_size];

    eprintln!("Running and learning:");
    for it in 0..iterations {
        if it % 1000 == 0 {
            eprint!(".");
        }

        let data = read_data(input_size + output_size, include_bias);
        let input = &data[..input_size];
        let output = &data[input_size..];

        if force_teacher && it < iterations / 2 {
            model.update(input, Some(&prev_output));
        } else {
            model.update(input, None);
        }

        if offline_training {
            training_data.push_state(&model, output.clone().to_vec());
        }

        let current_rate = if !offline_training && it < iterations / 2 {
            learning_rate
        } else {
            0.
        };

        let err;
        if !offline_training && use_lms {
            err = model.learn_online(output, current_rate);
        } else {
            err = model.learn_online(output, 0.);
            model.rls(20, &mut history, output, 0.9, current_rate);
        }

        // model.intrinsic_plastic(0., 0.5, current_rate / 10.);

        prev_output.as_mut_slice().copy_from_slice(&output);

        if print_output {
            for s in model.get_output_state() {
                print!("{s}");
            }
            println!("");
        } else {
            println!("{err}");
        }
    }

    if offline_training {
        training_data.offline_train(offline_skip, &mut model.output_weights, &mut model.input_output_weights);
        training_data.test(&mut model);
    }

    eprintln!("");
    eprintln!("all done");
}
