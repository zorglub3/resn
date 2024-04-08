use crate::mat;
use nalgebra::{DMatrix, Normed};
use rand::rngs::ThreadRng;
use rand::Rng;
use rand::thread_rng;
use rand::prelude::SliceRandom;

pub fn generate_dense<T, F>(rows: usize, cols: usize, mut f: F) -> mat::DenseRowMatrix<T> 
    where F: FnMut() -> T,
{
    let mut values = vec![];

    for _r in 0..rows {
        let mut row = vec![];

        for _c in 0..cols {
            row.push(f());
        }

        values.push(row);
    }

    mat::DenseRowMatrix {
        rows,
        cols,
        values,
    }
}

pub fn generate_random_dense_f64(rows: usize, cols: usize) -> mat::DenseRowMatrix<f64> {
    let mut rng = thread_rng();

    generate_dense(rows, cols, || random_weight_f64(&mut rng))
}

fn random_indices(rng: &mut ThreadRng, size: usize, connections: usize) -> Vec<usize> {
    let mut all_indices: Vec<usize> = (0..size).collect();
    all_indices.shuffle(rng);
    all_indices[..connections].to_vec()
}

fn random_weight_f64(rng: &mut ThreadRng) -> f64 {
    /*
    if rng.gen::<f32>() < 0.5 {
        0.4
    } else {
        -0.4
    }
    */
    rng.gen_range(-1. .. 1.)
}

pub fn generate_sparse_f64(size: usize, sparsity: f64, v: f64) -> mat::SparseRowMatrix<f64> {
    let mut rng = thread_rng();
    let mut col_indices: Vec<Vec<usize>> = vec![];
    let mut col_values: Vec<Vec<f64>> = vec![];

    for _row in 0 .. size {
        let mut row_indices = vec![];
        let mut row_values = vec![];

        for col in 0 .. size {
            if rng.gen::<f64>() >= sparsity {
                let value = if rng.gen::<bool>() {
                    v
                } else {
                    -v
                };

                row_indices.push(col);
                row_values.push(value);
            }
        }

        col_indices.push(row_indices);
        col_values.push(row_values);
    }

    mat::SparseRowMatrix {
        rows: size,
        cols: size,
        col_indices,
        col_values,
    }
}

pub fn generate_simple_f64(size: usize, _radius: f64) -> mat::SparseRowMatrix<f64> {
    let mut rng = thread_rng();
    let mut col_indices: Vec<Vec<usize>> = vec![];
    let mut col_values: Vec<Vec<f64>> = vec![];

    for _row in 0 .. size {
        let mut row_indices = vec![];
        let mut row_values = vec![];

        for col in 0 .. size {
            let x = rng.gen::<f64>();
            if x >= 0.85 {
                row_indices.push(col);
                let y = rng.gen::<f64>();
                if y > 0.5 {
                    row_values.push(0.7);
                } else {
                    row_values.push(-0.7);
                }
            }
        }

        col_indices.push(row_indices);
        col_values.push(row_values);
    }

    mat::SparseRowMatrix {
        rows: size,
        cols: size,
        col_indices,
        col_values,
    }
}

pub fn generate_esn_matrix_f64(size: usize, connections: usize, radius: f64) -> mat::SparseRowMatrix<f64> {
    let mut rng = thread_rng();
    let mut w: DMatrix<f64> = DMatrix::zeros(size, size);
    let mut col_indices: Vec<Vec<usize>> = vec![];

    for row in 0..size {
        let mut row_indices = vec![];
        for col in random_indices(&mut rng, size, connections) {
            w[(row, col)] = random_weight_f64(&mut rng);
            row_indices.push(col);
        }
        col_indices.push(row_indices);
    }

    let es = w.complex_eigenvalues().map(|x| x.norm());
    let spectral_radius = es.iter().max_by(|x, y| x.total_cmp(y)).unwrap();

    eprintln!("spectral radius mul: {0}", radius / spectral_radius);
    let x = w * (radius / spectral_radius);

    let mut col_values = vec![];

    for row in 0..size {
        let mut row_values = vec![];

        for col in &col_indices[row] {
            row_values.push(x[(row, *col)]);
        }

        col_values.push(row_values);
    }

    mat::SparseRowMatrix {
        rows: size,
        cols: size,
        col_indices,
        col_values,
    }
}
