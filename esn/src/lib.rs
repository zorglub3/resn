use std::ops::{AddAssign, Mul, Sub, Add};
use std::iter::Sum;
use std::collections::VecDeque;
use mat::{DenseRowMatrix, SparseRowMatrix};

pub mod mat;
pub mod generator;
pub mod offline;

pub trait ZeroOne where Self: Sized {
    fn zero() -> Self;
    fn one() -> Self;

    fn fill_zero(v: &mut [Self]) {
        for i in 0..v.len() {
            v[i] = Self::zero();
        }
    }
}

impl ZeroOne for f64 {
    fn zero() -> f64 {
        0.
    }

    fn one() -> f64 {
        1.
    }
}

pub struct ESN<T> {
    alpha: T,
    activation: fn(T) -> T,
    state_a: Vec<T>,
    state_b: Vec<T>,
    pub internal_state: Vec<T>,
    pub input_state: Vec<T>,
    pub output_state: Vec<T>,
    pub internal_weights: SparseRowMatrix<T>,
    pub input_weights: DenseRowMatrix<T>,
    pub feedback_weights: DenseRowMatrix<T>,
    pub output_weights: DenseRowMatrix<T>,
    pub input_output_weights: DenseRowMatrix<T>,
}

impl<T> ESN<T>
    where T: Mul<Output = T> + Add<Output = T> + AddAssign + Copy + ZeroOne + Sum + Sub<Output = T>
{
    pub fn new(
        alpha: T,
        activation: fn(T) -> T,
        internal_weights: SparseRowMatrix<T>,
        input_weights: DenseRowMatrix<T>,
        output_weights: DenseRowMatrix<T>,
        feedback_weights: DenseRowMatrix<T>,
        input_output_weights: DenseRowMatrix<T>,
    ) -> Self {
        let internal_state_size = internal_weights.rows;
        let input_size = input_weights.cols;
        let output_size = output_weights.rows;

        Self {
            alpha,
            activation,
            state_a: vec![T::one(); internal_state_size],
            state_b: vec![T::zero(); internal_state_size],
            internal_state: vec![T::zero(); internal_state_size],
            input_state: vec![T::zero(); input_size],
            output_state: vec![T::zero(); output_size],
            internal_weights,
            input_weights,
            feedback_weights,
            output_weights,
            input_output_weights,
        }
    }

    pub fn reset(&mut self) {
        T::fill_zero(&mut self.internal_state);
        T::fill_zero(&mut self.input_state);
        T::fill_zero(&mut self.output_state);

        self.state_a.fill(T::one());
        T::fill_zero(&mut self.state_b);
    }

    pub fn get_output_state(&self) -> &Vec<T> {
        &self.output_state
    }

    pub fn update(&mut self, input: &[T], teacher: Option<&[T]>) {
        let mut local_state = vec![T::zero(); self.internal_state.len()];

        self.internal_weights.mul_in_place(&self.internal_state, &mut local_state);
        self.input_weights.mul_in_place(input, &mut local_state);

        match teacher {
            Some(v) => self.feedback_weights.mul_in_place(v, &mut local_state),
            None => self.feedback_weights.mul_in_place(&self.output_state, &mut local_state),
        }

        self.map_activation(&mut local_state);

        Self::lin_interp(&mut self.internal_state, T::one() - self.alpha, &local_state, self.alpha);

        T::fill_zero(&mut self.output_state);
        self.output_weights.mul_in_place(&self.internal_state, &mut self.output_state);
        self.input_output_weights.mul_in_place(input, &mut self.output_state);

        T::fill_zero(&mut self.input_state);
        self.input_state.as_mut_slice().copy_from_slice(input);
    }

    fn map_activation(&self, s: &mut [T]) {
        for index in 0..s.len() {
            let u = s[index];
            s[index] = (self.activation)(self.state_a[index] * u + self.state_b[index]);
        }
    }

    fn lin_interp(target: &mut [T], a: T, new_state: &[T], b: T) {
        for index in 0..target.len() {
            target[index] = target[index] * a + new_state[index] * b;
        }
    }
}

impl ESN<f64> {
    pub fn error_square_sum(&self, target: &[f64]) -> f64 {
        let mut err_sum = 0.0_f64;

        for i in 0 .. target.len() {
            let diff = self.output_state[i] - target[i];
            err_sum += diff * diff;
        }

        err_sum
    }
    
    // least mean squares
    pub fn learn_online(&mut self, target: &[f64], rate: f64) -> f64 {
        debug_assert_eq!(target.len(), self.output_state.len());
        debug_assert!(rate >= 0.);

        let mut err_sum = 0.0_f64;

        for i in 0..target.len() {
            let diff = self.output_state[i] - target[i];
            err_sum += diff * diff;

            let effective_rate = rate / (self.input_state.len() + self.internal_state.len()) as f64;

            for j in 0..self.input_state.len() {
                *self.input_output_weights.get_mut(i, j) -= effective_rate * diff * self.input_state[j];
            }

            for j in 0..self.internal_state.len() {
                *self.output_weights.get_mut(i, j) -= effective_rate * diff * self.internal_state[j];
            }
        }

        err_sum
    }

    // recursive least squares
    pub fn rls(&mut self, n: usize, history: &mut VecDeque<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)>, target: &[f64], lambda: f64, rate: f64) {
        debug_assert_eq!(target.len(), self.output_state.len());
        debug_assert!(lambda >= 0. && lambda <= 1.);
        debug_assert!(rate >= 0. && rate <= 1.);

        if history.len() >= n {
            history.pop_back();
        }

        debug_assert!(history.len() < n);

        history.push_front((Vec::from(target), self.output_state.clone(), self.internal_state.clone(), self.input_state.clone()));

        let mut l = 1.;

        for (tgt, out, internal, inp) in history {
            for i in 0..tgt.len() {
                let diff = out[i] - tgt[i];

                for j in 0..self.input_state.len() {
                    *self.input_output_weights.get_mut(i, j) -= rate * l * diff * inp[j];
                }

                for j in 0..self.internal_state.len() {
                    *self.output_weights.get_mut(i, j) -= rate * l * diff * internal[j];
                }
            }

            l *= lambda;
        }
    }

    pub fn intrinsic_plastic(&mut self, u: f64, var: f64, rate: f64) {
        let var2 = var * var;

        for index in 0..self.internal_state.len() {
            let y = self.internal_state[index];
            let a = self.state_a[index];
            let delta_b = -rate * (u / var2 + y * (2. * var2 + 1. - y * y + u * y) / var2);
            let delta_a = rate / a + delta_b;

            self.state_a[index] += delta_a;
            self.state_b[index] += delta_b;
        }
    }
}

