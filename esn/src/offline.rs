//! Offline training
//!

use crate::ESN;
use crate::mat::DenseRowMatrix;
use nalgebra::DMatrix;

pub struct TrainingRecord<T> {
    pub input: Vec<T>,
    pub target: Vec<T>,
    pub state: Vec<T>,
}

impl<T> TrainingRecord<T> 
    where T: Clone
{
    pub fn from_model(esn: &ESN<T>, target: Vec<T>) -> Self {
        Self {
            input: esn.input_state.clone(),
            target,
            state: esn.internal_state.clone(),
        }
    }
}

pub struct TrainingData<T>(pub Vec<TrainingRecord<T>>);

impl<T> TrainingData<T> 
    where T: Clone
{
    pub fn new() -> Self {
        Self(vec![])
    }

    pub fn push_state(&mut self, esn: &ESN<T>, target: Vec<T>) {
        // TODO: In debug mode, check sizes match the records we already have 
        self.0.push(TrainingRecord::from_model(esn, target));
    }

    pub fn record_count(&self) -> usize {
        self.0.len()
    }

    pub fn input_size(&self) -> usize {
        self.0[0].input.len()
    }

    pub fn target_size(&self) -> usize {
        self.0[0].target.len()
    }

    pub fn state_size(&self) -> usize {
        self.0[0].state.len()
    }
}

impl TrainingData<f64> {
    pub fn offline_train(&self, skip: usize, internal_weights: &mut DenseRowMatrix<f64>, input_weights: &mut DenseRowMatrix<f64>) {
        eprintln!("Offline training...");
        for target_index in 0 .. self.target_size() {
            let mut x = DMatrix::zeros(self.record_count(), self.state_size() + self.input_size());
            let mut y = DMatrix::zeros(self.record_count(), 1);

            for record_index in skip .. self.0.len() {
                let record = &self.0[record_index];

                for state_index in 0..self.state_size() {
                    x[(record_index, state_index)] = record.state[state_index];
                }

                for input_index in 0..self.input_size() {
                    x[(record_index, self.state_size() + input_index)] = record.input[input_index];
                }

                y[(record_index, 0)] = record.target[target_index];
            }

            let w = (&x.transpose() * &x).try_inverse().unwrap() * &x.transpose() * y;

            // eprintln!("weights are now {0:?}", w);

            for i in 0..self.state_size() {
                *internal_weights.get_mut(target_index, i) = w[(i, 0)];
            }

            for i in 0..self.input_size() {
                *input_weights.get_mut(target_index, i) = w[(i + self.state_size(), 0)];
            }
        }
    }

    pub fn test(&self, esn: &mut ESN<f64>) {
        esn.reset();

        for record in &self.0 {
            esn.update(&record.input, None);
            let err = esn.error_square_sum(&record.target);

            println!("{err}");
        }
    }
}
