use std::ops::{Mul, AddAssign, MulAssign};
use std::iter::Sum;

pub fn scale_in_place<T>(scl: T, c: &mut [T])
    where T: MulAssign + Copy
{
    for v in c {
        *v *= scl;
    }
}

pub fn dense_dot_product<T>(a: &[T], b: &[T], acc: &mut T)
    where T: Mul<Output = T> + AddAssign + Sum + Copy
{
    debug_assert_eq!(a.len(), b.len());

    let result = a.iter().zip(b.iter()).map(|(x, y)| *x * *y).sum();
    *acc += result;
}

pub fn sparse_dense_dot_product<T>(cols: &[usize], values: &[T], b: &[T], acc: &mut T)
    where T: Mul<Output = T> + AddAssign + Copy
{
    debug_assert_eq!(cols.len(), values.len());
    // debug_assert!(cols.iter().max().unwrap() < &b.len());

    for (index, value) in cols.iter().zip(values.iter()) {
        *acc += *value * b[*index];
    }
}

pub struct SparseRowMatrix<T> {
    pub rows: usize,
    pub cols: usize,
    pub col_indices: Vec<Vec<usize>>,
    pub col_values: Vec<Vec<T>>,
}

impl<T> SparseRowMatrix<T>
    where T: Mul<Output = T> + AddAssign + Copy
{
    pub fn mul_in_place(&self, c: &[T], tgt: &mut [T]) {
        debug_assert_eq!(self.cols, c.len());
        debug_assert_eq!(self.rows, tgt.len());

        for r in 0..self.rows {
            sparse_dense_dot_product(&self.col_indices[r], &self.col_values[r], c, &mut tgt[r]);
        }
    }
}

pub struct DenseRowMatrix<T> {
    pub rows: usize,
    pub cols: usize,
    pub values: Vec<Vec<T>>,
}

impl<T> DenseRowMatrix<T>
    where T: Mul<Output = T> + AddAssign + Copy + Sum
{
    pub fn new(rows: usize, cols: usize, v: T) -> Self {
        Self {
            rows,
            cols,
            values: vec![vec![v; cols]; rows],
        }
    }

    pub fn mul_in_place(&self, c: &[T], tgt: &mut [T]) {
        debug_assert_eq!(self.cols, c.len());
        debug_assert_eq!(self.rows, tgt.len());

        for r in 0..self.rows {
            dense_dot_product(&self.values[r], c, &mut tgt[r]);
        }
    }

    pub fn get_mut(&mut self, r: usize, c: usize) -> &mut T {
        &mut self.values[r][c]
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_scale_in_place() {
        let mut v = vec![1, 2, 3];

        scale_in_place(-2, &mut v);

        assert_eq!(v[0], -2);
        assert_eq!(v[1], -4);
        assert_eq!(v[2], -6);
    }

    #[test]
    fn test_dense_dot_product() {
        let v = vec![-1, 2, 5];
        let u = vec![3, 4, 3];
        let mut acc = 3;

        dense_dot_product(&v, &u, &mut acc);

        assert_eq!(acc, 3 - 3 + 8 + 15);

        dense_dot_product(&v, &u, &mut acc);

        assert_eq!(acc, 23 + 20);
    }
    
    #[test]
    fn test_sparse_dense_dot_product() {
        let cols = vec![0, 1, 3];
        let vals = vec![3, 4, 5];
        let v    = vec![2, 3, 4, 5, 6, 7];
        let mut acc = 5;

        sparse_dense_dot_product(&cols, &vals, &v, &mut acc);

        assert_eq!(acc, 5 + 2 * 3 + 3 * 4 + 5 * 5);
    }
}
