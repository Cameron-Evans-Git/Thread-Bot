pub fn mse_weighted(
    target: &[u8],
    current: &[u8],
    weights: &[f64]
) -> f64 {
    let mut sum: f64 = 0.0;
    for i in 0..target.len() {
        let diff = target[i] as f64 - current[i] as f64;
        sum += diff * diff * weights[i];
    }
    sum
}

pub fn mse_subset_weighted(
    target: &[u8],
    current: &[u8],
    set: &[usize],
    weights: &[f64]
) -> f64 {
    let mut sum: f64 = 0.0;
    for i in set {
        let add = target[*i] as f64 - current[*i] as f64;
        sum += add * add * weights[*i];
    }
    sum
}