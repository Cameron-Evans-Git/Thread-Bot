// AI Generated Code

use ndarray::{Array2, ArrayView2};

/// Compute SSIM between two flattened grayscale images (Vec<u8>).
/// Assumes both images have the same dimensions (width × height).
pub fn ssim(
    img1: &[u8],
    img2: &[u8],
    resolution: usize,
    window_size: usize,
    k1: f64,
    k2: f64,
) -> f64 {
    // Convert Vec<u8> to 2D arrays (for sliding windows)
    let img1_array = Array2::from_shape_fn((resolution, resolution), |(y, x)| {
        img1[y * resolution + x] as f64
    });
    let img2_array = Array2::from_shape_fn((resolution, resolution), |(y, x)| {
        img2[y * resolution + x] as f64
    });

    let (c1, c2) = compute_ssim_constants(k1, k2);

    let mut total_ssim = 0.0;
    let mut window_count = 0;

    // Slide window over the images
    for y in 0..=(resolution - window_size) {
        for x in 0..=(resolution - window_size) {
            let window1 = img1_array.slice(ndarray::s![
                y..y + window_size,
                x..x + window_size
            ]);
            let window2 = img2_array.slice(ndarray::s![
                y..y + window_size,
                x..x + window_size
            ]);

            let ssim_val = compute_window_ssim(&window1, &window2, c1, c2);
            total_ssim += ssim_val;
            window_count += 1;
        }
    }

    // Average SSIM over all windows
    total_ssim / window_count as f64
}

/// Compute SSIM constants C1 and C2 (based on dynamic range)
fn compute_ssim_constants(k1: f64, k2: f64) -> (f64, f64) {
    let l = 255.0; // For 8-bit images (dynamic range = 255)
    let c1 = (k1 * l).powi(2);
    let c2 = (k2 * l).powi(2);
    (c1, c2)
}

/// Compute SSIM for a single window
fn compute_window_ssim(
    window1: &ArrayView2<f64>,
    window2: &ArrayView2<f64>,
    c1: f64,
    c2: f64,
) -> f64 {
    let mu1 = window1.mean().unwrap();
    let mu2 = window2.mean().unwrap();

    let var1 = window1.mapv(|x| (x - mu1).powi(2)).mean().unwrap();
    let var2 = window2.mapv(|x| (x - mu2).powi(2)).mean().unwrap();
    let covar = window1
        .iter()
        .zip(window2.iter())
        .map(|(&x1, &x2)| (x1 - mu1) * (x2 - mu2))
        .sum::<f64>()
        / (window1.len() as f64);

    // SSIM formula
    let numerator = (2.0 * mu1 * mu2 + c1) * (2.0 * covar + c2);
    let denominator = (mu1.powi(2) + mu2.powi(2) + c1) * (var1 + var2 + c2);
    numerator / denominator
}

