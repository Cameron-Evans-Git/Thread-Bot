use statrs::distribution::{Normal, ContinuousCDF};

/// Computes the integral (probability) from `a` to `b` under a normal distribution PDF.
fn integral_normal_pdf(a: f64, b: f64, mean: f64, sigma: f64) -> f64 {
    let normal = Normal::new(mean, sigma).expect("Invalid parameters");
    normal.cdf(b) - normal.cdf(a)
}

fn main() {
    let mean = 0.0;     // Mean (μ) of the normal distribution
    let sigma = 1.0;    // Standard deviation (σ)

    // Test cases: (a, b, expected_probability)
    let test_cases = [
        (0.0, 2.0, 0.682689492137),    // ±1σ (68.27%)
        (-2.0, 2.0, 0.954499736104),    // ±2σ (95.45%)
        (-3.0, 3.0, 0.997300203937),    // ±3σ (99.73%)
        (0.0, f64::INFINITY, 0.5),      // Half of the distribution
        (f64::NEG_INFINITY, 0.0, 0.5),  // Other half
    ];

    println!("Testing Integral of Normal PDF (μ = {}, σ = {})", mean, sigma);
    println!("{:<12} {:<12} {:<18} {:<12}", "a", "b", "Computed", "Expected");
    println!("cdf: {}", Normal::new(mean, sigma).expect("Invalid parameters").cdf(0.0));

    for &(a, b, expected) in &test_cases {
        let computed = integral_normal_pdf(a, b, mean, sigma);
        println!(
            "{:<12.2} {:<12.2} {:<18.12} {:<12.12}",
            a, b, computed, expected
        );
    }
}