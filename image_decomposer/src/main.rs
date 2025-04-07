// AI generated file
use serde::{Serialize, Deserialize};
use std::collections::{BTreeMap};
use std::f32::consts::PI;
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use std::sync::Mutex;
use image::{imageops, GrayImage, Luma};
use std::path::Path;
use statrs::distribution::{Normal, ContinuousCDF};


// Ideally resolution is at least double canvas size and nail count is over 200
const RESOLUTION: usize = 2048;
const NAIL_COUNT: usize = 200;
const THREAD_WIDTH: usize = 5000; // measured in units of how many threads are needed to cross the canvas

type SparseVector = BTreeMap<usize, f32>;
type DenseVector = Vec<f32>;

#[derive(Serialize, Deserialize)]
struct VectorDataset {
    resolution: usize,
    nail_count: usize,
    vectors: Vec<SparseVector>,
}

fn save_dense_vector_as_image( vector: &DenseVector, resolution: usize, output_path: &str, ) -> Result<(), Box<dyn std::error::Error>> {
    // Create a new grayscale image
    let mut img = GrayImage::new(resolution as u32, resolution as u32);

    for y in 0..resolution {
        for x in 0..resolution {
            let value = ((1.0 - vector[y * resolution + x].max(0.0).min(1.0)) * 255.0) as u8;
            // Put the pixel in the image
            img.put_pixel(x as u32, y as u32, Luma([value]));
        }
    }

    // Save the image
    img.save(Path::new(output_path))?;
    Ok(())
}

fn load_image_as_dense_vector( image_path: &str, resolution: usize, ) -> Result<DenseVector, Box<dyn std::error::Error>> {
    // Load the image
    let img = image::open(Path::new(image_path))?;
    
    // Convert to grayscale
    let gray_img = img.to_luma8();
    
    // Resize to the specified resolution
    let resized_img = imageops::resize(
        &gray_img,
        resolution as u32,
        resolution as u32,
        imageops::FilterType::Lanczos3,
    );
    
    let mut dense_vector = vec![0.0; resolution * resolution];
    
    // Center and radius for the circular crop
    let center = (resolution as f32 / 2.0, resolution as f32 / 2.0);
    let radius = (resolution as f32 / 2.0) - 1.0;
    
    // Convert image to vector with circular mask
    for y in 0..resolution {
        for x in 0..resolution {
            // Calculate distance from center
            let dx = x as f32 - center.0;
            let dy = y as f32 - center.1;
            let distance = (dx * dx + dy * dy).sqrt();
            
            // Only include pixels within the circle
            if distance <= radius {
                let pixel = resized_img.get_pixel(x as u32, y as u32);
                let value = 1.0 - (pixel[0] as f32 / 255.0); // Normalize to 0.0-1.0
                dense_vector[y * resolution + x] = value;
            }
        }
    }

    save_dense_vector_as_image(
        &dense_vector,
        resolution,
        "./images/input.png",
    )?;
    
    Ok(dense_vector)
}

// -1.0 to 1.0, -1 means line is bad, 1 means line is good, 0 means line is neutral
fn score_line(line: &SparseVector, target: &DenseVector, current_approximation: &DenseVector) -> f32 {
    let mut total_improvement = 0.0;
    let mut max_possible_improvement = 0.0;
    let mut count = 0;

    for (&i, &line_value) in line {
        let current = current_approximation[i];
        let target_value = target[i];
        
        // Calculate what the value would be if we add this line
        let new_value = (current + line_value).min(1.0);
        
        // Current error (distance from target)
        let current_error = (target_value - current).abs();
        
        // New error after adding this line
        let new_error = (target_value - new_value).abs();
        
        // Improvement is reduction in error
        let improvement = current_error - new_error;
        
        // Maximum possible improvement for this pixel
        let max_improvement = current_error;
        
        total_improvement += improvement;
        max_possible_improvement += max_improvement;
        count += 1;
    }

    if count == 0 {
        return 0.0;  // empty line is neutral
    }

    // Normalized score based on actual vs possible improvement
    if max_possible_improvement > 0.0 {
        (total_improvement / max_possible_improvement).clamp(-1.0, 1.0)
    } else {
        0.0  // no possible improvement means neutral score
    }
}

fn generate_line_vectors(resolution: usize, nail_count: usize, pb: &ProgressBar) -> VectorDataset {
    // Pre-compute all nail pairs first
    let nail_pairs: Vec<(usize, usize)> = (0..nail_count)
        .flat_map(|i| (i+1..nail_count).map(move |j| (i, j)))
        .collect();

    // Shared progress bar (wrapped in Mutex for thread safety)
    let pb_mutex = Mutex::new(pb);

    // Process nail pairs in parallel
    let vectors = nail_pairs
        .into_par_iter()
        .map(|(i, j)| {
            let vector = create_line_vector(resolution, nail_count, i, j);
            
            // Update progress bar
            if let Ok(pb) = pb_mutex.lock() {
                pb.inc(1);
            }
            
            vector
        })
        .collect();

    VectorDataset {
        resolution,
        nail_count,
        vectors,
    }
}

fn create_line_vector(resolution: usize, nail_count: usize, nail1: usize, nail2: usize) -> SparseVector {
    let mut vector = BTreeMap::new();
    
    // Calculate nail positions on unit circle
    let angle1 = 2.0 * PI * (nail1 as f32) / (nail_count as f32);
    let angle2 = 2.0 * PI * (nail2 as f32) / (nail_count as f32);
    
    let (x1, y1) = (angle1.cos(), angle1.sin());
    let (x2, y2) = (angle2.cos(), angle2.sin());
    
    // Scale to grid coordinates
    let scale = (resolution - 1) as f32 / 2.0;
    let x1_scaled = x1 * scale + scale;
    let y1_scaled = y1 * scale + scale;
    let x2_scaled = x2 * scale + scale;
    let y2_scaled = y2 * scale + scale;

    // Center and radius for the circular crop
    let center = (resolution as f32 / 2.0, resolution as f32 / 2.0);
    let radius = (resolution as f32 / 2.0) - 1.0;
    
    // For each point in the grid, calculate if it's under the line
    for y in 0..resolution {
        let mut found = false;
        for x in 0..resolution {
            // Calculate distance from center
            let dx = x as f32 - center.0;
            let dy = y as f32 - center.1;
            let distance = (dx * dx + dy * dy).sqrt();

            if distance > radius {
                continue;
            }


            let index = y * resolution + x;
            let value = point_under_line(
                x as f32, y as f32, 
                x1_scaled, y1_scaled, 
                x2_scaled, y2_scaled,
                1.0,
                2.0,
                0.6
            );
            
            if value > 0.0 {
                vector.insert(index, value);
                found = true;
            } else if found {
                break; // No need to check further in this row
            }
        }
    }
    
    vector
}

fn point_under_line(px: f32, py: f32, x1: f32, y1: f32, x2: f32, y2: f32, core_width: f32, falloff_width: f32, max_darkness: f32) -> f32 {
    // Vector from line start to point
    let dx = px - x1;
    let dy = py - y1;
    
    // Line direction vector
    let lx = x2 - x1;
    let ly = y2 - y1;
    
    // Calculate perpendicular distance
    let cross = dx * ly - dy * lx;
    let line_length_sq = lx * lx + ly * ly;
    
    // Early exit if line is degenerate (zero length)
    if line_length_sq < f32::EPSILON {
        return 0.0;
    }
    
    // Normalized distance from line (signed)
    let distance = cross / line_length_sq.sqrt();
    
    // Convert to absolute distance in "thread widths" units
    let thread_width = RESOLUTION as f32 / THREAD_WIDTH as f32;
    
    // The string is approximately 1 thread width wide
    // Use smooth falloff within ±1.5 thread widths
    let normalized_dist = distance.abs() / thread_width;
    
    if normalized_dist > (core_width + falloff_width) {
        return 0.0;
    }
    
    if normalized_dist <= core_width {
        max_darkness * (1.0 - (normalized_dist / core_width).powf(0.5))
    } else {
        let t = (normalized_dist - core_width) / falloff_width;
        max_darkness * 0.5 * (1.0 - t * t)
    }
}

fn integral_normal_pdf(a: f64, b: f64, mean: f64, sigma: f64) -> f64 {
    let normal = Normal::new(mean, sigma).expect("Invalid parameters");
    normal.cdf(b) - normal.cdf(a)
}

fn greedy_tophat(
    dataset: &VectorDataset,
    target: &DenseVector,
    max_lines: usize,
    selection_pb: &ProgressBar,
) -> Vec<usize> {
    let mut current_approximation = vec![0.0; dataset.resolution * dataset.resolution];
    let mut used_indices = vec![];
    
    // Precompute initial similarities for all lines
    println!("Precomputing initial similaritie scores...");
    let mut candidate_lines: Vec<(usize, f32)> = dataset.vectors.par_iter()
        .enumerate()
        .map(|(i, line)| {
            (i, score_line(line, target, &current_approximation))
        })
        .collect();
    // Sort candidates by initial similarity (highest first)
    candidate_lines.par_sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());


    println!("Making line selections...");
    for _ in 0..max_lines.min(candidate_lines.len()) {
        // Find the best available line
        let (best_index, best_similarity) = candidate_lines[0];

        // Stop if no line improves the approximation
        if best_similarity < 0.0 {
            break;
        }

        // Add the best line to our approximation
        add_sparse_to_dense(&mut current_approximation, &dataset.vectors[best_index]);
        candidate_lines.retain(|(i, _)| *i != best_index);
        used_indices.push(best_index);
        selection_pb.inc(1);
        
        // Update remaining candidates (only check top N to save time)
        let top_n = candidate_lines.len()/20;
        candidate_lines[..top_n]
            .par_iter_mut()
            .enumerate()
            .for_each(|(_i, (candidate_idx, similarity))| {
                let line = &dataset.vectors[*candidate_idx];
                *similarity = score_line(line, target, &current_approximation);
            });
        
        // Re-sort the top candidates
        candidate_lines[..top_n].par_sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    }

    selection_pb.finish_with_message("Line selection complete!");
    used_indices.into_iter().collect()
}

fn add_sparse_to_dense(dense: &mut DenseVector, sparse: &SparseVector) {
    for (&idx, &value) in sparse {
        if idx < dense.len() {
            dense[idx] = 1.0_f32.min(value+dense[idx]);
        }
    }
}

fn remove_sparse_from_dense(dense: &mut DenseVector, sparse: &SparseVector) {
    for (&idx, &value) in sparse {
        if idx < dense.len() {
            dense[idx] = (dense[idx] - value).max(0.0);
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parameters
    
    // Initialize progress bar
    let total_pairs = NAIL_COUNT * (NAIL_COUNT - 1) / 2; // Number of unique nail pairs
    let dataset_pb: ProgressBar = ProgressBar::new(total_pairs as u64);
    dataset_pb.set_style(
        indicatif::ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
            .unwrap()
            .progress_chars("#>-"),
    );
    
    // Generate all possible line vectors
    println!("Building database lines...");
    let dataset = generate_line_vectors(RESOLUTION, NAIL_COUNT, &dataset_pb);
            
    let query_vec = load_image_as_dense_vector("./images/star.jpg", RESOLUTION)?;
    
    // Find matching lines
    let max_lines = NAIL_COUNT*NAIL_COUNT/4;

    let selection_pb = ProgressBar::new(max_lines as u64);
    selection_pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) | {msg}")
            .unwrap()
            .progress_chars("#>-"),
    );

    
    let selected_lines = greedy_tophat(&dataset, &query_vec, max_lines, &selection_pb);

    println!("Selected {} lines:", selected_lines.len());


    // Create and save the approximation
    let mut approximation = vec![0.0; RESOLUTION * RESOLUTION];
    for idx in selected_lines {
        add_sparse_to_dense(&mut approximation, &dataset.vectors[idx]);
    }
    save_dense_vector_as_image(&approximation, RESOLUTION, "./images/approximation.png")?;

    Ok(())
}