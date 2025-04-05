use serde::{Serialize, Deserialize};
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::f32::consts::PI;
use indicatif::ProgressBar;
use rand::{Rng, rng};
use rand_distr::StandardNormal;
use std::hash::{Hash, Hasher};
use std::path::Path;
use rayon::prelude::*;
use std::sync::Mutex;

type SparseVector = BTreeMap<usize, f32>;
type DenseVector = Vec<f32>;

#[derive(Serialize, Deserialize)]
struct VectorDataset {
    resolution: usize,
    nail_count: usize,
    vectors: Vec<SparseVector>,
}

impl VectorDataset {
    /// Loads a VectorDataset from a JSON file
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let contents = fs::read_to_string(path)?;
        let dataset: VectorDataset = serde_json::from_str(&contents)?;
        Ok(dataset)
    }

    /// Saves the VectorDataset to a JSON file
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let serialized = serde_json::to_string_pretty(self)?;
        fs::write(path, serialized)?;
        Ok(())
    }
}

#[derive(Clone)]
struct HashableSparseVector(SparseVector);

impl PartialEq for HashableSparseVector {
    fn eq(&self, other: &Self) -> bool {
        if self.0.len() != other.0.len() {
            return false;
        }
        
        for ((k1, v1), (k2, v2)) in self.0.iter().zip(other.0.iter()) {
            if k1 != k2 || !float_equal(*v1, *v2) {
                return false;
            }
        }
        
        true
    }
}

impl Eq for HashableSparseVector {}

impl Hash for HashableSparseVector {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for (k, v) in &self.0 {
            k.hash(state);
            v.to_bits().hash(state);
        }
    }
}

fn float_equal(a: f32, b: f32) -> bool {
    (a - b).abs() < f32::EPSILON
}

pub struct CosineLSH {
    planes: Vec<DenseVector>,
    hash_tables: Vec<HashMap<String, Vec<SparseVector>>>,
    num_planes: usize,
    num_tables: usize,
}

impl CosineLSH {
    pub fn new(num_planes: usize, num_tables: usize, dim: usize) -> Self {
        let mut rng = rng();
        
        let planes = (0..num_tables * num_planes)
            .map(|_| {
                (0..dim).map(|_| rng.sample(StandardNormal)).collect()
            })
            .collect();
            
        CosineLSH {
            planes,
            hash_tables: vec![HashMap::new(); num_tables],
            num_planes,
            num_tables,
        }
    }
    
    fn hash_sparse(&self, vec: &SparseVector, table_idx: usize) -> String {
        let mut signature = String::with_capacity(self.num_planes);
        
        for i in 0..self.num_planes {
            let plane = &self.planes[table_idx * self.num_planes + i];
            let mut projection = 0.0;
            
            for (&dim, &val) in vec {
                projection += val * plane[dim];
            }
            
            signature.push(if projection >= 0.0 { '1' } else { '0' });
        }
        
        signature
    }
    
    fn hash_dense(&self, vec: &DenseVector, table_idx: usize) -> String {
        let mut signature = String::with_capacity(self.num_planes);
        
        for i in 0..self.num_planes {
            let plane = &self.planes[table_idx * self.num_planes + i];
            let projection = plane.iter().zip(vec.iter()).map(|(p, v)| p * v).sum::<f32>();
            signature.push(if projection >= 0.0 { '1' } else { '0' });
        }
        
        signature
    }
    
    pub fn index_vector(&mut self, vec: SparseVector) {
        for table_idx in 0..self.num_tables {
            let signature = self.hash_sparse(&vec, table_idx);
            self.hash_tables[table_idx]
                .entry(signature)
                .or_insert_with(Vec::new)
                .push(vec.clone());
        }
    }
    
    pub fn query(&self, query_vec: &DenseVector, max_results: usize) -> Vec<(SparseVector, f32)> {
        let seen = Mutex::new(HashMap::new());
        let results = Mutex::new(Vec::new());
    
        (0..self.num_tables).into_par_iter().for_each(|table_idx| {
            let signature = self.hash_dense(query_vec, table_idx);
            
            if let Some(bucket) = self.hash_tables[table_idx].get(&signature) {
                let mut local_results = Vec::new();
                
                for vec in bucket {
                    let hashable = HashableSparseVector(vec.clone());
                    let mut seen = seen.lock().unwrap();
                    
                    if !seen.contains_key(&hashable) {
                        seen.insert(hashable.clone(), ());
                        let similarity = cosine_similarity(vec, query_vec);
                        local_results.push((vec.clone(), similarity));
                    }
                }
                
                results.lock().unwrap().extend(local_results);
            }
        });
    
        let mut final_results = results.into_inner().unwrap();
        final_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        final_results.truncate(max_results);
        final_results
    }
}

fn cosine_similarity(sparse: &SparseVector, dense: &DenseVector) -> f32 {
    let dot = sparse.iter()
        .map(|(idx, &value)| {
            value * dense.get(*idx).unwrap_or(&0.0)
        })
        .sum::<f32>();
    
    let sparse_norm = sparse.values().map(|&v| v.powi(2)).sum::<f32>().sqrt();
    let dense_norm = dense.iter().map(|&v| v.powi(2)).sum::<f32>().sqrt();
    
    if sparse_norm == 0.0 || dense_norm == 0.0 {
        0.0
    } else {
        dot / (sparse_norm * dense_norm)
    }
}

fn get_dimension(sparse_vectors: &[SparseVector]) -> usize {
    sparse_vectors.iter()
        .flat_map(|v| v.keys())
        .max()
        .map_or(0, |&m| m + 1)
}

fn _generate_random_dense_vector(dim: usize) -> DenseVector {
    let mut rng = rng();
    (0..dim).map(|_| rng.random::<f32>()).collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parameters
    let r = 4096; // Resolution (grid size)
    let n = 401;  // Nail count
    let filename = format!("line_vectors_r{}_n{}.json", r, n);
    let dataset;

    // Check if file already exists
    if !Path::new(&filename).exists() {
        println!("Generating line vectors for resolution {} and {} nails...", r, n);
        
        // Initialize progress bar
        let total_pairs = n * (n - 1) / 2; // Number of unique nail pairs
        let pb = ProgressBar::new(total_pairs as u64);
        pb.set_style(
            indicatif::ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
                .unwrap()
                .progress_chars("#>-"),
        );
        
        // Generate all possible line vectors
        dataset = generate_line_vectors(r, n, &pb);
        
        pb.finish_with_message("Generation complete!");
        
        println!("Saving to {}...", filename);
        dataset.save_to_file(&filename)?;
        println!("Saved {} vectors to {}", dataset.vectors.len(), filename);
    } else {
        println!("Loading existing dataset from {}...", filename);
        dataset = VectorDataset::load_from_file(&filename)?;
        println!("Loaded {} vectors from file", dataset.vectors.len());
    }
    
    // Example of using the loaded data with LSH
    let dim = get_dimension(&dataset.vectors);
    assert!(dim == r * r, "Dimension mismatch: expected {}, got {}", r * r, dim);
    let mut lsh = CosineLSH::new(16, 8, dim); // 10 planes, 5 tables
    
    // Index all vectors
    for vec in dataset.vectors {
        lsh.index_vector(vec);
    }
    
    // Create a random query vector
    let query_vec = _generate_random_dense_vector(dim);
    
    // Perform a query
    let results = lsh.query(&query_vec, 5);
    println!("Top 5 similar vectors:");
    for (_vec, similarity) in results {
        println!("Similarity: {:.4}", similarity);
    }
    
    Ok(())
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
    
    // For each point in the grid, calculate if it's under the line
    for y in 0..resolution {
        for x in 0..resolution {
            let index = y * resolution + x;
            let value = point_under_line(
                x as f32, y as f32, 
                x1_scaled, y1_scaled, 
                x2_scaled, y2_scaled
            );
            
            if value > 0.0 {
                vector.insert(index, value);
            }
        }
    }
    
    vector
}

fn point_under_line(px: f32, py: f32, x1: f32, y1: f32, x2: f32, y2: f32) -> f32 {
    // Calculate the position relative to the line
    let d = (px - x1) * (y2 - y1) - (py - y1) * (x2 - x1);
    
    // Assumes thread weight is 1 pixel, uses triangular area to determine distance
    let line_length = ((x2 - x1).powi(2) + (y2 - y1).powi(2)).sqrt();
    let distance = d.abs() / line_length;
    
    // Value between 0 and 1 based on distance from line
    1.0 - distance.min(1.0)
}