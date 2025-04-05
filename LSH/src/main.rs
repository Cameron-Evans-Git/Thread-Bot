use std::collections::{HashMap, BTreeMap};
use rand::{Rng, rng};
use rand_distr::StandardNormal;
use std::hash::{Hash, Hasher};
use maplit::btreemap;
use std::time::Instant;

type SparseVector = BTreeMap<usize, f32>;
type DenseVector = Vec<f32>;

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
        let mut seen = HashMap::new();
        let mut results = Vec::new();
        
        for table_idx in 0..self.num_tables {
            let signature = self.hash_dense(query_vec, table_idx);
            
            if let Some(bucket) = self.hash_tables[table_idx].get(&signature) {
                for vec in bucket {
                    let hashable = HashableSparseVector(vec.clone());
                    if !seen.contains_key(&hashable) {
                        seen.insert(hashable.clone(), ());
                        let similarity = cosine_similarity(vec, query_vec);
                        results.push((vec.clone(), similarity));
                    }
                }
            }
        }
        
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results.truncate(max_results);
        results
    }
    
    pub fn batch_query(&self, queries: &[DenseVector], max_results: usize) -> Vec<Vec<(SparseVector, f32)>> {
        queries.iter().map(|q| self.query(q, max_results)).collect()
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

fn generate_random_sparse_vector(dim: usize, sparsity: f32) -> SparseVector {
    let mut rng = rng();
    let mut vec = BTreeMap::new();
    
    for i in 0..dim {
        if rng.random::<f32>() < sparsity {
            vec.insert(i, rng.random::<f32>());
        }
    }
    
    vec
}

fn generate_random_dense_vector(dim: usize) -> DenseVector {
    let mut rng = rng();
    (0..dim).map(|_| rng.random::<f32>()).collect()
}

fn main() {
    // Basic test case
    println!("=== Basic Test Case ===");
    let dataset = vec![
        btreemap! {0 => 0.5, 2 => 1.0, 5 => 0.3},
        btreemap! {1 => 0.8, 3 => 0.2},
        btreemap! {0 => 0.9, 2 => 0.1, 4 => 0.7},
        btreemap! {1 => 0.6, 5 => 0.4},
    ];

    let dim = get_dimension(&dataset);
    let mut lsh = CosineLSH::new(4, 2, dim);

    for vec in dataset {
        lsh.index_vector(vec);
    }

    let query = vec![0.5, 0.0, 0.9, 0.0, 0.0, 0.3];
    let results = lsh.query(&query, 3);

    for (vec, similarity) in results {
        println!("Similarity: {:.3}, Vector: {:?}", similarity, vec);
    }

    // Performance test with larger dataset
    println!("\n=== Performance Test ===");
    let dim = 1000;
    let num_vectors = 10_000;
    let sparsity = 0.01; // 1% non-zero elements
    
    // Generate random dataset
    let start_gen = Instant::now();
    let large_dataset: Vec<SparseVector> = (0..num_vectors)
        .map(|_| generate_random_sparse_vector(dim, sparsity))
        .collect();
    println!("Generated {} sparse vectors in {:?}", num_vectors, start_gen.elapsed());

    // Create LSH index
    let start_index = Instant::now();
    let mut lsh = CosineLSH::new(16, 8, dim); // More planes and tables for better accuracy
    for vec in &large_dataset {
        lsh.index_vector(vec.clone());
    }
    println!("Indexed vectors in {:?}", start_index.elapsed());

    // Generate random queries
    let num_queries = 10;
    let queries: Vec<DenseVector> = (0..num_queries)
        .map(|_| generate_random_dense_vector(dim))
        .collect();

    // Test query performance
    let start_query = Instant::now();
    let batch_results = lsh.batch_query(&queries, 5);
    println!("Processed {} queries in {:?}", num_queries, start_query.elapsed());

    // Print some results
    println!("\nSample query results:");
    for (i, results) in batch_results.iter().enumerate().take(2) {
        println!("Query {} top results:", i);
        for (vec, similarity) in results {
            println!("  Similarity: {:.3}, Vector (first 5 dims): {:?}", 
                similarity, 
                vec.iter().take(5).collect::<BTreeMap<_, _>>());
        }
    }

    // Accuracy test
    println!("\n=== Accuracy Test ===");
    let known_vectors = vec![
        btreemap! {0 => 1.0, 1 => 1.0}, // 45 degrees to axes
        btreemap! {0 => 1.0},          // X axis
        btreemap! {1 => 1.0},           // Y axis
        btreemap! {0 => -1.0},          // Negative X
    ];
    
    let dim = 2;
    let mut lsh = CosineLSH::new(8, 4, dim);
    for vec in &known_vectors {
        lsh.index_vector(vec.clone());
    }

    let test_queries = vec![
        (vec![1.0, 1.0], "45 degree query"),
        (vec![1.0, 0.0], "X axis query"),
        (vec![0.0, 1.0], "Y axis query"),
        (vec![-1.0, 0.0], "Negative X query"),
    ];

    for (query, desc) in test_queries {
        println!("\nTesting {}: {:?}", desc, query);
        let results = lsh.query(&query, 2);
        
        for (vec, similarity) in results {
            println!("  Found: similarity {:.3}, vector {:?}", similarity, vec);
        }
    }

    // Edge case testing
    println!("\n=== Edge Case Testing ===");
    let edge_vectors = vec![
        btreemap! {}, // Empty vector
        btreemap! {0 => 0.0, 1 => 0.0}, // Zero vector
        btreemap! {0 => f32::INFINITY}, // Infinite value
        btreemap! {0 => f32::NAN}, // NaN value
    ];
    
    let dim = 2;
    let mut lsh = CosineLSH::new(8, 4, dim);
    for vec in &edge_vectors {
        lsh.index_vector(vec.clone());
    }

    let edge_queries = vec![
        (vec![], "Empty query"),
        (vec![0.0, 0.0], "Zero query"),
    ];

    for (query, desc) in edge_queries {
        println!("\nTesting {}: {:?}", desc, query);
        let results = lsh.query(&query, 2);
        
        if results.is_empty() {
            println!("  No results found (expected for edge cases)");
        } else {
            for (vec, similarity) in results {
                println!("  Found: similarity {:.3}, vector {:?}", similarity, vec);
            }
        }
    }
}