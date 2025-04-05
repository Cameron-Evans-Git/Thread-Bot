use std::collections::{HashMap, HashSet};
use rand::{Rng, thread_rng};
use rand::distributions::StandardNormal;
use maplit::hashmap;


type SparseVector = HashMap<usize, f32>;
type DenseVector = Vec<f32>;

/// LSH structure for cosine similarity
pub struct CosineLSH {
    planes: Vec<DenseVector>,  // Random projection planes
    hash_tables: Vec<HashMap<String, Vec<SparseVector>>>,  // Multiple hash tables
    num_planes: usize,        // Number of hyperplanes per hash function
    num_tables: usize,        // Number of hash tables
}

impl CosineLSH {
    /// Create a new LSH index
    pub fn new(num_planes: usize, num_tables: usize, dim: usize) -> Self {
        let mut rng = thread_rng();
        
        // Generate random projection planes for all tables
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
    
    /// Hash a sparse vector into a signature
    fn hash_sparse(&self, vec: &SparseVector, table_idx: usize) -> String {
        let mut signature = String::with_capacity(self.num_planes);
        
        for i in 0..self.num_planes {
            let plane = &self.planes[table_idx * self.num_planes + i];
            let mut projection = 0.0;
            
            // Sparse dot product with the plane
            for (&dim, &val) in vec {
                projection += val * plane[dim];
            }
            
            signature.push(if projection >= 0.0 { '1' } else { '0' });
        }
        
        signature
    }
    
    /// Hash a dense vector into a signature
    fn hash_dense(&self, vec: &DenseVector, table_idx: usize) -> String {
        let mut signature = String::with_capacity(self.num_planes);
        
        for i in 0..self.num_planes {
            let plane = &self.planes[table_idx * self.num_planes + i];
            let projection = plane.iter().zip(vec.iter()).map(|(p, v)| p * v).sum::<f32>();
            signature.push(if projection >= 0.0 { '1' } else { '0' });
        }
        
        signature
    }
    
    /// Index a sparse vector in the LSH tables
    pub fn index_vector(&mut self, vec: SparseVector) {
        for table_idx in 0..self.num_tables {
            let signature = self.hash_sparse(&vec, table_idx);
            self.hash_tables[table_idx]
                .entry(signature)
                .or_insert_with(Vec::new)
                .push(vec.clone());
        }
    }
    
    /// Query similar vectors to a dense vector
    pub fn query(&self, query_vec: &DenseVector, max_results: usize) -> Vec<(SparseVector, f32)> {
        let mut candidates = HashSet::new();
        let mut results = Vec::new();
        
        // Get candidates from all hash tables
        for table_idx in 0..self.num_tables {
            let signature = self.hash_dense(query_vec, table_idx);
            
            if let Some(bucket) = self.hash_tables[table_idx].get(&signature) {
                for vec in bucket {
                    candidates.insert(vec);
                }
            }
        }
        
        // Convert HashSet to Vec and compute exact similarities
        for vec in candidates.into_iter() {
            let similarity = cosine_similarity(vec, query_vec);
            results.push((vec.clone(), similarity));
        }
        
        // Sort by similarity and return top results
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results.truncate(max_results);
        results
    }
    
    /// Batch query for multiple dense vectors
    pub fn batch_query(&self, queries: &[DenseVector], max_results: usize) -> Vec<Vec<(SparseVector, f32)>> {
        queries.iter().map(|q| self.query(q, max_results)).collect()
    }
}

// Helper function to get dimension from sparse vectors
fn get_dimension(sparse_vectors: &[SparseVector]) -> usize {
    sparse_vectors.iter()
        .flat_map(|v| v.keys())
        .max()
        .map_or(0, |&m| m + 1)
}

// Create a dataset of sparse vectors
let dataset = vec![
    hashmap! {0 => 0.5, 2 => 1.0, 5 => 0.3},
    hashmap! {1 => 0.8, 3 => 0.2},
    hashmap! {0 => 0.9, 2 => 0.1, 4 => 0.7},
    hashmap! {1 => 0.6, 5 => 0.4},
];

// Determine dimensionality (max index + 1)
let dim = dataset.iter()
    .flat_map(|v| v.keys())
    .max()
    .map_or(0, |&m| m + 1);

// Create LSH index (4 planes per table, 2 tables)
let mut lsh = CosineLSH::new(4, 2, dim);

// Index all vectors
for vec in dataset {
    lsh.index_vector(vec);
}

// Create a query vector
let query = vec![0.5, 0.0, 0.9, 0.0, 0.0, 0.3];

// Find top 3 similar vectors
let results = lsh.query(&query, 3);

for (vec, similarity) in results {
    println!("Similarity: {:.3}, Vector: {:?}", similarity, vec);
}