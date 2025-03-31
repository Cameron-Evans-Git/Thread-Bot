use std::collections::HashMap;
use bresenham::Bresenham;


pub struct StringCanvas {
    resolution: usize,
    nail_count: usize,
    stitch_cache: HashMap<usize, HashMap<usize, HashMap<usize, u8>>>, // index1, index2, (flat_index, ink)
    pub count: usize
}

impl StringCanvas {
    pub fn new(resolution: usize, nail_count: usize) -> Self {
        let mut stitch_cache = HashMap::new();
        let radius = resolution as f64 / 2.0;
        let pi = std::f64::consts::PI;
        let mut count = 0;
        for i1 in 0..nail_count {
            let mut middle_map = HashMap::new();
            let radians1 = 2.0 * pi * i1 as f64 / nail_count as f64;
            let xy1 = (radius * radians1.cos(), radius * radians1.sin());
            for j in 2..(nail_count+1)/2 {
                let mut inner_map = HashMap::new();
                let i2 = (i1 + j) % nail_count;
                let radians2 = 2.0 * pi * i2 as f64 / nail_count as f64;
                let xy2 = (radius * radians2.cos(), radius * radians2.sin());
                let true_distance = ((xy1.0 - xy2.0).powi(2) + (xy1.1 - xy2.1).powi(2)).sqrt();
                let bresenham: Vec<_> = Bresenham::new(
                    (xy1.0 as isize, xy1.1 as isize),
                    (xy2.0 as isize, xy2.1 as isize)
                ).collect();
                let ink = (255.0 * bresenham.len() as f64 / true_distance) as u8; // POTENTIAL FOR IMPROVEMENT, BASE INK ON DISTANCE TO TRUE LINE
                for (x, y) in bresenham {
                    let flat_index = ((x + radius as isize) + ((y - radius as isize) * -1)) as usize * resolution; // Wrong??
                    inner_map.insert(flat_index, ink);
                }
                middle_map.insert(i2, inner_map);
                count += 1;
            }
            stitch_cache.insert(i1, middle_map);
        }
        StringCanvas {
            resolution,
            nail_count,
            stitch_cache,
            count
        }
    }

    pub fn print_stitch_cache(&self) {
        for (i1, middle_map) in &self.stitch_cache {
            println!("Nail {}: ", i1);
            for (i2, inner_map) in middle_map {
                println!("  Nail {}: ", i2);
                for (flat_index, ink) in inner_map {
                    println!("    Index: {}, Ink: {:.3}", flat_index, ink);
                }
            }
        }
    }

    pub fn get_stitch(&self, source: usize, target: usize) -> &HashMap<usize, u8> {
        if let Some(middle_map) = self.stitch_cache.get(&source) {
            if let Some(inner_map) = middle_map.get(&target) {
                return inner_map;
            }
        }
        panic!("Stitch not found for source: {} and target: {}", source, target);
    }
}
