use std::collections::HashMap;
use bresenham::Bresenham;


pub struct StringCanvas {
    _resolution: usize,
    _nail_count: usize,
    stitch_cache: HashMap<usize, HashMap<usize, HashMap<usize, f64>>>, // index1, index2, (flat_index, ink)
    pub count: usize
}

impl StringCanvas {
    pub fn new(_resolution: usize, _nail_count: usize) -> Self {
        let mut stitch_cache = HashMap::new();
        let radius = _resolution as f64 / 2.0;
        let pi = std::f64::consts::PI;
        let mut count = 0;
        for i1 in 0.._nail_count {
            let mut middle_map = HashMap::new();
            let radians1 = 2.0 * pi * i1 as f64 / _nail_count as f64;
            let xy1 = ((radius - 1.0) * radians1.cos(), (radius - 1.0) * radians1.sin());
            for j in 1..(_nail_count+1)/2 {
                let mut inner_map = HashMap::new();
                let i2 = (i1 + j) % _nail_count;
                let radians2 = 2.0 * pi * i2 as f64 / _nail_count as f64;
                let xy2 = ((radius - 1.0) * radians2.cos(), (radius - 1.0) * radians2.sin());
                let true_distance = ((xy1.0 - xy2.0).powi(2) + (xy1.1 - xy2.1).powi(2)).sqrt();
                let bresenham: Vec<_> = Bresenham::new(
                    (xy1.0 as isize, xy1.1 as isize),
                    (xy2.0 as isize, xy2.1 as isize)
                ).collect();
                let ink = 255.0 * bresenham.len() as f64 / true_distance; // POTENTIAL FOR IMPROVEMENT, BASE INK ON DISTANCE TO TRUE LINE
                for (x, y) in bresenham {
                    let flat_index: usize = (x + radius as isize) as usize + (((y - radius as isize) * -1) as usize * _resolution);
                    assert!(flat_index < _resolution * _resolution);
                    inner_map.insert(flat_index, ink);
                }
                middle_map.insert(i2, inner_map);
                count += 1;
            }
            stitch_cache.insert(i1, middle_map);
        }
        println!("made {} connections from {} nails", count, _nail_count);
        StringCanvas {
            _resolution,
            _nail_count,
            stitch_cache,
            count
        }
    }

    pub fn _print_stitch_cache(&self) {
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

    pub fn get_stitch(&self, index_a: usize, index_b: usize) -> &HashMap<usize, f64> {
        let low = std::cmp::min(index_a, index_b);
        let high = std::cmp::max(index_a, index_b);

        let source = if low + (self._nail_count/2) < high {
            high
        } else { low };
        let target = if low + (self._nail_count/2) < high {
            low
        } else { high };

        if let Some(middle_map) = self.stitch_cache.get(&source) {
            if let Some(inner_map) = middle_map.get(&target) {
                return inner_map;
            }
        }
        panic!("Stitch not found for source: {} and target: {}!\nWhy are you asking?", source, target);
    }
}
