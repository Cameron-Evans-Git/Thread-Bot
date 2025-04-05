use std::collections::HashMap;


pub struct StringCanvas {
    _resolution: usize,
    _nail_count: usize,
    stitch_cache: HashMap<usize, HashMap<usize, HashMap<usize, f64>>>, // index1, index2, (flat_index, ink)
    pub count: usize,

    flat_index_to_stitch: HashMap<usize, (usize, usize)>
}

impl StringCanvas {
    pub fn new(_resolution: usize, _nail_count: usize) -> Self {
        let mut flat_index_to_stitch = HashMap::new();
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

                for point in 0.._resolution*_resolution {
                    let x_corner = point % _resolution;
                    let y_corner = point / _resolution;
                    let xy_cart = (x_corner as f64 - radius, (y_corner as f64 - radius) * -1.0);

                    let dist = distance_to_line_segment(xy1, xy2, xy_cart);
                    if dist > 1.0 {
                        continue;
                    }
                    inner_map.insert(point, (1.0 - dist) * 255.0);
                }
                middle_map.insert(i2, inner_map);
                flat_index_to_stitch.insert(count, (i1, i2));
                count += 1;
            }
            stitch_cache.insert(i1, middle_map);
        }
        println!("made {} connections from {} nails", count, _nail_count);
        StringCanvas {
            _resolution,
            _nail_count,
            stitch_cache,
            count,
            flat_index_to_stitch
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

    pub fn get_stitch_flat(&self, flat_index: usize) -> &HashMap<usize, f64> {
        let indexs = self.flat_index_to_stitch.get(&flat_index).unwrap();
        return self.get_stitch(indexs.0, indexs.1);
    }
}

pub fn distance_to_line_segment(
    xy1: (f64, f64),
    xy2: (f64, f64),
    point: (f64, f64),
) -> f64 {
    let (x1, y1) = xy1;
    let (x2, y2) = xy2;
    let (x0, y0) = point;

    let dx = x2 - x1;
    let dy = y2 - y1;

    // Handle case where the segment is just a point
    if dx == 0.0 && dy == 0.0 {
        return ((x0 - x1).powi(2) + (y0 - y1).powi(2)).sqrt();
    }

    // Projection factor (how far along the segment the closest point is)
    let t = ((x0 - x1) * dx + (y0 - y1) * dy) / (dx * dx + dy * dy);

    // Clamp to the segment (0 ≤ t ≤ 1)
    let t_clamped = t.max(0.0).min(1.0);

    // Closest point on the segment
    let closest_x = x1 + t_clamped * dx;
    let closest_y = y1 + t_clamped * dy;

    // Euclidean distance
    ((x0 - closest_x).powi(2) + (y0 - closest_y).powi(2)).sqrt()
}
