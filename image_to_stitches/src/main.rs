mod string_canvas;
mod ssim;
use image::ImageReader as ImageReader;
use std::path::Path;
use fixedbitset::FixedBitSet;
use rand::Rng;


const RESOLUTION: usize = 512;
const NAIL_COUNT: usize = 200;
const CANVAS_DIAMETER: u32 = 1000000; // micrometers
const THREAD_DIAMETER: u32 = 500; // micrometers

fn genetic_simulation(canvas:&string_canvas::StringCanvas, target_image:&Vec<u8>, parent_count: usize, generation_count: usize, mutation_chance: f64) {
    let mut rng = rand::rng();
    let ink_scalar: f64 = (THREAD_DIAMETER as f64 * RESOLUTION as f64) / CANVAS_DIAMETER as f64;
    let pop_size = parent_count + ((parent_count * (parent_count - 1)) / 2);
    let mut population = vec![(-1.0 as f64, FixedBitSet::with_capacity(canvas.count)); pop_size]; //fitness, stitch_key
    for i in 0..pop_size {
        population[i].1 = FixedBitSet::with_capacity(canvas.count);
        for j in 0..canvas.count {
            population[i].1.set(j, rand::random());
        }
    }

    for _gen in 0..generation_count {
        for individual in &mut population {
            let mut drawing = vec![0 as u8; target_image.len()];
            let mut source: usize = 0;
            let mut target: usize = 1;
            let mut step: usize = 0;
            for index in individual.1.ones() {
                for _ in 0..index-step {
                    target += 1;
                    if target == NAIL_COUNT {
                        source += 1;
                        target = source + 2;
                    }
                }
                for (flat_index, ink) in canvas.get_stitch(source, target).into_iter() {
                    drawing[*flat_index] += (ink * ink_scalar) as u8;
                }
                step = index;
            }
            individual.0 = ssim::ssim(&drawing, target_image, RESOLUTION, 11, 0.01, 0.03);
        }

        population.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        println!("Generation {}: Best fitness: {}", _gen, population[0].0);
        //keep parent_count best parents for next generation and cross-breed/mutate the rest
        for i in 0..parent_count {
            for j in i+1..parent_count {
                let mut child = (-1.0 as f64, population[i].1.clone());
                for k in 0..canvas.count {
                    if rng.random_bool(0.5) {
                        child.1.set(k, population[j].1[k]);
                    }
                    if rng.random_bool(mutation_chance) {
                        child.1.set(k, !child.1[k]);
                    }
                }
                population[(i*parent_count)+j+parent_count] = child;
            }
        }
    }
}

fn load_image_as_bytes(path: &Path) -> Result<Vec<u8>, image::ImageError> {
    // Load the image
    let img = ImageReader::open(path)?.decode()?;

    // Resize the image to the desired resolution
    let img = img.resize_exact(RESOLUTION as u32, RESOLUTION as u32, image::imageops::FilterType::Nearest);
    
    let rgb_img = img.into_luma8();
    
    // Get raw bytes
    Ok(rgb_img.as_raw().to_vec())
}

fn main() {
    let _canvas = string_canvas::StringCanvas::new(RESOLUTION, NAIL_COUNT);


    let path = Path::new("images/example.png");
    match load_image_as_bytes(path) {
        Ok(bytes) => {
            println!("Image converted to {} bytes", bytes.len());
            genetic_simulation(&_canvas, &bytes, 10, 10, 0.0001);
        }
        Err(e) => eprintln!("Error: {}", e),
    }
}
