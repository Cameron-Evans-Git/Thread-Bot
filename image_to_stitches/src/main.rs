mod string_canvas;
mod ssim;
use image::ImageReader as ImageReader;
use std::path::Path;
use fixedbitset::FixedBitSet;
use rand::Rng;
use rayon::prelude::*;


const RESOLUTION: usize = 512;
const NAIL_COUNT: usize = 21;
const CANVAS_DIAMETER: u32 = 1000000; // micrometers
const THREAD_DIAMETER: u32 = 500; // micrometers
const INK_SCALAR: f64 = (THREAD_DIAMETER as f64 * RESOLUTION as f64) / CANVAS_DIAMETER as f64;


const PARENT_COUNT: usize = 10;
const GENERATION_COUNT: usize = 100;

fn genetic_simulation(canvas:&string_canvas::StringCanvas, target_image:&Vec<u8>, mutation_chance: f64) {
    let mut rng = rand::rng();
    let pop_size = PARENT_COUNT + ((PARENT_COUNT * (PARENT_COUNT - 1)) / 2);
    let mut population = vec![(-1.0 as f64, FixedBitSet::with_capacity(canvas.count)); pop_size]; //fitness, stitch_key
    for i in 0..pop_size {
        population[i].1 = FixedBitSet::with_capacity(canvas.count);
        for j in 0..canvas.count {
            population[i].1.set(j, rand::random());
        }
    }

    for _gen in 0..GENERATION_COUNT {
        population.par_iter_mut().for_each(|individual| {
            let drawing = bitset_to_vec(&individual.1, canvas);
            individual.0 = ssim::ssim(&drawing, target_image, RESOLUTION, 11, 0.01, 0.03);
        });

        //sort population in decending order
        population.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        println!("Generation {}: Best fitness: {:.8}", _gen, population[0].0);
        // println!("Generation {}: Best fitness: {:.5} Population: {:?}", _gen, population[0].0, population);
        let _ = vec_to_image(&bitset_to_vec(&population[0].1, &canvas), Path::new("./../images/output.png"));
        //keep PARENT_COUNT best parents for next generation and cross-breed/mutate the rest
        let mut pop_ind = PARENT_COUNT;
        for i in 0..PARENT_COUNT {
            for j in i+1..PARENT_COUNT {
                let mut child = (-1.0 as f64, population[i].1.clone());
                for k in 0..canvas.count {
                    if rng.random_bool(0.5) {
                        child.1.set(k, population[j].1[k]);
                    }
                    if rng.random_bool(mutation_chance) {
                        child.1.set(k, !child.1[k]);
                    }
                }
                population[pop_ind] = child;
                pop_ind += 1;
            }
        }
    }
}

fn bitset_to_vec(bitset: &FixedBitSet, canvas: &string_canvas::StringCanvas) -> Vec<u8> {
    assert!(bitset.len() == canvas.count, "Bitset length does not match canvas count");
    let mut drawing = vec![0 as u8; RESOLUTION*RESOLUTION];
    let mut source: usize = 0;
    let mut target: usize = 1;
    let mut prev_ind: usize = 0;
    // 01 02 03 04 12 13 14 23 24 34
    for index in bitset.ones() {
        for _ in prev_ind..index {
            target += 1;
            if target == NAIL_COUNT {
                source += 1;
                target = source + 1;
            }
            if target == NAIL_COUNT {
                panic!("Target nail index out of bounds: {} == {}, Source: {}, index: {}", target, NAIL_COUNT, source, index);
            }
        }
        for (flat_index, ink) in canvas.get_stitch(source, target).into_iter() {
            drawing[*flat_index] += (ink * INK_SCALAR) as u8;
        }
        prev_ind = index;
    }
    drawing
}

fn vec_to_image(vec: &Vec<u8>, path: &Path) -> Result<(), image::ImageError> {
    let img = image::GrayImage::from_raw(RESOLUTION as u32, RESOLUTION as u32, vec.clone()).unwrap();
    img.save(path)?;
    Ok(())
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
    assert!(NAIL_COUNT % 2 == 1, "Nail count must be odd");
    let _canvas = string_canvas::StringCanvas::new(RESOLUTION, NAIL_COUNT);


    let path = Path::new("./../images/star.jpg");
    match load_image_as_bytes(path) {
        Ok(bytes) => {
            println!("Image converted to {} bytes", bytes.len());
            genetic_simulation(&_canvas, &bytes, 0.001);
        }
        Err(e) => eprintln!("Error: {}", e),
    }
}
