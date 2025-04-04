mod string_canvas;
mod mse;
use image::{ImageBuffer, Luma};
use image::ImageReader as ImageReader;
use std::{ path::Path};
use fixedbitset::FixedBitSet;
use rand::Rng;
use rand::prelude::SliceRandom;
use rayon::prelude::*;


const RESOLUTION: usize = 512;
const NAIL_COUNT: usize = 201;
const CANVAS_DIAMETER: u32 = 1000000; // micrometers
const THREAD_DIAMETER: u32 = 500; // micrometers
const INK_SCALAR: f64 = (THREAD_DIAMETER as f64 * RESOLUTION as f64) / CANVAS_DIAMETER as f64;


const PARENT_COUNT: usize = 20;
const GENERATION_COUNT: usize = 10000;

fn genetic_simulation(canvas:&string_canvas::StringCanvas, target_image:&Vec<u8>, mutation_chance: f64, seed: &FixedBitSet) {
    let mut rng = rand::rng();
    let pop_size = PARENT_COUNT + ((PARENT_COUNT * (PARENT_COUNT - 1)) / 2);
    let mut population = vec![(-1.0 as f64, FixedBitSet::with_capacity(canvas.count)); pop_size]; //fitness, stitch_key
    population[0].1 = seed.clone();
    for i in 1..pop_size {
        population[i].1 = FixedBitSet::with_capacity(canvas.count);
        for j in 0..canvas.count {
            population[i].1.set(j, rand::random_bool(0.2));
        }
    }

    for _gen in 0..GENERATION_COUNT {
        population.par_iter_mut().for_each(|individual| {
            let drawing = bitset_to_vec(&individual.1, canvas);
            // individual.0 = -mse::mse(&drawing, target_image); //negate to work with ssim implementation
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
    let mut drawing = vec![255 as u8; RESOLUTION*RESOLUTION];
    // 01 02 03 04 12 13 14 23 24 34
    for index in bitset.ones() {
        for (flat_index, ink) in canvas.get_stitch_flat(index).into_iter() {
            let ink_delta = (ink * INK_SCALAR) as u8;
            if drawing[*flat_index] < ink_delta {
                drawing[*flat_index] = 0;
            } else {
                drawing[*flat_index] -= ink_delta;
            }
        }
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

    // Convert to grayscale
    let grey_img = img.into_luma8();

    // Create a circular mask
    let (width, height) = grey_img.dimensions();
    let mut circular_img = ImageBuffer::new(width, height);
    
    let center_x = width as f32 / 2.0;
    let center_y = height as f32 / 2.0;
    let radius = (width.min(height) as f32 / 2.0).powi(2); // Using squared radius for comparison

    for (x, y, pixel) in circular_img.enumerate_pixels_mut() {
        let dx = x as f32 - center_x;
        let dy = y as f32 - center_y;
        let distance_squared = dx * dx + dy * dy;

        if distance_squared <= radius {
            *pixel = *grey_img.get_pixel(x, y);
        } else {
            *pixel = Luma([255u8]);
        }
    }

    let _ = circular_img.save("./../images/input.png");
    
    // Get raw bytes
    Ok(circular_img.as_raw().to_vec())
}

fn hill_climb(canvas: &string_canvas::StringCanvas, target_image:&Vec<u8>, weights:&Vec<f64>) -> FixedBitSet {
    //start with blank canvas state
    //while !changes_to_state
    //foreach bit in state, check if flipping will improve the output
    let mut state = FixedBitSet::with_capacity(canvas.count);
    let mut image = bitset_to_vec(&state, &canvas);
    loop {
        let mut changes = 0;
        let mut indexs: Vec<usize> = (0..canvas.count).collect();
        indexs.shuffle(&mut rand::rng());

        for i in indexs {
            let slice = canvas.get_stitch_flat(i);
            let old_fit_sliced = mse::mse_subset_weighted(&target_image, &image, &slice.keys().collect(), weights);

            state.toggle(i);
            image = bitset_to_vec(&state, &canvas);
            let new_fit_slice = mse::mse_subset_weighted(&target_image, &image, &slice.keys().collect(), weights);

            if new_fit_slice < old_fit_sliced {
                changes += 1;
            } else {
                state.toggle(i);
                image = bitset_to_vec(&state, &canvas);
            }
        }

        if changes < 10 { // use this to seed genetic? What to do from here?
            break;
        }
        println!("changes: {}", changes);
        let _ = vec_to_image(&bitset_to_vec(&state, &canvas), Path::new("./../images/output.png"));
    }
    state
}

fn main() {
    assert!(NAIL_COUNT % 2 == 1, "Nail count must be odd");
    let _canvas = string_canvas::StringCanvas::new(RESOLUTION, NAIL_COUNT);


    let path = Path::new("./../images/cat.png");
    match load_image_as_bytes(path) {
        Ok(bytes) => {
            println!("Image converted to {} bytes", bytes.len());
            let _seed = hill_climb(&_canvas, &bytes, &vec![1.0; bytes.len()]);
            // genetic_simulation(&_canvas, &bytes, 0.001);
        }
        Err(e) => eprintln!("Error: {}", e),
    }
}
