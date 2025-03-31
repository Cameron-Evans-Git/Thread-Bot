mod string_canvas;
mod ssim;
use image::ImageReader as ImageReader;
use std::path::Path;
use fixedbitset::FixedBitSet;


const RESOLUTION: usize = 512;
const NAIL_COUNT: usize = 200;
// const CANVAS_DIAMETER: u32 = 1000000; // micrometers
// const THREAD_DIAMETER: u32 = 500; // micrometers

fn genetic_simulation(canvas:&string_canvas::StringCanvas, target_image:&Vec<u8>, parent_count: usize) {
    // Initialize population
    let pop_size = parent_count + ((parent_count * (parent_count - 1)) / 2);
    let mut population = vec![FixedBitSet::with_capacity(canvas.count); pop_size];

    loop {
        //
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
            genetic_simulation(&_canvas, &bytes, 10);
        }
        Err(e) => eprintln!("Error: {}", e),
    }
}

// fitness DONE
// precompute DONE

//loop
    //make generation
    //evaluate fitness
    //select parents
    //crossover
    //mutate
    //replace population
