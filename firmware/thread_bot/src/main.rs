use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use std::io::{self, Write};
use std::thread;
use std::time::Duration;
mod tarjan;
mod hierholzer;

const DO_PRINT: bool = false;

fn main() {
    // Load & pre-process an image file
    // Send image to the pytorch server
    // unpack the .onnx
    // create thread bot instructions .tbi file
    // send .tbi file to thread bot as work order
    //println!("{:?}", unpack_onnx());
    // let path = generate_instructions(unpack_onnx());
    let onnx = unpack_onnx(0);
    let path = generate_instructions(&onnx);

    // let bar_width = 50; // Width of the loading bar
    // let total = 1000; // Total number of iterations

    // for i in 0..=total {
    //     // Calculate progress percentage
    //     let progress = i as f32 / total as f32;
    //     let filled = (progress * bar_width as f32) as usize;

    //     // Create the loading bar string
    //     let bar = format!(
    //         "[{}{}] {:.1}%",
    //         "=".repeat(filled),
    //         " ".repeat(bar_width - filled),
    //         progress * 100.0
    //     );

    //     // Print the loading bar
    //     print!("\r{}", bar); // `\r` moves the cursor to the start of the line
    //     io::stdout().flush().unwrap(); // Flush the output to ensure it's displayed

    //     generate_instructions(unpack_onnx(i));
    // }
}

fn unpack_onnx(seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    return (0..4950).map(|_| rng.gen_range(0.0..1.0)).collect();
    //return vec![];
}


fn generate_instructions(onnx: &Vec<f32>) -> Vec<u16> {
    // the thread bot will articulate the applicator just before reaching each nail
    // each instruction is a signed short containing the index of the nail to be visted next
    // positive if inter-canvas traversal should be made on the way to the target nail
    // negative if outer-canvas traversal should be made on the way to the target nail

    // Assume the bot only moves in the clockwise direction
    // assume the bot can only move 180 degrees between each applicator articulation

    // Each nail will have a number of incoming connections "i" and outgoing connections "o"
    // and incoming connection is defined as a connection that is with a nail > .5 revolutions awap
    // an outgoing connection is defined as a connection that is with a nail < .5 revolutions away
    // i and o must be made equal for each nail
    // this will be done by adding outer-canvas connections
        // These outer canvas connevtions will be made by finding the nail with the largest defecit and scanning clockwise
        // or counterclockwise to find an appropriate partner nail. repeat until canvas is balanced
    // once balanced, start with the nail with the largest number of connections. end with the smallest number of connections
        // out of the nails within a positive half rotation AND desire a connection with the current nail,
        // (this desired connection may be inter or outer, doesn't matter)
        // choose the target nail that has the largest number of connections. decrementing as you go
        // repeat until all connections are made.

    let confidence_cutoff: f32 = 0.5; // Adjust this based on desired "brightness" of final work piece
    let n: u16 = ((1.0 + ((1+8*onnx.len()) as f32).sqrt())/2.0) as u16; // number of nails
    if (n*(n-1)/2) != onnx.len() as u16 {
        panic!("Invalid ONNX file. Number of nails does not match number of connections");
    }
    // sum connections
    let mut st_tracking: Vec<(i16, i16, u16)> = vec![(0, 0, 0); n as usize]; // (target_count, source_count, nail_id) connections for each nail
    let mut inner_connections: Vec<(u16, u16)> = vec![]; // (target, source)
    let mut s_count = 0;
    for i in 0..n {
        st_tracking[i as usize].2 = i;
        for j in i+1..n { 
            let score: f32 = onnx[(s_count) as usize];
            s_count += 1;
            if score < confidence_cutoff {
                continue;
            }
            if j <= i+n/2 { // which nail is "left" of the other
                st_tracking[i as usize].1 += 1;
                st_tracking[j as usize].0 += 1;
                inner_connections.push((j, i));
            } else {
                st_tracking[i as usize].0 += 1;
                st_tracking[j as usize].1 += 1;
                inner_connections.push((i, j));
            }
        }
    }
    let sccs = tarjan::tarjan_scc(&inner_connections);
    if DO_PRINT {
        println!("tarjan: {:?}", sccs);
        println!("inner connections: {:?}", inner_connections);
        println!("pre balanced source target tracking: {:?}", st_tracking);
    }
    // balance connections
    let mut outer_connections: Vec<(u16, u16)> = vec![]; // (target, source)
    loop {
        st_tracking.sort_by(|a, b| (a.0-a.1).cmp(&(b.0-b.1))); // positive difference means too many incoming connections
        if (st_tracking[0].0 - st_tracking[0].1) == 0 && (st_tracking[n as usize - 1].0 - st_tracking[n as usize -1].1) == 0 {
            break;
        }
        let mut found_connection: bool = false;
        if -1*(st_tracking[0].0 - st_tracking[0].1) > (st_tracking[n as usize - 1].0 - st_tracking[n as usize -1].1) {
            for i in 1..n+1 { // find nearest match for most sourced nail
                let check_index = if i <= st_tracking[0].2 {st_tracking[0].2-i} else {n+st_tracking[0].2-i};
                for j in 0..n {
                    if st_tracking[j as usize].2 == check_index {
                        if st_tracking[j as usize].0 - st_tracking[j as usize].1 > 0 {
                            outer_connections.push((st_tracking[0].2, check_index));
                            st_tracking[j as usize].1 += 1;
                            st_tracking[0].0 += 1;
                            found_connection = true;
                        }
                        break;
                    }
                }
                if found_connection {
                    break;
                }
            }
            if !found_connection {
                panic!("No connection found for nail {}", st_tracking[0].2);
            }
        } else {
            for i in 1..n+1 as u16 { // find nearest match for most targeted nail
                let check_index = (st_tracking[n as usize - 1].2+i)%n;
                for j in 0..n {
                    if st_tracking[j as usize].2 == check_index {
                        if st_tracking[j as usize].0 - st_tracking[j as usize].1 < 0 {
                            outer_connections.push((check_index, st_tracking[n as usize - 1].2));
                            st_tracking[j as usize].0 += 1;
                            st_tracking[n as usize - 1].1 += 1;
                            found_connection = true;
                        }
                        break;
                    }
                }
                if found_connection {
                    break;
                }
            }
            if !found_connection {
                panic!("No connection found for nail {}", st_tracking[n as usize - 1].2);
            }
        }
    }

    if DO_PRINT {
        println!("Outer Connections {:?}", outer_connections);
        println!("post balanced source target tracking: {:?}", st_tracking);
    }
    let all_connections = inner_connections.iter().chain(outer_connections.iter()).cloned().collect::<Vec<(u16, u16)>>();

    let sccs = tarjan::tarjan_scc(&all_connections);
    if DO_PRINT {
        println!("updated tarjan: {:?}", sccs);
    }
    if sccs.len() > 1 {
        panic!("Invalid connection graph. Multiple strongly connected components");
    }

    let mut path = hierholzer::hierholzer(&all_connections);
    if DO_PRINT {
        println!("hierholzer: {:?}, len: {}", path, path.len());
        println!("inner con length: {} outer con length: {}", inner_connections.len(), outer_connections.len());
    }
    


    for i in 0..outer_connections.len() {
        let mut found = false;
        for j in 0..path.len()-1 {
            if outer_connections[i].1 == path[j] && outer_connections[i].0 == path[j+1] {
                path[j] += 0xF000;
                found = true;
                break;
            }
        }
        if !found {
            panic!("Outer connection not found in path");
        }
    }

    if DO_PRINT {
        println!("final path: {:?}", path);
    }

    // add something to validate the path
    

    return path;
}