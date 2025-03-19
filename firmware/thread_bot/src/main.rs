use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
mod tarjan;

fn main() {
    // Load & pre-process an image file
    // Send image to the pytorch server
    // unpack the .onnx
    // create thread bot instructions .tbi file
    // send .tbi file to thread bot as work order
    //println!("{:?}", unpack_onnx());
    generate_instructions(unpack_onnx());
}

fn unpack_onnx() -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(0);
    return (0..36).map(|_| rng.gen_range(0.0..1.0)).collect();
    //return vec![];
}


fn generate_instructions(onnx: Vec<f32>) -> Vec<i16> {
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
    println!("tarjan: {:?}", sccs);
    println!("inner connections: {:?}", inner_connections);
    println!("pre balanced source target tracking: {:?}", st_tracking);
    panic!("done");
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

    println!("Outer Connections {:?}", outer_connections);
    println!("post balanced source target tracking: {:?}", st_tracking);

    // develop ordered instructions from inner connections and outer_connections
    // assume thread begins fastened to nail 0
    // this will fail if nail 0 has no connections TODO
    let mut instructions: Vec<i16> = vec![];
    let mut source_nail: u16 = 0;
    loop {
        st_tracking.sort_by(|a, b| (b.0).cmp(&(a.0))); // sort by decending number of incoming connections remaining
        println!("source nail: {} st_tracking: {:?}", source_nail, st_tracking);
        if st_tracking[0].0 == 0 {
            break;
        }
        for i in 0..n {
            if st_tracking[i as usize].2 == source_nail {
                if i == n-1 {
                    panic!("No connection found for nail {}", source_nail);
                }
                continue;
            }
            if inner_connections.contains(&(st_tracking[i as usize].2, source_nail)) {
                println!("inner connections: {:?}", inner_connections);
                inner_connections.retain(|x| x != &(st_tracking[i as usize].2, source_nail));
                println!("i:{} source_nail:{} inner connections: {:?}", i, source_nail, inner_connections);
                instructions.push(st_tracking[i as usize].2 as i16);
                st_tracking[i as usize].0 -= 1;
                st_tracking[source_nail as usize].1 -= 1;
                source_nail = st_tracking[i as usize].2;
                break;
            }
            if outer_connections.contains(&(st_tracking[i as usize].2, source_nail)) {
                println!("outer connections: {:?}", outer_connections);
                outer_connections.retain(|x| x != &(st_tracking[i as usize].2, source_nail));
                println!("i:{} source_nail:{} outer connections: {:?}", i, source_nail, outer_connections);
                instructions.push(st_tracking[i as usize].2 as i16 * -1);
                st_tracking[i as usize].0 -= 1;
                st_tracking[source_nail as usize].1 -= 1;
                source_nail = st_tracking[i as usize].2;
                break;
            }
        }
    }
    println!("Instructions: {:?}", instructions);

    return instructions;
}
