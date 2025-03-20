use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use chrono::Local;
mod graph;

fn main() {
    // Load & pre-process an image file
    // Send image to the pytorch server
    // unpack the .onnx
    // create thread bot instructions .tbi file
    // send .tbi file to thread bot as work order
    //println!("{:?}", unpack_onnx());
    let mut graph = unpack_onnx();
    graph.make_eulerian();
    let path = graph.hierholzer(0);
    println!("path: {:?}", path);
}

fn unpack_onnx() -> graph::Graph {
    let n = 10;
    let mut graph = graph::Graph::new(n);

    let mut rng = StdRng::seed_from_u64(0);

    let mut s_count = 0;
    for i in 0..n {
        for j in i+1..n { 
            let score: f32 = rng.gen_range(0.0..1.0);
            s_count += 1;
            if score < 0.5 {
                continue;
            }
            if graph.edge_cost(i, j) > graph.edge_cost(j, i) { // which nail is "left" of the other
                graph.add_edge(j, i, graph::EdgeType::Original, 1);
            } else {
                graph.add_edge(i, j, graph::EdgeType::Original, 1);
            }
        }
    }

    graph.print();

    graph
}