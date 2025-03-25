use std::collections::{HashMap, VecDeque};

fn build_adjacency_list(edges: &[(u16, u16)]) -> HashMap<u16, VecDeque<(u16, u32)>> {
    let mut adjacency_list: HashMap<u16, VecDeque<(u16, u32)>> = HashMap::new();

    for &(source, target) in edges {
        // Check if the edge already exists
        if let Some(neighbors) = adjacency_list.get_mut(&source) {
            if let Some(edge) = neighbors.iter_mut().find(|(t, _)| *t == target) {
                // Increment the count if the edge already exists
                edge.1 += 1;
            } else {
                // Add a new edge with a count of 1
                neighbors.push_back((target, 1));
            }
        } else {
            // Add a new node with the edge and a count of 1
            adjacency_list.insert(source, VecDeque::from([(target, 1)]));
        }
    }

    adjacency_list
}

// (target, source)
pub fn hierholzer(edges: &[(u16, u16)]) -> Vec<u16> {
    // Build the adjacency list with support for duplicate edges
    let mut adjacency_list = build_adjacency_list(edges);

    // Start at any node (we'll use the first node in the adjacency list)
    let start_node = *adjacency_list.keys().next().unwrap();
    let mut stack = vec![start_node];
    let mut path = Vec::new();

    while let Some(&current) = stack.last() {
        if let Some(neighbors) = adjacency_list.get_mut(&current) {
            if !neighbors.is_empty() {
                // Get the next neighbor and decrement its count
                let (next, count) = neighbors.front_mut().unwrap();

                // Push the next node onto the stack
                stack.push(*next);
                
                *count -= 1;

                // If the count reaches 0, remove the edge
                if *count == 0 {
                    neighbors.pop_front();
                }
            } else {
                // No more neighbors, backtrack and add to path
                path.push(stack.pop().unwrap());
            }
        } else {
            // No neighbors, backtrack and add to path
            path.push(stack.pop().unwrap());
        }
    }

    path
}

// fn main() {
//     // Example edges of a graph with an Eulerian cycle and duplicate edges
//     let edges = vec![(0, 1), (1, 2), (2, 0), (1, 3), (3, 4), (4, 1), (1, 2)]; // Duplicate edge: (1, 2)

//     // Find the Eulerian cycle
//     let eulerian_cycle = hierholzer(&edges);
//     println!("Eulerian Cycle: {:?}", eulerian_cycle);
// }