use std::collections::HashSet;
use std::collections::HashMap;
pub fn tarjan_scc(edges: &Vec<(u16, u16)>) -> Vec<Vec<u16>> {
    // Step 1: Build adjacency list
    let mut adj_list: HashMap<u16, Vec<u16>> = HashMap::new();
    for &(src, dst) in edges {
        adj_list.entry(src).or_insert_with(Vec::new).push(dst);
    }

    // Step 2: Initialize variables for Tarjan's algorithm
    let mut index = 0; // Global index counter
    let mut indices: HashMap<u16, u16> = HashMap::new(); // Discovery time of each node
    let mut low_links: HashMap<u16, u16> = HashMap::new(); // Lowest reachable node
    let mut on_stack: HashSet<u16> = HashSet::new(); // Nodes currently on the stack
    let mut stack: Vec<u16> = Vec::new(); // Stack for DFS
    let mut sccs: Vec<Vec<u16>> = Vec::new(); // Resulting SCCs

    // Step 3: Define the recursive DFS function
    fn strongconnect(
        v: u16,
        adj_list: &HashMap<u16, Vec<u16>>,
        index: &mut u16,
        indices: &mut HashMap<u16, u16>,
        low_links: &mut HashMap<u16, u16>,
        on_stack: &mut HashSet<u16>,
        stack: &mut Vec<u16>,
        sccs: &mut Vec<Vec<u16>>,
    ) {
        // Set the depth index for v to the smallest unused index
        indices.insert(v, *index);
        low_links.insert(v, *index);
        *index += 1;
        stack.push(v);
        on_stack.insert(v);

        // Consider successors of v
        if let Some(neighbors) = adj_list.get(&v) {
            for &w in neighbors {
                if !indices.contains_key(&w) {
                    // Successor w has not yet been visited; recurse on it
                    strongconnect(w, adj_list, index, indices, low_links, on_stack, stack, sccs);
                    low_links.insert(v, std::cmp::min(low_links[&v], low_links[&w]));
                } else if on_stack.contains(&w) {
                    // Successor w is in the stack and hence in the current SCC
                    low_links.insert(v, std::cmp::min(low_links[&v], indices[&w]));
                }
            }
        }

        // If v is a root node, pop the stack and generate an SCC
        if low_links[&v] == indices[&v] {
            let mut scc = Vec::new();
            loop {
                let w = stack.pop().unwrap();
                on_stack.remove(&w);
                scc.push(w);
                if w == v {
                    break;
                }
            }
            sccs.push(scc);
        }
    }

    // Step 4: Iterate over all nodes and perform DFS if not visited
    for &(src, _) in edges {
        if !indices.contains_key(&src) {
            strongconnect(
                src,
                &adj_list,
                &mut index,
                &mut indices,
                &mut low_links,
                &mut on_stack,
                &mut stack,
                &mut sccs,
            );
        }
    }

    sccs
}

fn main() {
    // Example directed graph as a vector of edges
    let edges = vec![
        (0, 1),
        (1, 2),
        (2, 0),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 3),
    ];

    // Find all SCCs using Tarjan's algorithm
    let sccs = tarjan_scc(&edges);

    // Print the SCCs
    println!("Strongly Connected Components:");
    for scc in sccs {
        println!("{:?}", scc);
    }
}