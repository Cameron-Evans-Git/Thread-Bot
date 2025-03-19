use std::collections::{HashMap, HashSet, VecDeque};

#[derive(Debug, Clone, Copy)]
pub enum EdgeType {
    Original, // Original edge in the graph
    Added,    // Edge added by ensure_eulerian_cycle()
}

pub struct Graph {
    adjacency_list: HashMap<u16, Vec<(u16, EdgeType)>>, // Store edges with their type
}

impl Graph {
    pub fn new(n: u16) -> Self {
        let mut graph = Graph {
            adjacency_list: HashMap::new(),
        };
        for i in 0..n {
            graph.add_node(i);
        }
        graph
    }

    // Add a node to the graph (if it doesn't already exist)
    fn add_node(&mut self, node: u16) {
        self.adjacency_list.entry(node).or_insert_with(Vec::new);
    }

    // Add a directed edge from `source` to `target` with a specified type
    pub fn add_edge(&mut self, source: u16, target: u16, edge_type: EdgeType) {
        self.adjacency_list
            .entry(source)
            .or_insert_with(Vec::new)
            .push((target, edge_type));
    }

    // Get the neighbors of a node
    fn neighbors(&self, node: u16) -> Option<&Vec<(u16, EdgeType)>> {
        self.adjacency_list.get(&node)
    }

    // Print the graph
    pub fn print(&self) {
        for (node, neighbors) in &self.adjacency_list {
            println!("Node {} -> {:?}", node, neighbors);
        }
    }

    // Calculate the degree of each node
    fn calculate_degrees(&self) -> HashMap<u16, usize> {
        let mut degrees = HashMap::new();
        for (&node, neighbors) in &self.adjacency_list {
            degrees.insert(node, neighbors.len());
        }
        degrees
    }

    // Find nodes with odd degrees
    fn find_odd_degree_nodes(&self) -> Vec<u16> {
        let degrees = self.calculate_degrees();
        degrees
            .into_iter()
            .filter(|&(_, degree)| degree % 2 != 0)
            .map(|(node, _)| node)
            .collect()
    }

    // Calculate the cost of adding an edge between two nodes
    pub fn edge_cost(&self, node: u16, target: u16) -> u32 {
        if node == target {
            return u32::MAX; // Node cannot connect to itself
        }
    
        let len = self.adjacency_list.len() as u16;
        let half_len = len / 2;
    
        if node > target {
            if node - target <= half_len {
                u32::MAX
            } else {
                (target + len - node) as u32
            }
        } else {
            if target - node <= half_len {
                (target - node) as u32
            } else {
                u32::MAX
            }
        }
    }

    // Find the needed edges to make all degrees even
    fn find_needed_edges(&self) -> Vec<(u16, u16)> {
        let odd_nodes = self.find_odd_degree_nodes();
        let mut edges = Vec::new();

        // Pair the odd-degree nodes with minimum cost
        let mut paired = HashSet::new();
        for i in 0..odd_nodes.len() {
            if paired.contains(&odd_nodes[i]) {
                continue;
            }
            let mut min_cost = u32::MAX;
            let mut best_match = odd_nodes[i];
            for j in i + 1..odd_nodes.len() {
                if paired.contains(&odd_nodes[j]) {
                    continue;
                }
                let cost = self.edge_cost(odd_nodes[i], odd_nodes[j]);
                if cost < min_cost {
                    min_cost = cost;
                    best_match = odd_nodes[j];
                }
            }
            if min_cost != u32::MAX {
                edges.push((odd_nodes[i], best_match));
                paired.insert(odd_nodes[i]);
                paired.insert(best_match);
            }
        }

        edges
    }

    // Tarjan's algorithm to find SCCs
    fn tarjan_scc(&self) -> Vec<Vec<u16>> {
        let mut index = 0; // Unique index for each node
        let mut indices = HashMap::new(); // Stores the index of each node
        let mut low_links = HashMap::new(); // Stores the low_link of each node
        let mut stack = VecDeque::new(); // Stack to track nodes in the current SCC
        let mut on_stack = HashMap::new(); // Tracks whether a node is on the stack
        let mut sccs = Vec::new(); // Stores the final SCCs

        // Helper function for DFS
        fn strong_connect(
            node: u16,
            graph: &Graph,
            index: &mut usize,
            indices: &mut HashMap<u16, usize>,
            low_links: &mut HashMap<u16, usize>,
            stack: &mut VecDeque<u16>,
            on_stack: &mut HashMap<u16, bool>,
            sccs: &mut Vec<Vec<u16>>,
        ) {
            // Set the depth index and low_link for the node
            indices.insert(node, *index);
            low_links.insert(node, *index);
            *index += 1;

            // Push the node onto the stack
            stack.push_back(node);
            on_stack.insert(node, true);

            // Visit all neighbors
            if let Some(neighbors) = graph.neighbors(node) {
                for &(neighbor, _) in neighbors {
                    if !indices.contains_key(&neighbor) {
                        // Neighbor has not been visited, recurse
                        strong_connect(neighbor, graph, index, indices, low_links, stack, on_stack, sccs);
                        // Update low_link
                        low_links.insert(node, std::cmp::min(low_links[&node], low_links[&neighbor]));
                    } else if on_stack[&neighbor] {
                        // Neighbor is on the stack, update low_link
                        low_links.insert(node, std::cmp::min(low_links[&node], indices[&neighbor]));
                    }
                }
            }

            // If node is a root node, pop the stack and generate an SCC
            if low_links[&node] == indices[&node] {
                let mut scc = Vec::new();
                loop {
                    let neighbor = stack.pop_back().unwrap();
                    on_stack.insert(neighbor, false);
                    scc.push(neighbor);
                    if neighbor == node {
                        break;
                    }
                }
                sccs.push(scc);
            }
        }

        // Iterate over all nodes and perform DFS if not visited
        for &node in self.adjacency_list.keys() {
            if !indices.contains_key(&node) {
                strong_connect(node, self, &mut index, &mut indices, &mut low_links, &mut stack, &mut on_stack, &mut sccs);
            }
        }

        sccs
    }

    // Ensure the graph has an Eulerian cycle by modifying the adjacency_list
    pub fn ensure_eulerian_cycle(&mut self) {
        // Step 1: Make all node degrees even
        let needed_edges = self.find_needed_edges();
        for (source, target) in needed_edges {
            self.add_edge(source, target, EdgeType::Added); // Mark as added edge
        }

        // Step 2: Ensure the graph is strongly connected
        let sccs = self.tarjan_scc();
        if sccs.len() > 1 {
            // Connect SCCs with minimum cost edges
            for i in 0..sccs.len() - 1 {
                let scc1 = &sccs[i];
                let scc2 = &sccs[i + 1];
                let mut min_cost = u32::MAX;
                let mut best_edge = (0, 0);

                // Find the minimum cost edge between scc1 and scc2
                for &node1 in scc1 {
                    for &node2 in scc2 {
                        let cost = self.edge_cost(node1, node2);
                        if cost < min_cost {
                            min_cost = cost;
                            best_edge = (node1, node2);
                        }
                    }
                }

                // Add the edge to connect the SCCs
                self.add_edge(best_edge.0, best_edge.1, EdgeType::Added); // Mark as added edge
            }
        }
    }

    // Hierholzer's algorithm to find the Eulerian cycle
    pub fn hierholzer(&self, start: u16) -> Vec<(u16, EdgeType)> {
        let mut graph = self.adjacency_list.clone(); // Clone the graph to modify it
        let mut path = Vec::new();
        let mut stack = Vec::new();
        stack.push(start);

        while let Some(&current) = stack.last() {
            if let Some(neighbors) = graph.get_mut(&current) {
                if !neighbors.is_empty() {
                    // Push the next neighbor onto the stack
                    let (next, edge_type) = neighbors.pop().unwrap();
                    stack.push(next);
                    path.push((current, edge_type)); // Record the edge type
                } else {
                    // No more neighbors, backtrack
                    stack.pop();
                }
            } else {
                // No neighbors, backtrack
                stack.pop();
            }
        }

        path
    }
}

// fn main() {
//     let mut graph = Graph::new();

//     // Add nodes
//     graph.add_node(0);
//     graph.add_node(1);
//     graph.add_node(2);
//     graph.add_node(3);

//     // Add original edges
//     graph.add_edge(0, 1, EdgeType::Original);
//     graph.add_edge(1, 2, EdgeType::Original);
//     graph.add_edge(2, 3, EdgeType::Original);
//     graph.add_edge(3, 0, EdgeType::Original);

//     // Ensure the graph has an Eulerian cycle
//     graph.ensure_eulerian_cycle();

//     // Print the modified graph
//     graph.print();

//     // Find the Eulerian cycle starting at Node 0
//     let eulerian_cycle = graph.hierholzer(0);
//     println!("Eulerian Cycle with Edge Types: {:?}", eulerian_cycle);
// }