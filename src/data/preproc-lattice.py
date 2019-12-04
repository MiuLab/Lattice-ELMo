import math
import argparse
import re
import csv

from tqdm import tqdm


class LatticeFromPlfExtractor:
    def extract_to(self, dataset_file, lattice_file, out_file):

        with open(dataset_file) as csvfile:
            reader = csv.DictReader(csvfile)
            dataset = [row for row in reader]

        lattices = dict()
        with open(lattice_file) as f:
            for line in tqdm(f):
                id, line = line.strip().split('\t', maxsplit=1)
                id = id.replace('.plf', '')
                graph = LatticeFromPlfExtractor._Lattice()
                graph.read_plf_line(line)
                # graph.insert_initial_node()
                # graph.insert_final_node()
                graph.forward()
                graph2 = LatticeFromPlfExtractor._Lattice.convert_to_node_labeled_lattice(graph)
                if len(graph2.nodes) == 1:
                    graph2.insert_initial_node()
                graph2.resolve_final_nodes()
                serial = graph2.serialize_to_string()
                lattices[id] = serial
        
        new_dataset = []
        for i, row in enumerate(dataset):
            id = row["id"]
            if id in lattices:
                row["text"] = lattices[id]
                new_dataset.append(row)

        with open(args.out_file, 'w') as csvfile:
            writer = csv.DictWriter(
                csvfile,
                dataset[0].keys()
            )
            writer.writeheader()
            writer.writerows(new_dataset)

    class _Lattice(object):

        def __init__(self, nodes=None, edges=None):
            self.nodes = nodes
            self.edges = edges

        def serialize_to_string(self):
            node_ids = {}
            for n in self.nodes:
                node_ids[n] = len(node_ids)
            node_lst = [n.label for n in self.nodes]
            node_str = "[" + ", ".join(["(" + ", ".join(
                [("'" + str(labelItem).replace("'", "\\'") + "'" if type(labelItem) == str else str(labelItem)) for labelItem in node]) + ")" for
                                 node in node_lst]) + "]"
            numbered_edges = []
            for fromNode, toNode, _ in self.edges:
                from_id = node_ids[fromNode]
                to_id = node_ids[toNode]
                numbered_edges.append((from_id, to_id))
            edge_str = str(numbered_edges)
            return node_str + "," + edge_str

        def insert_initial_node(self):
            initial_node = LatticeFromPlfExtractor._LatticeLabel(label=("<s>", 0.0, 0.0, 0.0))
            if len(self.nodes) > 0:
                self.edges.insert(0, (initial_node, self.nodes[0], LatticeFromPlfExtractor._LatticeLabel(("<s>", 0.0))))
            self.nodes.insert(0, initial_node)

        def insert_final_node(self):
            final_node = LatticeFromPlfExtractor._LatticeLabel(label=("final-node", 0.0))
            self.edges.append((self.nodes[-1], final_node, LatticeFromPlfExtractor._LatticeLabel(("</s>", 0.0))))
            self.nodes.append(final_node)

        @staticmethod
        def convert_to_node_labeled_lattice(edge_labeled_lattice):
            word_nodes = []
            word_node_edges = []
            for edge in edge_labeled_lattice.edges:
                _, _, edge_label = edge
                new_word_node = edge_label
                word_nodes.append(new_word_node)
            for edge1 in edge_labeled_lattice.edges:
                _, edge1_to, edge1_label = edge1
                for edge2 in edge_labeled_lattice.edges:
                    edge2_from, _, edge2_label = edge2
                    if edge1_to == edge2_from:
                        word_node_edges.append((edge1_label, edge2_label, LatticeFromPlfExtractor._LatticeLabel()))
            return LatticeFromPlfExtractor._Lattice(nodes=word_nodes, edges=word_node_edges)

        def forward(self):
            self.nodes[0].marginal_log_prob = 0.0
            for edge in self.edges:
                from_node, to_node, edge_label = edge
                prev_sum = 0.0  # incomplete P(toNode)
                if not hasattr(to_node, 'marginal_log_prob'):
                    to_node.marginal_log_prob = 0.0
                else:
                    prev_sum = math.exp(to_node.marginal_log_prob)
                fwd_weight = math.exp(edge_label.label[1])  # lattice weight normalized across outgoing edges
                marginal_link_prob = math.exp(from_node.marginal_log_prob) * fwd_weight  # P(fromNode, toNode)
                to_node.marginal_log_prob = math.log(prev_sum + marginal_link_prob)  # (partially) completed P(toNode)
                to_node.label = (to_node.marginal_log_prob,)
                edge_label.label = tuple(list(edge_label.label) + [min(0.0, math.log(marginal_link_prob))])
            for node in self.nodes:
                incoming_edges = [edge for edge in self.edges if edge[1] == node]
                # incoming_sum = sum([math.exp(edge[0].marginal_log_prob) for edge in incoming_edges])
                incoming_sum = sum([math.exp(edge[2].label[2]) for edge in incoming_edges])
                for edge in incoming_edges:
                    from_node, to_node, edge_label = edge
                    # bwd_weight_log = min(0.0, edge[0].marginal_log_prob - math.log(incoming_sum))
                    bwd_weight_log = min(0.0, edge_label.label[2] - math.log(incoming_sum))
                    # bwd_weight_log = min(0.0, edge[0].marginal_log_prob + edge_label.label[1] - node.marginal_log_prob)
                    edge_label.label = tuple(list(edge_label.label) + [bwd_weight_log])

        def resolve_final_nodes(self):
            final_node = self.nodes[-1]
            changed_edges = []
            for i, node in enumerate(self.nodes):
                if node.label[0] == '</s>':
                    for j, (from_node, to_node, label) in enumerate(self.edges):
                        if to_node is node:
                            from_node.label = from_node.label[:-1] + (from_node.label[-1] + node.label[-1],)
                            changed_edges.append(j)
                    if i != len(self.nodes) - 1:
                        _ = self.nodes.pop(i)
            for j in changed_edges:
                from_node, to_node, label = self.edges[j]
                self.edges[j] = (from_node, final_node, label)
            self.nodes[-1].label = ('</s>', 0.0, 0.0, 0.0)

        def read_plf_line(self, line):
            parenth_depth = 0
            plf_nodes = []
            plf_edges = []

            for token in re.split("([()])", line):
                if len(token.strip()) > 0 and token.strip() != ",":
                    if token == "(":
                        parenth_depth += 1
                        if parenth_depth == 2:
                            new_node = LatticeFromPlfExtractor._LatticeLabel(label=None)
                            plf_nodes.append(new_node)
                    elif token == ")":
                        parenth_depth -= 1
                        if parenth_depth == 0:
                            new_node = LatticeFromPlfExtractor._LatticeLabel(label=None)
                            plf_nodes.append(new_node)
                            break  # end of the lattice
                    elif token[0] == "'":
                        word, score, distance = [eval(tt) for tt in token.split(",")]
                        word = word.lower()
                        cur_node_id = len(plf_nodes) - 1
                        edge_from = cur_node_id
                        edge_to = cur_node_id + distance
                        edge_label = LatticeFromPlfExtractor._LatticeLabel(label=(word, score))
                        plf_edges.append((edge_from, edge_to, edge_label))
            resolved_edges = []
            for edge in plf_edges:
                edge_from, edge_to, edge_label = edge
                resolved_edges.append((plf_nodes[edge_from], plf_nodes[edge_to], edge_label))
            self.nodes = plf_nodes
            self.edges = resolved_edges

    class _LatticeLabel(object):
        def __init__(self, label=None):
            self.label = label
        def __repr__(self):
            return str(self.label)


if __name__ == "__main__":
    """
    plf = "((('<s>', 0, 1),),(('YES', 0, 1),),(('I\\'D', 0, 1),),(('LIKE', 0, 1),),(('TO', 0, 1),),(('FIND', 0, 1),),(('A', 0, 1),),(('FLIGHT', 0, 1),),(('FROM', 0, 1),),(('MEMPHIS', 0, 1),),(('TO', 0, 1),),(('ALMA', 0, 1),),(('<UNK>', 0, 1),),(('PUMPING', -1.81475937, 1),('</s>', -0.177784383, 4),),(('AND', 0, 1),),(('<UNK>', 0, 1),),(('</s>', 0, 1),),)"
    graph = LatticeFromPlfExtractor._Lattice()
    graph.read_plf_line(plf)
    graph.forward()
    graph2 = LatticeFromPlfExtractor._Lattice.convert_to_node_labeled_lattice(graph)
    print(graph2.serialize_to_string())
    graph2.resolve_final_nodes()
    print(graph2.serialize_to_string())
    exit()
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_file")
    parser.add_argument("lattice_file")
    parser.add_argument("out_file")
    args = parser.parse_args()
    LatticeFromPlfExtractor().extract_to(args.dataset_file, args.lattice_file, args.out_file)
