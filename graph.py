from collections import deque
import networkx as nx
import json

# Load data
def create_graph(file_path):
    with open('captions_with_concepts_train2017.json') as f:
        data = json.load(f)

        # Create directed graph
        G = nx.DiGraph()

        # Add image nodes and concept level nodes
        for ann in data['annotations']:
            img_id = ann['image_id']
            levels = ann['concept_levels']
            
            # Add nodes
            G.add_node(img_id, type='image')
            G.add_node(levels['level1'], type='level1')
            G.add_node(levels['level2'], type='level2')
            
            # Add edges
            G.add_edge(img_id, levels['level1'])
            G.add_edge(levels['level1'], levels['level2'])
        return G


# nx.write_graphml(G, "concept_hierarchy.graphml")
# import pickle
# with open("concept_hierarchy.pkl", "wb") as f:
#     pickle.dump(G, f)

# with open("concept_hierarchy.pkl", "rb") as f:
#     G = pickle.load(f)

# Visualize
# pos = nx.spring_layout(G)
# nx.draw(G, pos, with_labels=True)

def find_and_merge_small_clusters(G):
   # Get all level1 nodes
   level1_nodes = [n for n,d in G.nodes(data=True) if d['type']=='level1']
   
   for concept in level1_nodes:
       # Do BFS to count leaves (image nodes)
       leaves = []
       visited = set()
       queue = deque([concept])
       
       while queue: #For each level1 concept, BFS to find all image nodes (leaves) under it:           
           node = queue.popleft()
           if G.nodes[node]['type'] == 'image':
               leaves.append(node)
           
           for neighbor in G.neighbors(node):
               if neighbor not in visited:
                   visited.add(neighbor)
                   queue.append(neighbor)
                   
       # If less than 5 leaves, merge to parent
       if len(leaves) < 5:
           parent_concept = list(G.predecessors(concept))[0]
           for leaf in leaves:
               # Remove old edges
               G.remove_edge(leaf, concept)
               # Add edge to parent
               G.add_edge(leaf, parent_concept)
           # Remove the small concept node
           G.remove_node(concept)

   return G