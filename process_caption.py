from pycocotools.coco import COCO
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import os
import re
from captions_loader import setup_coco
import networkx as nx
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate

############################################################################################################
def setup_coco(data_dir, data_type='train2017'):
    """
    Setup COCO dataset
    data_dir: root directory of COCO dataset
    data_type: 'train2017' or 'val2017'
    """
    annotation_file = os.path.join(data_dir, f'captions_{data_type}.json')
    coco = COCO(annotation_file)
    return coco



def extract_output_content(text):
   """Extract content between output tags"""
   
   pattern = '<output>(.*?)</output>'
   match = re.search(pattern, text)
   return match.group(1) if match else None

def extract_multiple_output_content(text):
   """Extract content between output tags"""
   text = extract_output_content(text)
   matches = re.findall(pattern, text)
   pattern = '<caption>(.*?)</caption>'
   return matches


def call_open_ai_api(prompt):
   """
   Call OpenAI API with the given prompt
   """
   from openai import OpenAI
   openai.api_key = ""
   client = OpenAI()

   return client.chat.completions.create(
      model="o1-mini",
      messages=[
         {"role": "system", "content": "You are a helpful assistant."},
         {
               "role": "user",
               "content": prompt
         }
      ]
   )["choices"][0]["message"]["content"]
   

def get_concept_level1(image_captions):
   """
   Get concept level 1 from caption
   """
   return call_open_ai_api(""" 
Definition:
A general concept is a broad idea or category that captures common attributes or qualities shared by multiple specific instances or objects, which may be concrete or abstract. It simplifies complex information by grouping similar instances together. For example: 1) The concept of “animal” encompasses all living organisms that can move and consume food. 2) The concept of “freedom” describes a state of being free from constraints or oppression, applicable in various social, political, and personal contexts.

Instruction:
Given a set of captions describing one single image, determine the first-level main general concept, so avoid being too general. The first-level concept should describe a general idea of the original caption. The concept should be a short phrase or a brief phrase that captures the general idea of the image (represented by the captions). Emphasize more on events or themes rather than objects or actions. Concepts MUST use canonical vocabulary (e.g. singular; present tense). Do not provide explanations. Write the concept in an <output> section.

Captions:
""" + "\n".join(image_captions))

def get_concept_level2(image_captions, concept_level1):
   """
   Get concept level 2 from caption
   """
   return call_open_ai_api(""" 
Definition:
A general concept is a broad idea or category that captures common attributes or qualities shared by multiple specific instances or objects, which may be concrete or abstract. It simplifies complex information by grouping similar instances together. For example: 1) The concept of “animal” encompasses all living organisms that can move and consume food. 2) The concept of “freedom” describes a state of being free from constraints or oppression, applicable in various social, political, and personal contexts.

Instruction:
Given a set of captions describing one single image, determine the second-level main general concept. The second-level concept should describe a general idea of the first-level concept. The concept should be a short phrase or a brief phrase that captures the general idea of the image (represented by the captions). Emphasize more on events or themes rather than objects or actions. Concepts MUST use canonical vocabulary (e.g. singular; present tense). Do not provide explanations. Write the concept in an <output> section.

First Level Concept: {concept_level1}
                           
Captions:
""" + "\n".join(image_captions))

def get_concept_level3(image_captions, concept_level1, concept_level2):
   """
   Get concept level 3 from caption
   """
   return call_open_ai_api(""" 
Definition:
A general concept is a broad idea or category that captures common attributes or qualities shared by multiple specific instances or objects, which may be concrete or abstract. It simplifies complex information by grouping similar instances together. For example: 1) The concept of “animal” encompasses all living organisms that can move and consume food. 2) The concept of “freedom” describes a state of being free from constraints or oppression, applicable in various social, political, and personal contexts.

Instruction:
Given a set of captions describing one single image, determine the third-level main general concept. The third-level concept should describe a general idea of the second-level concept. The concept should be a short phrase or a brief phrase that captures the general idea of the image (represented by the captions). Emphasize more on events or themes rather than objects or actions. Concepts MUST use canonical vocabulary (e.g. singular; present tense). Do not provide explanations. Write the concept in an <output> section.
First Level Concept: {concept_level1}
Second Level Concept: {concept_level2}                           
                           
Captions:
""" + "\n".join(image_captions))

def get_concept_level4(image_captions, concept_level1, concept_level2, concept_level3):
   """
   Get concept level 4 from caption
   """
   return call_open_ai_api(""" 
Definition:
A general concept is a broad idea or category that captures common attributes or qualities shared by multiple specific instances or objects, which may be concrete or abstract. It simplifies complex information by grouping similar instances together. For example: 1) The concept of “animal” encompasses all living organisms that can move and consume food. 2) The concept of “freedom” describes a state of being free from constraints or oppression, applicable in various social, political, and personal contexts.

Instruction:
Given a set of captions describing one single image, determine the fourth-level main general concept. The fourth-level concept should describe a general idea of the third-level concept. The concept should be a short phrase or a brief phrase that captures the general idea of the image (represented by the captions). Emphasize more on events or themes rather than objects or actions. Concepts MUST use canonical vocabulary (e.g. singular; present tense). Do not provide explanations. Write the concept in an <output> section.
First Level Concept: {concept_level1}
Second Level Concept: {concept_level2}      
Third Level Concept: {concept_level3}                     
                           
Captions:
""" + "\n".join(image_captions))

def get_new_merged_concept(concept, smallest_sibling, parent, images, smallest_sibling_imags):
   return call_open_ai_api("""
   Definition:
A general concept is a broad idea or category that captures common attributes or qualities shared by multiple specific instances or objects, which may be concrete or abstract. It simplifies complex information by grouping similar instances together. For example: 1) The concept of “animal” encompasses all living organisms that can move and consume food. 2) The concept of “freedom” describes a state of being free from constraints or oppression, applicable in various social, political, and personal contexts.

Instruction:
Group 1 of captions have the concept {concept} and group 2 of captions have the concept {smallest_sibling}. Combine two groups together and determine a new unified concept. The concept should be a short phrase or a brief phrase that captures the general idea of the image (represented by the captions). The new unified concept should appropriately represent all the captions from both groups and also belong to the higher-level general concept {parent}. Concepts MUST use canonical vocabulary (e.g. singular; present tense). The purpose of this is to streamline the concept hierarchy and reduce fragmentation. Do not provide explanations. Write the new unified general concept in an <output> section.

Group 1 Captions:                           
""" + "\n".join(images)
+ """
Group 1 Captions:      
"""+ "\n".join(smallest_sibling_imags))

def get_general_captions(captions, concept):
   return call_open_ai_api("""
   Definition:
A general concept is a broad idea or category that captures common attributes or qualities shared by multiple specific instances or objects, which may be concrete or abstract. It simplifies complex information by grouping similar instances together. For example: 1) The concept of “animal” encompasses all living organisms that can move and consume food. 2) The concept of “freedom” describes a state of being free from constraints or oppression, applicable in various social, political, and personal contexts.

Instruction:
Given a group of captions that share one common general concept {concept}, paraphrase the original captions. For each caption, consider a more generalized version of the caption that focuses on the general theme rather than specific details. Each new caption MUST be compatible with and be able to FULLY describe and represent every other caption of the original captions ACCURATELY. Each new caption should be related to the concept of {}. The captions should be around the same length as the original captions. Write all the new captions, separated using caption tags, in an <output> section.
                           
Captions:
   """ + "\n".join(captions))
def process_all_store():
   import json
   with open('captions_train2017.json') as f:
      data = json.load(f)
      for ann in data['annotations']:
         concept_levels = get_concept_levels(ann['caption'])
         concept_levels = extract_output_content(concept_levels)
         concept_levels = extract_levels_from_output(concept_levels)
         ann['concept_levels'] = concept_levels

   with open('captions_with_concepts_train2017.json', 'w') as f:
      json.dump(data, f)

def add_image_nodes(G, img_id, coco, concept_level1, concepts):
   annIds = coco.getAnnIds(imgIds=img_id)
   anns = coco.loadAnns(annIds)
   anns = [ann['caption'] for ann in anns]
   if concept_level1 not in concepts:
      concepts[concept_level1] = True
      concept_level2 = extract_output_content(get_concept_level2(anns, concept_level1))
      if concept_level2 not in concepts:
         concepts[concept_level2] = True
         concept_level3 = extract_output_content(get_concept_level3(anns, concept_level1, concept_level2))
         if concept_level3 not in concepts:
            concepts[concept_level3] = True
            concept_level4 = extract_output_content(get_concept_level4(anns, concept_level1, concept_level2, concept_level3))
            if concept_level4 not in concepts:
               concepts[concept_level4] = True
               G.add_node(concept_level4, type='level4')
               G.add_node(concept_level3, type='level3')
               G.add_node(concept_level2, type='level2')
               G.add_node(concept_level1, type='level1')
               G.add_node(img_id, type='image')
               G.add_edge(img_id, concept_level1)
               G.add_edge(concept_level1, concept_level2)
               G.add_edge(concept_level2, concept_level3)
               G.add_edge(concept_level3, concept_level4)
            else:
               G.add_node(concept_level3, type='level3')
               G.add_node(concept_level2, type='level2')
               G.add_node(concept_level1, type='level1')
               G.add_node(img_id, type='image')
               G.add_edge(img_id, concept_level1)
               G.add_edge(concept_level1, concept_level2)
               G.add_edge(concept_level2, concept_level3)
               G.add_edge(concept_level3, concept_level4)
         else:
            G.add_node(concept_level2, type='level2')
            G.add_node(concept_level1, type='level1')
            G.add_node(img_id, type='image')
            G.add_edge(img_id, concept_level1)
            G.add_edge(concept_level1, concept_level2)
            G.add_edge(concept_level2, concept_level3)
      else:
            G.add_node(concept_level1, type='level1')
            G.add_node(img_id, type='image')
            G.add_edge(img_id, concept_level1)
            G.add_edge(concept_level1, concept_level2)
   else:
         G.add_node(img_id, type='image')
         G.add_edge(img_id, concept_level1)
# Step 1
coco = setup_coco('./dataset')
imgIds = coco.getImgIds()
G = nx.DiGraph()
concepts = {}
for img_id in imgIds:
   
   annIds = coco.getAnnIds(imgIds=img_id)
   anns = coco.loadAnns(annIds)
   anns = [ann['caption'] for ann in anns]
   concept_level1 = extract_output_content(get_concept_level1(anns)) # get captions form the annotations
   add_image_nodes(G, img_id, coco, concept_level1)

nx.write_graphml(G, "concept_hierarchy.graphml")

# Step 2
G = nx.read_graphml("concept_hierarchy.graphml")
coco = setup_coco('./dataset')

def merge_concepts(coco):
   level1_nodes = [n for n,d in G.nodes(data=True) if d['type']=='level1']
   concepts = {}
   for d in G.nodes(data=True):
      if d['type'] != 'image':
         concepts[d] = True
   for concept in level1_nodes:
      # Do BFS to count leaves (image nodes)
      images = G.successors(concept)
      if len(images) < 5:
         # Merge to parent
         parents = list(G.predecessors(parent))
         siblings = list(G.successors(parents[0]))
         while len(parents) > 0 and len(siblings) == 0:
            parent = parents[0]
            children = G.successors(parent)
            while len(children) > 0:
               for child in children:
                  if children['type']  == 'level1':
                     siblings.append(children)
                  else:
                     children.extend(list(G.successors(child)))

            parents = list(G.predecessors(parent))
         if len(siblings) == 0:
            G.remove_node(concept)
         else:
            smallest_sibling = siblings[0]
            for sibling in siblings:
               if len(G.successors(sibling)) < len(G.successors(smallest_sibling)):
                  smallest_sibling = sibling
            smallest_sibling_imags = G.successors(smallest_sibling)
            group1_aan = []
            for i in images:
               annIds = coco.getAnnIds(imgIds=i)
               anns = coco.loadAnns(annIds)
               anns = [ann['caption'] for ann in anns]
               group1_aan.extend(anns)

            group2_aan = []
            for i in smallest_sibling_imags:
               annIds = coco.getAnnIds(imgIds=i)
               anns = coco.loadAnns(annIds)
               anns = [ann['caption'] for ann in anns]
               group2_aan.extend(anns)

            new_concept = extract_output_content(get_new_merged_concept(concept, smallest_sibling, parents[0], group1_aan, group2_aan))
            
            for image in images:
               G.remove_edge(image, concept)
               G.remove_node(concept)
               add_image_nodes(G, image, coco, new_concept, concepts)
            for image in smallest_sibling_imags:
               G.remove_edge(image, smallest_sibling)
               G.remove_node(smallest_sibling)
               add_image_nodes(G, image, coco, new_concept, concepts)
         return True
      else:
         return False
               
found_small_node = merge_concepts(coco)
while found_small_node:
   found_small_node = merge_concepts(coco)

nx.write_graphml(G, "concept_hierarchy.graphml")

# Step 3
G = nx.read_graphml("concept_hierarchy.graphml")
level1_nodes = [n for n,d in G.nodes(data=True) if d['type']=='level1']
dataset= []
for concept in level1_nodes:
   images = G.successors(concept)
   group1_aan = []
   for i in images:
      annIds = coco.getAnnIds(imgIds=i)
      anns = coco.loadAnns(annIds)
      anns = [ann['caption'] for ann in anns]
      group1_aan.extend(anns)

   level2= G.predecessors(concept)[0]
   level3= G.predecessors(level2)[0]
   level4= G.predecessors(level3)[0]
   dataset.append({
      "images": images,
      "general_captions_level1": extract_multiple_output_content(get_general_captions(group1_aan, concept)),
      "general_captions_level2": extract_multiple_output_content(get_general_captions(group1_aan, level2)),
      "general_captions_level3": extract_multiple_output_content(get_general_captions(group1_aan, level3)),
      "general_captions_level4": extract_multiple_output_content(get_general_captions(group1_aan, level4)),
      "original_category": group1_aan,
      "level1": concept,
      "level2": level2,
      "level3": level3,
      "level4": level4
   })

json.dump(dataset, open('dataset.json', 'w'))
