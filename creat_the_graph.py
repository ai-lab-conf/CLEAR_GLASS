from pycocotools.coco import COCO
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import os
import re
from captions_loader import setup_coco
import networkx as nx
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate

def setup_coco(data_dir, data_type='train2017'):
    """
    Setup COCO dataset
    data_dir: root directory of COCO dataset
    data_type: 'train2017' or 'val2017'
    """
    annotation_file = os.path.join(data_dir, f'captions_{data_type}.json')
    coco = COCO(annotation_file)
    return coco

os.environ['OPENAI_API_KEY'] = "XXXX"

from openai import OpenAI
import os
from typing import Optional
import logging

def call_openai_api(prompt, model="gpt-4o-mini"):
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
            
        client = OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Error calling OpenAI API: {str(e)}")
        return None
    
#PROMPT of generating concept reduce the example to 5
PROMPT_1 = """

Definition:
A general concept is a broad idea or category that captures common attributes or qualities shared by multiple specific instances or objects, which may be concrete or abstract. It simplifies complex information by grouping similar instances together. For example: 1) The concept of “animal” encompasses all living organisms that can move and consume food. 2) The concept of “freedom” describes a state of being free from constraints or oppression, applicable in various social, political, and personal contexts.

Instruction:
Given a set of captions describing one single image, determine the first-level main general concept, so avoid being too general. The first-level concept should describe a general idea of the original caption. The concept should be a short phrase or a brief phrase that captures the general idea of the image (represented by the captions). Emphasize more on events or themes rather than objects or actions. Concepts MUST use canonical vocabulary (e.g. singular; present tense). Do not provide explanations. Write the concept in an <output> section.

Example 1:
Captions:
The tray on the bed has a pastry and two mugs on it.
A tray with coffee and a pastry on it.
A tray is full of breakfast foods and drinks on a bed.
coffee cream and a croissant on a tray
A tray on a bed with food and drink.
Output:
<output>Breakfast</output>

Example 2:
Captions:
Man in dress shirt and orange tie standing inside a building.
a male with a beard and orange tie
A man wearing a neck tie and a white shirt.
A man posing for the picture in a building
A man dressed in a shirt and tie standing in a lobby.
Output:
<output>Business Attire</output>

Example 3:
Captions:
Many small children are posing together in the black and white photo.
A vintage school picture of grade school aged children.
A black and white photo of a group of kids.
A group of children standing next to each other.
A group of children standing and sitting beside each other.
Output:
<output>Education</output>

Example 4:
Captions:
there is a very tall giraffe standing in the wild
A giraffe is standing by some brush in a field.
A giraffe standing in a dirt field near a tree.
A single giraffe standing in a brushy area looking at the photographer
A giraffe standing in dry dead brush on the savannah.
Output:
<output>Wildlife</output>

Example 5:
Captions:
Ornate wedding cake ready at the hotel reception
A tall multi layer cake sitting on top of a blue table cloth.
A wedding cake with flowers in a banquet hall.
Nicely decorated three tier wedding cake with topper.
A very ornate, three layered wedding cake in a banquet room.
Output:
<output>Wedding</output>
"""
PROMPT_2 = """
Example 1:
Captions:
The tray on the bed has a pastry and two mugs on it.
A tray with coffee and a pastry on it.
A tray is full of breakfast foods and drinks on a bed.
coffee cream and a croissant on a tray
A tray on a bed with food and drink.
Level1 Concept: Breakfast
Output:
<output>Meal</output>

Example 2:
Captions:
Man in dress shirt and orange tie standing inside a building.
a male with a beard and orange tie
A man wearing a neck tie and a white shirt.
A man posing for the picture in a building
A man dressed in a shirt and tie standing in a lobby.
Level1 Concept: Business Attire
Output:
<output>Business</output>

Example 3:
Captions:
Many small children are posing together in the black and white photo.
A vintage school picture of grade school aged children.
A black and white photo of a group of kids.
A group of children standing next to each other.
A group of children standing and sitting beside each other.
Level1 Concept: Education
Output:
<output>Learning</output>

Example 4:
Captions:
there is a very tall giraffe standing in the wild
A giraffe is standing by some brush in a field.
A giraffe standing in a dirt field near a tree.
A single giraffe standing in a brushy area looking at the photographer
A giraffe standing in dry dead brush on the savannah.
Level1 Concept: Wildlife
Output:
<output>Nature</output>

Example 5:
Captions:
Ornate wedding cake ready at the hotel reception
A tall multi layer cake sitting on top of a blue table cloth.
A wedding cake with flowers in a banquet hall.
Nicely decorated three tier wedding cake with topper.
A very ornate, three layered wedding cake in a banquet room.
Level1 Concept: Wedding
Output:
<output>Celebration</output>
"""
PROMPT_3 = """Example 1:
Captions:
The tray on the bed has a pastry and two mugs on it.
A tray with coffee and a pastry on it.
A tray is full of breakfast foods and drinks on a bed.
coffee cream and a croissant on a tray
A tray on a bed with food and drink.
Level1 Concept: Breakfast
Level2 Concept: Meal
Output:
<output>Food</output>

Example 2:
Captions:
Man in dress shirt and orange tie standing inside a building.
a male with a beard and orange tie
A man wearing a neck tie and a white shirt.
A man posing for the picture in a building
A man dressed in a shirt and tie standing in a lobby.
Level1 Concept: Business Attire
Level2 Concept: Business
Output:
<output>Career</output>

Example 3:
Captions:
Many small children are posing together in the black and white photo.
A vintage school picture of grade school aged children.
A black and white photo of a group of kids.
A group of children standing next to each other.
A group of children standing and sitting beside each other.
Level1 Concept: Education
Level2 Concept: Learning
Output:
<output>Personal Development</output>

Example 4:
Captions:
there is a very tall giraffe standing in the wild
A giraffe is standing by some brush in a field.
A giraffe standing in a dirt field near a tree.
A single giraffe standing in a brushy area looking at the photographer
A giraffe standing in dry dead brush on the savannah.
Level1 Concept: Wildlife
Level2 Concept: Nature
Output:
<output>Life</output>

Example 5:
Captions:
Ornate wedding cake ready at the hotel reception
A tall multi layer cake sitting on top of a blue table cloth.
A wedding cake with flowers in a banquet hall.
Nicely decorated three tier wedding cake with topper.
A very ornate, three layered wedding cake in a banquet room.
Level1 Concept: Wedding
Level2 Concept: Celebration
Output:
<output>Tradition</output>
"""
PROMPT_4 = """
Example 1:
Captions:
The tray on the bed has a pastry and two mugs on it.
A tray with coffee and a pastry on it.
A tray is full of breakfast foods and drinks on a bed.
coffee cream and a croissant on a tray
A tray on a bed with food and drink.
Level1 Concept: Breakfast
Level2 Concept: Meal
Level3 Concept: Food
Output:
<output>Nutrition</output>

Example 2:
Captions:
Man in dress shirt and orange tie standing inside a building.
a male with a beard and orange tie
A man wearing a neck tie and a white shirt.
A man posing for the picture in a building
A man dressed in a shirt and tie standing in a lobby.
Level1 Concept: Business Attire
Level2 Concept: Business
Level3 Concept: Career
Output:
<output>Personal Development</output>

Example 3:
Captions:
Many small children are posing together in the black and white photo.
A vintage school picture of grade school aged children.
A black and white photo of a group of kids.
A group of children standing next to each other.
A group of children standing and sitting beside each other.
Level1 Concept: Education
Level2 Concept: Learning
Level3 Concept: Personal Development
Output:
<output>Human Growth</output>

Example 4:
Captions:
there is a very tall giraffe standing in the wild
A giraffe is standing by some brush in a field.
A giraffe standing in a dirt field near a tree.
A single giraffe standing in a brushy area looking at the photographer
A giraffe standing in dry dead brush on the savannah.
Level1 Concept: Wildlife
Level2 Concept: Nature
Level3 Concept: Life
Output:
<output>Ecosystem</output>

Example 5:
Captions:
Ornate wedding cake ready at the hotel reception
A tall multi layer cake sitting on top of a blue table cloth.
A wedding cake with flowers in a banquet hall.
Nicely decorated three tier wedding cake with topper.
A very ornate, three layered wedding cake in a banquet room.
Level1 Concept: Wedding
Level2 Concept: Celebration
Level3 Concept: Tradition
Output:
<output>Cultural Heritage</output>
"""


def get_concept_level1(image_captions):
    prompt = PROMPT_1
    return call_openai_api(prompt + "\n".join(image_captions))
def get_concept_level2(image_captions, concept_level1):
   """
   Get concept level 2 from caption
   
   """
   prompt = PROMPT_2
   image_captions = "\n".join(image_captions)
   return call_openai_api(f""" 
Definition:
A general concept is a broad idea or category that captures common attributes or qualities shared by multiple specific instances or objects, which may be concrete or abstract. It simplifies complex information by grouping similar instances together. For example: 1) The concept of “animal” encompasses all living organisms that can move and consume food. 2) The concept of “freedom” describes a state of being free from constraints or oppression, applicable in various social, political, and personal contexts.

Instruction:
Given a set of captions describing one single image, determine the second-level main general concept. The second-level concept should describe a general idea of the first-level concept. The concept should be a short phrase or a brief phrase that captures the general idea of the image (represented by the captions). Emphasize more on events or themes rather than objects or actions. Concepts MUST use canonical vocabulary (e.g. singular; present tense). Do not provide explanations. Write the concept in an <output> section.

Examples:\n
{prompt}
Given Captions:\n
{image_captions}
                          
First Level Concept: {concept_level1}""")

def get_concept_level3(image_captions, concept_level1, concept_level2):
   """
   Get concept level 3 from caption
   """
   prompt = PROMPT_3
   image_captions = "\n".join(image_captions)
   return call_openai_api(f""" 
Definition:
A general concept is a broad idea or category that captures common attributes or qualities shared by multiple specific instances or objects, which may be concrete or abstract. It simplifies complex information by grouping similar instances together. For example: 1) The concept of “animal” encompasses all living organisms that can move and consume food. 2) The concept of “freedom” describes a state of being free from constraints or oppression, applicable in various social, political, and personal contexts.

Instruction:
Given a set of captions describing one single image, determine the third-level main general concept. The third-level concept should describe a general idea of the second-level concept. The concept should be a short phrase or a brief phrase that captures the general idea of the image (represented by the captions). Emphasize more on events or themes rather than objects or actions. Concepts MUST use canonical vocabulary (e.g. singular; present tense). Do not provide explanations. Write the concept in an <output> section.
Examples:\n
{prompt}
                          
Given Captions:\n
{image_captions}
First Level Concept: {concept_level1}
Second Level Concept: {concept_level2}                          
""")

def get_concept_level4(image_captions, concept_level1, concept_level2, concept_level3):
   """
   Get concept level 4 from caption
   """
   prompt = PROMPT_4
   image_captions = "\n".join(image_captions)
   return call_openai_api(f""" 
Definition:
A general concept is a broad idea or category that captures common attributes or qualities shared by multiple specific instances or objects, which may be concrete or abstract. It simplifies complex information by grouping similar instances together. For example: 1) The concept of “animal” encompasses all living organisms that can move and consume food. 2) The concept of “freedom” describes a state of being free from constraints or oppression, applicable in various social, political, and personal contexts.

Instruction:
Given a set of captions describing one single image, determine the fourth-level main general concept. The fourth-level concept should describe a general idea of the third-level concept. The concept should be a short phrase or a brief phrase that captures the general idea of the image (represented by the captions). Emphasize more on events or themes rather than objects or actions. Concepts MUST use canonical vocabulary (e.g. singular; present tense). Do not provide explanations. Write the concept in an <output> section.
Examples:\n
{prompt}
Given Captions:\n
{image_captions}

First Level Concept: {concept_level1}
Second Level Concept: {concept_level2}      
Third Level Concept: {concept_level3}                     
""")

def extract_output_content(text):
   """Extract content between output tags"""
   pattern = '<output>(.*?)</output>'
   match = re.search(pattern, text)
   return match.group(1) if match else None

def extract_multiple_output_content(text):
   """Extract content between output tags"""
   pattern = '<caption>(.*?)</caption>'
   matches = re.findall(pattern, text)
   return matches

data_dir = './dataset'
coco = setup_coco(data_dir)

def add_image_nodes(G, img_id, coco, concept_level1, concepts, root_node = "HOME_ROOT"):
    try:
        annIds = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(annIds)
        anns = [ann['caption'] for ann in anns]

        if not concept_level1:
            logging.error(f"Invalid concept_level1 for image {img_id}")
            return

        # Always need to add image node
        print(img_id)
        new_nodes = [(img_id, {'type': 'image'})]
        new_edges = []
        if root_node not in G:
            new_nodes.append((root_node, {'type': 'level5'}))
  

        if concept_level1 not in concepts:
            concepts[concept_level1] = True
            concept_level2 = extract_output_content(get_concept_level2(anns, concept_level1))
            
            # Add level1 node and edge
            new_nodes.append((concept_level1, {'type': 'level1'}))
            new_edges.append((img_id, concept_level1))
            
            if concept_level2 not in concepts:
                concepts[concept_level2] = True
                concept_level3 = extract_output_content(get_concept_level3(anns, concept_level1, concept_level2))
                
                # Add level2 node and edge
                new_nodes.append((concept_level2, {'type': 'level2'}))
                new_edges.append((concept_level1, concept_level2))
                
                if concept_level3 not in concepts:
                    concepts[concept_level3] = True
                    concept_level4 = extract_output_content(get_concept_level4(anns, concept_level1, concept_level2, concept_level3))
                    
                    # Add level3 node and edge
                    new_nodes.append((concept_level3, {'type': 'level3'}))
                    new_edges.append((concept_level2, concept_level3))
                    
                    if concept_level4 not in concepts:
                        concepts[concept_level4] = True
                        # Add level4 node and edge
                        new_nodes.append((concept_level4, {'type': 'level4'}))
                        new_edges.append((concept_level3, concept_level4))
                        # add root node
                        new_edges.append((concept_level4, root_node))
                    else:
                        # Connect to existing level4
                        new_edges.append((concept_level3, concept_level4))
                else:
                    # Connect to existing level3
                    new_edges.append((concept_level2, concept_level3))
            else:
                # Connect to existing level2
                new_edges.append((concept_level1, concept_level2))
        else:
            # Connect to existing level1
            new_edges.append((img_id, concept_level1))

        # Add all new nodes and edges at once
        G.add_nodes_from(new_nodes)
        G.add_edges_from(new_edges)

            
    except Exception as e:
        logging.error(f"Error processing image {img_id}: {str(e)}")

G = nx.DiGraph()
concepts = {}
imgIds = coco.getImgIds()

for img_id in imgIds:
   annIds = coco.getAnnIds(imgIds=img_id)
   anns = coco.loadAnns(annIds)
   anns = [ann['caption'] for ann in anns]
   concept_level1 = extract_output_content(get_concept_level1(anns)) # get captions form the annotations
   add_image_nodes(G, img_id, coco, concept_level1, concepts)

nx.write_graphml(G, "concept_hierarchy.graphml")
