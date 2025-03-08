import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import json
import random
from typing import List, Dict, Optional, Tuple, Union, Any
from pathlib import Path
from dataclasses import dataclass
from open_clip import tokenize
from PIL import Image
import os
from multiprocessing import Value
from tqdm import tqdm


def get_image_id_string(image_id: Union[int, str]) -> str:
    """
    Convert an image ID to COCO format string (12 digits with leading zeros).
    """
    if isinstance(image_id, int):
        image_id = str(image_id)
    image_id = ''.join(filter(str.isdigit, image_id))
    return image_id.zfill(12)

class PretrainDataset(Dataset):
    def __init__(
        self,
        data_path: Union[str, Path],
        image_path: Union[str, Path],
        preprocess_fn: Optional[callable] = None,
        tokenizer: Any = None,
        num_old_captions: int = 5,
        num_new_captions: int = 5,
        split: str = "train"
    ):
        """
        Dataset for caption pretraining that returns old and new captions.
        """
        self.data_path = Path(data_path)
        self.image_path = Path(image_path)
        self.preprocess_fn = preprocess_fn
        self.split = split
        self.tokenizer = tokenizer
        self.num_old_captions = num_old_captions
        self.num_new_captions = num_new_captions
        
        self.data = self._load_data()
        
    def _load_data(self) -> List[Dict]:
        """Load and process the JSON data."""
        with open(self.data_path, 'r') as f:
            raw_data = json.load(f)
            
        processed_data = []
        for concept_data in raw_data:
            images = (concept_data['train_images'] if self.split == 'train' 
                     else concept_data['test_images'])
            
            for image_data in images:
                for old_caption in image_data['old_captions']:
                    for new_caption in image_data['new_captions']:
                        processed_data.append({
                            'old_caption': old_caption,
                            'new_caption': new_caption,
                        })
                    if image_data["level1"]:
                        processed_data.append({
                                'old_caption': old_caption,
                                'new_caption': image_data["level1"],
                            })
                    if image_data["level2"]:
                        processed_data.append({
                                'old_caption': old_caption,
                                'new_caption': image_data["level2"],
                            })
                    if image_data["level3"]:
                        processed_data.append({
                                'old_caption': old_caption,
                                'new_caption': image_data["level3"],
                            })
                    if image_data["level4"]:
                        processed_data.append({
                                'old_caption': old_caption,
                                'new_caption': image_data["level4"],
                            })
        

                
        return processed_data
        
    def __getitem__(self, idx: int) -> Tuple[List[str], List[str]]:
        """
        Get a single sample from the dataset.
        
        Returns:
            tuple: (old_captions, new_captions)
        """
        item = self.data[idx]
        old_caption = item['old_caption']
        new_caption = item['new_caption']
        
        # Tokenize if tokenizer is provided
   
        if self.tokenizer:
            old_caption = self.tokenizer(str(old_caption))[0]

        if self.tokenizer:
            new_caption = self.tokenizer(str(new_caption))[0]
            
        return old_caption, new_caption

    def __len__(self) -> int:
        return len(self.data)

class ConceptPretrainDataset(Dataset):
    def __init__(
        self,
        data_path: Union[str, Path],
        image_path: Union[str, Path],
        preprocess_fn: Optional[callable] = None,
        tokenizer: Any = None,
        split: str = "train"
    ):
        """
        Dataset for concept pretraining that returns flattened concept-caption pairs.
        """
        self.data_path = Path(data_path)
        self.image_path = Path(image_path)
        self.preprocess_fn = preprocess_fn
        self.split = split
        self.tokenizer = tokenizer
        
        self.data = self._load_data()
        
    def _load_data(self) -> List[Dict]:
        """
        Load and process the JSON data into flattened concept-caption pairs.
        Each item will be a single concept (from any level) paired with a caption.
        """
        with open(self.data_path, 'r') as f:
            raw_data = json.load(f)
            
        processed_data = []
        for concept_data in raw_data:
            images = (concept_data['train_images'] if self.split == 'train' 
                     else concept_data['test_images'])
            
            for image_data in images:
                # Get all concepts from the hierarchy
                concepts = [
                    image_data['level1'],
                    image_data['level2'],
                    image_data['level3'],
                    image_data['level4']
                ]
                
                # For each new caption, create pairs with each concept level
                for new_caption in image_data['new_captions']:
                    for old_caption in image_data["old_captions"]:
                        processed_data.append({
                            'concept': old_caption,
                            'caption': new_caption
                        })
            
                    for concept in concepts:
                        processed_data.append({
                            'concept': concept,
                            'caption': new_caption
                        })
                
        return processed_data
        
    def __getitem__(self, idx: int) -> Tuple[str, str]:
        """
        Get a single concept-caption pair.
        
        Returns:
            tuple: (concept, caption) - both tokenized if tokenizer is provided
        """
        item = self.data[idx]
        concept = item['concept']
        caption = item['caption']
        
        # Tokenize if tokenizer is provided
        if self.tokenizer:
            concept = self.tokenizer(str(concept))[0]
            caption = self.tokenizer(str(caption))[0]
            
        return concept, caption

    def __len__(self) -> int:
        return len(self.data)

class TrainConceptDataset(Dataset):
    def __init__(
        self,
        data_path: Union[str, Path],
        image_path: Union[str, Path],
        preprocess_fn: Optional[callable] = None,
        tokenizer: Any = None,
        group_size: int = 4,
        split: str = "train",
        batch_size: int = 1
    ):
        self.batch_size = batch_size
        self.data_path = Path(data_path)
        self.image_path = Path(image_path)
        self.preprocess_fn = preprocess_fn
        self.split = split
        self.group_size = group_size
        self.tokenizer = tokenizer
        self.data  = self._load_data_train()
    
    def _load_image(self, image_id: str) -> torch.Tensor:
        """
        Load and preprocess an image.
        """
        image_id = get_image_id_string(image_id)
        image_path = self.image_path / f"{image_id}.jpg"
        
        try:
            image = Image.open(image_path).convert('RGB')
            if self.preprocess_fn is not None:
                image = self.preprocess_fn(image)
            return image
        except Exception as e:
            print(f"Error loading image {image_id}: {str(e)}")
            if self.preprocess_fn is not None:
                return torch.zeros(3, 224, 224)
            return Image.new('RGB', (224, 224))

    
    
    def _load_data_train(self) -> List[Dict]:
        """
        Load and process the JSON data to create groups of size N by concept.
        
        Returns:
            List[Dict]: A list of processed data groups, each containing items and valid hard negatives.
        """
        # Load raw data from JSON file
        with open(self.data_path, 'r') as f:
            raw_data = json.load(f)
                
        processed_data_initial = []
        
        # Process each concept separately
        for concept_data in raw_data:
            # Skip concepts with insufficient hard negatives early
            if len(concept_data.get('hard_negative', [])) < self.batch_size - 1:
                continue
                
            concept_captions = []
            images = (concept_data['train_images'] if self.split == 'train' 
                    else concept_data['test_images'])
            
            # Collect all captions for this concept
            for image_data in images:
                image_id = image_data.get('image')
                if not image_id:
                    continue
                    
                for caption in image_data.get('new_captions', []):
                    if caption:  # Only add non-empty captions
                        concept_captions.append({
                            "image_id": image_id,
                            "caption": caption
                        })
            
            # Create groups of size N
            for i in range(0, len(concept_captions), self.group_size):
                if i + self.group_size <= len(concept_captions):  # Check if there are enough items for a complete group
                    group = {
                        "hard_negatives": concept_data.get('hard_negative', []),
                        "concept": concept_data.get('concept', ''),
                        "items": concept_captions[i:i+self.group_size]
                    }
                    
                    # Only add complete groups with sufficient hard negatives
                    if len(group["items"]) == self.group_size and len(group["hard_negatives"]) >= self.batch_size - 1:
                        processed_data_initial.append(group)
        
        # Build a lookup dictionary for faster validation of hard negatives
        concept_lookup = {group['concept']: group for group in processed_data_initial if group['concept']}
        
        # Validate hard negatives
        processed_data = []
        for group in tqdm(processed_data_initial, desc="Validating hard negatives for training"):
            # Only keep hard negatives that exist as concepts in the dataset
            valid_hard_negatives = [
                hard_neg for hard_neg in group['hard_negatives']
                if hard_neg in concept_lookup
            ]
            
            # Update group with validated hard negatives
            group['hard_negatives'] = valid_hard_negatives
            
            # Only include groups with sufficient valid hard negatives
            if len(valid_hard_negatives) >= self.batch_size - 1:
                processed_data.append(group)

        return processed_data

    def __len__(self) -> int:
        return len(self.data)
    
    def _process_group_items(self,group):
        
        # Load images and captions
        images = [self._load_image(item['image_id']) for item in group["items"]]
        captions = [item['caption'] for item in group["items"]]

        # Process with tokenizer if available
        if self.tokenizer:
            processed_captions = []
            for caption in captions:
                tokenizer_output = self.tokenizer(str(caption))
                if hasattr(tokenizer_output, 'input_ids'):
                    processed_captions.append(tokenizer_output.input_ids)
                else:
                    processed_captions.append(tokenizer_output[0])
            captions = processed_captions

        return images, captions


    def __getitem__(self, idx: int) -> Tuple[List[torch.Tensor], List[str]]:
        """
        Get a group sample from the dataset.

        """
        group = self.data[idx]
        if len(group["hard_negatives"]) < self.batch_size - 1:
            return [], []
        images, captions = self._process_group_items(group)
        hard_negatives = random.choices(group["hard_negatives"], k=self.batch_size - 1) # list of hard negative concepts

        batch_images = []
        batch_captions = []
        batch_images.append(images)
        batch_captions.append(captions)

        for hard_negative in hard_negatives:
            hard_negative_group = None
            for other_group in self.data:
                if hard_negative == other_group["concept"]:
                    hard_negative_group = other_group
                    break
            if hard_negative_group is None:
                #TODO: LOG
                return [], []
            else:
                other_group_images, other_group_captions = self._process_group_items(hard_negative_group)
                batch_images.append(other_group_images)
                batch_captions.append(other_group_captions)
            
        return batch_images, batch_captions


               
class EvaluationDataset(Dataset):
    def __init__(
        self,
        data_path: Union[str, Path],
        image_path: Union[str, Path],
        preprocess_fn: Optional[callable] = None,
        tokenizer: Any = None,
        split: str = "test", 
        eval_type: str = "new_caption", #TODO: parameter
        batch_size: int = 1
        
    ):
        self.data_path = Path(data_path)
        self.image_path = Path(image_path)
        self.preprocess_fn = preprocess_fn
        self.tokenizer = tokenizer
        self.split = split
        self.eval_type = eval_type
        self.batch_size = batch_size

        
        # Load and process the data
        self.data = self._load_data()
        
    def _load_data(self):
        """
        Load and process the JSON data.
        
        Returns:
            list: Processed data entries that have sufficient valid hard negatives.
        """
        # Load raw data from JSON file
        with open(self.data_path, 'r') as f:
            raw_data = json.load(f)
        
        
        processed_data_initial = []
        for concept_data in raw_data:
            # Select train or test images based on split
            images = (concept_data['train_images'] if self.split == 'train' 
                    else concept_data['test_images'])
            
            # Skip if not enough hard negatives for this concept
            if len(concept_data.get("hard_negative", [])) < self.batch_size - 1:
                continue
                
            for image_data in images:
                # Skip images without level1 concept
                if image_data.get('level1') is None:
                    continue
                    
                # Extract data and add to initial processing list
                processed_data_initial.append({
                    'image_id': image_data['image'],
                    # Take first caption or empty string if none available
                    'new_caption': image_data.get('new_captions', [""])[0] if image_data.get('new_captions') else "",
                    'old_caption': image_data.get('old_captions', [""])[0] if image_data.get('old_captions') else "",
                    # Extract concept levels with safe defaults
                    'concept_level1': image_data.get('level1', ""),
                    'concept_level2': image_data.get('level2', ""),
                    'concept_level3': image_data.get('level3', ""),
                    'concept_level4': image_data.get('level4', ""),
                    'hard_negatives': concept_data.get('hard_negative', [])
                })
        
        # Build a lookup dictionary for faster validation
        concept_lookup = {item['concept_level1']: item for item in processed_data_initial if item['concept_level1']}
        
        # Second-pass processing: validate hard negatives
        processed_data = []
        for entry in tqdm(processed_data_initial, desc="Validating hard negatives for evaluation"):
            # Only keep hard negatives that exist as concept_level1 in the dataset
            valid_hard_negatives = [
                hard_neg for hard_neg in entry['hard_negatives']
                if hard_neg in concept_lookup
            ]
            
            # Update entry with validated hard negatives
            entry['hard_negatives'] = valid_hard_negatives
            
            # Only include entries with sufficient valid hard negatives
            if len(valid_hard_negatives) >= self.batch_size - 1:
                processed_data.append(entry)
        
        return processed_data
        
    def _load_image(self, image_id: str) -> torch.Tensor:
        """Load and preprocess an image."""
        image_id = get_image_id_string(image_id)
        image_path = self.image_path / f"{image_id}.jpg"
        
        try:
            image = Image.open(image_path).convert('RGB')
            if self.preprocess_fn is not None:
                image = self.preprocess_fn(image)
            return image
        except Exception as e:
            print(f"Error loading image {image_id}: {str(e)}")
            if self.preprocess_fn is not None:
                return torch.zeros(3, 224, 224)
            return Image.new('RGB', (224, 224))
    
    def __len__(self):
        return len(self.data)

    def _process_item(self, item):
        image = self._load_image(item['image_id'])
        
        # Get text items and tokenize if needed
        eval_type_item = item[self.eval_type]
        
        # Handle different tokenizer types
        if self.tokenizer:
            # Get tokenizer output
            tokenizer_output = self.tokenizer(str(eval_type_item))
            
            # Check output type and handle accordingly
            if hasattr(tokenizer_output, 'input_ids'):
                # HuggingFace tokenizer output with input_ids attribute
                caption = tokenizer_output.input_ids
            else:
                caption = tokenizer_output[0]
        
        return image, caption

    
    def __getitem__(self, idx):
        item = self.data[idx]
        if len(item["hard_negatives"]) < self.batch_size - 1:
            return [], []
        image, caption = self._process_item(item)
        if caption is None:
            return [], []
        hard_negatives = random.choices(item["hard_negatives"], k=self.batch_size-1) # list of hard negative concepts

        batch_images = []
        batch_captions = []
        batch_images.append([image])
        batch_captions.append([caption])

        for hard_negative in hard_negatives:
            hard_negative_item = None
            for other_item in self.data:
                if hard_negative == other_item["concept_level1"]:
                    hard_negative_item = other_item
                    break
            if hard_negative_item is None:
                #TODO: LOG
                return [], []
            else:
                other_item_image, other_item_caption = self._process_item(hard_negative_item)
                batch_images.append([other_item_image])
                batch_captions.append([other_item_caption])
        return batch_images, batch_captions

        
class HierarcapsEvaluationDataset(Dataset):
    def __init__(
        self,
        data_path: Union[str, Path],
        image_path: Union[str, Path],
        preprocess_fn: Optional[callable] = None,
        tokenizer: Any = None,
        split: str = "test", 
        eval_type: str = "new_caption", #TODO: parameter
        batch_size: int = 1
        
    ):
        self.data_path = Path(data_path)
        self.image_path = Path(image_path)
        self.preprocess_fn = preprocess_fn
        self.tokenizer = tokenizer
        self.split = split
        self.eval_type = eval_type
        self.batch_size = batch_size

        
        # Load and process the data
        self.data = self._load_data()
        
    def _load_data(self):
        """
        Load and process the JSON data.
        
        Returns:
            list: Processed data entries that have sufficient valid hard negatives.
        """
        # Load raw data from JSON file
        with open(self.data_path, 'r') as f:
            raw_data = json.load(f)
        
        
        processed_data_initial = []
        for concept_data in raw_data:
            # Select train or test images based on split
            images = concept_data["images"]
            for image_id in images:
                # Skip images without level1 concept
                # Extract data and add to initial processing list
                processed_data_initial.append({
                    'image_id': image_id,
                    # Take first caption or empty string if none available
                    # Extract concept levels with safe defaults
                    'concept_level1': random.choice(concept_data["general_captions_level1"]),
                    'concept_level2': random.choice(concept_data["general_captions_level2"]),
                    'concept_level3': random.choice(concept_data["general_captions_level3"]),
                    'concept_level4': random.choice(concept_data["general_captions_level4"]),
                    'hard_negatives': [concept["concept"] for concept in random.choices(raw_data, k = self.batch_size - 1)]
                })
        
        return processed_data_initial
        
    def _load_image(self, image_id: str) -> torch.Tensor:
        """Load and preprocess an image."""
        image_id = get_image_id_string(image_id)
        image_path = self.image_path / f"{image_id}.jpg"
        
        try:
            image = Image.open(image_path).convert('RGB')
            if self.preprocess_fn is not None:
                image = self.preprocess_fn(image)
            return image
        except Exception as e:
            print(f"Error loading image {image_id}: {str(e)}")
            if self.preprocess_fn is not None:
                return torch.zeros(3, 224, 224)
            return Image.new('RGB', (224, 224))
    
    def __len__(self):
        return len(self.data)

    def _process_item(self, item):
        image = self._load_image(item['image_id'])
        
        # Get text items and tokenize if needed
        eval_type_item = item[self.eval_type]
        
        # Handle different tokenizer types
        if self.tokenizer:
            # Get tokenizer output
            tokenizer_output = self.tokenizer(str(eval_type_item))
            
            # Check output type and handle accordingly
            if hasattr(tokenizer_output, 'input_ids'):
                # HuggingFace tokenizer output with input_ids attribute
                caption = tokenizer_output.input_ids
            else:
                caption = tokenizer_output[0]
        
        return image, caption

    
    def __getitem__(self, idx):
        item = self.data[idx]
        if len(item["hard_negatives"]) < self.batch_size - 1:
            return [], []
        image, caption = self._process_item(item)
        if caption is None:
            return [], []
        hard_negatives = item["hard_negatives"]

        batch_images = []
        batch_captions = []
        batch_images.append([image])
        batch_captions.append([caption])

        for hard_negative in hard_negatives:
            hard_negative_item = None
            for other_item in self.data:
                if hard_negative == other_item["concept_level4"]:
                    hard_negative_item = other_item
                    break
            if hard_negative_item is None:
                #TODO: LOG
                return [], []
            else:
                other_item_image, other_item_caption = self._process_item(hard_negative_item)
                batch_images.append([other_item_image])
                batch_captions.append([other_item_caption])
        return batch_images, batch_captions

class BreedsEvaluationDataset(Dataset):
    def __init__(
        self,
        data_path1: Union[str, Path],
        data_path2: Union[str, Path],
        image_path: Union[str, Path],
        preprocess_fn: Optional[callable] = None,
        tokenizer: Any = None,
        split: str = "test", 
        eval_type: str = "new_caption", #TODO: parameter
        batch_size: int = 1
        
    ):
        self.data_path1 = Path(data_path1)
        self.data_path2 = Path(data_path2)
        self.image_path = Path(image_path)
        self.preprocess_fn = preprocess_fn
        self.tokenizer = tokenizer
        self.split = split
        self.eval_type = eval_type
        self.batch_size = batch_size

        
        # Load and process the data
        self.data = self._load_data()
        
    def _load_data(self):
        """
        Load and process the JSON data.
        
        Returns:
            list: Processed data entries that have sufficient valid hard negatives.
        """
        # Load raw data from JSON file
        with open(self.data_path1, 'r') as f:
            raw_data1 = json.load(f)
        with open(self.data_path2, 'r') as f:
            raw_data2 = json.load(f)
        combined_raw_data = raw_data1 + raw_data2
        processed_data_initial = []
        self.load_data_impel(combined_raw_data, processed_data_initial)
        return processed_data_initial
    
    def load_data_impel(self,raw_data, processed_data_initial):
        for concept_data in raw_data:
            # Select train or test images based on split
            images = concept_data["images"]
        
            for image_id in images:
                # Skip images without level1 concept
                # Extract data and add to initial processing list
                processed_data_initial.append({
                    'image_id': image_id,
                    # Take first caption or empty string if none available
                    # Extract concept levels with safe defaults
                    'concept_level1': random.choice(concept_data["general_captions_level1"]),
                    'concept_level2': concept_data["level1"],
                    'concept_level3': concept_data["level2"],
                    'concept_level4': concept_data["level3"],
                    'hard_negatives': [concept["concept"] for concept in random.choices(raw_data, k = self.batch_size - 1)]
                })
        
        
        
    def _load_image(self, image_id: str) -> torch.Tensor:
        """Load and preprocess an image."""
        # Handle ImageNet-style image IDs (e.g., n01631663_2291)
        
        # Check for ImageNet format directory structure (e.g., n01631663/n01631663_2291.JPEG)
        class_id = image_id.split('_')[0]  # Extract class ID (e.g., n01631663)
        image_path = self.image_path / class_id / f"{image_id}.JPEG"
        
        # If file doesn't exist, try alternate extensions and locations
        if not image_path.exists():
            print(f"the given image_path {image_path} does not exist")
        
        try:
            image = Image.open(image_path).convert('RGB')
            if self.preprocess_fn is not None:
                image = self.preprocess_fn(image)
            return image
        except Exception as e:
            print(f"Error loading image {image_id} from {image_path}: {str(e)}")
            if self.preprocess_fn is not None:
                return torch.zeros(3, 224, 224)
            return Image.new('RGB', (224, 224))
    
    def __len__(self):
        return len(self.data)

    def _process_item(self, item):
        image = self._load_image(item['image_id'])
        
        # Get text items and tokenize if needed
        eval_type_item = item[self.eval_type]
        
        # Handle different tokenizer types
        if self.tokenizer:
            # Get tokenizer output
            tokenizer_output = self.tokenizer(str(eval_type_item))
            
            # Check output type and handle accordingly
            if hasattr(tokenizer_output, 'input_ids'):
                # HuggingFace tokenizer output with input_ids attribute
                caption = tokenizer_output.input_ids
            else:
                caption = tokenizer_output[0]
        
        return image, caption

    def __getitem__(self, idx):
        item = self.data[idx]
        if len(item["hard_negatives"]) < self.batch_size - 1:
            return [], []
        image, caption = self._process_item(item)
        if caption is None:
            return [], []
        hard_negatives = item["hard_negatives"] # list of hard negative concepts

        batch_images = []
        batch_captions = []
        batch_images.append([image])
        batch_captions.append([caption])

        for hard_negative in hard_negatives:
            hard_negative_item = None
            for other_item in self.data:
                if hard_negative == other_item["concept_level2"]:
                    hard_negative_item = other_item
                    break
            if hard_negative_item is None:
                #TODO: LOG
                return [], []
            else:
                other_item_image, other_item_caption = self._process_item(hard_negative_item)
                batch_images.append([other_item_image])
                batch_captions.append([other_item_caption])
        return batch_images, batch_captions
        

# def evaluation_collate_fn(batch):


def train_collate_fn(batch):
    """
    Collate function for training that handles batches of images and captions.
    """
    images_batch = []
    captions_batch = []
    
    for images, captions in batch:
        for idx,image in enumerate(images):
            images_batch.extend(images[idx])
            captions_batch.extend(captions[idx])
    
    if len(images_batch) > 0:
        if isinstance(images_batch[0], torch.Tensor):
            images_batch = torch.stack(images_batch)
        
        if isinstance(captions_batch[0], torch.Tensor):
            captions_batch = torch.stack(captions_batch)
    else:
        a = 1
        
    return images_batch, captions_batch


def pretrain_collate_fn(batch):
    """
    Collate function for concept pretraining that handles batches of concept-caption pairs.
    """
    concepts_batch = []
    captions_batch = []
    
    for old_caption, new_caption in batch:
        concepts_batch.append(old_caption)
        captions_batch.append(new_caption)
    
    if isinstance(concepts_batch[0], torch.Tensor):
        concepts_batch = torch.stack(concepts_batch)
        captions_batch = torch.stack(captions_batch)
        
    return concepts_batch, captions_batch



class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value

@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)

def get_dataset(
    args: Any,
    data_path: Path,
    image_path: Path,
    preprocess_fn: callable,
    tokenizer: Any,
    is_train: bool = True,
    dataset_type: str = "caption_pretrain",
    epoch: int = 0,
    eval_type: Optional[str] = None,
    data_path2: Optional[Union[str, Path]] = "/users/mali37/ConceptAbstraction/imagenet_dataset_non_living.json"
) -> DataInfo:
    """
    Create a dataset and dataloader for concept data.
    """
    split = "train" if is_train else "test"
    data_loader_batch_size = args.batch_size
    if dataset_type == "evaluation":
        print("dataset_type == evaluation")
        data_loader_batch_size = 1
        current_eval_type = eval_type if eval_type is not None else args.eval_type
        print(f"Using eval_type: {current_eval_type}")
        dataset = EvaluationDataset(
            data_path=data_path,
            image_path=image_path,
            preprocess_fn=preprocess_fn,
            tokenizer=tokenizer,
            split=split,
            batch_size=args.batch_size,
            eval_type=current_eval_type
        )
        collate_fn = train_collate_fn
    elif dataset_type == "hierarcaps":
        print("dataset_type == hierarcpas")
        data_loader_batch_size = 1
        current_eval_type = eval_type if eval_type is not None else args.eval_type
        print(f"Using eval_type: {current_eval_type}")
        dataset = HierarcapsEvaluationDataset(
            data_path=data_path,
            image_path=image_path,
            preprocess_fn=preprocess_fn,
            tokenizer=tokenizer,
            split=split,
            batch_size=args.batch_size,
            eval_type=current_eval_type
        )
        collate_fn = train_collate_fn
    elif dataset_type == "breeds":
        print("dataset_type == breeds")
        data_loader_batch_size = 1
        current_eval_type = eval_type if eval_type is not None else args.eval_type
        print(f"Using eval_type: {current_eval_type}")
        dataset = BreedsEvaluationDataset(
            data_path1=data_path,
            data_path2=data_path2,
            image_path=image_path,
            preprocess_fn=preprocess_fn,
            tokenizer=tokenizer,
            split=split,
            batch_size=args.batch_size,
            eval_type=current_eval_type
        )
        collate_fn = train_collate_fn
    elif dataset_type == "pretrain":
        dataset_class = PretrainDataset
        dataset = dataset_class(
            data_path=data_path,
            image_path=image_path,
            preprocess_fn=preprocess_fn,
            tokenizer=tokenizer,
            num_old_captions=args.num_old_captions,
            num_new_captions=args.num_new_captions,
            split=split
        )
        collate_fn = pretrain_collate_fn
    elif dataset_type == "train":
        data_loader_batch_size = 1
        dataset_class = TrainConceptDataset
        dataset = dataset_class(
            data_path=data_path,
            image_path=image_path,
            preprocess_fn=preprocess_fn,
            tokenizer=tokenizer,
            group_size=args.group_size,
            split=split,
            batch_size=args.batch_size
        )
        collate_fn = train_collate_fn
    else:
        dataset_class = GeneralConceptDataset
        dataset = dataset_class(
            data_path=data_path,
            image_path=image_path,
            preprocess_fn=preprocess_fn,
            group_size=args.group_size,
            split=split
        )
        collate_fn = None
    
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None
    
    dataloader = DataLoader(
        dataset,
        batch_size=data_loader_batch_size,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=is_train,
        collate_fn=collate_fn
    )
    
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)
    
    return DataInfo(dataloader=dataloader, sampler=sampler)

def get_train_data(
    args: Any,
    preprocess_fn: callable,
    tokenizer: Any,
    dataset_type: str = "train",
    epoch: int = 0,
) -> Dict[str, DataInfo]:
    """
    Get training data loader.
    
    Args:
        args: Arguments containing training parameters
        preprocess_fn: Function to preprocess images
        tokenizer: Tokenizer function
        dataset_type: Type of dataset to create
        epoch: Current epoch number
        
    Returns:
        Dictionary containing training data loader
    """
    data = {}
    data_key = f"train_{dataset_type}"
    
    data[data_key] = get_dataset(
        args=args,
        data_path=args.captions,
        image_path=args.images,
        preprocess_fn=preprocess_fn,
        tokenizer=tokenizer,
        is_train=True,
        dataset_type=dataset_type,
        epoch=epoch
    )
    
    return data

def get_val_data(
    args: Any,
    preprocess_fn: callable,
    tokenizer: Any,
    epoch: int = 0,
    dataset_type: Optional[str] = None,
    eval_type: Optional[str] = None
) -> Dict[str, DataInfo]:
    """
    Get evaluation data loader.
    
    Args:
        args: Arguments containing evaluation parameters
        preprocess_fn: Function to preprocess images
        tokenizer: Tokenizer function
        epoch: Current epoch number
        eval_type: Type of evaluation to perform (overrides args.eval_type if provided)
        
    Returns:
        Dictionary containing evaluation data loader
    """
    data = {}
    
    dataset_type = dataset_type if dataset_type is not None else args.dataset_type
    data_key = f"val_{dataset_type}"
    data[data_key] = get_dataset(
        args=args,
        data_path=args.captions,
        data_path2=args.captions2,
        image_path=args.images,
        preprocess_fn=preprocess_fn,
        tokenizer=tokenizer,
        is_train=False,
        dataset_type=dataset_type,
        epoch=epoch,
        eval_type=eval_type
    )
    
    return data