from pycocotools.coco import COCO
import json
from collections import defaultdict
from typing import Dict, List, Any


class EnhancedCOCO(COCO):
    """
    Extends the COCO class to handle our custom concept levels
    """
    def __init__(self, annotation_file):
        super().__init__(annotation_file)
        
    def getCaptionWithConcepts(self, ann_id):
        """
        Get both caption and concept levels for an annotation
        """
        ann = self.anns[ann_id]
        return {
            'caption': ann['caption'],
            'concept_levels': ann['concept_levels'] 
        }
    
    def getImageCaptionsWithConcepts(self, img_id):
        """
        Get all captions and their concept levels for an image
        """
        ann_ids = self.getAnnIds(imgIds=img_id)
        return [self.getCaptionWithConcepts(ann_id) for ann_id in ann_ids]
# Example usage
def main():
    annotation_file = 'path/to/your/enhanced_captions.json'  # Update with your path
    coco = EnhancedCOCO(annotation_file)
    
    # Get all image IDs
    img_ids = coco.getImgIds()
    
    # Example: Get captions and concepts for first image
    first_img_id = img_ids[0]
    img_info = coco.loadImgs(first_img_id)[0]
    captions_with_concepts = coco.getImageCaptionsWithConcepts(first_img_id)
    
    print(f"Image {first_img_id} ({img_info['file_name']}):")
    for item in captions_with_concepts:
        print("\nCaption:", item['caption'])
        print("Concept Levels:")
        for level, concept in item['concept_levels'].items():
            print(f"  {level}: {concept}")

    # You can still use all standard COCO API functions
    # For example, getting annotation IDs for an image:
    ann_ids = coco.getAnnIds(imgIds=first_img_id)
    print(f"\nNumber of annotations for this image: {len(ann_ids)}")

if __name__ == "__main__":
    main()