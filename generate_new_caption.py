from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
def generate_level_captions(coco, img_id):

    # Get all captions for this image
    annIds = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(annIds)
    
    # Get all captions 
    captions = [ann['caption'] for ann in anns]
    concept_levels = anns[0]['concept_levels']
    
    # Generate a new general caption for each level
    level_captions = {}
    for level, concept in concept_levels.items():
        level_captions[level] = caption_generator(captions, concept)
    
    return level_captions

example_prompt = PromptTemplate.from_template("Question: {question}\n{answer}")

examples = [
    {
        "question": " ",
        "answer": """ """
    }
]


few_shot_prompt = FewShotPromptTemplate(
    examples=examples,  # List of examples
    example_prompt=example_prompt,
    suffix="Question: {input}",
    input_variables=["input"]
)

llm = ChatOpenAI(
    model_name="xxxx",
    temperature=0.7
)


def caption_generator(captions, concept):
    formatted_prompt = few_shot_prompt.invoke({"input": captions + concept})

    response = llm.invoke(formatted_prompt.to_string())
    
    return response.content

# The dataset structure will be:
# {
#     "images": [...],
#     "annotations": [
#         {
#             "id": int,
#             "image_id": int,
#             "caption": str, (multiple captions to desribe the image A), Cooking -> LLM -> paraphrase one specific caption realted to concept cooking)
#             "concept_levels": {
#                 "level1": "Cooking",
#                 "level2": "Food preparation",
#                 "level3": "Nourishment",
#                 "level4": "Sustenance"
#             },
#             "level_captions": {      
#                 "level1": "generated caption for level1", 
#                 "level2": "generated caption for level2", 
#                 "level3": "generated caption for level3",
#                 "level4": "generated caption for level4"
#             }
#         },
#         ...
#     ]
# }