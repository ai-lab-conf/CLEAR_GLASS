import os
import sys
import yaml
from training.main import main
import wandb



# config_path = os.path.join("/users/mali37/ConceptAbstraction/src/training", "sweep_configuration.yml")
# config_path = os.path.join("/users/mali37/ConceptAbstraction/src/training", "pretrain_configuration.yml")
config_path = os.path.join("/users/mali37/ConceptAbstraction/src/training", "evaluation_sweep.yml")

with open(config_path, "r") as file:
    sweep_config = yaml.safe_load(file)

# Specify your team name (entity) here
team_name = "conversational-ai-lab-2"

# Create the sweep under the specified team
sweep_id = wandb.sweep(
    sweep=sweep_config, 
    # project="Evaluationing Finetuned Models on Breeds",
    project="Evaluationing Finetuned Hierarcaps on Hierarcaps",
    entity=team_name
)

# The agent will also use the same team
wandb.agent(sweep_id, function=main, count=100)