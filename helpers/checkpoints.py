import os
from pathlib import Path
import torch
import wandb


class NoModelFoundException(Exception):
    def __init__(self, entity, project, runid, *args: object) -> None:
        super().__init__(*args)
        self.entity = entity
        self.project = project
        self.runid = runid

    def __str__(self):
        return f"Could not find any models locally or on wandb for {self.entity}/{self.project}/{self.runid}"

def get_model_path(entity, project, runid):

    #get artifact, to link it to this run
    api = wandb.Api()
    model_run = api.run(f"{entity}/{project}/{runid}")
    artifacts = model_run.logged_artifacts()
    artifact = None


    for a in artifacts:
        if a.type == 'model' and runid in a.name:
            artifact = a
            break
    if artifact is None:
        raise NoModelFoundException(entity, project, runid)
    
    wandb.use_artifact(artifact)

    local_path = os.path.join(project, runid, "checkpoints/last.ckpt")
    if os.path.exists(local_path):
        return local_path
    
    local_path = os.path.join(project, runid, "last.ckpt")
    if os.path.exists(local_path):
        return local_path
    
    local_path = os.path.join(project, runid, "model.ckpt")
    if os.path.exists(local_path):
        return local_path
    
    model_file_pattern = "*.pth"
    model_files = Path(os.path.join(project, runid)).rglob(model_file_pattern)
    model_files = list(model_files)
    if len(model_files) > 0:
        return model_files[0]
    
    #if the model ckpt file is not saved locally download it
    print(f"Downloading model of run {model_run.name} ({runid})")
    model_file = artifact.file(os.path.join(project, runid, "checkpoints"))

    return model_file

def get_state_dict(entity, project, runid):
    model_path = get_model_path(entity, project, runid)
    state_dict = torch.load(model_path)
    # handle ckpt file containing LightningModule not just model parameters
    if 'state_dict' in state_dict.keys():
        # remove "model." in front of the parameter keys
        return {k[6:]:v for k,v in state_dict['state_dict'].items() if k.startswith('model.')}
    return state_dict