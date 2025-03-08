import argparse
import os
import sys

def get_default_params(model_name):
    # Params from paper (https://arxiv.org/pdf/2103.00020.pdf)
    model_name = model_name.lower()
    if "vit" in model_name:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.98, "eps": 1.0e-6}
    else:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.999, "eps": 1.0e-8}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--images",
        type=str,
        default="",
        help="train image directory",
    )
    parser.add_argument(
        "--captions",
        type=str,
        default = ""
    )
    parser.add_argument(
        "--captions2",
        type=str,
        default= ""
    )
    parser.add_argument(
    "--num_old_captions",
    type=int,
    default=3,
    help="Number of old captions to use per sample"
)
    parser.add_argument(
        "--num_new_captions",
        type=int,
        default=3,
        help="Number of new captions to use per sample"
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        default='train',
        choices=['pretrain', 'train', 'evaluation', "hierarcaps", "breeds"],
        help="Type of dataset to use: caption_pretrain for old/new captions, concept_pretrain for concept/caption pairs, or train for grouped data"
    )
    parser.add_argument(
        "--group_size", type=int, default=3, help=" "
    )

    parser.add_argument(
        "--logs",
        type=str,
        default="./logs/",
        help="Where to store tensorboard logs. Use None to avoid storing logs.",
    )
    parser.add_argument(
        "--log-local",
        action="store_true",
        default=False,
        help="log files on local master, otherwise global master only.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Optional identifier for the experiment when storing logs. Otherwise use current time.",
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of dataloader workers per GPU."
    )
    parser.add_argument(
        "--batch-size", type=int, default=4, help="Batch size per GPU." #number of hard negatives - 1
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of epochs to train for."
    )
    parser.add_argument(
        "--alpha", type=float, default=0.5, help="Weight for CLIP loss term"
    )
    parser.add_argument(
        "--eval_type", type=str, default="concept_level1", help=""
    )
    parser.add_argument(
        "--tau", type=float, default=0.07, help=""
    )
    parser.add_argument(
        "--evaluate-trained-models", type=bool, default=True, help=" "
    )
    parser.add_argument(
        "--wandb-sweep", type=bool, default=True, help=" "
    )
    # parser.add_argument(
    #     "--test-random-captions", type=bool, default=False, help=" "
    # )
    parser.add_argument(
        "--loss", type=int, default=3, help="1 = scaled_pairwise_loss, 2 = scaled_centroid_based_loss, 3= clip_loss"
    )

    parser.add_argument("--lr", type=float, default=1.8748186634898305e-05, help="Learning rate.")
    parser.add_argument("--beta1", type=float, default=None, help="Adam beta 1.")
    parser.add_argument("--beta2", type=float, default=None, help="Adam beta 2.")
    parser.add_argument("--eps", type=float, default=None, help="Adam epsilon.")
    parser.add_argument("--wd", type=float, default=4.5269353639276774e-05, help="Weight decay.")
    parser.add_argument(
        "--warmup", type=int, default= 1430, help="Number of steps to warmup for."
    )
    parser.add_argument(
        "--use-bn-sync",
        default=False,
        action="store_true",
        help="Whether to use batch norm sync.")
    parser.add_argument(
        "--skip-scheduler",
        action="store_true",
        default=False,
        help="Use this flag to skip the learning rate decay.",
    )
    parser.add_argument(
        "--save-frequency", type=int, default=1, help="How often to save checkpoints."
    )
    parser.add_argument(
        "--save-most-recent",
        action="store_true",
        default=True,
        help="Always save the most recent model trained to epoch_latest.pt.",
    )
    parser.add_argument(
        "--val-frequency", type=int, default=1, help="How often to run evaluation with val data."
    )
    parser.add_argument(
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "--force-quick-gelu",
        default=False,
        action='store_true',
        help="Force use of QuickGELU activation for non-OpenAI transformer models.",
    )
    parser.add_argument(
        "--precision",
        choices=["amp", "fp16", "fp32"],
        default="amp",
        help="Floating point precision."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="hierarcaps",
        help="Name of the vision backbone to use.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        # default="/users/mali37/scratch/conceptAbstract/pretrain/-model_open_clip-b_64-gs_3-e_7-loss_1",
        default = None,
        help="path of the trained model you want to evaluate",
    )
    parser.add_argument(
        "--pretrained",
        default='openai',
        type=str,
        help="Use a pretrained CLIP model weights with the specified tag or file path.",
    )
    parser.add_argument(
        "--pretrained-image",
        default=False,
        action='store_true',
        help="Load imagenet pretrained weights for image tower backbone if available.",
    )
    parser.add_argument(
        "--lock-image",
        default=False,
        action='store_true',
        help="Lock full image tower by disabling gradients.",
    )
    parser.add_argument(
        "--lock-image-unlocked-groups",
        type=int,
        default=0,
        help="Leave last n image tower layer groups unlocked.",
    )
    parser.add_argument(
        "--lock-image-freeze-bn-stats",
        default=False,
        action='store_true',
        help="Freeze BatchNorm running stats in image tower for any locked layers.",
    )
    parser.add_argument(
        "--grad-checkpointing",
        default=False,
        action='store_true',
        help="Enable gradient checkpointing.",
    )
    parser.add_argument(
        "--local-loss",
        default=False,
        action="store_true",
        help="calculate loss w/ local features @ global (instead of realizing full global @ global matrix)"
    )
    parser.add_argument(
        "--gather-with-grad",
        default=True,
        action="store_true",
        help="enable full distributed gradient for feature gather"
    )
    parser.add_argument(
        "--torchscript",
        default=False,
        action='store_true',
        help="torch.jit.script the model, also uses jit version of OpenAI models if pretrained=='openai'",
    )
    parser.add_argument(
        "--trace",
        default=False,
        action='store_true',
        help="torch.jit.trace the model for inference / eval only",
    )
    # arguments for distributed training
    parser.add_argument(
        "--dist-url",
        default='env://',
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--report-to",
        default='wandb',
        type=str,
        help="Options are ['wandb', 'tensorboard', 'wandb,tensorboard']"
    )
    parser.add_argument(
        "--wandb-notes",
        default='',
        type=str,
        help="Notes if logging with wandb"
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="If true, more information is logged."
    )
    parser.add_argument(
        "--copy-codebase",
        default=False,
        action="store_true",
        help="If true, we copy the entire base on the log diretory, and execute from there."
    )
    parser.add_argument(
        "--ddp-static-graph",
        default=False,
        action='store_true',
        help="Enable static graph optimization for DDP in PyTorch >= 1.11.",
    )
    parser.add_argument(
        "--no-set-device-rank",
        default=False,
        action="store_true",
        help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc)."
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Default random seed."
    )
    parser.add_argument(
        "--norm_gradient_clip", type=float, default=1.0, help="Gradient clip."
    )
    args = parser.parse_args()
    if args.images:
        args.images = os.path.expanduser(args.images)
        
    # If some params are not passed, we use the default values based on model name.
    default_params = get_default_params(args.model)
    for name, val in default_params.items():
        if getattr(args, name) is None:
            setattr(args, name, val)

    return args