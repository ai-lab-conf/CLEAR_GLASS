import logging
import os
import gc
import random
from datetime import datetime
import numpy as np
import torch
from torch import optim
from torch.cuda.amp import GradScaler
import clip
from open_clip import tokenize

try:
    import wandb
except ImportError:
    wandb = None

try:
    import torch.utils.tensorboard as tensorboard
except ImportError:
    tensorboard = None

# # Custom modules
from training.data import get_train_data
from training.data import get_val_data
from training.distributed import is_master, init_distributed_device, world_info_from_env
from training.logger import setup_logging
from training.params import parse_args
from training.scheduler import cosine_lr
from training.train import train_one_epoch, evaluate
# from training.pretrain import train_one_epoch, evaluate
from training.model import get_model

# from data import get_train_data
# from data import get_val_data
# from distributed import is_master, init_distributed_device, world_info_from_env
# from logger import setup_logging
# from params import parse_args
# from scheduler import cosine_lr
# from train import train_one_epoch, evaluate
# # from pretrain import train_one_epoch, evaluate
# from model import get_model



def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def main():
    args = parse_args()
    rank = getattr(args, 'rank', 0)
    random_seed(args.seed, rank)
    random_seed(args.seed, 0)
    # os.environ['MASTER_ADDR'] = 'localhost'  # Use appropriate IP if running across multiple machines
    # os.environ['MASTER_PORT'] = '12355'  # Choose any free port
    # os.environ['WORLD_SIZE'] = '1'  # Set to number of processes
    # os.environ['RANK'] = '0'  # Set to the rank of this proce

    args.model = args.model.replace('/', '-')

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    if args.name is None:
        args.name = '-'.join([
            datetime.now().strftime("%Y_%m_%d-%H_%M_%S"),
            f"model_{args.model}",
            f"lr_{args.lr}",
            f"b_{args.batch_size}",
            f"e_{args.epochs}",
            f"d_t_{args.alpha}",
            f"l_{args.loss}"
        ])

    # args.distributed = False
    # args.distributed = False
    args.local_rank, args.rank, args.world_size = world_info_from_env()
    args.distributed = args.world_size > 1

    args.log_path = None
    if is_master(args, local=args.log_local):
        log_base_path = os.path.join(args.logs, args.name)
        os.makedirs(log_base_path, exist_ok=True)
        log_filename = f'out-{args.rank}' if args.log_local else 'out.log'
        args.log_path = os.path.join(log_base_path, log_filename)
        if os.path.exists(args.log_path):
            print(f"Error. Experiment already exists. Use --name {args.name} to specify a new experiment.")
            return -1

    args.log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(args.log_path, args.log_level)

    print("Initializing device...")
    device = init_distributed_device(args)
    print("Device initialized:", device)

    args.wandb = 'wandb' in args.report_to or 'all' in args.report_to
    args.tensorboard = 'tensorboard' in args.report_to or 'all' in args.report_to
    if is_master(args):
        args.tensorboard_path = os.path.join(args.logs, args.name, "tensorboard") if args.tensorboard else ''
        args.checkpoint_path = os.path.join("")

        for dirname in [args.tensorboard_path, args.checkpoint_path]:
            if dirname:
                os.makedirs(dirname, exist_ok=True)
    else:
        args.tensorboard_path = ''
        args.checkpoint_path = ''

    if args.copy_codebase:
        copy_codebase(args)


    assert args.precision in ['amp', 'fp32']
    if args.precision == 'fp16':
        logging.warning(
            'It is recommended to use AMP mixed-precision instead of FP16. '
            'FP16 support needs further verification and tuning, especially for training.')

    if args.distributed:
        logging.info(
            f'Running in distributed mode with multiple processes. Device: {args.device}.'
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
    else:
        logging.info(f'Running with a single process. Device {args.device}.')

    
    # random_seed(args.seed, 0)

    model, preprocess_train, preprocess_val, tokenizer = get_model(args, model_name=args.model, from_path=args.model_path, device=device)
    print("Model and preprocess functions loaded.")

    if args.lock_image:
        model.lock_image_tower(
            unlocked_groups=args.lock_image_unlocked_groups,
            freeze_bn_stats=args.lock_image_freeze_bn_stats)

    if args.grad_checkpointing:
        model.set_grad_checkpointing()

    if is_master(args):
        logging.info("Model:")
        logging.info(f"{str(model)}")
        logging.info("Params:")
        params_file = os.path.join(args.logs, args.name, "params.txt")
        with open(params_file, "w") as f:
            for name in sorted(vars(args)):
                val = getattr(args, name)
                logging.info(f"  {name}: {val}")
                f.write(f"{name}: {val}\n")

    if args.distributed:
        if args.use_bn_sync:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        ddp_args = {}
        if args.ddp_static_graph:
            ddp_args['static_graph'] = True
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], **ddp_args, find_unused_parameters=True)

    optimizer = None
    scaler = None
    if args.captions:
        assert not args.trace, 'Cannot train with traced model'

        exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
        include = lambda n, p: not exclude(n, p)

        named_parameters = list(model.named_parameters())
        gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
        rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]

        optimizer = optim.AdamW(
            [
                {"params": gain_or_bias_params, "weight_decay": 0.},
                {"params": rest_params, "weight_decay": args.wd},
            ],
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
        )

    
        assert args.precision in ['amp', 'fp32']
        if args.precision == 'amp':
            scaler = GradScaler()
        else:
            scaler = None

    start_epoch = 0
    if args.resume is not None:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location=device)
            if 'epoch' in checkpoint:
                start_epoch = checkpoint["epoch"]
                sd = checkpoint["state_dict"]
                if not args.distributed and next(iter(sd.items()))[0].startswith('module'):
                    sd = {k[len('module.'):]: v for k, v in sd.items()}
                model.load_state_dict(sd)
                if optimizer is not None:
                    optimizer.load_state_dict(checkpoint["optimizer"])
                if scaler is not None and 'scaler' in checkpoint:
                    scaler.load_state_dict(checkpoint['scaler'])
                logging.info(f"=> resuming checkpoint '{args.resume}' (epoch {start_epoch})")
            else:
                model.load_state_dict(checkpoint)
                logging.info(f"=> loaded checkpoint '{args.resume}' (epoch {start_epoch})")
        else:
            logging.info(f"=> no checkpoint found at '{args.resume}'")

   

    args.save_logs = args.logs and args.logs.lower() != 'none' and is_master(args)
    writer = None
    if args.save_logs and args.tensorboard:
        assert tensorboard is not None, "Please install tensorboard."
        writer = tensorboard.SummaryWriter(args.tensorboard_path)
    
    if args.wandb and is_master(args):
        assert wandb is not None, 'Please install wandb.'
        logging.debug('Starting wandb.')
        wandb.init(
            project="Finetuning v3 models on L14",  # Verify this project name
            entity="conversational-ai-brown",  # Update this if it's not correct
            group="RunClipEx_" + args.model,
            name = args.name,
            reinit=True
)
        if args.wandb_sweep:
            config = wandb.config
            batch_size = config.batch_size
            group_size = config.group_size
            learning_rate = config.learning_rate
            num_epochs = config.num_epochs
            alpha = config.alpha
            tau = config.tau
            loss = config.loss
            dataset_type = config.dataset_type
            

            # Construct the run name with f-string
            timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
            run_name = f"{timestamp}-model_{args.model}-lr_{learning_rate}-b_{batch_size}-gs_{group_size}-e_{num_epochs}-a_{alpha}-t_{tau}-loss_{loss}-dataset_type{dataset_type}"

            # Update args with values from config
            args.batch_size = batch_size
            args.dataset_type = dataset_type
            args.group_size = group_size
            args.lr = learning_rate
            args.epochs = num_epochs
            args.alpha = alpha
            args.tau = tau
            args.loss = loss

            if args.evaluate_trained_models:
                args.model_path = config.model_path
                args.eval_type = config.eval_type
                args.model = config.model
                # args.caption_mode = config.caption_mode
                model_name = config.eval_type + "_" + config.model_path
                wandb.run.name = model_name
            else:
                wandb.run.name = run_name
            wandb.run.save()

          # Updating the run summary with additional metadata
            wandb.run.summary["model"] = args.model
            # wandb.run.summary["train_size"] = args.train_sz
            # if args.val_captions is not None:
            #     wandb.run.summary["val_size"] = args.val_sz
            wandb.run.summary["loss"] = args.loss
            wandb.run.summary["dataset"] = ""

        if args.debug:
            wandb.watch(model, log='all')
        wandb.save(params_file)
        logging.debug('Finished loading wandb.')
    train_data = get_train_data(args, preprocess_fn=preprocess_train, tokenizer=tokenizer, dataset_type=args.dataset_type, epoch=start_epoch)
    
    assert len(train_data), 'At least one train or eval dataset must be specified.'
    # assert len(val_data), 'At least one train or eval dataset must be specified.'

    scheduler = None
    if optimizer is not None:
        total_steps = train_data[f"train_{args.dataset_type}"].dataloader.num_batches * args.epochs
        scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)

    if args.evaluate_trained_models:
        if args.model_path == "frozen_hierarCaps":
            trained_model, preprocess_train, preprocess_val, tokenizer = get_model(args, model_name="hierarcaps", device=device)
        elif args.model_path == "frozen_clip":
            trained_model, preprocess_train, preprocess_val, tokenizer = get_model(args, model_name="open_clip", device=device)
        elif args.model_path == "frozen_negClip":
            trained_model, preprocess_train, preprocess_val, tokenizer = get_model(args, model_name="negClip", device=device)
        elif args.model_path == "frozen_Clip_ViT-L-14":
            trained_model, preprocess_train, preprocess_val, tokenizer = get_model(args, model_name="Clip_ViT-L-14", device=device)
        elif args.model_path == "frozen_ConvNext":
            trained_model, preprocess_train, preprocess_val, tokenizer = get_model(args, model_name="ConvNext", device=device)
        elif args.model_path == "frozen_SigLip":
            trained_model, preprocess_train, preprocess_val, tokenizer = get_model(args, model_name="SigLip", device=device)
        elif args.model_path == "frozen_COCA":
            trained_model, preprocess_train, preprocess_val, tokenizer = get_model(args, model_name="COCA", device=device)
        elif args.model_path == "frozen_DFN":
            trained_model, preprocess_train, preprocess_val, tokenizer = get_model(args, model_name="DFN", device=device)
        elif args.model_path == "frozen_EVA02-B-16":
            trained_model, preprocess_train, preprocess_val, tokenizer= get_model(args, model_name="eva", device=device)
        elif args.model_path == "frozen_ce-clip":
            trained_model, preprocess_train, preprocess_val, tokenizer = get_model(args, model_name="ce-clip", device=device)
        elif args.model_path == "frozen_conclip":
            trained_model, preprocess_train, preprocess_val, tokenizer = get_model(args, model_name="conclip", device=device)
        else:
            model_path = f"{args.model_path}"
            trained_model, preprocess_train, preprocess_val, tokenizer= get_model(args, model_name=args.model, from_path=model_path, device=device)

        evaluate_data = get_val_data(args, preprocess_fn=preprocess_val,tokenizer=tokenizer,epoch=start_epoch)
        evaluate(trained_model, evaluate_data, start_epoch, args, dataset_type=args.dataset_type, eval_type = args.eval_type)
    else:
        for epoch in range(start_epoch, args.epochs):
            if is_master(args):
                logging.info(f'Start epoch {epoch}')

            train_one_epoch(model, train_data, epoch, optimizer, scaler, scheduler, args, writer)
            completed_epoch = epoch + 1

                
            eval_types = ["new_caption", "old_caption", "concept_level1", "concept_level2", "concept_level3", "concept_level4"]
            # eval_types = ["concept_level1", "concept_level2", "concept_level3", "concept_level4"]
            for eval_type in eval_types:
                val_data = get_val_data(args, eval_type=eval_type,preprocess_fn=preprocess_val,tokenizer=tokenizer,epoch=start_epoch, dataset_type="evaluation")
                evaluate(model, val_data,completed_epoch, args, eval_type=eval_type, dataset_type= "evaluation" )
            

            # Saving checkpoints.
            # Saving checkpoints
            if args.save_logs:
                checkpoint_dict = {
                    "epoch": completed_epoch,
                    "name": args.name,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }

                if scaler is not None:
                    checkpoint_dict["scaler"] = scaler.state_dict()
                
                run_name = f"model_{args.model}-b_{args.batch_size}-gs_{args.group_size}-e_{completed_epoch}-loss_{args.loss}"
                
                # Modified section for Hugging Face compatibility
                if hasattr(model, 'save_pretrained'):  # Detect Hugging Face model
                    # Create directory for Hugging Face format
                    model_dir = os.path.join(args.checkpoint_path, run_name)
                    os.makedirs(model_dir, exist_ok=True)
                    
                    # Save Hugging Face components
                    model.save_pretrained(model_dir)
                    if tokenizer is not None:  # Save tokenizer if available
                        tokenizer.save_pretrained(model_dir)
                    
                    # Save training state separately
                    torch.save(checkpoint_dict, os.path.join(model_dir, "training_state.pt"))
                else:
                    # Original saving logic for OpenAI-style models
                    ckpt_path = os.path.join(args.checkpoint_path, f"{run_name}.pt")
                    torch.save(checkpoint_dict, ckpt_path)

                # Common most-recent saving logic
                if args.save_most_recent:
                    if hasattr(model, 'save_pretrained'):
                        latest_dir = os.path.join(args.checkpoint_path, "latest")
                        os.makedirs(latest_dir, exist_ok=True)
                        model.save_pretrained(latest_dir)
                        if tokenizer is not None:
                            tokenizer.save_pretrained(latest_dir)
                        torch.save(checkpoint_dict, os.path.join(latest_dir, "training_state.pt"))
                    else:
                        torch.save(checkpoint_dict, os.path.join(args.checkpoint_path, "epoch_latest.pt"))

    if args.wandb and is_master(args):
        wandb.finish()

def copy_codebase(args):
    from shutil import copytree, ignore_patterns
    new_code_path = os.path.join(args.logs, args.name, "code")
    if os.path.exists(new_code_path):
        print(
            f"Error. Experiment already exists at {new_code_path}. Use --name to specify a new experiment."
        )
        return -1
    print(f"Copying codebase to {new_code_path}")
    current_code_path = os.path.realpath(__file__)
    for _ in range(3):
        current_code_path = os.path.dirname(current_code_path)
    copytree(current_code_path, new_code_path, ignore=ignore_patterns('log', 'logs', 'wandb'))
    print("Done copying code.")
    return 1

if __name__ == "__main__":
    main()