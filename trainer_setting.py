import transformers
import math
import os
import shutil
import sys
import time
import logging
import torch

from transformers.integrations import (
    hp_params,
)
import torch
import torch.distributed as dist
from packaging import version
from torch import nn
from torch.utils.data import RandomSampler
import torch.nn.functional as F
import numpy as np
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.integrations.deepspeed import deepspeed_init, deepspeed_load_checkpoint, is_deepspeed_available
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS, is_torch_less_than_1_11
from transformers.trainer_callback import (
    TrainerState,
)
from transformers.trainer_pt_utils import (
    get_dataloader_sampler,
    get_model_param_count,
    get_parameter_names,
)
from transformers.trainer_utils import (
    HPSearchBackend,
    TrainOutput,
    has_length,
    speed_metrics,
)
from transformers.training_args import ParallelMode, TrainingArguments
from transformers.utils import (
    is_sagemaker_mp_enabled,
    is_torch_tpu_available,
    logging,
)
from dataclasses import dataclass, field
import os
import sys
import torch
import transformers
from typing import List, Optional, Union
from dataclasses import dataclass, field
sys.path.append(os.path.join(os.getcwd(), "peft/src/"))
from peft.tuners import lora
from transformers import TrainingArguments 
import numpy as np
import torch.nn as nn
from functools import reduce
from transformers.utils import is_sagemaker_mp_enabled
from transformers.trainer_pt_utils import get_parameter_names
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
from lion_pytorch import Lion

logger = logging.get_logger(__name__)


@dataclass
class LoraPlusTrainingArguments(TrainingArguments):
    loraplus_lr_ratio: Optional[float] = field(
        default=4, metadata={"help": "loraplus learning rate ratio lr_B / lr_A."}
    )
    loraplus_lr_embedding: Optional[float] = field(
        default=1e-6,
        metadata={"help": "loraplus learning rate for lora embedding layers."},
    )

def get_module(name, opt_model):
    """
    Retrieve a module from a model using its parameter name.
    Args:
        name (str): Full name of the parameter, typically including module path.
        opt_model (torch.nn.Module): The model from which to retrieve the module.

    Returns:
        Module corresponding to the given name.
    """
    parent_idx = 2 if "lora" in name else 1
    module_names = name.split(sep=".")[:-parent_idx]
    module = reduce(getattr, module_names, opt_model)
    return module

def create_loraplus_optimizer(
    opt_model,
    optimizer_cls,
    optimizer_kwargs,
    loraplus_lr_ratio,
    loraplus_lr_embedding=None,
):
    """
    Creates an optimizer for the given model, applying LoRA-specific learning rate adjustments to different parameter groups.

    Args:
        opt_model (torch.nn.Module): The model for which the optimizer is being created.
        optimizer_cls (class): The class of the optimizer to be used (e.g., torch.optim.Adam).
        optimizer_kwargs (dict): A dictionary of keyword arguments for the optimizer's initialization.
        loraplus_lr_ratio (float): The learning rate ratio to be applied to LoRA parameters.
        loraplus_lr_embedding (float, optional): A specific learning rate for embedding parameters, with a default value if not provided.

    Returns:
        An instance of the specified optimizer class configured with the model's parameters organized into groups with custom learning rates.
    """

    assert loraplus_lr_ratio is not None, "loraplus_lr_ratio must be provided."

    if loraplus_lr_embedding is None:
        loraplus_lr_embedding = 1e-6

    decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    param_groups = {
        "groupA": {},
        "groupB": {},
        "groupB_no_decay": {},
        "embedding": {},
    }

    for name, param in opt_model.named_parameters():
        if not param.requires_grad:
            continue

        module = get_module(name, opt_model)
        if isinstance(module, torch.nn.Embedding):
            param_groups["embedding"][name] = param
        elif "lora_B" in name or param.ndim == 1:
            if name in decay_parameters:
                param_groups["groupB"][name] = param
            else:
                param_groups["groupB_no_decay"][name] = param
        else:
            param_groups["groupA"][name] = param

    assigned_param_groups = ""
    for group in param_groups:
        assigned_param_groups += f"{group}\n {list(param_groups[group].keys())}\n\n"
    print(assigned_param_groups)

    lr = optimizer_kwargs["lr"]
    weight_decay = optimizer_kwargs.get("weight_decay", 0.0)

    optimizer_grouped_parameters = [
        {
            "params": list(param_groups["groupA"].values()),
            "weight_decay": weight_decay,
            "lr": lr,
        },
        {
            "params": list(param_groups["embedding"].values()),
            "weight_decay": weight_decay,
            "lr": loraplus_lr_embedding,
        },
        {
            "params": list(param_groups["groupB"].values()),
            "weight_decay": weight_decay,
            "lr": lr * loraplus_lr_ratio,
        },
        {
            "params": list(param_groups["groupB_no_decay"].values()),
            "weight_decay": 0.0,
            "lr": lr * loraplus_lr_ratio,
        },
    ]

    optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
    if optimizer_cls.__name__ == "Adam8bit":
        import bitsandbytes

        manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

        skipped = 0
        for module in opt_model.modules():
            if isinstance(module, nn.Embedding):
                skipped += sum(
                    {p.data_ptr(): p.numel() for p in module.parameters()}.values()
                )
                print(f"skipped {module}: {skipped/2**20}M params")
                manager.register_module_override(module, "weight", {"optim_bits": 32})
        print(f"skipped: {skipped/2**20}M params")

    return optimizer

class StatesRecordTrainer(transformers.Trainer):
    def __init__(self, method_name=None, diagnostic_data=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.path_name = './' + method_name + '_results/'
        self.File1 = open(self.path_name + "Train_Hidden_States_Record.txt", 'a')
        self.File2 = open(self.path_name + "Train_Loss.txt", 'a')
        self.File3 = open(self.path_name + "Grad_Norm.txt", 'a')
        # self.File4 = open(self.path_name + "Diagnostic_Hidden_States_Record.txt", 'a')
        # self.File5 = open(self.path_name + "Grad_Similarity.txt", 'a')
        self.File6 = open(self.path_name + "Token_Dimension.txt", 'a')
        self.File7 = open(self.path_name + "Token_Outlier_Value.txt", 'a')
        self.method_name = method_name
        self.diagnostic_data = diagnostic_data
        self.outlier_value_name = ['rank: ', 'mekurtosis: ', 'mmt: ']
        self.prev_gradients = None
        self.kurtosis_top_k_layer = 5

    def create_optimizer(self):
        """
        Overrides the method to create an optimizer with LoRA+ specific adjustments.
        """
        if 'loraplus' not in self.method_name and 'lion' not in self.method_name:
            return super().create_optimizer()

        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model
        if 'loraplus' in self.method_name:
            if self.optimizer is None:
                optimizer_cls, optimizer_kwargs = transformers.Trainer.get_optimizer_cls_and_kwargs(
                    self.args
                )

                loraplus_lr_ratio = getattr(self.args, "loraplus_lr_ratio", None)
                loraplus_lr_embedding = getattr(self.args, "loraplus_lr_embedding", None)
                self.optimizer = create_loraplus_optimizer(
                    opt_model,
                    optimizer_cls,
                    optimizer_kwargs,
                    loraplus_lr_ratio,
                    loraplus_lr_embedding,
                )
        
        if 'lion' in self.method_name:
            if self.optimizer is None:
                self.optimizer = Lion(opt_model.parameters(), lr=self.args.learning_rate * (1/3), \
                                      weight_decay=self.args.weight_decay)

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False):
        if self.should_record_exponential(self.state.global_step):
            inputs['output_hidden_states'] = True

        result = super().compute_loss(model, inputs, return_outputs=True)
        loss, outputs = result
        self.File2.write(str(loss.item()) + '\n')
        self.File2.flush()

        if self.should_record_exponential(self.state.global_step):
            if hasattr(outputs, 'hidden_states'):
                outlier_value, layer_kurtosis, layer_seleted_token_values, layer_seleted_token_indices \
                    = self.analyze_hidden_states(outputs.hidden_states, [])
                sorted_layers_kurtosis = sorted(layer_kurtosis, key=lambda x: x[1], reverse=True)
                selected_layer_indices = [idx for idx, _ in sorted_layers_kurtosis[:self.kurtosis_top_k_layer]]
                 
                
                for layer_idx in range(len(outlier_value)):
                    self.File1.write('step: ' + str(self.state.global_step) + '\n')
                    self.File1.write('layer index: ' + str(layer_idx) + '\n')
                    for value_idx in range(len(outlier_value[layer_idx])):
                        self.File1.write(str(outlier_value[layer_idx][value_idx][0]) + '\n')
                    self.File1.write('\n')
                self.File1.flush() 

            
                self._extract_hidden_vectors(outputs.hidden_states, selected_layer_indices, layer_seleted_token_values, layer_seleted_token_indices)

            # self.scan_diagnostic_hidden_states(model, self.diagnostic_data)

        return (loss, outputs) if return_outputs else loss

    def _extract_hidden_vectors(self, hidden_states, selected_layer_indices, seleted_token_values, selected_token_indices):
        # vectors = []
        self.File6.write(str(self.state.global_step) + '\n')
        self.File7.write(str(self.state.global_step) + '\n')
        for layer_idx in selected_layer_indices:
            self.File6.write(str(layer_idx) + '\n')
            self.File7.write(str(layer_idx) + '\n')
            for i in range(len(seleted_token_values[layer_idx])):
                self.File6.write(str(selected_token_indices[layer_idx][i]) + '\n')
                self.File7.write(str(seleted_token_values[layer_idx][i]) + '\n')
            self.File6.write('\n')
            self.File7.write('\n')
            hidden = hidden_states[layer_idx].detach().cpu()
            token_vector = hidden[:, selected_token_indices[layer_idx], :]
            token_vector = token_vector.transpose(0, 1).reshape(len(selected_token_indices[layer_idx]), -1)
            token_vector = torch.norm(token_vector, dim=1)
            np.save(self.path_name + 'step' + str(self.state.global_step) + '_layer' + str(layer_idx) + '_' + 'outlier_token_distribution.npy', token_vector)
            # vectors.append(token_vector.view(-1))
            # token_vector1 = token_vector.unsqueeze(1)  
            # token_vector2 = token_vector.unsqueeze(0)
            # token_cosine_sim = F.cosine_similarity(token_vector1, token_vector2, dim=2) 
            # np.save(self.path_name + 'step' + str(self.state.global_step) + '_layer' + str(layer_idx) + '_' + 'outlier_token_cosine_similarity.npy', token_cosine_sim)
        # vectors =  torch.stack(vectors, dim=0)
        # vector1 = vectors.unsqueeze(1)  
        # vector2 = vectors.unsqueeze(0)
        # layer_cosine_sim = F.cosine_similarity(vector1, vector2, dim=2)
        # np.save(self.path_name + 'step' + str(self.state.global_step) + '_cross_layer_' + 'outlier_token_cosine_similarity.npy', layer_cosine_sim)
        self.File6.flush()
        self.File7.flush()

    # def should_record_exponential(self, step):
    #     if step == 0:
    #         return True
    #
    #     base_interval = 10
    #     growth_rate = 1.05
    #     scale = 1000
    #
    #     interval = base_interval * (growth_rate ** (step / scale))
    #     min_interval = 5
    #     max_interval = 1000
    #     interval = max(min_interval, min(max_interval, interval))
    #     return step % round(interval) == 0

    def should_record_exponential(self, step):
        stairs = [0, 50, 200, 800, 3200, 12800, 25600, 51200, 63800]
        for s in stairs:
            if step == s:
                return True
        return False

    # def scan_diagnostic_hidden_states(self, model, diagnostic_data):
    #     model.eval()
    #     outlier_value = []
    #     diagnostic_dataloader = self.get_eval_dataloader(diagnostic_data)
    #     all_batches = list(diagnostic_dataloader)
    #     total_batches = len(all_batches)

    #     import random
    #     n_to_sample = min(10, total_batches)
    #     selected_indices = random.sample(range(total_batches), n_to_sample)
    #     with torch.no_grad():
    #         for idx in selected_indices:
    #             batch = all_batches[idx]
    #             batch = self._prepare_inputs(batch)
    #             outputs = model(**batch, output_hidden_states=True)
    #             hidden_states = outputs.hidden_states
    #             outlier_value, _, _, _ = self.analyze_hidden_states(hidden_states, outlier_value)

    #     for layer_idx in range(len(outlier_value)):
    #         self.File4.write('step: ' + str(self.state.global_step) + '\n')
    #         self.File4.write('layer index: ' + str(layer_idx) + '\n')
    #         for value_idx in range(len(outlier_value[layer_idx])):
    #             self.File4.write(
    #                 str(sum(outlier_value[layer_idx][value_idx]) / len(outlier_value[layer_idx][value_idx])) + '\n')
    #         self.File4.write('\n')
    #     self.File4.flush()
    #     model.train()

    def analyze_hidden_states(self, hidden_states, outlier_value):
        layer_kurtosis_single = []
        layer_seleted_token_values = []
        layer_seleted_token_indices = []
        for layer_idx, hidden in enumerate(hidden_states):
            if not outlier_value:
                outlier_value = [[[] for _ in range(3)] for _ in range(len(hidden_states))]

            hidden = hidden.detach().cpu()
            rank = self.compute_effective_rank(hidden)
            outlier_value[layer_idx][0].append(rank.item())

            # kurtosis
            # x = hidden.reshape(-1, hidden.size(-1))
            # mean = x.mean(dim=0, keepdim=True)     
            # std = x.std(dim=0, keepdim=True, unbiased=False) + 1e-8  
            # z = (x - mean) / std            
            # kurtosis = (z ** 4).mean(dim=0) - 3.0 

            batch_size, seq_len, hidden_dim = hidden.shape
            x = hidden.transpose(0, 1).reshape(seq_len, -1)
            mean = x.mean(dim=1, keepdim=True)
            std = x.std(dim=1, keepdim=True, unbiased=False) + 1e-8
            z = (x - mean) / std
            kurtosis = (z ** 4).mean(dim=1) - 3.0

            sorted_tokrn_values, sorted_token_indices = torch.sort(kurtosis, descending=True)
            total_dims = kurtosis.size(0)
            num_top = int(total_dims * (100 - 90) / 100) 
            num_top = max(1, num_top)
            top_token_values = sorted_tokrn_values[:num_top]
            top_token_indices = sorted_token_indices[:num_top]
            layer_seleted_token_values.append(top_token_values.tolist())
            layer_seleted_token_indices.append(top_token_indices.tolist())

            kurtosis_single = torch.quantile(kurtosis, 0.95)
            layer_kurtosis_single.append((layer_idx, kurtosis_single))
            outlier_value[layer_idx][1].append(kurtosis_single.item())            

            # Max Median Ratio
            x_abs = torch.abs(hidden)
            x_flat = x_abs.reshape(-1)
            max_val = x_flat.max()
            median_val = torch.median(x_flat)
            std_val = x_flat.std()
            mmt = (max_val - median_val) / (std_val + 1e-12)
            outlier_value[layer_idx][2].append(mmt.item())

        return outlier_value, layer_kurtosis_single, layer_seleted_token_values, layer_seleted_token_indices


    # def compute_matrix_rank(self, matrix, eps=1e-6):
    #     try:
    #         s = torch.linalg.svdvals(matrix.float())
    #         # tol = eps * s[0] * max(matrix.shape)
    #         # return (s > tol).sum()
    #         return (s > eps * s.max()).sum()
    #     except:
    #         return 0

    def compute_effective_rank(self, hidden_states: torch.Tensor):
        H = hidden_states.reshape(-1, hidden_states.size(-1)) 
        U, S, Vh = torch.linalg.svd(H, full_matrices=False)
        
        sum_sigma = S.sum()
        sum_sigma_sq = (S ** 2).sum()
        stable_rank = (sum_sigma ** 2) / (sum_sigma_sq + 1e-12) 
        
        return stable_rank
    

    # def record_and_compute_grad_similarity(self, model):
    #     if (self.state.global_step + 1) % 100 == 0:
    #         current_gradients = self._flatten_gradients(model)
    #         self.prev_gradients = current_gradients

    #     if self.prev_gradients is not None and (self.state.global_step % 100 == 0):
    #         current_gradients = self._flatten_gradients(model)
            
    #         if current_gradients is None or self.prev_gradients is None:
    #             return

    #         if current_gradients.dim() != 1:
    #             current_gradients = current_gradients.flatten()
    #         if self.prev_gradients.dim() != 1:
    #             self.prev_gradients = self.prev_gradients.flatten()
            
    #         if current_gradients.numel() == 0 or self.prev_gradients.numel() == 0:
    #             return
            
    #         current_gradients_2d = current_gradients.unsqueeze(0)  # [1, n]
    #         prev_gradients_2d = self.prev_gradients.unsqueeze(0)   # [1, n]
            
    #         similarity = F.cosine_similarity(
    #             current_gradients_2d, 
    #             prev_gradients_2d,
    #             dim=1 
    #         )
    #         self.File5.write(str(self.state.global_step) + '\n')
    #         self.File5.write(str(similarity.item()) + '\n')
    #         self.File5.flush()

    
    def _flatten_gradients(self, model):
        grads = []
        for param in model.parameters():
            if param.grad is not None:
                grads.append(param.grad.detach().flatten())
        return torch.cat(grads) if grads else None


    def log_grad_norms(self, step, norm_type=2):
        self.File3.write(str(step) + '\n')
        total_norm = 0.0
        layer_norms = {}

        for name, param in self.model.named_parameters():
            if param.grad is not None and param.requires_grad:
                if norm_type == 2 or norm_type == 'fro':
                    param_norm = param.grad.norm(p='fro').item() 
                else:
                    param_norm = param.grad.norm(p=norm_type).item()
                
                total_norm += param_norm ** 2
                
                if "model.layers" in name:
                    parts = name.split(".")
                    for i, part in enumerate(parts):
                        if part == "layers" and i+1 < len(parts):
                            try:
                                layer_idx = int(parts[i+1])
                                layer_norms[layer_idx] = layer_norms.get(layer_idx, 0.0) + param_norm ** 2
                                break
                            except ValueError:
                                continue
        
        total_grad_norm = math.sqrt(total_norm) if total_norm > 0 else 0.0
        self.File3.write(str(math.sqrt(total_grad_norm)) + '\n')
        layer_grad_norms = {}
        for layer_idx, sq_norm in layer_norms.items():
            layer_grad_norms[layer_idx] = math.sqrt(sq_norm)
            self.File3.write(str(layer_norms[layer_idx]) + '\n')
        self.File3.write('\n')
        self.File3.flush()

    def _inner_training_loop(
            self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        self.accelerator.free_memory()
        self._train_batch_size = batch_size
        if self.args.auto_find_batch_size:
            self.state.train_batch_size = self._train_batch_size
        logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = self._train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        num_train_tokens = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
                if args.include_tokens_per_second:
                    num_train_tokens = (
                            self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
                    )
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
                if args.include_tokens_per_second:
                    num_train_tokens = self.num_tokens(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
            if args.include_tokens_per_second:
                num_train_tokens = self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torchrun or torch.distributed.launch (deprecated))."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = is_sagemaker_mp_enabled() or self.is_fsdp_xla_enabled or self.is_fsdp_enabled

        # We need to reset the scheduler, as its parameters may be different on subsequent calls
        if self._created_lr_scheduler:
            self.lr_scheduler = None
            self._created_lr_scheduler = False

        if self.is_deepspeed_enabled:
            self.optimizer, self.lr_scheduler = deepspeed_init(self, num_training_steps=max_steps)

        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None
        self.state.train_batch_size = self._train_batch_size

        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            if args.gradient_checkpointing_kwargs is None:
                gradient_checkpointing_kwargs = {}
            else:
                gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs

            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

        model = self._wrap_model(self.model_wrapped)

        # as the model is wrapped, don't use `accelerator.prepare`
        # this is for unhandled cases such as
        # FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
        use_accelerator_prepare = True if model is self.model else False

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # prepare using `accelerator` prepare
        if use_accelerator_prepare:
            self.model.train()
            if hasattr(self.lr_scheduler, "step"):
                if self.use_apex:
                    model = self.accelerator.prepare(self.model)
                else:
                    model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
            else:
                # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
                model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                    self.model, self.optimizer, self.lr_scheduler
                )

        if self.is_fsdp_enabled:
            self.model = self.model_wrapped = model

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model_wrapped

        # ckpt loading
        if resume_from_checkpoint is not None:
            if self.is_deepspeed_enabled:
                deepspeed_load_checkpoint(self.model_wrapped, resume_from_checkpoint)
            elif is_sagemaker_mp_enabled() or self.is_fsdp_enabled:
                self._load_from_checkpoint(resume_from_checkpoint, self.model_wrapped)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model),
        # FSDP(Transformers Model), Dynamo Optimized Module(Transformers Model) etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        if self.args.per_device_train_batch_size != self._train_batch_size:
            logger.info(f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
                os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first"
                    f" {steps_trained_in_current_epoch} batches in the first epoch."
                )

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if self.hp_name is not None and self._trial is not None:
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
            # parameter to Train when using DDP.
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                sampler = get_dataloader_sampler(train_dataloader)
                sampler_kinds = [RandomSampler]
                if version.parse(accelerate_version) > version.parse("0.23.0"):
                    sampler_kinds.append(SeedableRandomSampler)
                is_random_sampler = isinstance(sampler, tuple(sampler_kinds))
                if is_torch_less_than_1_11 or not is_random_sampler:
                    # We just need to begin an iteration to create the randomization of the sampler.
                    for _ in train_dataloader:
                        break
                else:
                    # Otherwise we need to call the whooooole sampler cause there is some random operation added
                    # AT THE VERY END!
                    sampler = sampler if sampler is not None else []
                    _ = list(sampler)

        total_batched_samples = 0
        for epoch in range(epochs_trained, num_train_epochs):
            epoch_iterator = train_dataloader
            if hasattr(epoch_iterator, "set_epoch"):
                epoch_iterator.set_epoch(epoch)

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            rng_to_sync = False
            steps_skipped = 0
            if steps_trained_in_current_epoch > 0:
                epoch_iterator = skip_first_batches(epoch_iterator, steps_trained_in_current_epoch)
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0
                rng_to_sync = True

            step = -1
            for step, inputs in enumerate(epoch_iterator):
                total_batched_samples += 1

                if self.args.include_num_input_tokens_seen:
                    main_input_name = getattr(self.model, "main_input_name", "input_ids")
                    if main_input_name not in inputs:
                        logger.warning(
                            "Tried to track the number of tokens seen, however the current model is "
                            "not configured properly to know what item is the input. To fix this, add "
                            "a `main_input_name` attribute to the model class you are using."
                        )
                    else:
                        self.state.num_input_tokens_seen += self.accelerator.gather(inputs[main_input_name]).numel()
                if rng_to_sync:
                    self._load_rng_state(resume_from_checkpoint)
                    rng_to_sync = False

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                with self.accelerator.accumulate(model):
                    tr_loss_step = self.training_step(model, inputs)

                if (
                        args.logging_nan_inf_filter
                        and not is_torch_tpu_available()
                        and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))

                is_last_step_and_steps_less_than_grad_acc = (
                        steps_in_epoch <= args.gradient_accumulation_steps and (step + 1) == steps_in_epoch
                )

                if (
                        total_batched_samples % args.gradient_accumulation_steps == 0
                        or
                        # last step in epoch but step is always smaller than gradient_accumulation_steps
                        is_last_step_and_steps_less_than_grad_acc
                ):
                    # the `or` condition of `is_last_step_and_steps_less_than_grad_acc` is not covered
                    # in accelerate. So, explicitly enable sync gradients to True in that case.
                    if is_last_step_and_steps_less_than_grad_acc:
                        self.accelerator.gradient_state._set_sync_gradients(True)

                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0:
                        # deepspeed does its own clipping

                        if is_sagemaker_mp_enabled() and args.fp16:
                            self.optimizer.clip_master_grads(args.max_grad_norm)
                        elif self.use_apex:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            nn.utils.clip_grad_norm_(
                                amp.master_params(self.optimizer),
                                args.max_grad_norm,
                            )
                        else:
                            self.accelerator.clip_grad_norm_(
                                model.parameters(),
                                args.max_grad_norm,
                            )

                    if self.should_record_exponential(self.state.global_step):
                        self.log_grad_norms(self.state.global_step)
                    # if ((self.state.global_step+1) % 100 == 0) or (self.state.global_step % 100 == 0):
                    #     self.record_and_compute_grad_similarity(model)

                    # Optimizer step
                    self.optimizer.step()
                    optimizer_was_run = not self.accelerator.optimizer_step_was_skipped
                    if optimizer_was_run:
                        # Delay optimizer scheduling until metrics are generated
                        if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            self.lr_scheduler.step()

                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                    self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break
            if step < 0:
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_tpu_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sure the model has been saved by process 0.
            if is_torch_tpu_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.parallel_mode == ParallelMode.DISTRIBUTED:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics(
            "train",
            start_time,
            num_samples=num_train_samples,
            num_steps=self.state.max_steps,
            num_tokens=num_train_tokens,
        )
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        # Wait for the checkpoint to be uploaded.
        self._finish_current_push()

        # After training we make sure to retrieve back the original forward pass method
        # for the embedding layer by removing the forward post hook.
        if self.neftune_noise_alpha is not None:
            self._deactivate_neftune(self.model)

        return TrainOutput(self.state.global_step, train_loss, metrics)