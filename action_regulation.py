from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from peft import get_peft_model, LoraConfig
import torch.nn as nn
import torch

def inject_neuroreg(model, adapter_config=None):
    layers = model.model.layers if hasattr(model, 'model') else model.layers

    for i, layer in enumerate(layers):
        if not isinstance(layer, LlamaDecoderLayer):
            continue

        hidden_size = layer.self_attn.hidden_size
        device = layer.self_attn.q_proj.weight.device
            
        layer.sensitivity = nn.Parameter(
            torch.ones(hidden_size, device=device) * 0.5
        ) 
        original_forward = layer.forward

        def make_smart_forward(original_fn, layer_instance):
            def smart_forward(*args, **kwargs):
                outputs = original_fn(*args, **kwargs)
                
                if isinstance(outputs, tuple):
                    hidden = outputs[0]
                    other_outputs = outputs[1:]
                else:
                    hidden = outputs
                    other_outputs = None
                
                # with torch.no_grad():
                abs_h = torch.abs(hidden)
                median_val = abs_h.median()
                
                percentile_90 = torch.quantile(abs_h.float().flatten(), 0.90)
                threshold = max(percentile_90.item(), median_val.item() * 3, 1.0)

                scale = max(threshold, 0.5)
                
                sensitivity = torch.sigmoid(layer_instance.sensitivity).view(1, 1, -1)               
                normalized = hidden / (scale + 1e-8)
                mask = (abs_h > threshold).float()
                
                compressed = torch.where(
                    mask > 0.5,
                    sensitivity * normalized,  
                    normalized  
                )
                
                recovered = compressed * scale 
                residual_ratio = 0.2
                output = residual_ratio * hidden + (1 - residual_ratio) * recovered
                
                if other_outputs is not None:
                    return (output,) + other_outputs
                else:
                    return output
            
            return smart_forward
        
        layer.forward = make_smart_forward(original_forward, layer)
        print(f"{i}-th layer with fixed tanh compression")

    if adapter_config is not None:
        model = get_peft_model(model, adapter_config)
        
        for name, param in model.named_parameters():
            if any(key in name for key in ["sensitivity"]):
                param.requires_grad = True
                if hasattr(model, '_adapter_parameters'):
                    model._adapter_parameters.append(param)
                if hasattr(model, '_trainable_adapter_parameters'):
                    model._trainable_adapter_parameters.append(param)

    return model

# def inject_neuroreg(model, adapter_config=None):
#     layers = model.model.layers if hasattr(model, 'model') else model.layers

#     for i, layer in enumerate(layers):
#         if not isinstance(layer, LlamaDecoderLayer):
#             continue

#         hidden_size = layer.self_attn.hidden_size
#         device = layer.self_attn.q_proj.weight.device
        
#         layer.sensitivity = nn.Parameter(torch.ones(hidden_size, device=device) * 0.5)      
#         original_forward = layer.forward

#         def make_smart_forward(original_fn, layer_instance):
#             def smart_forward(*args, **kwargs):
#                 outputs = original_fn(*args, **kwargs)
                
#                 if isinstance(outputs, tuple):
#                     hidden = outputs[0]
#                     other_outputs = outputs[1:]
#                 else:
#                     hidden = outputs
#                     other_outputs = None
                
#                 with torch.no_grad():
#                     abs_h = torch.abs(hidden)  
#                     median_val = abs_h.median()       
                    
#                     percentile_90 = torch.quantile(abs_h.float().flatten(), 0.90)
#                     threshold = max(percentile_90.item(), median_val.item() * 3, 1.0)
#                     scale = max(threshold, 0.5)
                
#                 mask = (abs_h > threshold).float().detach()
                
#                 sensitivity_coef = torch.sigmoid(layer_instance.sensitivity) 
#                 sensitivity_coef = sensitivity_coef.view(1, 1, -1)   
                
#                 normalized = hidden / (scale + 1e-8)
#                 compressed_abnormal = sensitivity_coef * normalized
                
#                 regulated = mask * compressed_abnormal + (1 - mask) * hidden
#                 regulated_output = regulated * scale

#                 residual_ratio = 0.1 
#                 output = residual_ratio * hidden + (1 - residual_ratio) * regulated_output
                
#                 if other_outputs is not None:
#                     return (output,) + other_outputs
#                 else:
#                     return output
            
#             return smart_forward
        
#         layer.forward = make_smart_forward(original_forward, layer)
#         print(f"{i}-th layer with fixed tanh compression")

#     if adapter_config is not None:
#         model = get_peft_model(model, adapter_config)
        
#         for name, param in model.named_parameters():
#             if any(key in name for key in ["sensitivity"]):
#                 param.requires_grad = True
#                 if hasattr(model, '_adapter_parameters'):
#                     model._adapter_parameters.append(param)
#                 if hasattr(model, '_trainable_adapter_parameters'):
#                     model._trainable_adapter_parameters.append(param)

#     return model

# def inject_neuroreg(model, adapter_config=None, alpha=0.2):
#     layers = model.model.layers if hasattr(model, 'model') else model.layers
    
#     for i, layer in enumerate(layers):
#         if not isinstance(layer, LlamaDecoderLayer):
#             continue
        
#         hidden_size = layer.self_attn.hidden_size
#         device = layer.self_attn.q_proj.weight.device
        
#         layer.scale_factor = nn.Parameter(torch.ones(hidden_size, device=device))
        
#         original_forward = layer.forward
        
#         def make_smart_forward(original_fn, layer_instance):
#             def smart_forward(*args, **kwargs):
#                 outputs = original_fn(*args, **kwargs)
                
#                 if isinstance(outputs, tuple):
#                     hidden = outputs[0]
#                     other_outputs = outputs[1:]
#                 else:
#                     hidden = outputs
#                     other_outputs = None
                
#                 abs_h = torch.abs(hidden)
#                 rms = torch.sqrt(torch.mean(abs_h ** 2))  
                
#                 scale_factor = torch.sigmoid(layer_instance.scale_factor).view(1, 1, -1)
                
#                 scale_base = rms.detach()  
#                 normalized = hidden / (scale_base + 1e-8)
                

#                 scaling = 1.0 + (scale_factor - 0.5) * torch.nn.functional.softplus(abs_h / scale_base - 1.0)
#                 scaled = normalized * scaling

#                 recovered = scaled * scale_base
#                 output = alpha * hidden + (1 - alpha) * recovered
                
#                 if other_outputs is not None:
#                     return (output,) + other_outputs
#                 else:
#                     return output
            
#             return smart_forward
        
#         layer.forward = make_smart_forward(original_forward, layer)
#         print(f"{i}-th layer: NeuroReg-simple (alpha={alpha})")
    
#     if adapter_config is not None:
#         model = get_peft_model(model, adapter_config)
        
#         for name, param in model.named_parameters():
#             if "scale_factor" in name:
#                 param.requires_grad = True
#                 if hasattr(model, '_adapter_parameters'):
#                     model._adapter_parameters.append(param)
    
#     return model