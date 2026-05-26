import argparse
import os
import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM

def find_pair(lora_state_dict,):
    pairs = {}
    for key in lora_state_dict.keys():
        prefix = "base_model.model."
        suffix = ".lora_A.weight"
        base_key = key[len(prefix):-len(suffix)] + ".weight"
        pairs[key] = base_key 
    return pairs



def center_matrix(X):
    mean = torch.mean(X, dim=0, keepdim=True)
    centered_X = X - mean
    return centered_X, mean

def select_n_components(eigenvalues, thr):
    total_variance = torch.sum(eigenvalues)
    cumulative_variance = 0.0
    for k, eigenvalue in enumerate(eigenvalues):
        cumulative_variance += eigenvalue
        if cumulative_variance / total_variance >= thr:
            return k + 1
    return len(eigenvalues)


def pca_reduction(X, n_components):

    cov_matrix = torch.matmul(X, X.T) / (X.shape[1] - 1)

    eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)

    sorted_indices = torch.argsort(eigenvalues, descending=True)
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    principal_components = sorted_eigenvectors[:, :n_components]

    X_reduced = torch.matmul(principal_components.T, X)
    return X_reduced, principal_components

def pca_compress(x, y, dim, thr=0.8):
    # pca_num = int(select_lora_rank.split("_")[-1])
    dtype = x.dtype
    A_centered, x_mean = center_matrix(x.float())
    B_centered, y_mean = center_matrix(y.T.float())
    cov_A = torch.matmul(A_centered, A_centered.T) / (A_centered.shape[1] - 1)
    eigenvalues_A, _ = torch.linalg.eigh(cov_A)
    sorted_eigenvalues_A = torch.sort(eigenvalues_A, descending=True).values

    cov_B = torch.matmul(B_centered, B_centered.T) / (B_centered.shape[1] - 1)
    eigenvalues_B, _ = torch.linalg.eigh(cov_B)
    sorted_eigenvalues_B = torch.sort(eigenvalues_B, descending=True).values

    r_A = select_n_components(sorted_eigenvalues_A, thr)
    r_B = select_n_components(sorted_eigenvalues_B, thr)
    r_prime = min(r_A, r_B)

    A_reduced, A_components = pca_reduction(A_centered, r_prime)
    B_reduced, B_components = pca_reduction(B_centered, r_prime)
    assert A_reduced.shape[0] == B_reduced.shape[0]
    new_x = A_reduced.to(dtype).contiguous()
    new_y = B_reduced.T.to(dtype).contiguous()
                # standard_dim = 1 - dim 

    return new_x, new_y 

def svd_compress(x, y, dim, thr):
    assert x.shape[dim] == y.shape[1], "Dimension mismatch between x and y."
    ori_dtype = x.dtype
    x = x.to(torch.float32)
    y = y.to(torch.float32)
    # Compute the product of x and y
    product = torch.matmul(y, x)  # Shape: [d', d]

    # Parameters
    d_prime, d = product.shape
    total_singular_values = x.shape[dim]
    retain_count = max(int(thr * total_singular_values), 1)

    # Step 1: Create a random Gaussian matrix
    random_matrix = torch.randn(d, retain_count, device=product.device)

    # Step 2: Form Y = A * random_matrix
    Y = torch.matmul(product, random_matrix)

    # Step 3: Compute an orthonormal basis Q for the range of Y using QR decomposition
    Q, _ = torch.linalg.qr(Y, mode='reduced')

    # Step 4: Form B = Q.T * A
    B = torch.matmul(Q.T, product)

    # Step 5: Compute the SVD of the smaller matrix B
    U_hat, S, Vt = torch.linalg.svd(B, full_matrices=False)

    # Step 6: Form the approximate singular vectors
    U = torch.matmul(Q, U_hat)

    # Truncate the singular values and corresponding vectors
    U_truncated = U[:, :retain_count]  # Shape: [d', c]
    S_truncated = S[:retain_count]     # Shape: [c]
    Vt_truncated = Vt[:retain_count, :]  # Shape: [c, d]

    # Reconstruct the truncated x and y
    new_x = Vt_truncated  # Shape: [c, d]
    new_y = torch.matmul(U_truncated, torch.diag(S_truncated))  # Shape: [d', c]

    return new_x.to(ori_dtype), new_y.to(ori_dtype)

def minorsvd_compress(x, y, dim, thr):
    assert x.shape[dim] == y.shape[1], "Dimension mismatch between x and y."
    ori_dtype = x.dtype
    x = x.to(torch.float32)
    y = y.to(torch.float32)
    # Compute the product of x and y
    product = torch.matmul(y, x)  # Shape: [d', d]

    # Parameters
    d_prime, d = product.shape
    total_singular_values = x.shape[dim]
    retain_count = max(int(thr * total_singular_values), 1)

    # Step 1: Create a random Gaussian matrix
    random_matrix = torch.randn(d, retain_count, device=product.device)

    # Step 2: Form Y = A * random_matrix
    Y = torch.matmul(product, random_matrix)

    # Step 3: Compute an orthonormal basis Q for the range of Y using QR decomposition
    Q, _ = torch.linalg.qr(Y, mode='reduced')

    # Step 4: Form B = Q.T * A
    B = torch.matmul(Q.T, product)

    # Step 5: Compute the SVD of the smaller matrix B
    U_hat, S, Vt = torch.linalg.svd(B, full_matrices=False)

    # Step 6: Form the approximate singular vectors
    U = torch.matmul(Q, U_hat)

    # Truncate the singular values and corresponding vectors
    U_truncated = U[:, -retain_count:]  # Shape: [d', c]
    S_truncated = S[-retain_count:]     # Shape: [c]
    Vt_truncated = Vt[-retain_count:, :]  # Shape: [c, d]

    # Reconstruct the truncated x and y
    new_x = Vt_truncated  # Shape: [c, d]
    new_y = torch.matmul(U_truncated, torch.diag(S_truncated))  # Shape: [d', c]

    return new_x.to(ori_dtype), new_y.to(ori_dtype)

def lora_convert(x, y, dim, base_weight, step, save_name, range_start=5):
    r = x.shape[dim]
    cos_dim = 1 - dim
    ratio_list = [step * i for i in range(range_start, 10)] + [1]
    remaining_list = [max(1, int(r * ratio)) for ratio in ratio_list] 
    
    sim = -1
    u, s, vt = torch.svd(base_weight.to(torch.float))
    u_r = u[:, :r]
    ret_x = ret_y = None
    for thr in ratio_list:
        if thr == 1:
            new_x, new_y = x, y
        else:
            new_x, new_y = svd_compress(x, y, dim, thr) 
        new_param = (new_y @ new_x)
        nu, ns, nvt = torch.svd(new_param.to(torch.float))
        nu_r = nu[:, :r]
        phi = torch.norm(nu_r.T @ u_r,) ** 2 / r
        if phi > sim:
            sim = phi 
            ret_x = new_x 
            ret_y = new_y 

    return ret_x, ret_y

def l2lora_convert(x, y, dim, base_weight, step, save_name, range_start=5):
    r = x.shape[dim]
    cos_dim = 1 - dim
    ratio_list = [step * i for i in range(range_start, 10)]
    remaining_list = [max(1, int(r * ratio)) for ratio in ratio_list] 
    
    distance = 1e9
    # u, s, vt = torch.svd(base_weight.to(torch.float))
    # u_r = u[:, :r]
    ret_x = ret_y = None
    for thr in ratio_list:
        new_x, new_y = svd_compress(x, y, dim, thr) 
        new_param = (new_y @ new_x)
        phi = torch.norm(torch.abs(new_param-base_weight.to(torch.float)))
        if phi < distance:
            distance = phi 
            ret_x = new_x 
            ret_y = new_y 

    return ret_x, ret_y

def maxcoslora_convert(x, y, dim, base_weight, step, save_name, range_start=5):
    r = x.shape[dim]
    cos_dim = 1 - dim
    ratio_list = [step * i for i in range(range_start, 10)]
    remaining_list = [max(1, int(r * ratio)) for ratio in ratio_list] 
    
    sim = -1
    base_weight = base_weight.to(torch.bfloat16)
    # u, s, vt = torch.svd(base_weight.to(torch.float))
    # u_r = u[:, :r]
    ret_x = ret_y = None
    for thr in ratio_list:
        new_x, new_y = svd_compress(x, y, dim, thr) 
        new_param = (new_y @ new_x)
        product = new_param @ base_weight.T  # d'xd'
        phi = product.sum() / (new_param.norm() * base_weight.norm())
        if phi > sim:
            sim = phi 
            ret_x = new_x 
            ret_y = new_y 

    return ret_x, ret_y

def minlora_convert(x, y, dim, base_weight, step, save_name, range_start=5):
    r = x.shape[dim]
    cos_dim = 1 - dim
    ratio_list = [step * i for i in range(range_start, 10)]
    remaining_list = [max(1, int(r * ratio)) for ratio in ratio_list] 
    
    sim = 1e9
    u, s, vt = torch.svd(base_weight.to(torch.float))
    u_r = u[:, :r]
    ret_x = ret_y = None
    for thr in ratio_list:
        new_x, new_y = svd_compress(x, y, dim, thr) 
        new_param = (new_y @ new_x)
        nu, ns, nvt = torch.svd(new_param.to(torch.float))
        nu_r = nu[:, :r]
        phi = torch.norm(nu_r.T @ u_r,) ** 2 / r
        if phi < sim:
            sim = phi 
            ret_x = new_x 
            ret_y = new_y 

    return ret_x, ret_y

def minorlora_convert(x, y, dim, base_weight, step, save_name, range_start=5):
    r = x.shape[dim]
    cos_dim = 1 - dim
    ratio_list = [step * i for i in range(range_start, 10)]
    remaining_list = [max(1, int(r * ratio)) for ratio in ratio_list] 
    
    sim = -1
    u, s, vt = torch.svd(base_weight.to(torch.float))
    u_r = u[:, :r]
    ret_x = ret_y = None
    for thr in ratio_list:
        new_x, new_y = minorsvd_compress(x, y, dim, thr) 
        new_param = (new_y @ new_x)
        nu, ns, nvt = torch.svd(new_param.to(torch.float))
        nu_r = nu[:, :r]
        phi = torch.norm(nu_r.T @ u_r,) ** 2 / r
        if phi > sim:
            sim = phi 
            ret_x = new_x 
            ret_y = new_y 

    return ret_x, ret_y

def pcalora_convert(x, y, dim, base_weight, step, save_name, range_start=5):
    r = x.shape[dim]
    cos_dim = 1 - dim
    ratio_list = [step * i for i in range(range_start, 10)]
    remaining_list = [max(1, int(r * ratio)) for ratio in ratio_list] 
    
    sim = -1
    # print(base_weight.shape)
    u, s, vt = torch.svd(base_weight.to(torch.float))
    u_r = u[:, :r]
    ret_x = ret_y = None
    for thr in ratio_list:
        new_x, new_y = pca_compress(x, y, dim, thr) 
        # print(new_x.shape, new_y.shape)
        new_param = (new_y @ new_x)
        # print(new_param.shape)
        nu, ns, nvt = torch.svd(new_param.to(torch.float))
        nu_r = nu[:, :r]
        phi = torch.norm(nu_r.T @ u_r,) ** 2 / r
        if phi > sim:
            sim = phi 
            ret_x = new_x 
            ret_y = new_y 

    return ret_x, ret_y

def my_convert(x, y, dim, base_weight, step, save_name, range_start=5):
    r = x.shape[dim]
    cos_dim = 1 - dim
    ratio_list = [step * i for i in range(range_start, 10)]
    remaining_list = [max(int(r * ratio), 1) for ratio in ratio_list] 
    
    sim = -1
    u, s, vt = torch.svd(base_weight.to(torch.float))
    ret_x = ret_y = None
    for thr in ratio_list:
        new_x, new_y = svd_compress(x, y, dim, thr) 
        new_r = new_x.shape[0]
        u_r = u[:, :new_r]
        
        # new_param = (new_y @ new_x)
        # nu, ns, nvt = torch.svd(new_param.to(torch.float))
        # nu_r = nu[:, :r]
        phi = torch.norm(new_y.T.to(u_r) @ u_r,) ** 2 / new_r
        if phi > sim:
            sim = phi 
            ret_x = new_x 
            ret_y = new_y 

    return ret_x, ret_y



def convert_to_automodel(model_path, model_base, save_name=None, step=0.1, add_module=False, select_method='lora', range_start=5):
    # disable_torch_init()
    # assert model_path contains adapter_model.bin and non_lora_trainables.bin two files
    # model_name = get_model_name_from_path(model_path)
    base_model = AutoModelForCausalLM.from_pretrained(model_base)
    # lora_config = LoraConfig.from_pretrained(model_path)
    base_state_dict = base_model.state_dict()
    model = base_model
    state_dict = load_file(os.path.join(model_path, "adapter_model.safetensors"))
    state_dict = {key: value.to(base_model.device) for key, value in state_dict.items()}
    # tokenizer, model, context_len, tokenizer_with_adapter = load_molora_pretrained_model(model_path, model_base, model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_base)
    pairs = find_pair(state_dict)
    mapping = {
        "lora": lora_convert,
        "my": my_convert,
        "minlora": minlora_convert,
        "minorlora": minorlora_convert,
        "l2lora": l2lora_convert,
        "maxcoslora": maxcoslora_convert,
        "pcalora": pcalora_convert,
    }
    convert_func = mapping[select_method]
    if save_name is None:
        return
    new_state_dict = {}
    processed_set = set()
    for key, param in tqdm(state_dict.items()):
        if key in processed_set:
            continue
        if "lora_A" in key:
            r = param.shape[0]
            pair_key = key.replace("lora_A", "lora_B")
            pair_param = state_dict[pair_key]
            base_key = pairs[key]
            kwargs = {
                "dim": 0 if param.shape[0] == r else 1,
                "base_weight": base_state_dict[base_key],
                "step": step,
                'save_name': save_name,
                'range_start': range_start
                # "module": key.split(".")[-3]
            }
            if add_module:
                kwargs['module'] = key.split(".")[-3]
            param_A, param_B = convert_func(param, pair_param, **kwargs)
            # new_param = convert_func(param, dim=0, select_lora_rank=select_lora_rank)
            cur_r = param_A.shape[0] if param.shape[0] == r else param_A.shape[1]
            new_state_dict[key] = param_A 
            new_state_dict[pair_key] = param_B 
        else:
            r = param.shape[1]
            pair_key = key.replace("lora_B", "lora_A")
            pair_param = state_dict[pair_key]
            base_key = pairs[key]
            kwargs = {
                "dim": 0 if param.shape[0] == r else 1,
                "base_weight": base_state_dict[base_key],
                "step": step,
                'save_name': save_name,
                'range_start': range_start
                # "module": key.split(".")[-3]
            }
            if add_module:
                kwargs['module'] = key.split(".")[-3]
            param_A, param_B = convert_func(pair_param, param, **kwargs)
            # new_param = convert_func(param, dim=0, select_lora_rank=select_lora_rank)
            cur_r = param_A.shape[0] if param.shape[0] == r else param_A.shape[1]
            new_state_dict[key] = param_B 
            new_state_dict[pair_key] = param_A
        processed_set.add(key)
        processed_set.add(pair_key)
    # new_state_dict = {k[:-7] + ".default.weight": v for k, v in new_state_dict.items()}
    # print("Saving to original lora path")
    # save_file(new_state_dict, os.path.join(model_path, f"{save_name}.safetensors"))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="model path or lora")
    parser.add_argument("--model_base", type=str, help="the path of the base model that a peft model has")
    parser.add_argument("--save_name", type=str, help='the full model\'s save name')
    parser.add_argument('--add_module', action='store_true')
    parser.add_argument("--select_method", type=str, default='lora')
    parser.add_argument("--step", type=float, default=0.1, help="the search step")
    parser.add_argument("--range_start", type=int, default=5)
    args = parser.parse_args()
    model_path = args.model_path
    model_base = args.model_base
    save_name = args.save_name
    # save_path = args.save_path
    torch.set_default_device("cuda")
    convert_to_automodel(model_path, model_base, save_name=save_name, step=args.step, add_module=args.add_module, select_method=args.select_method, range_start=args.range_start)



