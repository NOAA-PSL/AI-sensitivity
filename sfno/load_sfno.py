import torch
import ai_models_fourcastnetv2.fourcastnetv2 as nvs

def load_sfno(checkpoint_file, device):
    model = nvs.FourierNeuralOperatorNet()

    model.zero_grad()
    # Load weights

    checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)

    weights = checkpoint["model_state"]
    drop_vars = ["module.norm.weight", "module.norm.bias"]
    weights = {k: v for k, v in weights.items() if k not in drop_vars}

    # Make sure the parameter names are the same as the checkpoint
    # need to use strict = False to avoid this error message when
    # using sfno_76ch::
    # RuntimeError: Error(s) in loading state_dict for Wrapper:
    # Missing key(s) in state_dict: "module.trans_down.weights",
    # "module.itrans_up.pct",
    try:
        # Try adding model weights as dictionary
        new_state_dict = dict()
        for k, v in checkpoint["model_state"].items():
            name = k[7:]
            if name != "ged":
                new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    except Exception:
        model.load_state_dict(checkpoint["model_state"])

    # Set model to eval mode and return
    model.eval()
    model.to(device)

    return model

