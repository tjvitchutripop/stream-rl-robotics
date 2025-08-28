import torch
def relabel_checkpoint(old_ckpt_path, new_ckpt_path):
    # Load checkpoint
    checkpoint = torch.load(old_ckpt_path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"]
    new_state_dict = {}
    # Mapping rules
    critic_map = {
        "0": "fc1",
        "2": "fc2",
        "4": "fc3",
        "6": "value"
    }
    actor_map = {
        "0": "fc1",
        "2": "fc2",
        "4": "fc3",
        "6": "fc4"
    }
    for key, val in state_dict.items():
        if key.startswith("critic."):
            _, idx, param = key.split(".")
            new_key = f"critic.{critic_map[idx]}.{param}"
        elif key.startswith("actor_mean."):
            _, idx, param = key.split(".")
            new_key = f"actor_mean.{actor_map[idx]}.{param}"
        else:
            new_key = key  # keep unchanged (like actor_logstd)
        new_state_dict[new_key] = val
    # Replace in checkpoint and save
    checkpoint["model_state_dict"] = new_state_dict
    torch.save(checkpoint, new_ckpt_path)
    print(f"Relabeled checkpoint saved to {new_ckpt_path}")
# Example usage:
relabel_checkpoint("obgd_ppo_pretrain_old.pt", "obgd_ppo_pretrain_old2.pt")