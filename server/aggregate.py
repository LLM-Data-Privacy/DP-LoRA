import torch

def aggregate_updates(client_updates):
    """
    Aggregate model parameter updates from multiple clients.
    Args:
        client_updates (list of dict): An updated list of model parameters from the client.
    Returns:
        dict: Model parameter update after aggregation
    """
    aggregated_update = {}
    for update in client_updates:
        for name, param in update.items():
            if name in aggregated_update:
                aggregated_update[name] += param
            else:
                aggregated_update[name] = param.clone()

    # 计算平均值
    for name in aggregated_update:
        aggregated_update[name] /= len(client_updates)

    return aggregated_update

def apply_updates_to_model(model, aggregated_update):
    """
    Apply aggregated updates to existing models.
    Args:
        model (torch.nn.Module): Models to be updated。
        aggregated_update (dict): Model parameter update after aggregation。
    """
    current_state_dict = model.state_dict()
    updated_state_dict = {name: current_state_dict[name] + aggregated_update.get(name, torch.zeros_like(current_state_dict[name]))
                          for name in current_state_dict}
    model.load_state_dict(updated_state_dict)
