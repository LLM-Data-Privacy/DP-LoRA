def aggregate_updates(client_updates):
    aggregated_update = {}
    for update in client_updates:
        for name, param in update.items():
            if name in aggregated_update:
                aggregated_update[name] += param
            else:
                aggregated_update[name] = param

    # 计算平均值
    for name in aggregated_update:
        aggregated_update[name] /= len(client_updates)

    return aggregated_update