import ijson

filename = "/apdcephfs_cq10/share_1567347/arisyhzhang/Restart_HyperInstrucT/datasets/P3/test_features.json"

def count_items(filename):
    with open(filename, 'rb') as f:
        objects = ijson.items(f, 'item')
        count = sum(1 for _ in objects)
    return count

print(count_items(filename))