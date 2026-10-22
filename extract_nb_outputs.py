import json

nb = json.load(open('notebooks/agg_clustering.ipynb', 'r', encoding='utf-8'))
cells = nb['cells']

for cell_idx in [49, 53]:
    print(f"\n{'='*60}")
    print(f"Cell {cell_idx} outputs:")
    print('='*60)
    outputs = cells[cell_idx].get('outputs', [])
    for j, o in enumerate(outputs):
        otype = o.get('output_type', '')
        print(f"\n--- Output {j} ({otype}) ---")
        if 'text' in o:
            print(''.join(o['text'][:10]))
        if 'data' in o:
            if 'text/plain' in o['data']:
                plain = o['data']['text/plain']
                if isinstance(plain, list):
                    print(''.join(plain[:30]))
                else:
                    print(plain[:3000])
