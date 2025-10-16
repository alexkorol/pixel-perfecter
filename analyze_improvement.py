import json
from pathlib import Path

from ml.inference import suggest_parameters
from src.pixel_reconstructor import _evaluate_with_parameters

metrics_path = Path('output/metrics.csv')
baseline = {}
with metrics_path.open() as fh:
    header = fh.readline().strip().split(',')
    for line in fh:
        parts = [p.strip() for p in line.strip().split(',')]
        if len(parts) < 8:
            continue
        record = dict(zip(header, parts))
        name = record['image']
        baseline[name] = float(record['percent_diff'])

improvements = []
no_suggestions = []
results = []

input_dir = Path('input')
for image_path in sorted(input_dir.glob('*.png')):
    name = image_path.name
    base = baseline.get(name)
    suggestions = suggest_parameters(image_path, top_k=5)
    if not suggestions:
        no_suggestions.append(name)
        continue
    best_diff = None
    best_sug = None
    for sug in suggestions:
        eval_result = _evaluate_with_parameters(
            image_path=image_path,
            cell_size=sug.cell_size,
            offset=sug.offset,
            debug=False,
        )
        diff = eval_result['metrics']['percent_diff']
        if best_diff is None or diff < best_diff:
            best_diff = diff
            best_sug = sug
    if base is not None and best_diff is not None:
        delta = base - best_diff
        results.append((name, base, best_diff, delta, best_sug))
        if delta > 0.1:
            improvements.append((name, delta, best_sug))

print(f"Images evaluated: {len(results)}")
print(f"Suggestions without improvement >0.1: {len(results) - len(improvements)}")
print(f"Suggestions improving >0.1: {len(improvements)}")
if improvements:
    print("Top improvements:")
    for name, delta, sug in sorted(improvements, key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {name}: delta={delta:.2f}% -> cell={sug.cell_size} off={sug.offset} conf={sug.confidence:.2f}")
