"""Convert an extension JSONL to the same 530 inputs, plus a calibration report.

This does not train: NEAT learns through interaction with the local simulator.
"""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from .schema import record_inputs, record_outputs, contract
from .io import write_json

def convert(source, output):
    inputs, outputs, times, speed_ratios, turns, radii, food_counts = [], [], [], [], [], [], []
    previous, previous_boost = None, False
    with Path(source).open(encoding='utf-8-sig') as stream:
        for line in stream:
            record = json.loads(line)
            if record.get('type') != 'sample': continue
            inputs.append(record_inputs(record, previous_boost))
            outputs.append(record_outputs(record))
            times.append(record['wallTimeMs'])
            p = record['lidar']['player']
            radii.append(p['bodyRadiusEstimate'])
            food_counts.append(len(record['lidar']['visibleFood']))
            if previous:
                q = previous['lidar']['player']
                dt = (record['wallTimeMs'] - previous['wallTimeMs']) / 1000
                if dt <= 0: raise ValueError('Non-increasing timestamps')
                if dt <= .3:
                    speed = math.hypot(p['x']-q['x'], p['y']-q['y']) / dt
                    if q['speedRaw'] > 0: speed_ratios.append(speed / q['speedRaw'])
                    diff = p['heading']-q['heading']
                    turns.append(abs(math.atan2(math.sin(diff), math.cos(diff))) / dt)
            previous, previous_boost = record, record['action']['boost']
    if not inputs: raise ValueError('No samples')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, inputs=np.stack(inputs), outputs=np.stack(outputs), wall_time_ms=np.array(times))
    report = dict(samples=len(inputs), duration_seconds=(times[-1]-times[0])/1000, schema=contract(),
                  median_world_speed_per_speed_raw=float(np.median(speed_ratios)) if speed_ratios else None,
                  observed_turn_rate_p95=float(np.quantile(turns, .95)) if turns else None,
                  median_body_radius=float(np.median(radii)), mean_visible_food=float(np.mean(food_counts)),
                  note='Observed turns are human actions, not a measurement of maximum turn rate. Review before adjusting physics.')
    write_json(output.with_suffix('.json'), report)
    return report

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source')
    parser.add_argument('--out', default='runs/recorded-inputs.npz')
    args = parser.parse_args()
    print(json.dumps(convert(args.source, args.out), indent=2))
