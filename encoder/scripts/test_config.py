#!/usr/bin/env python3
"""
Test script to verify hyperparameters are being loaded correctly
"""

import sys
sys.path.insert(0, 'encoder')

from config.configurator import configs

print("=" * 80)
print("Current Configuration for mult_vae_mddm_mmd")
print("=" * 80)
print(f"Model name: {configs['model']['name']}")
print(f"Dataset: {configs['data']['name']}")
print()

dataset = configs['data']['name']
if dataset in configs['model']:
    print(f"Dataset-specific parameters for '{dataset}':")
    for key, value in configs['model'][dataset].items():
        print(f"  {key}: {value}")
else:
    print(f"No dataset-specific parameters found for '{dataset}'")

print()
print("Global model parameters:")
for key, value in configs['model'].items():
    if key != 'name' and not isinstance(value, dict):
        print(f"  {key}: {value}")

print("=" * 80)
