"""Outcome-Anchored Supervision (OAS) — NISP's outcome-anchored contract-learning mechanism.

OAS uses the high-integrity formal safety outcome (the "Anchor") to supervise
the neural intent signal. It minimizes the dissonance (D) by updating the
decoder weights to better predict the physical safety facts.

Invariant:
    Formal axioms (Lean 4) and physical thresholds (thresholds.yaml) are NEVER
    mutated. Learning only happens in the neuro-symbolic decoding bridge
    (Linear Probes / Adapter).
"""
import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Dict
from src.nisp.core.contracts import CyclePayload

class OASReplay:
    """Replays a NISP log and performs supervised contract adaptation."""
    
    def __init__(self, probe_weights_path: str = "artifacts/probe_weights.pt", 
                 lr: float = 1e-4):
        self.probe_weights_path = probe_weights_path
        self.lr = lr
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load current weights if they exist, otherwise initialize a new probe
        # (Assuming K=8 probes on DiFuMo-1024)
        self.model = nn.Linear(1024, 8)
        if os.path.exists(probe_weights_path):
            try:
                state_dict = torch.load(probe_weights_path, map_location="cpu")
                # Handle possible key differences between script and model
                if 'linear.weight' in state_dict:
                    self.model.weight.data = state_dict['linear.weight']
                    self.model.bias.data = state_dict['linear.bias']
                else:
                    self.model.load_state_dict(state_dict)
            except Exception as e:
                print(f"  [OAS] Warning: could not load existing weights ({e})")
        
        self.model.to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()

    def train_on_log(self, log_path: str, epochs: int = 5) -> List[float]:
        """Runs N epochs of training on a single JSONL log file.
        
        Returns: list of mean loss per epoch (monotonicity check).
        """
        # 1. Load cycles from log
        X, y = [], []
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                if obj.get("_header"): continue
                
                # We need the latent E (pre-projection) or at least the Z and facts.
                # In the current log format, we don't log E. 
                # This means OAS must run against the precomputed latents folder.
                # For this PoC, we map cycle_id -> artifacts/latents/cycle_i_difumo.npy
                cycle_id = obj["cycle_id"]
                latent_path = f"artifacts/latents/cycle_{cycle_id}_difumo.npy"
                
                if os.path.exists(latent_path):
                    E = np.load(latent_path)
                    X.append(E)
                    
                    # Target y is the "Anchor": the primary trigger index
                    # 0=prox, 1=torque, 2=geo, 3=vel, 7=nominal (approx)
                    # We use the index of the first True fact or 7 if none.
                    facts = obj.get("facts", [])
                    label = 7
                    for idx, (prop, val, reading, thresh) in enumerate(facts):
                        if val:
                            # Map proposition name to predicate index
                            # (Mirroring src/nisp/core/orchestrator.py)
                            mapping = {
                                "proximity_violation": 0,
                                "torque_exceedance": 1,
                                "geofence_breach": 2,
                                "velocity_limit": 3
                            }
                            label = mapping.get(prop, 7)
                            break
                    y.append(label)

        if not X:
            return []

        X_tensor = torch.tensor(np.array(X), dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(np.array(y), dtype=torch.long).to(self.device)

        history = []
        for epoch in range(epochs):
            self.model.train()
            logits = self.model(X_tensor)
            loss = self.criterion(logits, y_tensor)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            history.append(float(loss.item()))
            
        return history

    def save_weights(self, path: str = None):
        if path is None: path = self.probe_weights_path
        torch.save(self.model.state_dict(), path)
        print(f"  [OAS] Saved updated weights to {path}")
