#!/usr/bin/env python
"""
Upload trained PAW model to Hugging Face Hub
"""
import argparse
from pathlib import Path
from huggingface_hub import HfApi, create_repo

def upload_model_to_hf(
    model_path: str,
    repo_id: str,
    repo_type: str = "model",
    private: bool = True
):
    """Upload model to Hugging Face Hub."""
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    print(f"\n{'='*70}")
    print(f"Uploading PAW Model to Hugging Face")
    print(f"{'='*70}")
    print(f"Model: {model_path}")
    print(f"Repo:  {repo_id}")
    print(f"Private: {private}\n")
    
    api = HfApi()
    print("Creating/accessing repository...")
    try:
        create_repo(
            repo_id=repo_id,
            repo_type=repo_type,
            private=private,
            exist_ok=True
        )
        print(f"✓ Repository ready: https://huggingface.co/{repo_id}\n")
    except Exception as e:
        print(f"Warning: {e}\n")
    
    print("Uploading model weights...")
    api.upload_file(
        path_or_fileobj=str(model_path),
        path_in_repo=model_path.name,
        repo_id=repo_id,
        repo_type=repo_type,
    )
    print(f"✓ Uploaded: {model_path.name}\n")
    
    readme_content = f"""---
license: mit
tags:
- seismology
- phase-detection
- pytorch
---

# PAW Pretrained Model

Pretrained PAW (Picker for Arrival of Waves) model for seismic phase detection.

## Model Details

- **Architecture:** CNN + LSTM + Transformer hybrid
- **Sampling Rate:** 40 Hz (0.025s per sample)
- **Input:** 5-second waveforms with 0.5s padding on each side
- **Output:** Binary masks indicating phase arrival windows

## Usage

```python
from pawlib import PAW

# Load pretrained model from Hugging Face
model = PAW.from_pretrained('hf://suroRitch/pawlib-pretrained/paw_corrected.pt')

# Make predictions
predictions = model.predict(waveforms)
```

## License

MIT
"""
    
    print("Creating README...")
    try:
        api.upload_file(
            path_or_fileobj=readme_content.encode(),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type=repo_type,
        )
        print(f"✓ README created\n")
    except Exception as e:
        print(f"Note: {e}\n")
    
    print(f"{'='*70}")
    print(f"✅ Upload Complete!")
    print(f"{'='*70}")
    print(f"\nModel available at: https://huggingface.co/{repo_id}")
    print(f"\nTo use in code:")
    print(f"  from pawlib import PAW")
    print(f"  model = PAW.from_pretrained('hf://{repo_id}/{model_path.name}')")
    print()


def main():
    parser = argparse.ArgumentParser(description="Upload PAW model to Hugging Face")
    parser.add_argument(
        "--model",
        type=str,
        default="checkpoints_fixed/paw_corrected.pt",
        help="Path to trained model file"
    )
    parser.add_argument(
        "--repo",
        type=str,
        default="ArianaVillegas/pawlib-pretrained",
        help="Hugging Face repo ID (username/repo-name)"
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Make repository public (default: private)"
    )
    
    args = parser.parse_args()
    
    upload_model_to_hf(
        model_path=args.model,
        repo_id=args.repo,
        private=not args.public
    )


if __name__ == "__main__":
    main()
