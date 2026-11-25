"""Score one precomputed four-feature TC-VAD vector with a trusted SVM artifact."""
import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--d-center", required=True, type=float)
    parser.add_argument("--d-iqr", required=True, type=float)
    parser.add_argument("--d-tail", required=True, type=float)
    parser.add_argument("--d-q95", required=True, type=float)
    parser.add_argument("--image-id", help="Required for an OOF replay fold, not deployment")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "runtime/blip"))
    from integration.detector import load_detector, score_features

    artifact = load_detector(args.model)
    values = {"d_center": args.d_center, "d_iqr": args.d_iqr,
              "d_tail": args.d_tail, "d_q95": args.d_q95}
    result = score_features(artifact, values, image_id=args.image_id)
    print(json.dumps({"caption_model": artifact.get("caption_model", "blip"),
                      "features": values, **result}, ensure_ascii=False))


if __name__ == "__main__":
    main()
