"""Run stage-1, risk analysis, SSS theta search, gaussian stage-2, flip detector."""
import argparse
from pathlib import Path

from analyze_risk_results import run_analysis
from answer_flip_detector import run as run_flip_detector
from mitigation_summary import run as run_mitigation_summary
from evaluate_best_stage1_new import EXPERIMENT_ROOT, METHODS, run_stage1
from evaluate_stage2_gaussian import run_stage2
from sss_denoise import run_sss


def main():
    parser = argparse.ArgumentParser(description="TC-VAD stage1 -> analysis -> SSS -> stage2")
    parser.add_argument("--experiment_root", default=str(EXPERIMENT_ROOT))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stage2_method", default="rotate_5", choices=METHODS)
    parser.add_argument("--stage1_methods", nargs="+", default=["rotate_5"], choices=METHODS)
    parser.add_argument("--skip_stage1", action="store_true")
    parser.add_argument("--stage1_limit", type=int, default=None, help="only first N image pairs")
    parser.add_argument("--skip_sss", action="store_true")
    parser.add_argument("--skip_flip_detector", action="store_true")
    parser.add_argument("--skip_mitigation_summary", action="store_true")
    parser.add_argument("--sss_kernels", nargs="+", type=int, default=[5])
    parser.add_argument("--sss_sigmas", nargs="+", type=float, default=[0.3, 0.5, 0.8, 1.0, 1.5])
    parser.add_argument("--theta_mode", default="set", choices=["set", "per_sample"])
    args = parser.parse_args()
    # Stage helpers chdir into BLIP_ROOT to load BLIP, so resolve before anything runs.
    root = str(Path(args.experiment_root).resolve())

    print("===== PIPELINE START: stage1 -> analysis -> SSS -> stage2 gaussian =====", flush=True)
    if not args.skip_stage1:
        run_stage1(
            experiment_root=root,
            device=args.device,
            methods=args.stage1_methods,
            limit=args.stage1_limit,
        )
        print("===== STAGE1 DONE, starting analysis =====", flush=True)
    else:
        print("===== SKIP stage1, starting analysis =====", flush=True)
    run_analysis(experiment_root=root, method=args.stage2_method)

    theta_json = None
    if not args.skip_sss:
        print("===== ANALYSIS DONE, starting SSS theta search =====", flush=True)
        result = run_sss(
            experiment_root=root,
            device=args.device,
            method=args.stage2_method,
            kernels=args.sss_kernels,
            sigmas=args.sss_sigmas,
        )
        if result is not None:
            candidate = Path(root).resolve() / "sss_best_theta.json"
            theta_json = str(candidate) if candidate.is_file() else None

    print("===== starting stage2 gaussian =====", flush=True)
    run_stage2(
        experiment_root=root,
        device=args.device,
        method=args.stage2_method,
        theta_json=theta_json,
        theta_mode=args.theta_mode,
    )

    if not args.skip_flip_detector:
        print("===== starting answer-flip detector (theta_det sweep) =====", flush=True)
        run_flip_detector(experiment_root=root)

    if not args.skip_mitigation_summary:
        print("===== starting mitigation ASR summary (theta*) =====", flush=True)
        try:
            run_mitigation_summary(experiment_root=root)
        except FileNotFoundError as e:
            print(f"mitigation_summary skipped: {e}", flush=True)
    print("===== PIPELINE COMPLETE =====", flush=True)


if __name__ == "__main__":
    main()
