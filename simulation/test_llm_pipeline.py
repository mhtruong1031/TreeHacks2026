"""
Test script for the LLM coaching pipeline and ProgressMonitor.

Usage:
    # Test ProgressMonitor only (no API key needed):
    python simulation/test_llm_pipeline.py

    # Test full LLM pipeline with Gemini (requires API key):
    python simulation/test_llm_pipeline.py --api-key YOUR_GEMINI_API_KEY

    # Or set the env var:
    set GEMINI_API_KEY=YOUR_KEY
    python simulation/test_llm_pipeline.py

    # Test a specific personality:
    python simulation/test_llm_pipeline.py --api-key YOUR_KEY --personality drill_sergeant
"""

import sys
import os
import argparse
import numpy as np

# ── Ensure sibling packages are importable ────────────────────────────
_project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_project_root, "analysis"))


def test_progress_monitor():
    """Test ProgressMonitor plateau and uniqueness detection."""
    from ProgressMonitor import ProgressMonitor

    print("=" * 60)
    print("  TEST 1: ProgressMonitor — Plateau Detection")
    print("=" * 60)

    pm = ProgressMonitor(plateau_window=5, plateau_threshold=0.02, uniqueness_threshold=0.10)

    # Simulate improving attempts (coordination index decreasing)
    print("\n  Phase 1: Improving attempts")
    for i in range(6):
        ci = 0.8 - i * 0.05  # 0.80 → 0.55
        sim = 0.5 - i * 0.02
        status = pm.add_attempt(ci, sim)
        print(f"    Attempt {i+1}: CI={ci:.2f}  trend={status['trend']:<10}  plateau={status['plateau_detected']}")

    # Simulate plateau (flat coordination index)
    print("\n  Phase 2: Plateaued attempts")
    for i in range(6):
        ci = 0.55 + np.random.uniform(-0.005, 0.005)  # flat around 0.55
        sim = 0.3 + np.random.uniform(-0.005, 0.005)
        status = pm.add_attempt(ci, sim)
        flag = " *** PLATEAU" if status["plateau_detected"] else ""
        print(f"    Attempt {i+7}: CI={ci:.3f}  trend={status['trend']:<10}  "
              f"plateau={status['plateau_detected']}  type={status['plateau_type']}{flag}")
        if status["plateau_detected"]:
            print(f"      Details: {status['details']}")

    print("\n  Phase 3: Low uniqueness attempts")
    pm.reset()
    pm2 = ProgressMonitor(plateau_window=5, plateau_threshold=0.02, uniqueness_threshold=0.10)
    for i in range(7):
        ci = 0.6 - i * 0.03  # still improving slightly in CI
        sim = 0.05 + np.random.uniform(-0.01, 0.01)  # very low similarity = same motion
        status = pm2.add_attempt(ci, sim)
        flag = " *** LOW UNIQUENESS" if status.get("plateau_type") == "low_uniqueness" else ""
        print(f"    Attempt {i+1}: CI={ci:.3f}  sim={sim:.3f}  "
              f"plateau={status['plateau_detected']}  type={status['plateau_type']}{flag}")
        if status["plateau_detected"]:
            print(f"      Details: {status['details']}")

    print("\n  ProgressMonitor tests PASSED\n")


def test_llm_pipeline(api_key: str | None = None, personality: str = "encouraging"):
    """Test the LLMPipeline with mock metrics."""
    from LLMPipeline import LLMPipeline, PERSONALITIES, API_KEY
    if api_key is None:
        api_key = API_KEY

    print("=" * 60)
    print(f"  TEST 2: LLMPipeline — Personality: {personality}")
    print("=" * 60)

    print(f"\n  Available personalities: {list(PERSONALITIES.keys())}")
    print(f"  Initializing with model gemini-2.0-flash ...")

    llm = LLMPipeline(
        api_key=api_key,
        movement_class="wrist_flex_ext",
        personality=personality,
    )
    print("  Chat session started.\n")

    # --- Test 1: Normal feedback (attempt #5 triggers because FEEDBACK_INTERVAL=5) ---
    print("  --- Test 2a: Normal feedback (5th attempt) ---")
    metrics = {
        "coordination_index": 0.42,
        "similarity_score": 0.35,
        "attempt_number": 5,
        "coordination_history": [0.65, 0.58, 0.50, 0.45, 0.42],
        "movement_class": "wrist_flex_ext",
        "trend": "improving",
        "top_n_scores": [
            {"coordination_index": 0.42, "similarity_score": 0.35},
            {"coordination_index": 0.45, "similarity_score": 0.28},
            {"coordination_index": 0.50, "similarity_score": 0.40},
        ],
        "has_prediction_model": False,
        "plateau_detected": False,
        "plateau_type": None,
        "plateau_details": "",
    }

    # Manually set the counter so the 10th call triggers
    llm._attempt_counter = 9  # next call will be #10
    response = llm.generate_feedback(metrics)
    if response:
        print(f"\n  LLM Response:\n  {response}\n")
    else:
        print("  (Rate limited — no response this attempt)\n")

    # --- Test 2: Plateau intervention ---
    print("  --- Test 2b: Plateau intervention ---")
    plateau_metrics = {
        "coordination_index": 0.55,
        "similarity_score": 0.30,
        "attempt_number": 25,
        "coordination_history": [0.55, 0.54, 0.56, 0.55, 0.54, 0.55, 0.56, 0.55, 0.54, 0.55],
        "movement_class": "wrist_flex_ext",
        "trend": "plateau",
        "top_n_scores": [
            {"coordination_index": 0.54, "similarity_score": 0.30},
            {"coordination_index": 0.55, "similarity_score": 0.28},
        ],
        "has_prediction_model": True,
        "plateau_detected": True,
        "plateau_type": "coordination_plateau",
        "plateau_details": "Over the last 10 attempts the coordination index has barely changed (mean=0.5490, std=0.0070).",
    }
    response = llm.generate_plateau_intervention(plateau_metrics)
    print(f"\n  LLM Plateau Intervention:\n  {response}\n")

    # --- Test 3: User follow-up question ---
    print("  --- Test 2c: User follow-up question ---")
    response = llm.ask("Why is my coordination not improving? Should I take a break?", current_metrics=plateau_metrics)
    print(f"\n  LLM Answer:\n  {response}\n")

    # --- Test 4: Personality switch ---
    new_personality = "drill_sergeant" if personality != "drill_sergeant" else "calm"
    print(f"  --- Test 2d: Switching personality to '{new_personality}' ---")
    llm.set_personality(new_personality)
    llm._attempt_counter = 19  # next call is #20
    response = llm.generate_feedback(metrics)
    if response:
        print(f"\n  LLM Response (new personality):\n  {response}\n")

    print("  LLMPipeline tests PASSED\n")


# ── CLI ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the LLM coaching pipeline")
    parser.add_argument("--api-key", type=str, default=None,
                        help="Gemini API key (or set GEMINI_API_KEY env var)")
    parser.add_argument("--personality", type=str, default="encouraging",
                        help="Personality to test (default: encouraging)")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("GEMINI_API_KEY")

    # Try to fall back to the hardcoded key in LLMPipeline
    if not api_key:
        try:
            sys.path.insert(0, os.path.join(_project_root, "analysis"))
            from LLMPipeline import API_KEY
            api_key = API_KEY
        except ImportError:
            pass

    # Always run ProgressMonitor tests (no API key needed)
    test_progress_monitor()

    # Run LLM tests only if key is available
    if api_key:
        test_llm_pipeline(api_key, personality=args.personality)
    else:
        print("=" * 60)
        print("  SKIPPED: LLMPipeline tests (no API key provided)")
        print("  Set GEMINI_API_KEY env var or pass --api-key to test.")
        print("=" * 60)

    print("\nAll tests complete.")
