"""
Quick test script for the inference pipeline.

This script tests the inference pipeline with a simple example.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.inference_pipeline import InferencePipeline
from src.predictor import load_predictor


def test_inference():
    """Test the inference pipeline with a simple SMILES."""
    print("Testing CancerAg Inference Pipeline...")
    print("=" * 60)

    try:
        # Load predictor
        print("\n1. Loading predictor...")
        predictor = load_predictor(model_name="random_forest")
        print("   ✓ Predictor loaded successfully")

        # Create pipeline
        print("\n2. Creating inference pipeline...")
        pipeline = InferencePipeline(predictor)
        print("   ✓ Pipeline created successfully")

        # Test prediction
        print("\n3. Testing prediction...")
        test_smiles = "CCO"  # Ethanol
        print(f"   Input SMILES: {test_smiles}")

        result = pipeline.predict_from_smiles(test_smiles)

        if result["success"]:
            print("   ✓ Prediction successful!")
            print(f"   Predicted Class: {result['predicted_class']}")
            print("   Probabilities:")
            for class_name, prob in sorted(
                result["probabilities"].items(), key=lambda x: x[1], reverse=True
            ):
                print(f"     - {class_name}: {prob:.1%}")
        else:
            print(f"   ✗ Prediction failed: {result.get('error', 'Unknown error')}")
            return False

        print("\n" + "=" * 60)
        print("All tests passed! ✓")
        return True

    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_inference()
    sys.exit(0 if success else 1)
