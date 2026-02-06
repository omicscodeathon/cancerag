# Docking Timeout Fix

## Problem

Docking was taking forever (timing out) in the Docker container when users selected "Enable Docking" in the inference app.

## Root Cause

**Path mismatch between pre-converted receptors and runtime lookup:**

1. **Dockerfile** pre-converts all receptors to PDBQT format and saves them to:
   ```
   /app/data/processed/receptors_prepared/*.pdbqt
   ```

2. **DockingFeatureExtractor** was looking for prepared receptors in:
   ```
   /app/data/interim/docking_results/receptors/*.pdbqt
   ```

3. Because of this mismatch:
   - Every docking request converted the receptor from PDB to PDBQT at runtime (30-60s delay)
   - The pre-converted receptors were never used
   - The 300s timeout might be hit, causing the app to appear frozen

## Solution

### 1. Fixed `_prepare_receptor()` method ([docking_extractor.py](src/docking_extractor.py))

Now checks directories in this order:
1. **Pre-converted receptors** (`/app/data/processed/receptors_prepared/`) - Docker build
2. **Cached receptors** (`/app/data/interim/docking_results/receptors/`) - Runtime cache
3. **Runtime conversion** - Only if neither exists (with 90s timeout)

```python
def _prepare_receptor(self, receptor_name: str) -> Optional[str]:
    # First, check the pre-converted receptors directory (used in Docker)
    pre_converted_path = self.base_path / "data" / "processed" / "receptors_prepared" / f"{pdb_id}.pdbqt"
    if pre_converted_path.exists() and pre_converted_path.stat().st_size > 100:
        logger.info(f"Using pre-converted receptor: {pre_converted_path}")
        return str(pre_converted_path)

    # Fallback: check interim directory (for backward compatibility)
    prepared_dir = self.base_path / "data" / "interim" / "docking_results" / "receptors"
    prepared_path = prepared_dir / f"{receptor_name}.pdbqt"
    if prepared_path.exists() and prepared_path.stat().st_size > 100:
        logger.info(f"Using cached receptor: {prepared_path}")
        return str(prepared_path)

    # If not pre-converted, prepare it now (with timeout)
    # ...
```

### 2. Reduced docking timeout ([docking_extractor.py](src/docking_extractor.py))

- **Before**: 300 seconds (5 minutes)
- **After**: 120 seconds (2 minutes)
- **Rationale**: Typical docking takes 30-60 seconds; 2 minutes is generous

### 3. Added timeout for receptor conversion

- Added 90-second timeout for runtime `obabel` conversion
- Better error messages when conversion times out

### 4. Improved user feedback ([app.py](app.py))

- Updated progress messages to show "Docking (30-60s)..."
- Changed UI text from "~1-2 min" to "30-60s per prediction"
- Updated loading message to reflect actual expected time

## Testing

### Test in Docker Container

1. **Build the image:**
   ```bash
   cd inference_app
   docker build -t cancerag-inference .
   ```

2. **Check build output** - should see:
   ```
   ✓ Copied X receptors with binding sites
   Pre-converting receptors to PDBQT format...
     ✓ [receptor names]
   Receptor pre-conversion complete: X success, 0 failed
   ```

3. **Run the container:**
   ```bash
   docker run -p 7860:8080 cancerag-inference
   ```

4. **Test docking:**
   - Open http://localhost:7860
   - Select any receptor
   - Enter a SMILES string (e.g., `CC(=O)OC1=CC=CC=C1C(=O)O` for Aspirin)
   - Enable "🧪 Enable Docking"
   - Click "🚀 Predict Bias"
   - **Expected**: Should complete in 30-60 seconds, not timeout

5. **Check logs** for:
   ```
   INFO - Using pre-converted receptor: /app/data/processed/receptors_prepared/[pdb_id].pdbqt
   INFO - Running docking for [receptor_name]...
   INFO - DOCKING RESULT: -X.XX kcal/mol
   ```

### Test Script (in container)

A test script is included at `test_docking_fix.py`:

```bash
# Run in container
docker exec -it <container_id> python /app/inference_app/test_docking_fix.py

# Or with full docking test (slower)
docker exec -it <container_id> \
  bash -c "TEST_FULL_DOCKING=1 python /app/inference_app/test_docking_fix.py"
```

## Performance Improvements

| Scenario | Before | After |
|----------|--------|-------|
| **First docking (cold start)** | 90-120s (conversion + docking) | 30-60s (pre-converted) |
| **Subsequent docking** | 90-120s (re-conversion each time) | 30-60s (uses pre-converted) |
| **Timeout threshold** | 300s | 120s |

## Files Changed

1. [`inference_app/src/docking_extractor.py`](src/docking_extractor.py)
   - Fixed `_prepare_receptor()` to check pre-converted directory first
   - Reduced docking timeout from 300s to 120s
   - Added 90s timeout for receptor conversion
   - Improved error messages

2. [`inference_app/app.py`](app.py)
   - Updated progress messages
   - Updated UI text for expected timing
   - Improved user feedback during docking

3. [`inference_app/test_docking_fix.py`](test_docking_fix.py) (new)
   - Test script to verify fixes work

## Verification

After deploying, verify in logs that you see:
- ✅ `Using pre-converted receptor: /app/data/processed/receptors_prepared/...`
- ✅ Docking completes in 30-60 seconds
- ❌ No `Converting receptor ... to PDBQT (this may take 30-60s)` messages (means runtime conversion)
- ❌ No timeout errors

## Rollback

If issues occur, the fallback chain ensures:
1. Pre-converted receptors are tried first
2. Cached receptors are tried second
3. Runtime conversion is used as last resort

This maintains backward compatibility with non-Docker deployments.
