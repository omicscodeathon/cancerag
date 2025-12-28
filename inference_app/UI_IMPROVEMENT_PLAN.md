# UI and Functionality Improvement Plan

## 1. Docking Failure Issues

### Root Causes:
1. **Silent Vina Failures**: The subprocess redirects output but doesn't properly log errors
2. **Missing Dependencies**: Vina binary might not be in PATH or accessible
3. **File Permission Issues**: Temporary files might not have correct permissions
4. **Timeout Issues**: 5-minute timeout might be too short for slower systems

### Fixes:
- Add better error logging with actual vina output
- Check vina availability at startup
- Improve subprocess error handling
- Add retry logic for transient failures
- Display actual error messages to users instead of generic "docking failed"

## 2. Dark Mode Compatibility Issues

### Current Problems:
1. **Hardcoded light colors**: Some inline styles use `#f8f9fa` (light gray) backgrounds
2. **Insufficient contrast**: `#666` gray text may be hard to read in dark mode
3. **Gradient backgrounds**: Purple gradient (`#667eea`, `#764ba2`) might clash with dark mode
4. **Warning colors**: `#fff3cd` (light yellow) background doesn't work in dark mode
5. **Missing CSS variables**: Not all elements use Gradio's CSS variables

### Fixes:
- Replace all hardcoded colors with CSS variables
- Use semantic color names (`--primary`, `--secondary`, `--background`, `--text`)
- Add dark mode specific overrides
- Test all states (error, warning, success, info) in both modes

## 3. UI Structure Improvements

### Current Issues:
1. **Visual Hierarchy**: Headers and sections could be more distinct
2. **Information Density**: Too much text in some areas
3. **Mobile Responsiveness**: Some layouts may not work well on smaller screens
4. **Loading States**: No clear indication when operations are in progress
5. **Empty States**: Generic messages when no data is available

### Improvements:
- Add loading spinners for long operations
- Better empty state designs
- Clearer visual separation between sections
- Improved spacing and padding
- Better typography hierarchy
- Progress indicators for multi-step operations

## 4. Specific UI Elements to Fix

### Colors to Replace:
```python
# Light mode colors (BAD)
background: #f8f9fa
color: #666
color: #333
border: #ddd
background: #fff

# Should use (GOOD)
background: var(--background-fill-primary)
color: var(--body-text-color)
border: var(--border-color-primary)
```

### HTML Templates to Improve:
1. Empty receptor state
2. Error messages
3. Docking results display
4. Ligand visualization info
5. Receptor information cards

## 5. Implementation Priority

### Phase 1 (Critical):
1. Fix docking failures with better error handling
2. Fix dark mode colors on critical elements

### Phase 2 (Important):
3. Improve all HTML templates for dark mode
4. Add loading states

### Phase 3 (Nice to have):
5. Enhanced visual design
6. Better animations and transitions
