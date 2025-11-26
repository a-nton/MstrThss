# Implementation Plan: Dashboard Interactive Enhancements

## Overview
Add two new interactive features to the Bokeh dashboard:
1. **Tooltips for Whale Plot** - Show detailed prediction information on hover
2. **Interactive Confidence Highlighting** - Highlight predictions in both plots when hovering over confidence thresholds

## Phase 1: Add Tooltips to Whale Plot (Simple)

### Current State
The whale plot (magnitude analysis scatter) exists at [modules/viz.py:326-367](modules/viz.py#L326-L367) but has no HoverTool configured. The time series plot already has a working tooltip implementation we can use as reference.

### Implementation Steps

**Step 1.1: Enhance Whale Plot Data Source**
- Location: [modules/viz.py:338-344](modules/viz.py#L338-L344)
- Current `whale_source` only contains: `pred_abs`, `actual_abs`, `color`, `date`
- Add additional fields needed for rich tooltip:
  - `headline_text` - First 100 chars of headlines
  - `justification` - Agent reasoning summary
  - `ticker` - Stock symbol
  - `status` - Prediction outcome (correct/incorrect)
  - `confidence` - Model confidence level
  - `pred_return_pct` - Signed prediction (for direction info)
  - `actual_return_pct` - Signed actual return

**Step 1.2: Create Tooltip Template**
- Reference existing `base_tooltip` from [modules/viz.py:294-304](modules/viz.py#L294-L304)
- Create new `magnitude_tooltip` with format:
  ```
  Date: {date}
  Ticker: {ticker}
  Predicted: {pred_abs}% | Actual: {actual_abs}%
  Direction: {pred_return_pct} → {actual_return_pct}
  Confidence: {confidence}
  Status: {status}
  Headlines: {headline_text}
  Reasoning: {justification}
  ```

**Step 1.3: Add HoverTool to Whale Plot**
- Location: After [modules/viz.py:349](modules/viz.py#L349)
- Create HoverTool instance with `magnitude_tooltip`
- Configure formatters for date field: `{"@date": "datetime"}`
- Set mode to 'mouse' for better UX
- Add to whale_plot: `whale_plot.add_tools(hover)`

**Expected Outcome**: Hovering over any dot in the whale plot displays detailed prediction information.

---

## Phase 2: Interactive Confidence Highlighting (Complex)

### Goal
When hovering over a confidence threshold row in the calibration table (e.g., "≥70%"), highlight all predictions with confidence ≥70% in both the time series plot and whale plot by increasing their size and opacity.

### Technical Challenges
1. **HTML Div Limitation**: Current calibration table is a static HTML Div ([modules/viz.py:432-455](modules/viz.py#L432-L455)), which cannot trigger Bokeh callbacks
2. **Cross-Plot Synchronization**: Need to update glyphs in two separate plots simultaneously
3. **Performance**: Highlighting logic must be fast enough for smooth hover interactions

### Solution Architecture

**Approach A: Convert Calibration Table to Bokeh DataTable (Recommended)**
- Replace HTML Div with Bokeh DataTable widget
- DataTable supports TapTool and selection callbacks
- Can use `.selected` property to trigger CustomJS

**Approach B: Add Clickable Button Widgets**
- Keep HTML table for display
- Add small Button widgets next to each threshold row
- Button hover triggers CustomJS callback
- Less elegant but easier to implement

### Implementation Steps (Approach A - DataTable)

**Step 2.1: Prepare Data for Interactive Highlighting**
- Location: [modules/viz.py:338-344](modules/viz.py#L338-L344) and [modules/viz.py:281-287](modules/viz.py#L281-L287)
- Add `confidence` column to both ColumnDataSources:
  - `ts_source` (time series plot data)
  - `whale_source` (whale plot data)
- Store confidence as decimal (0.0-1.0) for easy filtering

**Step 2.2: Create Calibration DataTable**
- Location: Replace [modules/viz.py:432-470](modules/viz.py#L432-L470)
- Build DataFrame with columns: `threshold`, `count`, `accuracy`
- Convert to ColumnDataSource: `calibration_source`
- Create DataTable widget with columns:
  - `threshold` (string, e.g., "≥70%")
  - `count` (integer)
  - `accuracy` (formatted percentage)
- Style using TableColumn formatters

**Step 2.3: Implement CustomJS Callback**
- Create JavaScript callback that:
  1. Reads selected row from `calibration_source`
  2. Extracts threshold value (e.g., 0.7 from "≥70%")
  3. Filters `ts_source.data.confidence` and `whale_source.data.confidence`
  4. Creates boolean masks for points meeting threshold
  5. Updates glyph properties:
     - **Highlighted**: `size=12`, `alpha=1.0`, `line_width=2`
     - **Dimmed**: `size=6`, `alpha=0.2`, `line_width=1`
- Attach callback to DataTable's `source.selected` change event

**Step 2.4: Handle Hover vs Click**
- **Option 1 (Hover)**: Use TapTool with `behavior='inspect'` for hover-like behavior
- **Option 2 (Click)**: Use SelectTool, require click to activate highlighting
- **Option 3 (Hybrid)**: Add "Clear Highlight" button to reset

**Step 2.5: Synchronize Both Plots**
- Both plots share the same confidence filtering logic
- Time series plot: Update circle renderer sizes/alphas
- Whale plot: Update scatter renderer sizes/alphas
- Ensure consistent visual feedback across both plots

**Step 2.6: Add Visual Feedback**
- Selected calibration row: Change background color to highlight active threshold
- Add subtle animation (optional): Smooth transitions using CSS or Bokeh's animate property
- Display active threshold in plot titles: "Time Series (Highlighting ≥70%)"

### Alternative Implementation (Approach B - Button Widgets)

If DataTable proves too complex:

**Step 2B.1: Keep HTML Table**
- Maintain existing HTML Div with calibration data
- Add unique IDs to each row: `<tr id="cal-threshold-70">`

**Step 2B.2: Add Button Column**
- Add buttons next to each threshold: `[👁️ Highlight]`
- Create Bokeh Button widgets: `button_70 = Button(label="70%", width=60)`
- Arrange in column layout next to HTML table

**Step 2B.3: Button Callback**
- Each button's `.on_click()` triggers CustomJS
- CustomJS implementation same as Approach A, but threshold value is hardcoded per button

**Step 2B.4: Layout**
- Row layout: `[calibration_html_div, button_column]`
- Buttons aligned with table rows using spacer Divs

---

## Implementation Order

1. **Phase 1** (1 hour): Add whale plot tooltips
   - Low risk, immediate value
   - Reuses existing tooltip patterns
   - No complex interactions

2. **Phase 2A - Prototype** (2 hours): Test DataTable approach
   - Create simple prototype with one plot
   - Verify callback performance
   - Validate user experience

3. **Phase 2B - Full Implementation** (3 hours): Complete interactive highlighting
   - Extend to both plots
   - Add visual polish (transitions, colors)
   - Test edge cases (no data, all thresholds)

4. **Fallback** (if Phase 2A fails): Implement Approach B with buttons

---

## Files to Modify

### Primary File
- **[modules/viz.py](modules/viz.py)** - Dashboard generation
  - Lines 281-287: Enhance `ts_source` with confidence data
  - Lines 326-367: Enhance whale plot with tooltip
  - Lines 432-470: Replace calibration HTML with DataTable
  - New section: Add CustomJS callback functions

### No Changes Needed
- [config.py](config.py) - Configuration (already updated)
- [modules/data_sources/gdelt_bigquery.py](modules/data_sources/gdelt_bigquery.py) - News fetching (already fixed)
- [modules/llm_configs/maker_consensus.py](modules/llm_configs/maker_consensus.py) - Already provides confidence data

---

## Testing Plan

### Phase 1 Testing
- [ ] Tooltip displays on hover over whale plot dots
- [ ] All fields render correctly (date, confidence, headlines, etc.)
- [ ] Tooltip formatting matches time series tooltip style
- [ ] No performance degradation with 90+ data points

### Phase 2 Testing
- [ ] Selecting calibration threshold highlights correct dots
- [ ] Highlighting works in both time series and whale plots simultaneously
- [ ] Visual feedback is clear (highlighted dots are obviously different)
- [ ] Deselecting threshold resets all dots to normal state
- [ ] Performance is smooth (no lag when hovering/clicking)
- [ ] Edge cases handled:
  - No predictions meet threshold → show message or dim all
  - All predictions meet threshold → highlight all
  - Multiple rapid selections → no visual glitches

---

## Risk Assessment

### Phase 1 Risks: **LOW**
- Tooltip implementation is straightforward
- Existing pattern to follow from time series plot
- No state management or callbacks needed

### Phase 2 Risks: **MEDIUM**
- **DataTable styling**: May be difficult to match existing HTML table appearance
- **Callback complexity**: CustomJS debugging can be time-consuming
- **Performance**: Need to ensure smooth updates with 90+ points per plot
- **Bokeh version compatibility**: CustomJS API may vary between versions

### Mitigation Strategies
- Start with Phase 1 to deliver immediate value
- Prototype Phase 2 with single plot before full implementation
- Keep Approach B (buttons) as fallback if DataTable proves problematic
- Test with maximum expected data volume (7 tickers × 90 days = 630 points)

---

## Expected Benefits

### Phase 1 Benefits
- **Better exploration**: Users can inspect individual predictions in detail
- **Debugging aid**: Easier to identify why certain predictions failed
- **Consistency**: Matches existing time series plot UX

### Phase 2 Benefits
- **Confidence analysis**: Visually understand which predictions were high/low confidence
- **Calibration validation**: Quickly see if high-confidence predictions cluster correctly
- **Pattern discovery**: Identify if certain confidence levels correlate with magnitude errors
- **Interactive exploration**: More engaging and informative than static tables

---

## Success Criteria

### Phase 1
✅ Whale plot tooltip shows all relevant prediction details
✅ Tooltip formatting is consistent with existing dashboard style
✅ No performance issues

### Phase 2
✅ Hovering/clicking confidence threshold highlights predictions in both plots
✅ Visual highlighting is clear and intuitive
✅ Interaction is smooth and responsive
✅ Feature adds genuine analytical value (not just visual flair)

---

## Notes

- Consider adding keyboard shortcuts for Phase 2 (e.g., press 1-9 to select threshold)
- Future enhancement: Add slider widget to dynamically adjust threshold
- Future enhancement: Add "Show only highlighted" toggle to hide dimmed points
- Ensure accessibility: Highlighted state should be clear even without color (size/opacity)
