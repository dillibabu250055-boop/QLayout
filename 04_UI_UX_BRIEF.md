# 04_UI_UX_BRIEF.md

## Visual Concept
Professional engineering application (EDA tool). Dark mode preferred (cyberpunk/synthwave accents).

## Layout Structure
```text
┌──────────────────────────────────────────────────────┐
│ Q-LAYOUT                         Project: Demo Chip  │
├────────────┬───────────────────────────┬─────────────┤
│            │                           │             │
│ PROJECT    │                           │   RISK      │
│            │       CHIP CANVAS         │   PANEL     │
│ - Layout   │                           │             │
│ - Circuit  │        Q0   Q1            │ [🔴] Q2-Q5  │
│ - Rules    │             ╲             │ [⚠] Q1-Q4   │
│            │              Q5           │ [✓] Q0-Q1   │
│            │                           │             │
├────────────┴───────────────────────────┴─────────────┤
│ LQS: 87        DRC: 2 Warnings      Optimize [▶]    │
└──────────────────────────────────────────────────────┘
```

## Colors & Indicators
- **High Risk:** Stark Coral / Red (#ef4444)
- **Medium Risk:** Amber / Yellow (#f59e0b)
- **Low Risk/Safe:** Teal / Green (#10b981)
- **Background:** Deep Charcoal / Slate (#0f172a)

## Interactions
- **UI Inputs:** Use Streamlit sliders and number inputs for coordinates.
- **Hover:** Show coordinate tooltips and frequency bands on qubits.
- **Click/Select:** Selecting a risk pair populates the explanation panel.
