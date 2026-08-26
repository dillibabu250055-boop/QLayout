# 01_PRD.md

## Problem
Quantum hardware designers waste hours on computationally expensive electromagnetic (EM) simulations only to discover basic spatial or frequency collisions in their superconducting qubit layouts.

## Target Users
Hardware engineers designing superconducting quantum chip layouts who need rapid, early-stage feedback.

## Existing Solutions
High-fidelity EM simulators like Ansys HFSS and Qiskit Metal. They are highly accurate but computationally heavy and slow.

## Product Positioning
Q-Layout is a physics-informed pre-simulation EDA copilot for superconducting quantum-chip layouts. It bridges architectural design and expensive EM validation.

## Core Innovation
Coupled Spatial-Spectral Risk Analysis combined with a human-in-the-loop interactive optimization engine.

## Non-Goals
Q-Layout does not replace high-fidelity EM simulation. It uses a physics-informed distance-dependent proxy for spatial coupling risk, rather than attempting to accurately calculate exact capacitive coupling. For the MVP, Q-Layout does *not* auto-allocate frequencies.

## Features
- Coordinate-based interactive chip canvas (Plotly)
- Circuit-aware layout analysis (intended connectivity vs. crosstalk)
- Real-time Layout Quality Score (LQS)
- Quantum Design Rule Checker (Q-DRC)
- Explainable Risk Dashboard
- "Auto-Optimize" hill-climbing assistant

## Success Metrics
- Calculate LQS and risk pairs in <= 200 ms for pairwise analysis of an 8-qubit demo layout on a standard development laptop.
- Successfully resolve intentional collisions via Auto-Optimize
