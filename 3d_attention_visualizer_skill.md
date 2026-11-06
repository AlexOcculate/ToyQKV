# Skill Definition: 3D Geometric Attention Visualizer

## Overview
This skill specializes in generating clean, robust, and highly production-optimized **3D Manim animations** designed to explain geometric LLM attention mechanics, token transformations, and mathematical vector warps in spatial environments.

## Core Capabilities
* **Perspective Multi-Panel Layouts:** Setting up localized, low-contrast 2D orthographic tracking projection planes (XY, XZ, YZ) mapped smoothly into a 3D scene grid layout.
* **Camera-Facing Billboards:** Using native `add_fixed_orientation_mobjects` loops to ensure clean typography labels and dynamic matrices always remain horizontal and legible regardless of camera orientation sweeps.
* **Synchronized Dynamic Updaters:** Engineering robust multi-object tracking callback patterns using `.add_updater()` hook sequences to tie parent 3D movements to child projection nodes in real-time.
* **Token Division Sequencing:** Animating context-dependent attention transformations via cell-division birth events (cloning origin anchors while moving context updates to target destinations).

## Operational Guardrails & Best Practices
1. **Homogeneous Array Safe Packing:** Never multiply a 3D NumPy array inside a nested array structure directly when mixed with raw scalar values. Pre-multiply individual scalar tracking terms component-by-component (`x_scaled`, `y_scaled`, `z_scaled`) to strictly avoid `ValueError: setting an array element with a sequence`.
2. **Color Constants Check:** Avoid using missing color palette suffixes such as `ORANGE_E`. Fall back cleanly to custom raw hex mappings (`"#5c2e00"`) or standard verified defaults (`GRAY_E`).
3. **Muted Visual Hierarchy:** Keep grid structures, axes, and multi-panel backdrops thinned and dark (`stroke_width=1`, opacity `0.2` or lower) to prevent background elements from distracting from focal animation tokens.
4. **Orientation Invariance:** Keep all metadata names, matrices, and axis guides registered through camera-invariance arrays (`add_fixed_orientation_mobjects`) instead of pinning hard-coded 3D plane angle tilts.