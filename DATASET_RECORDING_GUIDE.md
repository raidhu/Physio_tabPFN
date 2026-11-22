# Physio-TabPFN Dataset Recording Guide

This guide details best practices for collecting high-quality, fault-free kinematic training data for **TabPFN** exercise classification.

---

## 1. Target Exercise Classes

| Key | Class | Primary Motion Signature |
|:---:|:---|:---|
| **`[1]`** | **Squats** | Symmetrical leg flexion/extension, upright-to-leaning trunk. |
| **`[2]`** | **Step-Ups** | Strong asymmetrical leg elevation, one foot higher than the other. |
| **`[0]`** | **Idle** | Standing still, walking in place, minor shifting, resting. |

> **Important:** **Do not skip the `Idle` class!** Machine learning models require negative examples to know when a person is resting or preparing, preventing false positive exercise predictions.

---

## 2. Environment & Camera Setup

To prevent MediaPipe tracking dropouts and jitter:

```
          [ Webcam / Laptop ] (Height: 0.9m - 1.2m)
                  |
                  |  2.5m - 3.5m Distance
                  v
          [ Full Body In Frame ]
        (Head to Toes clearly visible)
```

1. **Full-Body Visibility:**
   - Both feet and the top of the head **must remain in frame** throughout the full range of motion.
   - For **Step-Ups**, ensure the box/step and raised foot do not clip outside the frame.
2. **Camera Height & Angle:**
   - Place camera at hip/chest level (~1 meter high).
   - Avoid looking from high angles pointing down or ground level pointing up.
3. **Lighting & Contrast:**
   - Face towards a light source; avoid bright backlighting (e.g. standing directly in front of a sunny window).
   - Wear clothes that contrast with your room background (loose baggy clothing can obscure joint centers).
4. **Camera Angle Relative to You:**
   - A **45-degree front-diagonal angle** or **pure side profile** captures knee flexion and step depth significantly better than a flat front-facing view.

---

## 3. Keyboard Controls

| Key | Action | Details |
|:---:|:---|:---|
| **`Spacebar`** | **Start / Pause Recording** | Triggers a **3-second visual countdown** before recording begins. |
| **`1`** | Select **Squats** | Switched label displays on top bar. |
| **`2`** | Select **Step-Ups** | Sets label to `Step-Ups`. |
| **`0`** | Select **Idle** | Sets label to `Idle` (standing/resting). |
| **`s`** | **Save to CSV** | Appends recorded session data to `data/exercise_dataset.csv`. |
| **`c`** | **Clear Buffer** | Clears unsaved frames in the current session. |
| **`q`** | **Save & Quit** | Automatically flushes remaining samples and exits. |

---

## 4. Step-by-Step Recording Procedure

Follow this routine to ensure clean, label-consistent data:

### Step 1: Launch the Recorder
```bash
python scripts/record_dataset.py
```

### Step 2: Record "Idle" First (~300–500 frames)
1. Press **`0`** to select `Idle`.
2. Press **`Spacebar`**. You have 3 seconds to step into position.
3. Stand naturally, shift weight slightly, turn slightly, or breathe normally for ~10–15 seconds.
4. Press **`Spacebar`** to pause.

### Step 3: Record Exercise Repetitions (~400–600 frames per exercise)
For each exercise (**Squats `1`**, **Step-Ups `2`**):
1. Press the corresponding number key to update the active label.
2. Press **`Spacebar`**.
3. During the 3-second countdown, get into the starting posture.
4. When the on-screen text turns **`● RECORDING`**, perform continuous, controlled repetitions.
5. As soon as you finish your set, press **`Spacebar` immediately** to stop recording.
   *(Do not walk over to the laptop while recording is still active. That walk would be falsely labeled as an exercise. Use the 3-second countdown to position yourself, and hit Spacebar as soon as you finish).*

### Step 4: Save & Check Balance
- Press **`s`** periodically to write data to disk.
- Look at the on-screen **"RECORDED SAMPLES"** sidebar to ensure balanced numbers across all classes.

---

## 5. Recommended Target Sample Counts

For **TabPFN**, you only need small-to-moderate high-quality datasets:

| Exercise | Target Frames | Approx. Duration | Reps Equivalent |
|:---|:---:|:---:|:---:|
| **Idle** | ~500 frames | ~15-20 sec | Natural standing |
| **Squats** | ~500 frames | ~15-20 sec | 6–10 smooth reps |
| **Step-Ups** | ~500 frames | ~15-20 sec | 6–10 alternating reps |
| **Total** | **~1,500 frames** | **~1-2 minutes** | Balanced dataset |

---

## 6. Common Pitfalls & How to Avoid Them

| Mistake | Consequence | Prevention |
|:---|:---|:---|
| **Recording the transition walk** | Walking to/from keyboard gets labeled as squats/hops. | Always pause before walking; use the 3-second countdown. |
| **Feet clipped out of camera** | `foot_height_diff` and knee angles become erratic. | Push camera back until entire body + shoes are visible. |
| **Only 1 repetition recorded** | Model overfits to a single posture state. | Perform continuous reps so the model sees all phases (eccentric, bottom, concentric). |
| **Class imbalance (e.g. 2000 squats, 100 idle)** | Model predicts squats for everything. | Keep frame counts roughly even (~400-600 each). |
| **Loose/Baggy trousers** | MediaPipe landmark jitter around knees. | Wear fitted athletic pants or shorts for crisp joint tracking. |
