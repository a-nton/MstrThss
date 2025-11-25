# MAKER Framework Implementation Log: Evolution of the "Council of Agents"

## 1. The Initial Configuration: "The Veto Model"
**Architecture:**
* **Agent A (Fundamental):** Analyzes news for business value.
* **Agent B (Technical):** Analyzes price history for trends.
* **Agent C (Risk Manager):** Scans news for catastrophic risks (Fraud, Lawsuits).
* **Consensus Logic:** Fundamental + Technical vote on direction; Risk Manager has a "Hard Veto" power to override both if risk > threshold.

### The Finding (Systematic Error)
During testing, the model exhibited a specific failure mode:
* **Observation:** The predicted probability of an "Up" move (`prob_up`) froze at **0.14** across nearly all trading days, regardless of the stock or the varying magnitude predictions (-3% vs -4%).
* **Diagnosis:** The Risk Manager agent, prompted to be "pessimistic" and "look for risks," fell victim to **Anchoring Bias**. In the absence of true catastrophes (e.g., bankruptcy), it flagged standard corporate risks (competition, regulatory updates) as "High Risk."
* **Mathematical Consequence:** The "Hard Veto" logic (`if risk > 7 then Bearish`) triggered every day. This forced the signal to "Bearish" with high confidence (0.9), resulting in a fixed probability calculation: `0.5 - (0.4 * 0.9) = 0.14`.

**Key Lesson:** A consensus system fails if one agent possesses "Dictator" privileges (Veto power) without a sufficiently high threshold. The decomposition was imbalanced; the "Risk" task was too broad for a veto role in day-to-day trading.

---

## 2. The Strategic Pivot: "The Trading Trinity"
**Reasoning:**
For short-term price prediction (1-day to 1-week horizons), "Existential Risk" is a rare event driver. The vast majority of price movement is driven by the interplay between **Value** (Fundamentals), **Trend** (Technical), and **Hype** (Sentiment).

**New Architecture:**
We shifted from "Risk Veto" to **"Active Consensus"** by redefining the decomposition.

* **Agent A (The Fundamentalist):** Focuses on *facts* and long-term value drivers (Earnings, Products).
* **Agent B (The Technician):** Focuses on *price action* and momentum. (Blind to news).
* **Agent C (The Psychologist):** Focuses on *emotion* and crowd reaction (Hype, Panic, Greed). Replaces "Risk."

### Improvements
1.  **Active Voting:** Unlike the Risk agent (which stayed silent/neutral 95% of the time in a healthy market), the Psychologist provides an active directional vote every day.
2.  **Orthogonality:** The agents represent distinct, often conflicting market forces. A stock can be Fundamentally bad (Voting Down) but Psychologically hyped (Voting Up).
3.  **Tie-Breaking:** This configuration ensures a mathematical majority (2 vs 1) is always possible, preventing stalemates.

---

## 3. Theoretical Alignment with MAKER Logic
This evolution illustrates the core principles of the MAKER (Decomposition + Consensus) framework:

### A. Decomposition (The "Atomic" Unit)
Instead of asking one LLM "Will the stock go up?", which encourages hallucination, we decomposed the problem into **Orthogonal Functional Perspectives**:
* *Value Perspective* (Is it worth more?)
* *Momentum Perspective* (Is it moving up?)
* *Sentiment Perspective* (Are people excited?)

This reduces the cognitive load on each individual agent. The Technical agent, for example, cannot be biased by bad news because it is structurally "blind" to the headlines.

### B. Consensus (The "Filter")
We moved from a **Hierarchical Consensus** (Veto system) to a **Democratic Consensus** (Majority Vote).
* **Error Correction:** If the "Psychologist" hallucinates hype where there is none, the "Fundamentalist" and "Technician" can outvote it.
* **Calibration:** By averaging the confidence of only the *winning* bloc, we filter out the noise from the dissenting agent, resulting in a cleaner probability signal.

### C. Debiasing via Architecture
The shift corrected the "Pessimism Bias" not by prompt engineering alone, but by **architectural design**. By replacing a "Risk Manager" (who looks for negatives) with a "Psychologist" (who looks for *reaction*, positive or negative), we removed the structural anchor that was dragging the probability to 0.14.