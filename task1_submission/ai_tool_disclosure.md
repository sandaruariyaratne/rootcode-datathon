# Tech-Triathlon 2026: AI Tool Disclosure

This disclosure provides full transparency regarding the use of AI tools in our Datathon Task 1 solution, in strict accordance with the competition rules outlined on Page 22 of the *Challenge Booklet*.

---

## 1. Compliance Statement
- **No Pre-trained Models:** Our models were trained strictly from scratch on the provided training records. No pre-trained foundation models or proprietary API-based predictive endpoints were utilized.
- **No AutoML / No-Code Tools:** Proprietary low-code/no-code AutoML platforms (e.g. DataRobot, Google AutoML, AWS SageMaker Autopilot) were **not** used. All models are open-source Python implementations (`catboost`, `scipy`, `pandas`, `numpy`, `scikit-learn`).

---

## 2. Work Breakdown: Human-Led vs. AI-Assisted

### A. Non-AI / Human-Led Work
1. **Business Problem Formulation & Domain Strategy:**
   - Understanding Waypoint Logistics' multi-temperature delivery network, depot structure (Peliyagoda and Kandy), and vehicle access constraints.
2. **Strict Zero-Leakage Label Definition:**
   - Formulating the business logic for service time: identifying that waiting before an outlet opens is not handling time, ensuring $\text{service\_start} = \max(\text{actual\_arrival}, \text{window\_open})$.
   - Defining lateness strictly as $\text{actual\_arrival} > \text{window\_close}$.
3. **Feature Engineering Design:**
   - Identifying critical physical variables: cargo density, volume fill, planned arrival slack buffer, cumulative route workload, and monsoon disruption factors.
4. **Validation Architecture:**
   - Choosing an out-of-time chronological 80/10/10 split to simulate future deployment and prevent optimistic early-stopping bias.
5. **Quality Assurance & Verification:**
   - Verifying all 5,014 test predictions against template constraints (0 nulls, correct sequence, non-negative service duration, probability bounds).

---

### B. AI-Assisted Work
1. **Interactive Pair-Programming & Scripting:**
   - AI coding assistants (Antigravity IDE / pair-programming assistant) were used to accelerate writing repetitive data processing code, merge operations, and validation assertion scripts.
2. **Plotting & Visualization Scripts:**
   - Assisted in formatting matplotlib visualization code for residual distributions, calibration decile plots, and system architecture diagrams.
3. **Documentation Structuring:**
   - Assisted in structuring markdown summaries and compiling benchmark tables from raw log outputs.
