## SCIENTIFIC METHODOLOGY & RIGOR

**Project:** ToM-NAS (Theory of Mind Neural Architecture Search)
**Student:** Oscar [19286667]
**Institution:** Oxford Brookes University
**Last Updated:** November 27, 2025

---

## 📋 Table of Contents

1. [Research Questions & Hypotheses](#research-questions--hypotheses)
2. [Experimental Design](#experimental-design)
3. [Statistical Analysis Plan](#statistical-analysis-plan)
4. [Validation Protocols](#validation-protocols)
5. [Limitations & Assumptions](#limitations--assumptions)
6. [Result Interpretation Guidelines](#result-interpretation-guidelines)
7. [Reproducibility](#reproducibility)
8. [Ethical Considerations](#ethical-considerations)

---

## 1. Research Questions & Hypotheses

### Primary Research Question

**RQ1:** Can neural architectures be evolved to demonstrate genuine Theory of Mind (ToM) capabilities as measured by established psychological benchmarks?

### Secondary Research Questions

**RQ2:** Do different neural architectures (TRN, RSAN, Transformer) show varying levels of ToM capability?

**RQ3:** Does explicit recursive structure (RSAN) lead to better higher-order ToM performance compared to implicit recursion (Transformer)?

**RQ4:** Can evolved architectures approach human-level performance on ToM tasks?

### Hypotheses

**H1 (Primary):** Neural architectures trained on social interaction tasks will demonstrate above-chance performance on standard ToM tests (Sally-Anne, false belief tasks).

- **Null Hypothesis (H0):** Performance will not differ significantly from chance (50%).
- **Alternative (H1):** Performance > 50% with p < 0.05 and medium effect size (Cohen's d > 0.5).

**H2 (Architecture Comparison):** RSAN will outperform TRN on higher-order ToM tasks (3rd order+) due to explicit recursive structure.

- **H0:** No significant difference between architectures.
- **H1:** RSAN accuracy > TRN accuracy with p < 0.05 and small-to-medium effect size.

**H3 (Evolutionary Benefit):** Evolved architectures will outperform baseline (non-evolved) architectures on ToM benchmarks.

- **H0:** No significant improvement from evolution.
- **H1:** Evolved performance > Baseline with p < 0.05 and medium effect size.

**H4 (Human Comparison):** On first-order ToM, architectures will approach but not exceed human adult performance (~85%).

- **H0:** Performance significantly below human baseline.
- **H1:** Performance within 10 percentage points of human baseline.

---

## 2. Experimental Design

### 2.1 Overview

This study employs a **within-subjects experimental design** with multiple conditions:

- **Factor 1:** Architecture Type (TRN, RSAN, Transformer, Hybrid)
- **Factor 2:** Training Regime (Baseline, Evolved)
- **Factor 3:** ToM Order (1st, 2nd, 3rd, 4th, 5th)

**Design:** 4 (Architecture) × 2 (Training) × 5 (ToM Order) = 40 conditions

### 2.2 Sample Sizes

**Power Analysis:**
- Target power: 0.80
- Alpha: 0.05
- Expected effect size: d = 0.5 (medium)
- Required n per condition: **30-50 trials**

**Actual Sample Sizes:**
- Per ToM test: **100 trials** (exceeds minimum)
- Per architecture: **500 total trials** (100 per ToM order)
- Total dataset: **2,000 trials** across all conditions

### 2.3 Control Conditions

**Baseline Controls:**
1. **Random Agent:** Random responses (expected 50% accuracy)
2. **Heuristic Agent:** Simple rule-based system (expected 60-70%)
3. **Pre-training Checkpoint:** Agent before evolution (expected 55-65%)

**Experimental Controls:**
1. **Fixed random seed:** Reproducibility across runs
2. **Consistent test instances:** Same scenarios for all architectures
3. **Balanced test set:** Equal representation of each scenario type
4. **Counterbalancing:** Order of tests randomized

### 2.4 Dependent Variables

**Primary DV:**
- **Accuracy:** Proportion of correct ToM inferences

**Secondary DVs:**
- **Response time:** Inference latency (computational cost)
- **Confidence calibration:** Alignment between confidence and correctness
- **Generalization:** Performance on novel test scenarios

**Tertiary DVs:**
- **Belief coherence:** Internal consistency of belief representations
- **Architectural efficiency:** Performance per parameter count

### 2.5 Independent Variables

**Manipulated:**
- Architecture type
- Training regime (evolution on/off)
- Number of training generations

**Measured:**
- ToM task order
- Task difficulty
- Social world complexity

---

## 3. Statistical Analysis Plan

### 3.1 Pre-Registered Analyses

**Note:** This section constitutes a pre-registration of planned statistical analyses to prevent p-hacking and HARKing (Hypothesizing After Results are Known).

### 3.2 Primary Analysis

**Test:** One-sample t-test comparing each architecture's accuracy to chance (50%)

**Assumptions:**
- Independence of observations
- Approximate normality of accuracy distribution
- No extreme outliers

**If assumptions violated:** Use Wilcoxon signed-rank test (non-parametric)

**Multiple comparisons:** Bonferroni correction for 4 architectures (α = 0.05/4 = 0.0125)

### 3.3 Secondary Analyses

**Architecture Comparison:**
- **Test:** Repeated-measures ANOVA (or Friedman test if non-normal)
- **Post-hoc:** Tukey HSD for pairwise comparisons
- **Effect size:** Partial eta-squared (η²p)

**Evolution Benefit:**
- **Test:** Paired t-test (Baseline vs. Evolved for same architecture)
- **Effect size:** Cohen's d
- **Power:** Calculated post-hoc

**ToM Order Effects:**
- **Test:** Linear mixed-effects model with ToM order as predictor
- **Random effects:** Architecture (intercept and slope)
- **Expected pattern:** Decreasing accuracy with increasing order

### 3.4 Effect Sizes

**All tests must report:**
- Cohen's d for t-tests
- Partial η² for ANOVA
- Correlation coefficients where applicable

**Interpretation (Cohen, 1988):**
- Small: d = 0.2, η² = 0.01
- Medium: d = 0.5, η² = 0.06
- Large: d = 0.8, η² = 0.14

### 3.5 Confidence Intervals

**All point estimates must include 95% CI:**
- Accuracy: Wilson score interval (for proportions)
- Mean differences: Bootstrap confidence intervals
- Effect sizes: Non-parametric bootstrap

### 3.6 Corrections for Multiple Comparisons

**Family-wise error rate (FWER) control:**
- Bonferroni correction for planned comparisons
- Holm-Bonferroni for exploratory analyses
- False Discovery Rate (FDR) control for large-scale comparisons

---

## 4. Validation Protocols

### 4.1 Construct Validity

**Question:** Do our tests actually measure Theory of Mind?

**Evidence Required:**
1. **Convergent validity:** Correlations with established ToM measures
   - Sally-Anne test (Baron-Cohen et al., 1985)
   - Strange Stories (Happé, 1994)
   - Reading the Mind in the Eyes (Baron-Cohen et al., 2001)

2. **Discriminant validity:** No correlation with non-ToM measures
   - Pattern matching tasks
   - Memory tests
   - General reasoning (without social component)

3. **Face validity:** Expert review of test design

**Criteria for Accepting Construct:**
- Convergent correlations: r > 0.5, p < 0.01
- Discriminant correlations: r < 0.3, non-significant
- Expert consensus: ≥80% agreement

### 4.2 Internal Validity

**Threats:**
1. **Selection bias:** All architectures tested on identical scenarios
2. **History effects:** Controlled via fixed random seed
3. **Maturation:** Not applicable (neural networks)
4. **Testing effects:** Each test instance used only once per architecture
5. **Instrumentation:** Consistent evaluation code across all conditions

**Mitigation:**
- Randomization of test order
- Counterbalancing
- Blind evaluation (where possible)

### 4.3 External Validity

**Generalization Concerns:**

1. **Population validity:**
   - Do results generalize to other architectures?
   - **Test:** Validate on architectures not included in training

2. **Ecological validity:**
   - Do results generalize to real-world ToM scenarios?
   - **Test:** Human-in-the-loop validation in game environment

3. **Temporal validity:**
   - Do results hold over time?
   - **Test:** Re-run experiments after code changes

**Limitations to Generalization:**
- Results specific to current task distribution
- May not generalize to vastly different social scenarios
- Human performance baselines from Western populations

### 4.4 Statistical Conclusion Validity

**Threats:**
1. **Low statistical power:** Mitigated by n=100 per test (exceeds minimum)
2. **Violated assumptions:** Checked via diagnostic tests
3. **Fishing/p-hacking:** Prevented via pre-registration
4. **Unreliability of measures:** Mitigated by multiple trials

**Checks:**
- Power analysis before and after data collection
- Assumption testing (normality, homogeneity, independence)
- Sensitivity analysis (robustness to outliers)

---

## 5. Limitations & Assumptions

### 5.1 Acknowledged Limitations

**1. Simplified ToM Tasks**

**Limitation:** Our ToM tests use simplified scenarios compared to real-world social reasoning.

**Impact:** May overestimate or underestimate true ToM capability.

**Mitigation:**
- Use multiple test types (false belief, deception, strange stories)
- Validate in game environment with naturalistic scenarios
- Report confidence intervals and effect sizes, not just p-values

**2. Limited Comparison to Human Performance**

**Limitation:** Human baseline data from literature, not collected in this study.

**Impact:** Comparisons may not be perfectly matched.

**Mitigation:**
- Cite original sources for human baselines
- Note population differences (children vs. adults, cultural factors)
- Conduct human validation study in game environment (future work)

**3. Computational Resource Constraints**

**Limitation:** Limited GPU resources may constrain architecture search space.

**Impact:** May not find globally optimal architecture.

**Mitigation:**
- Report computational budget clearly
- Use efficient NAS methods (gradient-based where possible)
- Compare to baselines achievable with same compute

**4. Training Data Distribution**

**Limitation:** Social World 4 may not capture full complexity of human social reasoning.

**Impact:** Agents may overfit to task distribution.

**Mitigation:**
- Test on out-of-distribution scenarios
- Report generalization metrics
- Validate in diverse game scenarios

**5. Evaluation Metrics**

**Limitation:** Accuracy alone may not capture full ToM capability.

**Impact:** May miss important aspects (confidence, reasoning process).

**Mitigation:**
- Report multiple metrics (accuracy, calibration, coherence)
- Analyze failure cases qualitatively
- Use attention visualization for interpretability

### 5.2 Assumptions

**Assumption 1: Behavioral Equivalence**

**Statement:** If an agent passes ToM tests behaviorally, it demonstrates ToM capability.

**Justification:** Standard in cognitive science (Turing test logic).

**Challenge:** Philosophical zombie problem - behavior without understanding.

**Response:**
- Implement "zombie detection" tests for behavioral inconsistency
- Require transparent reasoning (attention weights, belief traces)
- Multiple independent test types increase confidence

**Assumption 2: Task Adequacy**

**Statement:** Our ToM task battery adequately samples the ToM construct.

**Justification:** Based on established psychological literature.

**Challenge:** ToM is multifaceted; no single battery is complete.

**Response:**
- Use tasks spanning 1st through 5th order
- Include cognitive and affective ToM
- Cite construct validity evidence

**Assumption 3: Architecture Comparability**

**Statement:** Different architectures can be fairly compared on the same tasks.

**Justification:** Standard practice in ML research.

**Challenge:** Architectures may have different inductive biases.

**Response:**
- Control for parameter count (report performance per parameter)
- Use consistent training regimes
- Report multiple metrics, not just accuracy

**Assumption 4: Statistical Independence**

**Statement:** Test trials are independent observations.

**Justification:** Different test scenarios with fixed random seed.

**Challenge:** Agent state may carry over between trials.

**Response:**
- Reset agent state between trials
- Randomize trial order
- Check for autocorrelation in residuals

---

## 6. Result Interpretation Guidelines

### 6.1 Criteria for Scientific Claims

**To claim "genuine Theory of Mind," must demonstrate:**

1. **Above-chance performance:**
   - Sally-Anne: >75% (p < 0.05, d > 0.5)
   - Second-order: >60% (p < 0.05, d > 0.3)

2. **Generalization:**
   - Performance on held-out test set within 5% of training set
   - No evidence of memorization

3. **Architectural consistency:**
   - ToM-relevant components show importance in ablation studies
   - Attention patterns align with ToM reasoning

4. **Behavioral coherence:**
   - Consistent performance across task variants
   - Failure modes match human limitations (harder tasks = lower performance)

**To claim "superior architecture," must demonstrate:**

1. **Statistical significance:** p < 0.05 with Bonferroni correction
2. **Practical significance:** Effect size d > 0.3 (small-to-medium)
3. **Robustness:** Holds across multiple ToM task types
4. **Replicability:** Consistent across multiple training runs (n ≥ 5)

### 6.2 Interpretation Checklist

Before making a scientific claim, verify:

- [ ] Statistical significance (p < 0.05 after corrections)
- [ ] Effect size reported and meaningful (d > 0.3 or η² > 0.06)
- [ ] Confidence intervals reported
- [ ] Assumptions checked and violations noted
- [ ] Multiple comparison corrections applied
- [ ] Baseline comparisons included
- [ ] Sample size adequate (power > 0.80)
- [ ] Results replicated (at minimum n=3 runs)
- [ ] Generalization tested
- [ ] Limitations acknowledged

### 6.3 Common Interpretation Errors to Avoid

**Error 1: Confusing Statistical and Practical Significance**

- ❌ Wrong: "p < 0.05, therefore the effect is important"
- ✅ Right: "p < 0.05 AND d = 0.6 (medium effect), suggesting meaningful improvement"

**Error 2: Ignoring Effect Size**

- ❌ Wrong: "Significant difference (p = 0.03)"
- ✅ Right: "Significant but small difference (p = 0.03, d = 0.15, 95% CI [0.01, 0.29])"

**Error 3: HARKing (Hypothesizing After Results Known)**

- ❌ Wrong: Finding unexpected pattern and claiming it was predicted
- ✅ Right: Label as exploratory finding requiring confirmation

**Error 4: p-Hacking**

- ❌ Wrong: Trying multiple analyses until p < 0.05
- ✅ Right: Pre-register analyses, report all tests conducted

**Error 5: Ignoring Multiple Comparisons**

- ❌ Wrong: 10 t-tests, report those with p < 0.05
- ✅ Right: Apply Bonferroni correction (α = 0.05/10 = 0.005)

**Error 6: Overgeneralizing**

- ❌ Wrong: "This proves agents have consciousness"
- ✅ Right: "Agents demonstrate ToM-consistent behavior on tested scenarios"

### 6.4 Reporting Standards

**All results must include:**

1. **Test statistic and p-value:** "t(98) = 3.45, p = 0.001"
2. **Effect size with CI:** "d = 0.54, 95% CI [0.22, 0.86]"
3. **Sample size:** "n = 100 trials per condition"
4. **Descriptive statistics:** "M = 0.78, SD = 0.12"
5. **Assumption checks:** "Data approximately normal (Shapiro-Wilk p = 0.23)"
6. **Power analysis:** "Post-hoc power = 0.92"

**Format:** Follow APA 7th edition reporting standards.

---

## 7. Reproducibility

### 7.1 Computational Reproducibility

**Requirements:**
- All random seeds fixed and documented
- Software versions specified
- Hardware specifications reported
- Complete code and data available

**Implementation:**

```python
from src.evaluation.scientific_validation import ReproducibilityManager

# Set up reproducibility
repro = ReproducibilityManager(seed=42)
repro.save_config('experiment_config.json')

# ... run experiments ...

# Verify reproducibility
repro.load_config('experiment_config.json')
# Re-run should produce identical results
```

### 7.2 Replication Studies

**Internal replication:**
- Minimum 3 independent training runs per architecture
- Report mean and standard deviation across runs
- Flag non-replicable results (high variance across runs)

**External replication:**
- Code and data publicly available on GitHub
- Docker container with fixed environment
- Step-by-step reproduction guide

### 7.3 Open Science Practices

**Commitment:**
- Pre-registration of primary hypotheses
- Open data (with appropriate privacy protections)
- Open code (MIT license)
- Pre-print publication (arXiv)

**Data Sharing:**
- Anonymized experimental data
- Model checkpoints
- Evaluation scripts
- Visualization code

---

## 8. Ethical Considerations

### 8.1 Research Ethics

**IRB Status:**
- Core ToM-NAS research: No human subjects (computational only)
- Game integration with telemetry: **IRB approval required before data collection**

**Ethical Principles:**
1. **Beneficence:** Research advances understanding of AI and cognition
2. **Non-maleficence:** Minimal risks (computational research)
3. **Autonomy:** Future human participants provide informed consent
4. **Justice:** Results and code freely available to all

### 8.2 AI Ethics

**Concerns Addressed:**

**1. Anthropomorphization**
- **Risk:** Claiming "consciousness" or "understanding" beyond evidence
- **Mitigation:** Careful language, acknowledge behavioral equivalence only

**2. Deceptive AI**
- **Risk:** Training agents to deceive
- **Mitigation:** Deception detection is for validation, not deployment

**3. Dual Use**
- **Risk:** Manipulative AI in real-world applications
- **Mitigation:**
  - Ethical use guidelines in documentation
  - Transparency about capabilities and limitations
  - No military/surveillance applications

**4. Reproducibility and Trust**
- **Risk:** Non-reproducible results undermine scientific progress
- **Mitigation:** Full reproducibility infrastructure

### 8.3 Publication Ethics

**Authorship:** Follow CRediT (Contributor Roles Taxonomy)
- Oscar [19286667]: Conceptualization, Methodology, Software, Validation, Analysis, Writing
- Prof. Fabio Cuzzolin: Supervision, Funding acquisition, Review

**Conflicts of Interest:** None declared

**Data Integrity:**
- No fabrication, falsification, or plagiarism
- All data and code preserved for 10 years
- Errors corrected via published errata

---

## 9. Validation Checklist (Pre-Submission)

Before submitting for publication, verify:

**Experimental Design:**
- [ ] Hypotheses clearly stated and justified
- [ ] Sample size adequate (power analysis)
- [ ] Control conditions included
- [ ] Randomization and counterbalancing implemented

**Statistical Analysis:**
- [ ] Analyses pre-registered (or labeled exploratory)
- [ ] Assumptions checked
- [ ] Multiple comparisons corrected
- [ ] Effect sizes and CIs reported
- [ ] Power analysis reported

**Reproducibility:**
- [ ] Code publicly available
- [ ] Data publicly available (or justification for restrictions)
- [ ] Random seeds documented
- [ ] Software versions specified
- [ ] Results replicated (minimum 3 runs)

**Reporting:**
- [ ] Follows APA 7th edition standards
- [ ] CONSORT/STROBE checklist completed (if applicable)
- [ ] Limitations clearly stated
- [ ] Assumptions explicitly noted
- [ ] Interpretation appropriately cautious

**Ethics:**
- [ ] IRB approval obtained (if human subjects)
- [ ] Informed consent process documented
- [ ] Conflicts of interest declared
- [ ] Authorship appropriate

---

## 10. References

**Statistical Methods:**
- Cohen, J. (1988). Statistical power analysis for the behavioral sciences (2nd ed.). Lawrence Erlbaum Associates.
- Field, A., Miles, J., & Field, Z. (2012). Discovering statistics using R. Sage publications.
- Cumming, G. (2014). The new statistics: Why and how. Psychological science, 25(1), 7-29.

**Theory of Mind Assessment:**
- Baron-Cohen, S., Leslie, A. M., & Frith, U. (1985). Does the autistic child have a "theory of mind"? Cognition, 21(1), 37-46.
- Wimmer, H., & Perner, J. (1983). Beliefs about beliefs: Representation and constraining function of wrong beliefs in young children's understanding of deception. Cognition, 13(1), 103-128.
- Happé, F. G. (1994). An advanced test of theory of mind: Understanding of story characters' thoughts and feelings by able autistic, mentally handicapped, and normal children and adults. Journal of autism and Developmental disorders, 24(2), 129-154.

**Research Methodology:**
- Nosek, B. A., et al. (2018). The preregistration revolution. Proceedings of the National Academy of Sciences, 115(11), 2600-2606.
- Simmons, J. P., Nelson, L. D., & Simonsohn, U. (2011). False-positive psychology: Undisclosed flexibility in data collection and analysis allows presenting anything as significant. Psychological science, 22(11), 1359-1366.

**AI Ethics:**
- Jobin, A., Ienca, M., & Vayena, E. (2019). The global landscape of AI ethics guidelines. Nature Machine Intelligence, 1(9), 389-399.

---

**Document Version:** 1.0
**Last Review:** November 27, 2025
**Next Review:** Before paper submission
**Maintained by:** Oscar [19286667]
