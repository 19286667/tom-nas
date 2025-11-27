"""
Statistical Analysis Tools for ToM-NAS Experiments

Provides:
1. Distribution comparison tests
2. Hypothesis testing framework
3. Effect size calculations
4. Correlation analysis
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

# Statistical testing
try:
    from scipy import stats
    from scipy.stats import ttest_ind, mannwhitneyu, pearsonr, spearmanr, chi2_contingency
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@dataclass
class TestResult:
    """Result of a statistical test"""
    test_name: str
    statistic: float
    p_value: float
    significant: bool
    effect_size: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    interpretation: str = ""

    def to_dict(self) -> Dict:
        return {
            'test_name': self.test_name,
            'statistic': self.statistic,
            'p_value': self.p_value,
            'significant': self.significant,
            'effect_size': self.effect_size,
            'confidence_interval': self.confidence_interval,
            'interpretation': self.interpretation,
        }


def compare_distributions(
    group1: List[float],
    group2: List[float],
    test: str = 'auto',
    alpha: float = 0.05,
) -> TestResult:
    """
    Compare two distributions statistically.

    Args:
        group1: First group of values
        group2: Second group of values
        test: Test type ('ttest', 'mannwhitney', 'auto')
        alpha: Significance level

    Returns:
        TestResult with statistics and interpretation
    """
    if not HAS_SCIPY:
        return _compare_distributions_simple(group1, group2, alpha)

    arr1 = np.array(group1)
    arr2 = np.array(group2)

    # Auto-select test based on normality
    if test == 'auto':
        # Shapiro-Wilk test for normality (if sample size allows)
        if len(arr1) >= 8 and len(arr2) >= 8:
            _, p1 = stats.shapiro(arr1[:50])  # Limit for computational efficiency
            _, p2 = stats.shapiro(arr2[:50])
            test = 'ttest' if (p1 > 0.05 and p2 > 0.05) else 'mannwhitney'
        else:
            test = 'ttest'  # Default to t-test for small samples

    if test == 'ttest':
        statistic, p_value = ttest_ind(arr1, arr2)
        test_name = "Independent t-test"
    else:  # mannwhitney
        statistic, p_value = mannwhitneyu(arr1, arr2, alternative='two-sided')
        test_name = "Mann-Whitney U test"

    # Effect size (Cohen's d)
    effect_size = _cohens_d(arr1, arr2)

    # Interpretation
    significant = p_value < alpha
    effect_interp = _interpret_effect_size(effect_size)

    if significant:
        direction = "higher" if np.mean(arr1) > np.mean(arr2) else "lower"
        interpretation = (
            f"Significant difference detected (p={p_value:.4f}). "
            f"Group 1 mean is {direction} ({np.mean(arr1):.4f} vs {np.mean(arr2):.4f}). "
            f"Effect size: {effect_interp} (d={effect_size:.3f})."
        )
    else:
        interpretation = (
            f"No significant difference (p={p_value:.4f}). "
            f"Means: {np.mean(arr1):.4f} vs {np.mean(arr2):.4f}."
        )

    return TestResult(
        test_name=test_name,
        statistic=float(statistic),
        p_value=float(p_value),
        significant=significant,
        effect_size=float(effect_size),
        interpretation=interpretation,
    )


def _compare_distributions_simple(
    group1: List[float],
    group2: List[float],
    alpha: float,
) -> TestResult:
    """Simple distribution comparison without scipy"""
    arr1 = np.array(group1)
    arr2 = np.array(group2)

    mean1, mean2 = np.mean(arr1), np.mean(arr2)
    std1, std2 = np.std(arr1, ddof=1), np.std(arr2, ddof=1)
    n1, n2 = len(arr1), len(arr2)

    # Simple t-statistic
    pooled_se = np.sqrt(std1**2/n1 + std2**2/n2)
    if pooled_se == 0:
        t_stat = 0
    else:
        t_stat = (mean1 - mean2) / pooled_se

    # Rough p-value approximation (assumes normal distribution)
    df = min(n1, n2) - 1
    p_value = 2 * (1 - _normal_cdf(abs(t_stat)))

    effect_size = _cohens_d(arr1, arr2)
    significant = p_value < alpha

    return TestResult(
        test_name="Simple t-test (approximate)",
        statistic=float(t_stat),
        p_value=float(p_value),
        significant=significant,
        effect_size=float(effect_size),
        interpretation=f"Means: {mean1:.4f} vs {mean2:.4f}",
    )


def _normal_cdf(x: float) -> float:
    """Approximate normal CDF"""
    return 0.5 * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))


def _cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Calculate Cohen's d effect size"""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    return float((np.mean(group1) - np.mean(group2)) / pooled_std)


def _interpret_effect_size(d: float) -> str:
    """Interpret Cohen's d effect size"""
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


def run_hypothesis_tests(
    results: Dict[str, Any],
    alpha: float = 0.05,
) -> Dict[str, TestResult]:
    """
    Run all hypothesis tests for ToM-NAS experiments.

    Args:
        results: Experiment results containing data by task type
        alpha: Significance level

    Returns:
        Dictionary of test results for each hypothesis
    """
    tests = {}

    # Extract data
    tom_data = _extract_tom_data(results)
    control_data = _extract_control_data(results)

    # H1: Skip connections in ToM vs Control
    if tom_data.get('skip_connections') and control_data.get('skip_connections'):
        tests['H1_skip_connections'] = compare_distributions(
            tom_data['skip_connections'],
            control_data['skip_connections'],
            alpha=alpha,
        )
        tests['H1_skip_connections'].interpretation = (
            f"H1: {tests['H1_skip_connections'].interpretation}"
        )

    # H2: Attention in ToM vs Control
    if tom_data.get('attention_ops') and control_data.get('attention_ops'):
        tests['H2_attention'] = compare_distributions(
            tom_data['attention_ops'],
            control_data['attention_ops'],
            alpha=alpha,
        )
        tests['H2_attention'].interpretation = (
            f"H2: {tests['H2_attention'].interpretation}"
        )

    # H3: Correlation between ToM order and complexity
    if tom_data.get('tom_orders') and tom_data.get('effective_depth'):
        tests['H3_complexity'] = _correlation_test(
            tom_data['tom_orders'],
            tom_data['effective_depth'],
            'H3: ToM order vs effective depth',
        )

    # H4: Recursive structure in higher-order ToM
    if tom_data.get('recursive_depth'):
        higher_order = [d for o, d in zip(tom_data.get('tom_orders', []), tom_data['recursive_depth'])
                       if o >= 2]
        first_order = [d for o, d in zip(tom_data.get('tom_orders', []), tom_data['recursive_depth'])
                      if o == 1]

        if higher_order and first_order:
            tests['H4_recursive'] = compare_distributions(
                higher_order, first_order, alpha=alpha
            )
            tests['H4_recursive'].interpretation = (
                f"H4: {tests['H4_recursive'].interpretation}"
            )

    return tests


def _extract_tom_data(results: Dict) -> Dict[str, List[float]]:
    """Extract ToM task data from results"""
    data = {
        'skip_connections': [],
        'attention_ops': [],
        'effective_depth': [],
        'recursive_depth': [],
        'tom_orders': [],
        'fitness': [],
    }

    tom_tasks = ['tomi', 'bigtom', 'hitom', 'opentom', 'socialqa']

    for task_name, task_results in results.items():
        if not any(t in task_name.lower() for t in tom_tasks):
            continue

        if isinstance(task_results, list):
            for r in task_results:
                _extract_single_result(r, data, task_name)
        elif isinstance(task_results, dict):
            _extract_single_result(task_results, data, task_name)

    return data


def _extract_control_data(results: Dict) -> Dict[str, List[float]]:
    """Extract control task data from results"""
    data = {
        'skip_connections': [],
        'attention_ops': [],
        'effective_depth': [],
        'recursive_depth': [],
        'fitness': [],
    }

    control_tasks = ['simple', 'babi', 'relational', 'sequence']
    tom_tasks = ['tomi', 'bigtom', 'hitom', 'opentom', 'socialqa']

    for task_name, task_results in results.items():
        if any(t in task_name.lower() for t in tom_tasks):
            continue
        if not any(t in task_name.lower() for t in control_tasks):
            continue

        if isinstance(task_results, list):
            for r in task_results:
                _extract_single_result(r, data, task_name)
        elif isinstance(task_results, dict):
            _extract_single_result(task_results, data, task_name)

    return data


def _extract_single_result(result: Dict, data: Dict, task_name: str):
    """Extract metrics from single result"""
    metrics = result.get('final_metrics', result.get('metrics', {}))

    if 'num_skip_connections' in metrics:
        data['skip_connections'].append(metrics['num_skip_connections'])
    if 'num_attention_ops' in metrics:
        data['attention_ops'].append(metrics['num_attention_ops'])
    if 'effective_depth' in metrics:
        data['effective_depth'].append(metrics['effective_depth'])
    if 'recursive_depth' in metrics:
        data['recursive_depth'].append(metrics['recursive_depth'])
    if 'best_fitness' in result:
        data['fitness'].append(result['best_fitness'])

    # Determine ToM order from task name
    if 'tom_orders' in data:
        if 'hitom_4' in task_name.lower():
            data['tom_orders'].append(4)
        elif 'hitom_3' in task_name.lower():
            data['tom_orders'].append(3)
        elif 'hitom_2' in task_name.lower():
            data['tom_orders'].append(2)
        else:
            data['tom_orders'].append(1)


def _correlation_test(x: List[float], y: List[float], description: str) -> TestResult:
    """Run correlation test"""
    if not HAS_SCIPY:
        corr = np.corrcoef(x, y)[0, 1]
        return TestResult(
            test_name="Pearson correlation",
            statistic=float(corr),
            p_value=0.05,  # Unknown without scipy
            significant=abs(corr) > 0.3,
            effect_size=float(corr),
            interpretation=f"{description}: r={corr:.3f}",
        )

    corr, p_value = pearsonr(x, y)

    interpretation = f"{description}: "
    if abs(corr) < 0.3:
        interpretation += "weak correlation"
    elif abs(corr) < 0.7:
        interpretation += "moderate correlation"
    else:
        interpretation += "strong correlation"
    interpretation += f" (r={corr:.3f}, p={p_value:.4f})"

    return TestResult(
        test_name="Pearson correlation",
        statistic=float(corr),
        p_value=float(p_value),
        significant=p_value < 0.05,
        effect_size=float(corr),
        interpretation=interpretation,
    )


def compute_effect_sizes(
    results: Dict[str, Any],
) -> Dict[str, Dict[str, float]]:
    """
    Compute effect sizes for all comparisons.

    Returns:
        Effect sizes for each comparison
    """
    effect_sizes = {}

    tom_data = _extract_tom_data(results)
    control_data = _extract_control_data(results)

    metrics = ['skip_connections', 'attention_ops', 'effective_depth', 'recursive_depth']

    for metric in metrics:
        tom_values = tom_data.get(metric, [])
        control_values = control_data.get(metric, [])

        if tom_values and control_values:
            d = _cohens_d(np.array(tom_values), np.array(control_values))
            effect_sizes[metric] = {
                'cohens_d': float(d),
                'interpretation': _interpret_effect_size(d),
                'tom_mean': float(np.mean(tom_values)),
                'control_mean': float(np.mean(control_values)),
            }

    return effect_sizes


def generate_statistical_report(
    results: Dict[str, Any],
    alpha: float = 0.05,
) -> str:
    """
    Generate comprehensive statistical report.

    Args:
        results: Experiment results
        alpha: Significance level

    Returns:
        Markdown formatted report
    """
    lines = []
    lines.append("# Statistical Analysis Report")
    lines.append("")
    lines.append(f"Significance level: alpha = {alpha}")
    lines.append("")

    # Run hypothesis tests
    tests = run_hypothesis_tests(results, alpha)

    lines.append("## Hypothesis Tests")
    lines.append("")

    for test_name, test_result in tests.items():
        lines.append(f"### {test_name}")
        lines.append(f"- Test: {test_result.test_name}")
        lines.append(f"- Statistic: {test_result.statistic:.4f}")
        lines.append(f"- p-value: {test_result.p_value:.4f}")
        lines.append(f"- Significant: {'Yes' if test_result.significant else 'No'}")
        if test_result.effect_size is not None:
            lines.append(f"- Effect size: {test_result.effect_size:.3f} ({_interpret_effect_size(test_result.effect_size)})")
        lines.append(f"- Interpretation: {test_result.interpretation}")
        lines.append("")

    # Effect sizes
    lines.append("## Effect Sizes")
    lines.append("")

    effect_sizes = compute_effect_sizes(results)
    for metric, data in effect_sizes.items():
        lines.append(f"### {metric}")
        lines.append(f"- Cohen's d: {data['cohens_d']:.3f} ({data['interpretation']})")
        lines.append(f"- ToM mean: {data['tom_mean']:.3f}")
        lines.append(f"- Control mean: {data['control_mean']:.3f}")
        lines.append("")

    # Summary
    lines.append("## Summary")
    lines.append("")

    significant_tests = [name for name, result in tests.items() if result.significant]
    if significant_tests:
        lines.append(f"Significant findings: {', '.join(significant_tests)}")
    else:
        lines.append("No significant differences found.")

    large_effects = [metric for metric, data in effect_sizes.items()
                    if data['interpretation'] in ['large', 'medium']]
    if large_effects:
        lines.append(f"Notable effect sizes: {', '.join(large_effects)}")

    return '\n'.join(lines)
