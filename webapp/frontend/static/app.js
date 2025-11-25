/**
 * ToM-NAS Web Application
 * Making Neural Architecture Search for Theory of Mind feel like common sense
 */

// ============================================================================
// State Management
// ============================================================================

const AppState = {
    currentExperiment: null,
    pollingInterval: null,
    fitnessChart: null,
    diversityChart: null,
    concepts: null
};

// ============================================================================
// Initialization
// ============================================================================

document.addEventListener('DOMContentLoaded', async () => {
    await checkStatus();
    await loadConcepts();
    setupCharts();
});

async function checkStatus() {
    try {
        const response = await fetch('/api/status');
        const data = await response.json();

        const indicator = document.getElementById('status-indicator');
        const dot = indicator.querySelector('.status-dot');
        const text = indicator.querySelector('.status-text');

        if (data.torch_available) {
            dot.classList.add('ready');
            text.textContent = 'Ready';
        } else {
            dot.classList.add('ready');
            text.textContent = 'Demo Mode';
        }
    } catch (error) {
        console.error('Status check failed:', error);
    }
}

async function loadConcepts() {
    try {
        const response = await fetch('/api/concepts');
        AppState.concepts = await response.json();
    } catch (error) {
        console.error('Failed to load concepts:', error);
    }
}

function setupCharts() {
    // Add SVG gradient definition
    const svg = document.querySelector('.progress-ring');
    if (svg) {
        const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
        defs.innerHTML = `
            <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" style="stop-color:#6366f1"/>
                <stop offset="100%" style="stop-color:#8b5cf6"/>
            </linearGradient>
        `;
        svg.insertBefore(defs, svg.firstChild);
    }

    // Fitness Chart
    const fitnessCtx = document.getElementById('fitness-chart');
    if (fitnessCtx) {
        AppState.fitnessChart = new Chart(fitnessCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Best Fitness',
                        data: [],
                        borderColor: '#10b981',
                        backgroundColor: 'rgba(16, 185, 129, 0.1)',
                        fill: true,
                        tension: 0.4
                    },
                    {
                        label: 'Average Fitness',
                        data: [],
                        borderColor: '#6366f1',
                        backgroundColor: 'rgba(99, 102, 241, 0.1)',
                        fill: true,
                        tension: 0.4
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: {
                        labels: { color: '#94a3b8' }
                    }
                },
                scales: {
                    x: {
                        grid: { color: '#334155' },
                        ticks: { color: '#94a3b8' }
                    },
                    y: {
                        grid: { color: '#334155' },
                        ticks: { color: '#94a3b8' },
                        min: 0,
                        max: 1
                    }
                }
            }
        });
    }

    // Diversity Chart
    const diversityCtx = document.getElementById('diversity-chart');
    if (diversityCtx) {
        AppState.diversityChart = new Chart(diversityCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Genetic Diversity',
                    data: [],
                    borderColor: '#f59e0b',
                    backgroundColor: 'rgba(245, 158, 11, 0.1)',
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: {
                        labels: { color: '#94a3b8' }
                    }
                },
                scales: {
                    x: {
                        grid: { color: '#334155' },
                        ticks: { color: '#94a3b8' }
                    },
                    y: {
                        grid: { color: '#334155' },
                        ticks: { color: '#94a3b8' },
                        min: 0,
                        max: 1
                    }
                }
            }
        });
    }
}

// ============================================================================
// Quick Start
// ============================================================================

async function quickStart(mode) {
    try {
        const button = document.querySelector(`[data-mode="${mode}"]`);
        button.disabled = true;
        button.innerHTML = '<span class="mode-icon">⏳</span><span class="mode-name">Starting...</span>';

        const response = await fetch('/api/experiments/quick-start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ mode })
        });

        const data = await response.json();
        AppState.currentExperiment = data.experiment_id;

        // Switch to evolution view
        showSection('evolution');
        document.getElementById('experiment-name').textContent = data.message;

        // Start polling for updates
        startPolling();

    } catch (error) {
        console.error('Quick start failed:', error);
        alert('Failed to start experiment. Please try again.');
    }
}

// ============================================================================
// Polling & Updates
// ============================================================================

function startPolling() {
    if (AppState.pollingInterval) {
        clearInterval(AppState.pollingInterval);
    }

    AppState.pollingInterval = setInterval(updateExperiment, 1000);
    updateExperiment(); // Immediate first update
}

function stopPolling() {
    if (AppState.pollingInterval) {
        clearInterval(AppState.pollingInterval);
        AppState.pollingInterval = null;
    }
}

async function updateExperiment() {
    if (!AppState.currentExperiment) return;

    try {
        const response = await fetch(`/api/experiments/${AppState.currentExperiment}/stream`);
        const data = await response.json();

        // Update progress ring
        const progress = data.progress;
        const circumference = 2 * Math.PI * 54;
        const offset = circumference - (progress / 100) * circumference;
        const ring = document.getElementById('progress-ring-fill');
        if (ring) {
            ring.style.strokeDashoffset = offset;
        }
        document.getElementById('progress-percent').textContent = progress;

        // Update stats
        document.getElementById('current-generation').textContent = data.current_generation;
        document.getElementById('best-fitness').textContent = data.best_fitness.toFixed(4);
        document.getElementById('experiment-status').textContent = capitalizeFirst(data.status);

        // Update charts
        updateCharts(data.visualization_data);

        // Update log
        updateLog(data.latest_logs);

        // Update explainer
        updateExplainer(data.current_generation, data.best_fitness);

        // Check if complete
        if (data.status === 'complete') {
            stopPolling();
            showResults(data);
        } else if (data.status === 'error') {
            stopPolling();
            alert('Experiment encountered an error. Check the logs for details.');
        }

    } catch (error) {
        console.error('Update failed:', error);
    }
}

function updateCharts(vizData) {
    if (!vizData) return;

    // Update fitness chart
    if (AppState.fitnessChart && vizData.fitness_history) {
        const labels = vizData.fitness_history.map(d => d.generation);
        const bestData = vizData.fitness_history.map(d => d.best);
        const avgData = vizData.fitness_history.map(d => d.average);

        AppState.fitnessChart.data.labels = labels;
        AppState.fitnessChart.data.datasets[0].data = bestData;
        AppState.fitnessChart.data.datasets[1].data = avgData;
        AppState.fitnessChart.update('none');
    }

    // Update diversity chart
    if (AppState.diversityChart && vizData.diversity_history) {
        const labels = vizData.diversity_history.map(d => d.generation);
        const divData = vizData.diversity_history.map(d => d.diversity);

        AppState.diversityChart.data.labels = labels;
        AppState.diversityChart.data.datasets[0].data = divData;
        AppState.diversityChart.update('none');
    }
}

function updateLog(logs) {
    const logContainer = document.getElementById('evolution-log');
    if (!logContainer || !logs) return;

    logContainer.innerHTML = logs.map(log => `
        <div class="log-entry">
            <span class="time">${formatTime(log.time)}</span>
            <span class="message">${highlightLog(log.message)}</span>
        </div>
    `).join('');

    logContainer.scrollTop = logContainer.scrollHeight;
}

function updateExplainer(generation, fitness) {
    const explainer = document.getElementById('explainer-content');
    if (!explainer) return;

    const explanations = [
        { gen: 1, text: "Creating initial population with diverse architectures - TRNs for interpretability, RSANs for recursive reasoning, Transformers for pattern recognition..." },
        { gen: 3, text: "Natural selection at work! Networks that better predict others' beliefs are surviving and reproducing. Watch the fitness climb!" },
        { gen: 5, text: "Crossover is combining successful traits - maybe the attention mechanism from one network with the gating from another..." },
        { gen: 8, text: "Mutation introduces novel variations. Some will fail, but others might discover unexpected solutions!" },
        { gen: 10, text: "The population is specializing. Some networks excel at Sally-Anne tests, others at cooperation games..." },
        { gen: 15, text: "Higher-order Theory of Mind is emerging - networks are starting to model what others think others think!" },
        { gen: 20, text: "Zombie detection is improving - the networks can distinguish genuine mind-reading from mere pattern matching!" }
    ];

    // Find appropriate explanation
    let explanation = explanations[0];
    for (const exp of explanations) {
        if (generation >= exp.gen) {
            explanation = exp;
        }
    }

    // Add fitness-based commentary
    let fitnessComment = '';
    if (fitness > 0.8) {
        fitnessComment = ' <strong>Excellent progress!</strong> The population has developed strong ToM capabilities.';
    } else if (fitness > 0.6) {
        fitnessComment = ' Good progress - basic false belief understanding is emerging.';
    } else if (fitness > 0.4) {
        fitnessComment = ' The networks are learning - patterns are starting to form.';
    }

    explainer.innerHTML = `<p>${explanation.text}${fitnessComment}</p>`;
}

// ============================================================================
// Results Display
// ============================================================================

async function showResults(data) {
    showSection('results');

    // Get full experiment data
    const expResponse = await fetch(`/api/experiments/${AppState.currentExperiment}`);
    const experiment = await expResponse.json();

    const vizData = experiment.visualization_data || {};
    const arch = vizData.best_architecture || {};

    // Update winner panel
    document.getElementById('winner-type').textContent = `${arch.type || 'Evolved'} Architecture`;
    document.getElementById('winner-fitness').textContent = (arch.fitness || data.best_fitness).toFixed(4);
    document.getElementById('winner-hidden').textContent = arch.hidden_dim || 256;
    document.getElementById('winner-layers').textContent = arch.num_layers || 3;

    // Generate benchmark results display
    await displayBenchmarks();

    // Get interpretability insights
    await displayInterpretability();
}

async function displayBenchmarks() {
    // Request benchmark run
    const response = await fetch('/api/benchmarks/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            experiment_id: AppState.currentExperiment,
            tests: ['sally_anne', 'higher_order', 'zombie_detection', 'cooperation']
        })
    });

    const data = await response.json();
    const results = data.results || {};

    const container = document.getElementById('benchmark-results');
    container.innerHTML = '';

    // Sally-Anne
    if (results.sally_anne) {
        addBenchmarkItem(container, 'Sally-Anne Basic', results.sally_anne.basic);
        addBenchmarkItem(container, 'Sally-Anne 2nd Order', results.sally_anne.second_order);
    }

    // Higher-order ToM
    if (results.higher_order) {
        for (let i = 1; i <= 5; i++) {
            const key = `order_${i}`;
            if (results.higher_order[key]) {
                addBenchmarkItem(container, `ToM Order ${i}`, results.higher_order[key]);
            }
        }
    }

    // Zombie detection
    if (results.zombie_detection) {
        for (const [type, result] of Object.entries(results.zombie_detection)) {
            addBenchmarkItem(container, `Zombie: ${capitalizeFirst(type)}`, result);
        }
    }

    // Cooperation
    if (results.cooperation) {
        addBenchmarkItem(container, 'Cooperation', {
            score: results.cooperation.cooperation_rate,
            passed: results.cooperation.passed
        });
    }
}

function addBenchmarkItem(container, name, result) {
    const item = document.createElement('div');
    item.className = 'benchmark-item';

    const score = (result.score * 100).toFixed(0);
    const passed = result.passed;

    item.innerHTML = `
        <div class="benchmark-name">${name}</div>
        <div class="benchmark-score ${passed ? 'passed' : 'failed'}">${score}%</div>
        <div class="benchmark-status">${passed ? '✓ Passed' : '✗ Failed'}</div>
    `;

    container.appendChild(item);
}

async function displayInterpretability() {
    try {
        const response = await fetch(`/api/interpretability/${AppState.currentExperiment}`);
        const data = await response.json();

        // Update insight cards based on results
        const insights = data.insights || {};
        const cards = document.getElementById('insight-cards');

        if (data.recommendations && data.recommendations.length > 0) {
            // Add recommendations as an additional card
            const recCard = document.createElement('div');
            recCard.className = 'insight-card';
            recCard.innerHTML = `
                <h4>📋 Key Findings</h4>
                <ul style="padding-left: 1.2rem; color: var(--text-secondary);">
                    ${data.recommendations.map(r => `<li>${r}</li>`).join('')}
                </ul>
            `;
            cards.appendChild(recCard);
        }
    } catch (error) {
        console.error('Failed to load interpretability:', error);
    }
}

// ============================================================================
// Concept Modal
// ============================================================================

function showConceptDetail(conceptKey) {
    if (!AppState.concepts) return;

    const concept = AppState.concepts[conceptKey];
    if (!concept) return;

    const modal = document.getElementById('concept-modal');
    const title = document.getElementById('modal-title');
    const body = document.getElementById('modal-body');

    // Format title
    title.textContent = conceptKey.split('_').map(capitalizeFirst).join(' ');

    // Format body
    let html = '';

    if (concept.simple) {
        html += `<p><strong>In Simple Terms:</strong> ${concept.simple}</p>`;
    }

    if (concept.example) {
        html += `<p><strong>Example:</strong> ${concept.example}</p>`;
    }

    if (concept.why_it_matters) {
        html += `<p><strong>Why It Matters:</strong> ${concept.why_it_matters}</p>`;
    }

    if (concept.answer) {
        html += `<p><strong>Answer:</strong> ${concept.answer}</p>`;
    }

    if (concept.what_it_tests) {
        html += `<p><strong>What It Tests:</strong> ${concept.what_it_tests}</p>`;
    }

    // For architectures, show all options
    if (conceptKey === 'architectures') {
        html = '<p>ToM-NAS evolves multiple types of neural architectures:</p>';
        for (const [key, arch] of Object.entries(concept)) {
            html += `
                <div style="margin: 1rem 0; padding: 1rem; background: var(--bg-dark); border-radius: 0.5rem;">
                    <h4>${arch.name}</h4>
                    <p><strong>Simple:</strong> ${arch.simple}</p>
                    <p><strong>Strength:</strong> ${arch.strength}</p>
                </div>
            `;
        }
    }

    body.innerHTML = html;
    modal.classList.remove('hidden');
}

function closeModal() {
    document.getElementById('concept-modal').classList.add('hidden');
}

// Close modal on outside click
document.getElementById('concept-modal')?.addEventListener('click', (e) => {
    if (e.target.id === 'concept-modal') {
        closeModal();
    }
});

// ============================================================================
// Section Navigation
// ============================================================================

function showSection(sectionName) {
    // Hide all sections
    document.querySelectorAll('.section').forEach(s => s.classList.add('hidden'));

    // Show target section
    const section = document.getElementById(`${sectionName}-section`);
    if (section) {
        section.classList.remove('hidden');
    }

    // Reset mode buttons when returning to welcome
    if (sectionName === 'welcome') {
        document.querySelectorAll('.mode-btn').forEach(btn => {
            btn.disabled = false;
            const mode = btn.dataset.mode;
            const icons = { quick: '⚡', balanced: '⚖️', thorough: '🔬' };
            const names = { quick: 'Quick', balanced: 'Balanced', thorough: 'Thorough' };
            const descs = { quick: '~2 minutes', balanced: '~10 minutes', thorough: '~30 minutes' };
            btn.innerHTML = `
                <span class="mode-icon">${icons[mode]}</span>
                <span class="mode-name">${names[mode]}</span>
                <span class="mode-desc">${descs[mode]}</span>
                ${mode === 'balanced' ? '<span class="recommended-badge">Recommended</span>' : ''}
            `;
        });

        // Reset charts
        if (AppState.fitnessChart) {
            AppState.fitnessChart.data.labels = [];
            AppState.fitnessChart.data.datasets.forEach(ds => ds.data = []);
            AppState.fitnessChart.update();
        }
        if (AppState.diversityChart) {
            AppState.diversityChart.data.labels = [];
            AppState.diversityChart.data.datasets.forEach(ds => ds.data = []);
            AppState.diversityChart.update();
        }
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

function capitalizeFirst(str) {
    if (!str) return '';
    return str.charAt(0).toUpperCase() + str.slice(1);
}

function formatTime(isoString) {
    if (!isoString) return '';
    const date = new Date(isoString);
    return date.toLocaleTimeString();
}

function highlightLog(message) {
    return message
        .replace(/Best fitness = ([\d.]+)/g, 'Best fitness = <span class="highlight">$1</span>')
        .replace(/Generation (\d+)/g, 'Generation <span class="highlight">$1</span>');
}
