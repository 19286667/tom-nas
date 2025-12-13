/**
 * HUD.js - Heads Up Display Management
 *
 * Updates all UI elements: stats, nearby info, minimap, panels.
 */

class HUD {
    constructor() {
        // Cache DOM elements
        this.elements = {
            time: document.getElementById('stat-time'),
            agents: document.getElementById('stat-agents'),
            sims: document.getElementById('stat-sims'),
            realm: document.getElementById('stat-realm'),
            nearbyName: document.getElementById('nearby-name'),
            nearbyThought: document.getElementById('nearby-thought'),
            recentEvent: document.getElementById('recent-event'),
            interactionPrompt: document.getElementById('interaction-prompt'),
            statusDot: document.getElementById('status-dot'),
            statusText: document.getElementById('status-text'),
            minimapCanvas: document.getElementById('minimap-canvas'),
            agentPanel: document.getElementById('agent-panel'),
            panelName: document.getElementById('panel-agent-name'),
            panelResearch: document.getElementById('panel-research'),
            panelBeliefs: document.getElementById('panel-beliefs'),
            panelPublications: document.getElementById('panel-publications'),
        };

        // Minimap context
        this.minimapCtx = this.elements.minimapCanvas?.getContext('2d');
        if (this.minimapCtx) {
            this.elements.minimapCanvas.width = 200;
            this.elements.minimapCanvas.height = 200;
        }

        // State
        this.nearbyAgent = null;
        this.recentEvents = [];
        this.isPanelOpen = false;
    }

    /**
     * Update stats display
     */
    updateStats(stats) {
        if (this.elements.time) {
            this.elements.time.textContent = stats.ticks || 0;
        }
        if (this.elements.agents) {
            this.elements.agents.textContent = stats.agents || 0;
        }
        if (this.elements.sims) {
            this.elements.sims.textContent = stats.active_sims || 0;
        }
    }

    /**
     * Update current realm display
     */
    updateRealm(realm) {
        if (this.elements.realm) {
            const realmNames = {
                hollow: 'THE HOLLOW',
                market: 'THE MARKET',
                ministry: 'THE MINISTRY',
                court: 'THE COURT',
                temple: 'THE TEMPLE'
            };
            this.elements.realm.textContent = realmNames[realm] || realm.toUpperCase();
        }
    }

    /**
     * Update nearby agent info
     */
    updateNearbyAgent(agent) {
        this.nearbyAgent = agent;

        if (agent) {
            if (this.elements.nearbyName) {
                this.elements.nearbyName.textContent = `${agent.name} (${agent.specialty || 'General'})`;
            }
            if (this.elements.nearbyThought) {
                const thought = agent.thought || agent.activity || 'Contemplating...';
                this.elements.nearbyThought.textContent = `- "${thought}"`;
            }
            if (this.elements.interactionPrompt) {
                this.elements.interactionPrompt.style.display = 'block';
            }
        } else {
            if (this.elements.nearbyName) {
                this.elements.nearbyName.textContent = '-';
            }
            if (this.elements.nearbyThought) {
                this.elements.nearbyThought.textContent = '';
            }
            if (this.elements.interactionPrompt) {
                this.elements.interactionPrompt.style.display = 'none';
            }
        }
    }

    /**
     * Add event to recent events
     */
    addEvent(event) {
        this.recentEvents.unshift(event);
        if (this.recentEvents.length > 10) {
            this.recentEvents.pop();
        }

        if (this.elements.recentEvent && this.recentEvents.length > 0) {
            const latest = this.recentEvents[0];
            this.elements.recentEvent.textContent =
                `RECENT: ${latest.agent_id || 'System'} - ${latest.action_type || latest.type || 'Event'}`;
        }
    }

    /**
     * Update connection status
     */
    updateConnectionStatus(connected) {
        if (this.elements.statusDot) {
            this.elements.statusDot.classList.toggle('connected', connected);
        }
        if (this.elements.statusText) {
            this.elements.statusText.textContent = connected ? 'Connected' : 'Disconnected';
        }
    }

    /**
     * Draw minimap
     */
    updateMinimap(playerPos, agents, realm) {
        if (!this.minimapCtx) return;

        const ctx = this.minimapCtx;
        const w = 200;
        const h = 200;
        const scale = 0.4; // World units to pixels

        // Clear
        ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
        ctx.fillRect(0, 0, w, h);

        // Draw realm boundaries
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
        ctx.lineWidth = 1;

        const realmCenters = {
            hollow: { x: 0, z: 0, color: '#4466aa' },
            market: { x: 80, z: 0, color: '#44aa66' },
            ministry: { x: 60, z: -80, color: '#aa6644' },
            court: { x: -80, z: 0, color: '#aa4466' },
            temple: { x: 0, z: 80, color: '#6644aa' }
        };

        // Draw realm circles
        for (const [name, data] of Object.entries(realmCenters)) {
            const x = w/2 + data.x * scale;
            const y = h/2 + data.z * scale;

            ctx.beginPath();
            ctx.arc(x, y, 15, 0, Math.PI * 2);
            ctx.strokeStyle = data.color;
            ctx.stroke();

            // Current realm highlight
            if (name === realm) {
                ctx.fillStyle = data.color + '40';
                ctx.fill();
            }
        }

        // Draw agents
        if (agents) {
            for (const agent of agents) {
                const ax = w/2 + (agent.position?.x || 0) * scale;
                const ay = h/2 + (agent.position?.z || 0) * scale;

                ctx.beginPath();
                ctx.arc(ax, ay, 2, 0, Math.PI * 2);
                ctx.fillStyle = realmCenters[agent.realm]?.color || '#888';
                ctx.fill();
            }
        }

        // Draw player
        const px = w/2 + playerPos.x * scale;
        const py = h/2 + playerPos.z * scale;

        ctx.beginPath();
        ctx.arc(px, py, 4, 0, Math.PI * 2);
        ctx.fillStyle = '#00ff88';
        ctx.fill();

        // Player direction indicator
        const dir = window.game?.controls?.getLookDirection?.() || new THREE.Vector3(0, 0, -1);
        ctx.beginPath();
        ctx.moveTo(px, py);
        ctx.lineTo(px + dir.x * 10, py + dir.z * 10);
        ctx.strokeStyle = '#00ff88';
        ctx.lineWidth = 2;
        ctx.stroke();
    }

    /**
     * Show agent panel
     */
    showAgentPanel(agentData) {
        if (!agentData) return;

        this.isPanelOpen = true;

        if (this.elements.panelName) {
            this.elements.panelName.textContent = agentData.name || 'Unknown Agent';
        }

        if (this.elements.panelResearch) {
            this.elements.panelResearch.textContent =
                agentData.current_research || agentData.thought || 'No active research';
        }

        if (this.elements.panelBeliefs) {
            const beliefs = agentData.beliefs || {};
            const beliefText = Object.entries(beliefs)
                .map(([k, v]) => `${k}: ${typeof v === 'object' ? JSON.stringify(v) : v}`)
                .join('\n') || 'No recorded beliefs';
            this.elements.panelBeliefs.textContent = beliefText;
        }

        if (this.elements.panelPublications) {
            const pubs = agentData.publications || [];
            this.elements.panelPublications.innerHTML = '';
            pubs.slice(0, 5).forEach(pub => {
                const li = document.createElement('li');
                li.textContent = pub.title || pub;
                this.elements.panelPublications.appendChild(li);
            });
            if (pubs.length === 0) {
                const li = document.createElement('li');
                li.textContent = 'No publications yet';
                this.elements.panelPublications.appendChild(li);
            }
        }

        if (this.elements.agentPanel) {
            this.elements.agentPanel.classList.add('visible');
        }
    }

    /**
     * Close agent panel
     */
    closeAgentPanel() {
        this.isPanelOpen = false;
        if (this.elements.agentPanel) {
            this.elements.agentPanel.classList.remove('visible');
        }
    }

    /**
     * Show notification
     */
    showNotification(message, type = 'info') {
        // Create notification element
        const notif = document.createElement('div');
        notif.className = `notification notification-${type}`;
        notif.textContent = message;
        notif.style.cssText = `
            position: fixed;
            top: 100px;
            left: 50%;
            transform: translateX(-50%);
            padding: 12px 24px;
            background: rgba(0, 0, 0, 0.8);
            border: 1px solid ${type === 'error' ? '#ff4444' : '#00aaff'};
            color: white;
            font-size: 14px;
            border-radius: 4px;
            z-index: 1000;
            animation: fadeInOut 3s ease-in-out;
        `;

        document.body.appendChild(notif);

        setTimeout(() => {
            notif.remove();
        }, 3000);
    }
}

// Global close function
window.closeAgentPanel = function() {
    if (window.game?.hud) {
        window.game.hud.closeAgentPanel();
    }
};

// Export
window.HUD = HUD;
