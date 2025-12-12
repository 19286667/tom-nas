/**
 * Main.js - Game Entry Point
 *
 * Initializes and runs the ToM-NAS exploration experience.
 */

class Game {
    constructor() {
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.clock = null;

        // Managers
        this.world = null;
        this.agents = null;
        this.controls = null;
        this.hud = null;
        this.network = null;

        // State
        this.isPaused = false;
        this.isMapOpen = false;
        this.isInventoryOpen = false;
        this.currentRealm = 'hollow';

        // Latest server state
        this.serverState = null;
    }

    /**
     * Initialize the game
     */
    async init() {
        console.log('Initializing ToM-NAS...');

        // Show loading
        document.getElementById('loading').classList.remove('hidden');

        // Initialize Three.js
        this.initThree();

        // Initialize managers
        this.world = new LiminalWorld(this.scene);
        this.agents = new AgentManager(this.scene);
        this.controls = new PlayerControls(this.camera, this.renderer.domElement);
        this.hud = new HUD();

        // Build world
        this.world.build();

        // Add lighting
        this.initLighting();

        // Initialize network
        await this.initNetwork();

        // Hide loading
        document.getElementById('loading').classList.add('hidden');

        // Expose globally for controls
        window.camera = this.camera;
        window.game = this;

        // Start render loop
        this.animate();

        console.log('ToM-NAS initialized');
    }

    initThree() {
        // Scene
        this.scene = new THREE.Scene();
        this.scene.fog = new THREE.FogExp2(0x0a0a0f, 0.008);

        // Camera
        this.camera = new THREE.PerspectiveCamera(
            75,
            window.innerWidth / window.innerHeight,
            0.1,
            1000
        );
        this.camera.position.set(0, 2, 10);

        // Renderer
        const canvas = document.getElementById('canvas');
        this.renderer = new THREE.WebGLRenderer({
            canvas: canvas,
            antialias: true
        });
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;

        // Clock
        this.clock = new THREE.Clock();

        // Handle resize
        window.addEventListener('resize', () => this.onResize());
    }

    initLighting() {
        // Ambient light
        const ambient = new THREE.AmbientLight(0x404060, 0.5);
        this.scene.add(ambient);

        // Directional light (moon-like)
        const directional = new THREE.DirectionalLight(0x6060ff, 0.3);
        directional.position.set(50, 100, 50);
        directional.castShadow = true;
        directional.shadow.mapSize.width = 2048;
        directional.shadow.mapSize.height = 2048;
        this.scene.add(directional);

        // Point lights in each realm
        const realmLights = [
            { pos: [0, 10, 0], color: 0x4466aa },      // Hollow
            { pos: [80, 10, 0], color: 0x44aa66 },     // Market
            { pos: [60, 10, -80], color: 0xaa6644 },   // Ministry
            { pos: [-80, 10, 0], color: 0xaa4466 },    // Court
            { pos: [0, 10, 80], color: 0x6644aa },     // Temple
        ];

        realmLights.forEach(light => {
            const pointLight = new THREE.PointLight(light.color, 1, 50);
            pointLight.position.set(...light.pos);
            this.scene.add(pointLight);
        });
    }

    async initNetwork() {
        this.network = new NetworkManager();

        // Set up callbacks
        this.network.onConnect = () => {
            this.hud.updateConnectionStatus(true);
            this.hud.showNotification('Connected to simulation', 'success');
        };

        this.network.onDisconnect = () => {
            this.hud.updateConnectionStatus(false);
            this.hud.showNotification('Disconnected from simulation', 'error');
        };

        this.network.onStateUpdate = (state) => {
            this.handleStateUpdate(state);
        };

        this.network.onError = (error) => {
            console.error('Network error:', error);
        };

        // Try to connect
        try {
            await this.network.connect();
        } catch (error) {
            console.warn('Could not connect to simulation server. Running in offline mode.');
            this.hud.showNotification('Running in offline mode', 'info');
            // Generate some demo agents
            this.generateDemoAgents();
        }
    }

    /**
     * Generate demo agents for offline mode
     */
    generateDemoAgents() {
        const realms = ['hollow', 'market', 'ministry', 'court', 'temple'];
        const activities = ['researching', 'publishing', 'simulating', 'collaborating', 'idle'];

        const demoAgents = [];
        realms.forEach((realm, ri) => {
            const realmCenters = {
                hollow: { x: 0, z: 0 },
                market: { x: 80, z: 0 },
                ministry: { x: 60, z: -80 },
                court: { x: -80, z: 0 },
                temple: { x: 0, z: 80 }
            };

            for (let i = 0; i < 5; i++) {
                const center = realmCenters[realm];
                demoAgents.push({
                    id: `demo_${realm}_${i}`,
                    name: `Dr. ${realm.charAt(0).toUpperCase() + realm.slice(1)} ${i + 1}`,
                    realm: realm,
                    specialty: realm,
                    position: {
                        x: center.x + (Math.random() - 0.5) * 30,
                        y: 0,
                        z: center.z + (Math.random() - 0.5) * 30
                    },
                    activity: activities[Math.floor(Math.random() * activities.length)],
                    thought: 'Contemplating the nature of recursive belief...'
                });
            }
        });

        this.agents.updateFromServer(demoAgents);
        this.hud.updateStats({ ticks: 0, agents: demoAgents.length, active_sims: 0 });
    }

    /**
     * Handle state update from server
     */
    handleStateUpdate(state) {
        this.serverState = state;

        // Update agents
        if (state.agents) {
            this.agents.updateFromServer(state.agents);
        }

        // Update stats
        if (state.stats) {
            this.hud.updateStats(state.stats);
        }

        // Handle events
        if (state.events) {
            state.events.forEach(event => {
                this.hud.addEvent(event);
            });
        }
    }

    /**
     * Main render loop
     */
    animate() {
        requestAnimationFrame(() => this.animate());

        if (this.isPaused) return;

        const deltaTime = this.clock.getDelta();
        const time = this.clock.getElapsedTime() * 1000;

        // Update controls
        this.controls.update(deltaTime);

        // Update world animations
        this.world.update(time);

        // Update agents
        this.agents.update(deltaTime, time);

        // Get player position
        const playerPos = this.controls.getPosition();

        // Update current realm
        const newRealm = this.world.getRealmAt(playerPos);
        if (newRealm !== this.currentRealm) {
            this.currentRealm = newRealm;
            this.hud.updateRealm(newRealm);
        }

        // Check for nearby agents
        const nearestAgent = this.agents.getNearestAgent(playerPos, 5);
        this.hud.updateNearbyAgent(nearestAgent?.userData?.data);

        // Update minimap
        const agentsData = this.serverState?.agents ||
            Array.from(this.agents.agents.values()).map(a => a.userData.data);
        this.hud.updateMinimap(playerPos, agentsData, this.currentRealm);

        // Check for portal proximity
        const portal = this.world.getPortalAt(playerPos);
        if (portal) {
            // Could show portal destination hint
        }

        // Render
        this.renderer.render(this.scene, this.camera);
    }

    /**
     * Handle window resize
     */
    onResize() {
        this.camera.aspect = window.innerWidth / window.innerHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(window.innerWidth, window.innerHeight);
    }

    /**
     * Interact with nearest entity
     */
    interact() {
        const playerPos = this.controls.getPosition();
        const nearestAgent = this.agents.getNearestAgent(playerPos, 5);

        if (nearestAgent) {
            const agentData = nearestAgent.userData.data;
            this.agents.selectAgent(nearestAgent);
            this.hud.showAgentPanel(agentData);

            // Request detailed info from server
            if (this.network.isConnected) {
                this.network.requestAgentDetails(agentData.id);
            }
        }
    }

    /**
     * Toggle map overlay
     */
    toggleMap() {
        this.isMapOpen = !this.isMapOpen;
        // Could show full-screen map view
        console.log('Map toggled:', this.isMapOpen);
    }

    /**
     * Toggle inventory (collected publications)
     */
    toggleInventory() {
        this.isInventoryOpen = !this.isInventoryOpen;
        console.log('Inventory toggled:', this.isInventoryOpen);
    }

    /**
     * Toggle pause
     */
    togglePause() {
        this.isPaused = !this.isPaused;

        if (this.network.isConnected) {
            if (this.isPaused) {
                this.network.pause();
            } else {
                this.network.resume();
            }
        }

        this.hud.showNotification(
            this.isPaused ? 'Simulation Paused' : 'Simulation Resumed',
            'info'
        );
    }

    /**
     * Toggle console
     */
    toggleConsole() {
        console.log('Console toggled');
        // Could show debug console
    }
}

// Menu functions
function startExploration() {
    document.getElementById('main-menu').classList.add('hidden');
    window.game.controls.lock();
}

function showRealms() {
    // Could show realm selection screen
    console.log('Realms screen');
}

function showSettings() {
    // Could show settings panel
    console.log('Settings screen');
}

// Initialize on load
window.addEventListener('DOMContentLoaded', async () => {
    const game = new Game();
    await game.init();
});

// Export
window.Game = Game;
window.startExploration = startExploration;
window.showRealms = showRealms;
window.showSettings = showSettings;
