/**
 * Agents.js - Agent Visualization and Behavior
 *
 * Renders agents as visual entities in the world with
 * thought bubbles, activities, and interactions.
 */

class AgentManager {
    constructor(scene) {
        this.scene = scene;
        this.agents = new Map();
        this.selectedAgent = null;
        this.hoveredAgent = null;

        // Agent appearance templates
        this.appearances = {
            hollow: { color: 0x4466aa, emissive: 0x223355 },
            market: { color: 0x44aa66, emissive: 0x225533 },
            ministry: { color: 0xaa6644, emissive: 0x553322 },
            court: { color: 0xaa4466, emissive: 0x552233 },
            temple: { color: 0x6644aa, emissive: 0x332255 }
        };
    }

    /**
     * Create visual representation for an agent
     */
    createAgentMesh(agentData) {
        const group = new THREE.Group();
        group.userData = { agentId: agentData.id, data: agentData };

        // Get appearance based on realm/specialty
        const appearance = this.appearances[agentData.realm] || this.appearances.hollow;

        // Body (floating humanoid abstraction)
        const bodyGeo = new THREE.CapsuleGeometry(0.4, 1.2, 8, 16);
        const bodyMat = new THREE.MeshStandardMaterial({
            color: appearance.color,
            emissive: appearance.emissive,
            roughness: 0.5,
            metalness: 0.3
        });
        const body = new THREE.Mesh(bodyGeo, bodyMat);
        body.position.y = 1.5;
        group.add(body);

        // Head (glowing orb representing consciousness)
        const headGeo = new THREE.SphereGeometry(0.3, 16, 16);
        const headMat = new THREE.MeshStandardMaterial({
            color: 0xffffff,
            emissive: appearance.color,
            emissiveIntensity: 0.5,
            roughness: 0.2
        });
        const head = new THREE.Mesh(headGeo, headMat);
        head.position.y = 2.5;
        group.add(head);

        // Thought bubble (invisible until thinking)
        const thoughtBubble = this.createThoughtBubble();
        thoughtBubble.visible = false;
        thoughtBubble.position.set(0.5, 3.2, 0);
        group.add(thoughtBubble);

        // Activity indicator ring
        const ringGeo = new THREE.TorusGeometry(0.8, 0.05, 8, 32);
        const ringMat = new THREE.MeshBasicMaterial({
            color: appearance.color,
            transparent: true,
            opacity: 0.5
        });
        const ring = new THREE.Mesh(ringGeo, ringMat);
        ring.rotation.x = Math.PI / 2;
        ring.position.y = 0.1;
        group.add(ring);

        // Name label (simplified - in production use sprite text)
        const labelSprite = this.createNameLabel(agentData.name);
        labelSprite.position.y = 3.5;
        group.add(labelSprite);

        // Store references
        group.userData.body = body;
        group.userData.head = head;
        group.userData.thoughtBubble = thoughtBubble;
        group.userData.ring = ring;
        group.userData.label = labelSprite;

        return group;
    }

    createThoughtBubble() {
        const group = new THREE.Group();

        // Main bubble
        const bubbleGeo = new THREE.SphereGeometry(0.4, 16, 16);
        const bubbleMat = new THREE.MeshBasicMaterial({
            color: 0xffffff,
            transparent: true,
            opacity: 0.8
        });
        const bubble = new THREE.Mesh(bubbleGeo, bubbleMat);
        group.add(bubble);

        // Smaller bubbles
        const smallGeo = new THREE.SphereGeometry(0.15, 8, 8);
        const small1 = new THREE.Mesh(smallGeo, bubbleMat);
        small1.position.set(-0.3, -0.3, 0);
        group.add(small1);

        const tinyGeo = new THREE.SphereGeometry(0.08, 8, 8);
        const tiny = new THREE.Mesh(tinyGeo, bubbleMat);
        tiny.position.set(-0.5, -0.5, 0);
        group.add(tiny);

        return group;
    }

    createNameLabel(name) {
        // Create canvas for text
        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d');
        canvas.width = 256;
        canvas.height = 64;

        context.fillStyle = 'rgba(0, 0, 0, 0.5)';
        context.fillRect(0, 0, canvas.width, canvas.height);

        context.font = 'Bold 24px Arial';
        context.fillStyle = 'white';
        context.textAlign = 'center';
        context.fillText(name || 'Agent', canvas.width / 2, 40);

        const texture = new THREE.CanvasTexture(canvas);
        const material = new THREE.SpriteMaterial({
            map: texture,
            transparent: true
        });

        const sprite = new THREE.Sprite(material);
        sprite.scale.set(2, 0.5, 1);
        return sprite;
    }

    /**
     * Update agents from server state
     */
    updateFromServer(agentsData) {
        const currentIds = new Set(this.agents.keys());
        const newIds = new Set(agentsData.map(a => a.id));

        // Remove agents that no longer exist
        for (const id of currentIds) {
            if (!newIds.has(id)) {
                this.removeAgent(id);
            }
        }

        // Add or update agents
        for (const data of agentsData) {
            if (this.agents.has(data.id)) {
                this.updateAgent(data);
            } else {
                this.addAgent(data);
            }
        }
    }

    /**
     * Add a new agent to the world
     */
    addAgent(data) {
        const mesh = this.createAgentMesh(data);

        // Set initial position
        const pos = data.position || { x: 0, y: 0, z: 0 };
        mesh.position.set(pos.x, pos.y, pos.z);

        // Add movement properties
        mesh.userData.targetPosition = mesh.position.clone();
        mesh.userData.velocity = new THREE.Vector3();

        this.scene.add(mesh);
        this.agents.set(data.id, mesh);

        return mesh;
    }

    /**
     * Update existing agent
     */
    updateAgent(data) {
        const mesh = this.agents.get(data.id);
        if (!mesh) return;

        // Update position (smooth interpolation)
        if (data.position) {
            mesh.userData.targetPosition.set(
                data.position.x,
                data.position.y,
                data.position.z
            );
        }

        // Update activity visualization
        this.updateActivityVisualization(mesh, data);

        // Update data reference
        mesh.userData.data = data;
    }

    /**
     * Update visual based on activity
     */
    updateActivityVisualization(mesh, data) {
        const ring = mesh.userData.ring;
        const thoughtBubble = mesh.userData.thoughtBubble;
        const head = mesh.userData.head;

        const activity = data.activity || 'idle';

        switch (activity) {
            case 'researching':
                ring.material.color.setHex(0x00ff88);
                thoughtBubble.visible = true;
                head.material.emissiveIntensity = 0.8;
                break;

            case 'publishing':
                ring.material.color.setHex(0xffaa00);
                thoughtBubble.visible = true;
                head.material.emissiveIntensity = 1.0;
                break;

            case 'simulating':
                ring.material.color.setHex(0xff00ff);
                thoughtBubble.visible = true;
                head.material.emissiveIntensity = 1.2;
                // Add particle effect for simulation
                break;

            case 'collaborating':
                ring.material.color.setHex(0x00aaff);
                thoughtBubble.visible = false;
                head.material.emissiveIntensity = 0.6;
                break;

            case 'moving':
                ring.material.color.setHex(0x888888);
                thoughtBubble.visible = false;
                head.material.emissiveIntensity = 0.3;
                break;

            default: // idle
                ring.material.color.setHex(0x444444);
                thoughtBubble.visible = false;
                head.material.emissiveIntensity = 0.2;
        }
    }

    /**
     * Remove agent from world
     */
    removeAgent(id) {
        const mesh = this.agents.get(id);
        if (mesh) {
            this.scene.remove(mesh);
            this.agents.delete(id);
        }
    }

    /**
     * Animate all agents
     */
    update(deltaTime, time) {
        for (const [id, mesh] of this.agents) {
            // Smooth position interpolation
            const target = mesh.userData.targetPosition;
            mesh.position.lerp(target, 0.05);

            // Idle animation
            mesh.position.y = Math.sin(time * 0.002 + mesh.userData.data.id.charCodeAt(0)) * 0.1;

            // Rotate ring
            const ring = mesh.userData.ring;
            if (ring) {
                ring.rotation.z += 0.01;
            }

            // Pulse thought bubble
            const bubble = mesh.userData.thoughtBubble;
            if (bubble && bubble.visible) {
                bubble.scale.setScalar(1 + Math.sin(time * 0.005) * 0.1);
            }

            // Face camera (billboard effect for labels)
            const label = mesh.userData.label;
            if (label && window.camera) {
                label.lookAt(window.camera.position);
            }
        }
    }

    /**
     * Get agent at screen position (for interaction)
     */
    getAgentAtScreenPosition(screenX, screenY, camera) {
        const raycaster = new THREE.Raycaster();
        const mouse = new THREE.Vector2(
            (screenX / window.innerWidth) * 2 - 1,
            -(screenY / window.innerHeight) * 2 + 1
        );

        raycaster.setFromCamera(mouse, camera);

        const meshes = Array.from(this.agents.values());
        const intersects = raycaster.intersectObjects(meshes, true);

        if (intersects.length > 0) {
            // Find parent group
            let obj = intersects[0].object;
            while (obj.parent && !obj.userData.agentId) {
                obj = obj.parent;
            }
            return obj.userData.agentId ? this.agents.get(obj.userData.agentId) : null;
        }

        return null;
    }

    /**
     * Get nearest agent to position
     */
    getNearestAgent(position, maxDistance = 10) {
        let nearest = null;
        let nearestDist = maxDistance;

        for (const [id, mesh] of this.agents) {
            const dist = position.distanceTo(mesh.position);
            if (dist < nearestDist) {
                nearestDist = dist;
                nearest = mesh;
            }
        }

        return nearest;
    }

    /**
     * Highlight an agent (for hover/selection)
     */
    highlightAgent(agentMesh, highlight = true) {
        if (!agentMesh) return;

        const body = agentMesh.userData.body;
        const ring = agentMesh.userData.ring;

        if (highlight) {
            body.material.emissiveIntensity = 1.0;
            ring.material.opacity = 1.0;
            ring.scale.setScalar(1.2);
        } else {
            body.material.emissiveIntensity = 0.3;
            ring.material.opacity = 0.5;
            ring.scale.setScalar(1.0);
        }
    }

    /**
     * Select an agent
     */
    selectAgent(agentMesh) {
        // Deselect previous
        if (this.selectedAgent) {
            this.highlightAgent(this.selectedAgent, false);
        }

        this.selectedAgent = agentMesh;

        if (agentMesh) {
            this.highlightAgent(agentMesh, true);
            return agentMesh.userData.data;
        }

        return null;
    }

    /**
     * Get all agents in a realm
     */
    getAgentsInRealm(realm) {
        return Array.from(this.agents.values())
            .filter(mesh => mesh.userData.data.realm === realm);
    }
}

// Export
window.AgentManager = AgentManager;
