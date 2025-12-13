/**
 * World.js - 3D World Construction
 *
 * Creates the immersive environment for exploring the simulation.
 * Five realms: The Hollow, The Market, The Ministry, The Court, The Temple
 */

class LiminalWorld {
    constructor(scene) {
        this.scene = scene;
        this.realms = {};
        this.currentRealm = 'hollow';
        this.portals = [];
        this.landmarks = [];
    }

    /**
     * Build the complete world
     */
    build() {
        this.createSkybox();
        this.createGround();
        this.createRealms();
        this.createPortals();
        this.createAmbientEffects();
    }

    createSkybox() {
        // Gradient skybox for ethereal atmosphere
        const vertexShader = `
            varying vec3 vWorldPosition;
            void main() {
                vec4 worldPosition = modelMatrix * vec4(position, 1.0);
                vWorldPosition = worldPosition.xyz;
                gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
            }
        `;

        const fragmentShader = `
            uniform vec3 topColor;
            uniform vec3 bottomColor;
            uniform float offset;
            uniform float exponent;
            varying vec3 vWorldPosition;
            void main() {
                float h = normalize(vWorldPosition + offset).y;
                gl_FragColor = vec4(mix(bottomColor, topColor, max(pow(max(h, 0.0), exponent), 0.0)), 1.0);
            }
        `;

        const uniforms = {
            topColor: { value: new THREE.Color(0x0a0a20) },
            bottomColor: { value: new THREE.Color(0x000010) },
            offset: { value: 400 },
            exponent: { value: 0.6 }
        };

        const skyGeo = new THREE.SphereGeometry(1000, 32, 15);
        const skyMat = new THREE.ShaderMaterial({
            uniforms: uniforms,
            vertexShader: vertexShader,
            fragmentShader: fragmentShader,
            side: THREE.BackSide
        });

        const sky = new THREE.Mesh(skyGeo, skyMat);
        this.scene.add(sky);
    }

    createGround() {
        // Procedural ground with grid effect
        const groundGeo = new THREE.PlaneGeometry(500, 500, 50, 50);

        // Displace vertices for terrain
        const vertices = groundGeo.attributes.position.array;
        for (let i = 0; i < vertices.length; i += 3) {
            vertices[i + 2] = Math.sin(vertices[i] * 0.05) * Math.cos(vertices[i + 1] * 0.05) * 2;
        }
        groundGeo.computeVertexNormals();

        const groundMat = new THREE.MeshStandardMaterial({
            color: 0x111122,
            roughness: 0.9,
            metalness: 0.1,
            wireframe: false
        });

        const ground = new THREE.Mesh(groundGeo, groundMat);
        ground.rotation.x = -Math.PI / 2;
        ground.receiveShadow = true;
        this.scene.add(ground);

        // Grid overlay
        const gridHelper = new THREE.GridHelper(500, 100, 0x222244, 0x111133);
        gridHelper.position.y = 0.1;
        this.scene.add(gridHelper);
    }

    createRealms() {
        // The Hollow - Pure Research (center)
        this.realms.hollow = this.createHollow(0, 0);

        // The Market - Trade Ideas (east)
        this.realms.market = this.createMarket(80, 0);

        // The Ministry - Rules & Policy (north-east)
        this.realms.ministry = this.createMinistry(60, -80);

        // The Court - Power Dynamics (west)
        this.realms.court = this.createCourt(-80, 0);

        // The Temple - Meaning & Purpose (south)
        this.realms.temple = this.createTemple(0, 80);
    }

    createHollow(x, z) {
        const group = new THREE.Group();
        group.position.set(x, 0, z);

        // Central research tower
        const towerGeo = new THREE.CylinderGeometry(8, 12, 40, 8);
        const towerMat = new THREE.MeshStandardMaterial({
            color: 0x1a1a2e,
            emissive: 0x0a0a1e,
            roughness: 0.3
        });
        const tower = new THREE.Mesh(towerGeo, towerMat);
        tower.position.y = 20;
        group.add(tower);

        // Floating research pods
        for (let i = 0; i < 6; i++) {
            const angle = (i / 6) * Math.PI * 2;
            const pod = this.createResearchPod();
            pod.position.set(
                Math.cos(angle) * 25,
                10 + Math.sin(i) * 5,
                Math.sin(angle) * 25
            );
            pod.userData.floatOffset = i;
            group.add(pod);
        }

        // Publication boards
        const board = this.createPublicationBoard();
        board.position.set(15, 3, 0);
        group.add(board);

        this.scene.add(group);
        return group;
    }

    createMarket(x, z) {
        const group = new THREE.Group();
        group.position.set(x, 0, z);

        // Market stalls
        for (let i = 0; i < 8; i++) {
            const stall = this.createMarketStall();
            const angle = (i / 8) * Math.PI * 2;
            stall.position.set(
                Math.cos(angle) * 15,
                0,
                Math.sin(angle) * 15
            );
            stall.rotation.y = angle + Math.PI;
            group.add(stall);
        }

        // Central exchange
        const exchangeGeo = new THREE.OctahedronGeometry(6, 0);
        const exchangeMat = new THREE.MeshStandardMaterial({
            color: 0x2a4a2a,
            emissive: 0x1a3a1a,
            wireframe: true
        });
        const exchange = new THREE.Mesh(exchangeGeo, exchangeMat);
        exchange.position.y = 8;
        group.add(exchange);

        this.scene.add(group);
        return group;
    }

    createMinistry(x, z) {
        const group = new THREE.Group();
        group.position.set(x, 0, z);

        // Bureaucratic tower
        const towerGeo = new THREE.BoxGeometry(20, 50, 20);
        const towerMat = new THREE.MeshStandardMaterial({
            color: 0x2a2a3a,
            roughness: 0.8
        });
        const tower = new THREE.Mesh(towerGeo, towerMat);
        tower.position.y = 25;
        group.add(tower);

        // Flying papers effect
        for (let i = 0; i < 20; i++) {
            const paper = this.createFloatingPaper();
            paper.position.set(
                (Math.random() - 0.5) * 30,
                5 + Math.random() * 40,
                (Math.random() - 0.5) * 30
            );
            paper.userData.floatSpeed = 0.5 + Math.random();
            group.add(paper);
        }

        this.scene.add(group);
        return group;
    }

    createCourt(x, z) {
        const group = new THREE.Group();
        group.position.set(x, 0, z);

        // Throne platform
        const platformGeo = new THREE.CylinderGeometry(15, 18, 3, 6);
        const platformMat = new THREE.MeshStandardMaterial({
            color: 0x3a2a2a,
            roughness: 0.7
        });
        const platform = new THREE.Mesh(platformGeo, platformMat);
        platform.position.y = 1.5;
        group.add(platform);

        // Columns
        for (let i = 0; i < 6; i++) {
            const column = this.createColumn();
            const angle = (i / 6) * Math.PI * 2;
            column.position.set(
                Math.cos(angle) * 12,
                0,
                Math.sin(angle) * 12
            );
            group.add(column);
        }

        // Central throne
        const throneGeo = new THREE.BoxGeometry(4, 8, 4);
        const throneMat = new THREE.MeshStandardMaterial({
            color: 0x4a3a1a,
            emissive: 0x2a1a0a
        });
        const throne = new THREE.Mesh(throneGeo, throneMat);
        throne.position.y = 7;
        group.add(throne);

        this.scene.add(group);
        return group;
    }

    createTemple(x, z) {
        const group = new THREE.Group();
        group.position.set(x, 0, z);

        // Infinity symbol structure
        const curve = new THREE.CatmullRomCurve3([
            new THREE.Vector3(-10, 5, 0),
            new THREE.Vector3(0, 10, -5),
            new THREE.Vector3(10, 5, 0),
            new THREE.Vector3(0, 10, 5),
            new THREE.Vector3(-10, 5, 0)
        ], true);

        const tubeGeo = new THREE.TubeGeometry(curve, 100, 1, 8, true);
        const tubeMat = new THREE.MeshStandardMaterial({
            color: 0x2a2a4a,
            emissive: 0x1a1a3a
        });
        const infinity = new THREE.Mesh(tubeGeo, tubeMat);
        infinity.scale.set(2, 2, 2);
        group.add(infinity);

        // Meditation platforms
        for (let i = 0; i < 4; i++) {
            const angle = (i / 4) * Math.PI * 2 + Math.PI / 4;
            const platform = this.createMeditationPlatform();
            platform.position.set(
                Math.cos(angle) * 20,
                0,
                Math.sin(angle) * 20
            );
            group.add(platform);
        }

        this.scene.add(group);
        return group;
    }

    createPortals() {
        const portalPositions = [
            { from: 'hollow', to: 'market', pos: new THREE.Vector3(40, 0, 0) },
            { from: 'hollow', to: 'ministry', pos: new THREE.Vector3(30, 0, -40) },
            { from: 'hollow', to: 'court', pos: new THREE.Vector3(-40, 0, 0) },
            { from: 'hollow', to: 'temple', pos: new THREE.Vector3(0, 0, 40) },
        ];

        portalPositions.forEach(p => {
            const portal = this.createPortal(p.to);
            portal.position.copy(p.pos);
            portal.userData = { from: p.from, to: p.to };
            this.portals.push(portal);
            this.scene.add(portal);
        });
    }

    createPortal(destination) {
        const group = new THREE.Group();

        // Portal ring
        const ringGeo = new THREE.TorusGeometry(4, 0.5, 16, 32);
        const ringMat = new THREE.MeshStandardMaterial({
            color: 0x00aaff,
            emissive: 0x0066aa
        });
        const ring = new THREE.Mesh(ringGeo, ringMat);
        ring.rotation.x = Math.PI / 2;
        ring.position.y = 4;
        group.add(ring);

        // Portal surface
        const surfaceGeo = new THREE.CircleGeometry(3.5, 32);
        const surfaceMat = new THREE.MeshBasicMaterial({
            color: 0x001133,
            transparent: true,
            opacity: 0.5
        });
        const surface = new THREE.Mesh(surfaceGeo, surfaceMat);
        surface.rotation.x = -Math.PI / 2;
        surface.position.y = 4;
        group.add(surface);

        // Label
        // (In production, use TextGeometry or sprite)

        return group;
    }

    createResearchPod() {
        const geo = new THREE.DodecahedronGeometry(3, 0);
        const mat = new THREE.MeshStandardMaterial({
            color: 0x1a2a4a,
            emissive: 0x0a1a2a,
            wireframe: false
        });
        return new THREE.Mesh(geo, mat);
    }

    createPublicationBoard() {
        const group = new THREE.Group();

        const boardGeo = new THREE.BoxGeometry(8, 5, 0.2);
        const boardMat = new THREE.MeshStandardMaterial({
            color: 0x2a2a2a
        });
        const board = new THREE.Mesh(boardGeo, boardMat);
        board.position.y = 2.5;
        group.add(board);

        // Posts
        const postGeo = new THREE.CylinderGeometry(0.1, 0.1, 5);
        const postMat = new THREE.MeshStandardMaterial({ color: 0x3a3a3a });

        const post1 = new THREE.Mesh(postGeo, postMat);
        post1.position.set(-3.5, 2.5, 0);
        group.add(post1);

        const post2 = new THREE.Mesh(postGeo, postMat);
        post2.position.set(3.5, 2.5, 0);
        group.add(post2);

        return group;
    }

    createMarketStall() {
        const group = new THREE.Group();

        // Counter
        const counterGeo = new THREE.BoxGeometry(4, 1, 2);
        const counterMat = new THREE.MeshStandardMaterial({ color: 0x3a2a1a });
        const counter = new THREE.Mesh(counterGeo, counterMat);
        counter.position.y = 0.5;
        group.add(counter);

        // Canopy
        const canopyGeo = new THREE.BoxGeometry(5, 0.1, 3);
        const canopyMat = new THREE.MeshStandardMaterial({ color: 0x1a3a1a });
        const canopy = new THREE.Mesh(canopyGeo, canopyMat);
        canopy.position.y = 3;
        group.add(canopy);

        return group;
    }

    createFloatingPaper() {
        const geo = new THREE.PlaneGeometry(0.5, 0.7);
        const mat = new THREE.MeshBasicMaterial({
            color: 0xeeeeee,
            side: THREE.DoubleSide
        });
        return new THREE.Mesh(geo, mat);
    }

    createColumn() {
        const group = new THREE.Group();

        const columnGeo = new THREE.CylinderGeometry(0.8, 1, 15, 8);
        const columnMat = new THREE.MeshStandardMaterial({ color: 0x4a4a4a });
        const column = new THREE.Mesh(columnGeo, columnMat);
        column.position.y = 7.5;
        group.add(column);

        return group;
    }

    createMeditationPlatform() {
        const geo = new THREE.CylinderGeometry(3, 3, 0.5, 16);
        const mat = new THREE.MeshStandardMaterial({
            color: 0x2a2a3a,
            emissive: 0x1a1a2a
        });
        const platform = new THREE.Mesh(geo, mat);
        platform.position.y = 0.25;
        return platform;
    }

    createAmbientEffects() {
        // Ambient particles
        const particleCount = 500;
        const particles = new THREE.BufferGeometry();
        const positions = new Float32Array(particleCount * 3);

        for (let i = 0; i < particleCount * 3; i += 3) {
            positions[i] = (Math.random() - 0.5) * 200;
            positions[i + 1] = Math.random() * 50;
            positions[i + 2] = (Math.random() - 0.5) * 200;
        }

        particles.setAttribute('position', new THREE.BufferAttribute(positions, 3));

        const particleMat = new THREE.PointsMaterial({
            color: 0x4466aa,
            size: 0.5,
            transparent: true,
            opacity: 0.6
        });

        const particleSystem = new THREE.Points(particles, particleMat);
        this.scene.add(particleSystem);
        this.particles = particleSystem;
    }

    /**
     * Update animations
     */
    update(time) {
        // Animate floating research pods
        if (this.realms.hollow) {
            this.realms.hollow.children.forEach(child => {
                if (child.userData.floatOffset !== undefined) {
                    child.position.y = 10 + Math.sin(time * 0.001 + child.userData.floatOffset) * 2;
                    child.rotation.y += 0.005;
                }
            });
        }

        // Animate floating papers in Ministry
        if (this.realms.ministry) {
            this.realms.ministry.children.forEach(child => {
                if (child.userData.floatSpeed) {
                    child.position.y += Math.sin(time * 0.001 * child.userData.floatSpeed) * 0.02;
                    child.rotation.z = Math.sin(time * 0.002) * 0.3;
                }
            });
        }

        // Animate particles
        if (this.particles) {
            this.particles.rotation.y += 0.0001;
        }

        // Pulse portal rings
        this.portals.forEach((portal, i) => {
            const ring = portal.children[0];
            if (ring) {
                ring.material.emissiveIntensity = 0.5 + Math.sin(time * 0.003 + i) * 0.3;
            }
        });
    }

    /**
     * Get portal at position
     */
    getPortalAt(position, threshold = 5) {
        for (const portal of this.portals) {
            const dist = position.distanceTo(portal.position);
            if (dist < threshold) {
                return portal.userData;
            }
        }
        return null;
    }

    /**
     * Get current realm based on position
     */
    getRealmAt(position) {
        const realmCenters = {
            hollow: new THREE.Vector3(0, 0, 0),
            market: new THREE.Vector3(80, 0, 0),
            ministry: new THREE.Vector3(60, 0, -80),
            court: new THREE.Vector3(-80, 0, 0),
            temple: new THREE.Vector3(0, 0, 80)
        };

        let closest = 'hollow';
        let closestDist = Infinity;

        for (const [name, center] of Object.entries(realmCenters)) {
            const dist = position.distanceTo(center);
            if (dist < closestDist) {
                closestDist = dist;
                closest = name;
            }
        }

        return closest;
    }
}

// Export
window.LiminalWorld = LiminalWorld;
