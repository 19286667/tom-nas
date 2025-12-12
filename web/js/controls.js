/**
 * Controls.js - First-Person Walking Simulator Controls
 *
 * WASD movement, mouse look, interactions.
 * Designed for immersive exploration.
 */

class PlayerControls {
    constructor(camera, domElement) {
        this.camera = camera;
        this.domElement = domElement;

        // Movement state
        this.moveForward = false;
        this.moveBackward = false;
        this.moveLeft = false;
        this.moveRight = false;
        this.canJump = false;

        // Movement parameters
        this.moveSpeed = 15.0;
        this.lookSpeed = 0.002;
        this.jumpForce = 10.0;
        this.gravity = 30.0;

        // Player state
        this.velocity = new THREE.Vector3();
        this.direction = new THREE.Vector3();
        this.position = new THREE.Vector3(0, 2, 10);

        // Camera rotation
        this.euler = new THREE.Euler(0, 0, 0, 'YXZ');
        this.minPolarAngle = 0.1;
        this.maxPolarAngle = Math.PI - 0.1;

        // Pointer lock state
        this.isLocked = false;

        // Input state
        this.keys = {};

        // Bind methods
        this.onMouseMove = this.onMouseMove.bind(this);
        this.onKeyDown = this.onKeyDown.bind(this);
        this.onKeyUp = this.onKeyUp.bind(this);
        this.onPointerLockChange = this.onPointerLockChange.bind(this);
        this.onPointerLockError = this.onPointerLockError.bind(this);

        this.init();
    }

    init() {
        // Set initial camera position
        this.camera.position.copy(this.position);

        // Event listeners
        document.addEventListener('mousemove', this.onMouseMove);
        document.addEventListener('keydown', this.onKeyDown);
        document.addEventListener('keyup', this.onKeyUp);

        // Pointer lock
        document.addEventListener('pointerlockchange', this.onPointerLockChange);
        document.addEventListener('pointerlockerror', this.onPointerLockError);

        // Click to lock
        this.domElement.addEventListener('click', () => {
            if (!this.isLocked) {
                this.lock();
            }
        });
    }

    lock() {
        this.domElement.requestPointerLock();
    }

    unlock() {
        document.exitPointerLock();
    }

    onPointerLockChange() {
        this.isLocked = document.pointerLockElement === this.domElement;

        if (this.isLocked) {
            document.getElementById('main-menu')?.classList.add('hidden');
        }
    }

    onPointerLockError() {
        console.error('Pointer lock failed');
    }

    onMouseMove(event) {
        if (!this.isLocked) return;

        const movementX = event.movementX || 0;
        const movementY = event.movementY || 0;

        this.euler.setFromQuaternion(this.camera.quaternion);

        this.euler.y -= movementX * this.lookSpeed;
        this.euler.x -= movementY * this.lookSpeed;

        // Clamp vertical rotation
        this.euler.x = Math.max(
            Math.PI / 2 - this.maxPolarAngle,
            Math.min(Math.PI / 2 - this.minPolarAngle, this.euler.x)
        );

        this.camera.quaternion.setFromEuler(this.euler);
    }

    onKeyDown(event) {
        this.keys[event.code] = true;

        switch (event.code) {
            case 'KeyW':
            case 'ArrowUp':
                this.moveForward = true;
                break;

            case 'KeyS':
            case 'ArrowDown':
                this.moveBackward = true;
                break;

            case 'KeyA':
            case 'ArrowLeft':
                this.moveLeft = true;
                break;

            case 'KeyD':
            case 'ArrowRight':
                this.moveRight = true;
                break;

            case 'Space':
                if (this.canJump) {
                    this.velocity.y = this.jumpForce;
                    this.canJump = false;
                }
                break;

            case 'KeyE':
                // Interaction - handled by main.js
                if (window.game) {
                    window.game.interact();
                }
                break;

            case 'Tab':
                // Toggle map
                event.preventDefault();
                if (window.game) {
                    window.game.toggleMap();
                }
                break;

            case 'KeyI':
                // Inventory
                if (window.game) {
                    window.game.toggleInventory();
                }
                break;

            case 'KeyP':
                // Pause
                if (window.game) {
                    window.game.togglePause();
                }
                break;

            case 'Backquote':
                // Console
                if (window.game) {
                    window.game.toggleConsole();
                }
                break;

            case 'Escape':
                this.unlock();
                document.getElementById('main-menu')?.classList.remove('hidden');
                break;
        }
    }

    onKeyUp(event) {
        this.keys[event.code] = false;

        switch (event.code) {
            case 'KeyW':
            case 'ArrowUp':
                this.moveForward = false;
                break;

            case 'KeyS':
            case 'ArrowDown':
                this.moveBackward = false;
                break;

            case 'KeyA':
            case 'ArrowLeft':
                this.moveLeft = false;
                break;

            case 'KeyD':
            case 'ArrowRight':
                this.moveRight = false;
                break;
        }
    }

    update(deltaTime) {
        if (!this.isLocked) return;

        // Apply gravity
        this.velocity.y -= this.gravity * deltaTime;

        // Calculate movement direction
        this.direction.z = Number(this.moveForward) - Number(this.moveBackward);
        this.direction.x = Number(this.moveRight) - Number(this.moveLeft);
        this.direction.normalize();

        // Get camera forward and right vectors
        const forward = new THREE.Vector3();
        const right = new THREE.Vector3();

        this.camera.getWorldDirection(forward);
        forward.y = 0;
        forward.normalize();

        right.crossVectors(forward, new THREE.Vector3(0, 1, 0));

        // Calculate velocity
        if (this.moveForward || this.moveBackward) {
            this.velocity.z = this.direction.z * this.moveSpeed;
        } else {
            this.velocity.z = 0;
        }

        if (this.moveLeft || this.moveRight) {
            this.velocity.x = this.direction.x * this.moveSpeed;
        } else {
            this.velocity.x = 0;
        }

        // Apply movement
        const movement = new THREE.Vector3();
        movement.addScaledVector(forward, this.velocity.z * deltaTime);
        movement.addScaledVector(right, this.velocity.x * deltaTime);
        movement.y = this.velocity.y * deltaTime;

        this.position.add(movement);

        // Ground collision (simple plane at y=2)
        if (this.position.y < 2) {
            this.position.y = 2;
            this.velocity.y = 0;
            this.canJump = true;
        }

        // World bounds
        const bounds = 240;
        this.position.x = Math.max(-bounds, Math.min(bounds, this.position.x));
        this.position.z = Math.max(-bounds, Math.min(bounds, this.position.z));

        // Update camera
        this.camera.position.copy(this.position);
    }

    getPosition() {
        return this.position.clone();
    }

    setPosition(x, y, z) {
        this.position.set(x, y, z);
        this.camera.position.copy(this.position);
    }

    getLookDirection() {
        const direction = new THREE.Vector3();
        this.camera.getWorldDirection(direction);
        return direction;
    }

    dispose() {
        document.removeEventListener('mousemove', this.onMouseMove);
        document.removeEventListener('keydown', this.onKeyDown);
        document.removeEventListener('keyup', this.onKeyUp);
        document.removeEventListener('pointerlockchange', this.onPointerLockChange);
        document.removeEventListener('pointerlockerror', this.onPointerLockError);
    }
}

// Export
window.PlayerControls = PlayerControls;
