/**
 * Network.js - WebSocket Connection to Simulation
 *
 * Handles real-time state synchronization with the simulation engine.
 */

class NetworkManager {
    constructor(url) {
        this.url = url || this.getDefaultUrl();
        this.ws = null;
        this.isConnected = false;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 10;
        this.reconnectDelay = 1000;

        // Callbacks
        this.onStateUpdate = null;
        this.onConnect = null;
        this.onDisconnect = null;
        this.onError = null;

        // Message queue for when disconnected
        this.messageQueue = [];
    }

    getDefaultUrl() {
        // Determine WebSocket URL based on current location
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const host = window.location.hostname || 'localhost';
        const port = 8765; // Bridge port

        return `${protocol}//${host}:${port}/ws`;
    }

    /**
     * Connect to simulation server
     */
    connect() {
        return new Promise((resolve, reject) => {
            try {
                console.log(`Connecting to ${this.url}...`);
                this.ws = new WebSocket(this.url);

                this.ws.onopen = () => {
                    console.log('Connected to simulation');
                    this.isConnected = true;
                    this.reconnectAttempts = 0;

                    // Flush message queue
                    while (this.messageQueue.length > 0) {
                        const msg = this.messageQueue.shift();
                        this.send(msg);
                    }

                    if (this.onConnect) {
                        this.onConnect();
                    }

                    resolve();
                };

                this.ws.onclose = (event) => {
                    console.log('Disconnected from simulation');
                    this.isConnected = false;

                    if (this.onDisconnect) {
                        this.onDisconnect();
                    }

                    // Attempt reconnect
                    this.scheduleReconnect();
                };

                this.ws.onerror = (error) => {
                    console.error('WebSocket error:', error);
                    this.isConnected = false;

                    if (this.onError) {
                        this.onError(error);
                    }

                    reject(error);
                };

                this.ws.onmessage = (event) => {
                    try {
                        const data = JSON.parse(event.data);
                        this.handleMessage(data);
                    } catch (e) {
                        console.error('Failed to parse message:', e);
                    }
                };

            } catch (error) {
                console.error('Failed to create WebSocket:', error);
                reject(error);
            }
        });
    }

    /**
     * Handle incoming message
     */
    handleMessage(data) {
        switch (data.type) {
            case 'state_update':
                if (this.onStateUpdate) {
                    this.onStateUpdate(data);
                }
                break;

            case 'agent_detail':
                if (this.onAgentDetail) {
                    this.onAgentDetail(data);
                }
                break;

            case 'event':
                if (this.onEvent) {
                    this.onEvent(data);
                }
                break;

            case 'error':
                console.error('Server error:', data.message);
                if (this.onError) {
                    this.onError(data);
                }
                break;

            default:
                console.log('Unknown message type:', data.type);
        }
    }

    /**
     * Send message to server
     */
    send(data) {
        if (this.isConnected && this.ws) {
            this.ws.send(JSON.stringify(data));
        } else {
            // Queue message for when connected
            this.messageQueue.push(data);
        }
    }

    /**
     * Send command to simulation
     */
    sendCommand(type, params = {}) {
        this.send({
            type: type,
            ...params,
            timestamp: Date.now()
        });
    }

    /**
     * Request agent details
     */
    requestAgentDetails(agentId) {
        this.sendCommand('query_agent', { agent_id: agentId });
    }

    /**
     * Send interaction
     */
    interact(targetId, interactionType) {
        this.sendCommand('interact', {
            target_id: targetId,
            interaction: interactionType
        });
    }

    /**
     * Pause simulation
     */
    pause() {
        this.sendCommand('pause');
    }

    /**
     * Resume simulation
     */
    resume() {
        this.sendCommand('resume');
    }

    /**
     * Schedule reconnection attempt
     */
    scheduleReconnect() {
        if (this.reconnectAttempts >= this.maxReconnectAttempts) {
            console.error('Max reconnection attempts reached');
            return;
        }

        this.reconnectAttempts++;
        const delay = this.reconnectDelay * Math.pow(1.5, this.reconnectAttempts - 1);

        console.log(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);

        setTimeout(() => {
            this.connect().catch(() => {
                // Will trigger another reconnect via onclose
            });
        }, delay);
    }

    /**
     * Disconnect
     */
    disconnect() {
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
        this.isConnected = false;
    }
}

// Export
window.NetworkManager = NetworkManager;
