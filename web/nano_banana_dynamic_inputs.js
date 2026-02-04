/**
 * Dynamic image inputs for Nano Banana Nodes
 * Based on cozy_ex_dynamic pattern for clean dynamic inputs
 */

import { app } from "../../../scripts/app.js"
import { api } from "../../../scripts/api.js"

const TypeSlot = {
    Input: 1,
    Output: 2,
};

const TypeSlotEvent = {
    Connect: true,
    Disconnect: false,
};

const NODE_IDS = ["NanoBananaNode", "NanoBananaMultipleOutputsNode"];
const PREFIX = "image";
const TYPE = "IMAGE";
const MAX_IMAGES = 14; // Nano Banana Pro limit

/**
 * Applies dynamic image input behavior to a node
 */
function applyDynamicImageInputs(nodeType) {
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
        const me = onNodeCreated?.apply(this);
        
        // Start with a new dynamic input - exactly like cozy example
        this.addInput(PREFIX, TYPE);
        
        // Ensure the new slot has proper appearance
        const slot = this.inputs[this.inputs.length - 1];
        if (slot) {
            slot.color_off = "#666";
        }
        
        // Initialize aspect ratio display text and usage metadata
        this.aspectRatioText = null;
        this.usageMetadata = null;

        return me;
    }
    
    // Override onDrawForeground to display aspect ratio and usage metadata on the node
    const onDrawForeground = nodeType.prototype.onDrawForeground;
    nodeType.prototype.onDrawForeground = function(ctx) {
        const ret = onDrawForeground?.apply(this, arguments);

        ctx.save();
        ctx.font = "11px Arial";
        ctx.textAlign = "center";

        // Position below the title, centered horizontally
        let yOffset = 18;  // Start below the title bar
        const lineHeight = 14;
        const xPos = this.size[0] / 2;  // Center of node

        // Draw aspect ratio (green) - existing functionality
        if (this.aspectRatioText) {
            ctx.fillStyle = "#8f8";
            ctx.fillText(this.aspectRatioText, xPos, yOffset);
            yOffset += lineHeight;
        }

        // Draw usage metadata (top to bottom)
        if (this.usageMetadata) {
            const meta = this.usageMetadata;

            // Processing time (cyan) - with parallel indicator if applicable
            if (meta.processingTimeMs) {
                ctx.fillStyle = "#8ff";
                const timeText = meta.parallelRequests && meta.parallelRequests > 1
                    ? `${(meta.processingTimeMs / 1000).toFixed(1)}s (${meta.parallelRequests}x)`
                    : `${(meta.processingTimeMs / 1000).toFixed(1)}s`;
                ctx.fillText(timeText, xPos, yOffset);
                yOffset += lineHeight;
            }

            // Candidates/Output tokens (yellow)
            if (meta.candidatesTokenCount) {
                ctx.fillStyle = "#ff8";
                ctx.fillText(`Out: ${meta.candidatesTokenCount}`, xPos, yOffset);
                yOffset += lineHeight;
            }

            // Thoughts tokens (magenta, if present - indicates reasoning model)
            if (meta.thoughtsTokenCount && meta.thoughtsTokenCount > 0) {
                ctx.fillStyle = "#f8f";
                ctx.fillText(`Think: ${meta.thoughtsTokenCount}`, xPos, yOffset);
                yOffset += lineHeight;
            }

            // Total tokens (yellow)
            if (meta.totalTokenCount) {
                ctx.fillStyle = "#ff8";
                ctx.fillText(`Total: ${meta.totalTokenCount}`, xPos, yOffset);
                yOffset += lineHeight;
            }
        }

        ctx.restore();
        return ret;
    }

    const onConnectionsChange = nodeType.prototype.onConnectionsChange
    nodeType.prototype.onConnectionsChange = function (slotType, slot_idx, event, link_info, node_slot) {
        const me = onConnectionsChange?.apply(this, arguments);

        if (slotType === TypeSlot.Input) {
            // Only process image inputs
            if (node_slot && !node_slot.name.startsWith(PREFIX)) {
                return me;
            }
            
            if (link_info && event === TypeSlotEvent.Connect) {
                // Get the parent (left side node) from the link
                const fromNode = this.graph._nodes.find(
                    (otherNode) => otherNode.id == link_info.origin_id
                )

                if (fromNode) {
                    // Make sure there is a parent for the link
                    const parent_link = fromNode.outputs[link_info.origin_slot];
                    if (parent_link) {
                        node_slot.type = parent_link.type;
                        node_slot.name = `${PREFIX}_`;
                    }
                }
            } else if (event === TypeSlotEvent.Disconnect) {
                // Don't remove the slot immediately, let the cleanup below handle it
            }

            // Track each slot name so we can index the uniques
            let idx = 0;
            let slot_tracker = {};
            let toRemove = [];
            
            for(const slot of this.inputs) {
                // Skip non-image inputs
                if (!slot.name.startsWith(PREFIX)) {
                    idx += 1;
                    continue;
                }
                
                // Mark empty image slots for removal (except the last one)
                if (slot.link === null && idx < this.inputs.length - 1) {
                    toRemove.push(idx);
                } else if (slot.link !== null) {
                    // Connected slot - update its name with proper index
                    const name = slot.name.split('_')[0];
                    let count = (slot_tracker[name] || 0) + 1;
                    slot_tracker[name] = count;
                    slot.name = `${name}_${count}`;
                }
                idx += 1;
            }
            
            // Remove empty slots from highest index to lowest
            toRemove.reverse();
            for (const removeIdx of toRemove) {
                this.removeInput(removeIdx);
            }

            // Check if the last input is an image input
            let lastInput = null;
            for (let i = this.inputs.length - 1; i >= 0; i--) {
                if (this.inputs[i].name.startsWith(PREFIX)) {
                    lastInput = this.inputs[i];
                    break;
                }
            }
            
            // Count current connected image inputs
            const connectedImageCount = this.inputs.filter(
                slot => slot.name.startsWith(PREFIX) && slot.link !== null
            ).length;
            
            // If there's no empty image slot at the end, or no image slots at all, add one
            // But only if we haven't reached the maximum
            if (!lastInput || lastInput.link !== null) {
                if (connectedImageCount < MAX_IMAGES) {
                    this.addInput(PREFIX, TYPE);
                    // Set the unconnected slot to appear gray
                    const newSlot = this.inputs[this.inputs.length - 1];
                    if (newSlot) {
                        newSlot.color_off = "#666";
                    }
                }
            }

            // Force the node to resize itself for the new/deleted connections
            this?.graph?.setDirtyCanvas(true);
            return me;
        }
    }
    
    return nodeType;
}

app.registerExtension({
    name: 'NanoBanana.DynamicImageInputs',
    async setup() {
        // Listen for aspect ratio messages from the server
        api.addEventListener("nanobanana.aspect_ratio", (event) => {
            const { aspect_ratio, was_auto, node_type } = event.detail;
            
            // Find the currently executing node (or most recently executed Nano Banana node)
            const runningNodeId = app.runningNodeId;
            if (runningNodeId) {
                const node = app.graph.getNodeById(runningNodeId);
                if (node && NODE_IDS.includes(node.comfyClass)) {
                    // Update the aspect ratio text on the node
                    if (was_auto) {
                        node.aspectRatioText = `Auto → ${aspect_ratio}`;
                    } else {
                        node.aspectRatioText = aspect_ratio !== "None" ? aspect_ratio : "";
                    }
                    // Redraw the node
                    app.graph.setDirtyCanvas(true);
                }
            }
        });

        // Listen for usage metadata messages from the server
        api.addEventListener("nanobanana.usage_metadata", (event) => {
            const { node_id, metadata, model, node_type } = event.detail;

            // Find the currently executing node
            const runningNodeId = app.runningNodeId;
            if (runningNodeId) {
                const node = app.graph.getNodeById(runningNodeId);
                if (node && NODE_IDS.includes(node.comfyClass)) {
                    // Store metadata for display
                    node.usageMetadata = metadata;

                    // Log to browser console for debugging
                    console.log(`[NanoBanana] Node ${runningNodeId} metadata:`, metadata);

                    // Redraw the node
                    app.graph.setDirtyCanvas(true);
                }
            }
        });
    },
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // Skip if not one of our nodes
        if (!NODE_IDS.includes(nodeData.name)) {
            return
        }

        applyDynamicImageInputs(nodeType);
        
        return nodeType;
    },
})