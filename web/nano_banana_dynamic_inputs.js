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
const PERMANENT_IMAGES = 4; // image_1 through image_4 are permanent (defined in Python)

/**
 * Applies dynamic image input behavior to a node
 * image_1 through image_4 are permanent (defined in Python INPUT_TYPES)
 * image_5 onwards are dynamically added/removed
 */
function applyDynamicImageInputs(nodeType) {
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
        const me = onNodeCreated?.apply(this);
        
        // Don't add initial dynamic input - we already have image_1 to image_4 from Python
        // Just initialize the aspect ratio display text
        this.aspectRatioText = null;
        
        return me;
    }
    
    // Override onDrawForeground to display aspect ratio on the node
    const onDrawForeground = nodeType.prototype.onDrawForeground;
    nodeType.prototype.onDrawForeground = function(ctx) {
        const ret = onDrawForeground?.apply(this, arguments);
        
        // Draw aspect ratio text if available
        if (this.aspectRatioText) {
            ctx.save();
            ctx.font = "12px Arial";
            ctx.fillStyle = "#8f8";
            ctx.textAlign = "right";
            // Position at top-right of node
            ctx.fillText(this.aspectRatioText, this.size[0] - 10, -8);
            ctx.restore();
        }
        
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
            
            // Helper to check if a slot is a permanent image input (image_1 through image_4)
            const isPermanentSlot = (slotName) => {
                const match = slotName.match(/^image_(\d+)$/);
                if (match) {
                    const num = parseInt(match[1]);
                    return num >= 1 && num <= PERMANENT_IMAGES;
                }
                return false;
            };
            
            // Helper to check if a slot is dynamic (image_5+)
            const isDynamicSlot = (slotName) => {
                // Slots named just "image" or "image_" are dynamic (pending naming)
                if (slotName === PREFIX || slotName === `${PREFIX}_`) {
                    return true;
                }
                const match = slotName.match(/^image_(\d+)$/);
                if (match) {
                    const num = parseInt(match[1]);
                    return num > PERMANENT_IMAGES;
                }
                return false;
            };
            
            if (link_info && event === TypeSlotEvent.Connect) {
                // Get the parent (left side node) from the link
                const fromNode = this.graph._nodes.find(
                    (otherNode) => otherNode.id == link_info.origin_id
                )

                if (fromNode) {
                    // Make sure there is a parent for the link
                    const parent_link = fromNode.outputs[link_info.origin_slot];
                    if (parent_link && isDynamicSlot(node_slot.name)) {
                        node_slot.type = parent_link.type;
                        node_slot.name = `${PREFIX}_`;
                    }
                }
            }

            // Count connected images and track dynamic slots
            let dynamicSlotCount = 0;
            let toRemove = [];
            
            for (let idx = 0; idx < this.inputs.length; idx++) {
                const slot = this.inputs[idx];
                
                // Skip non-image inputs
                if (!slot.name.startsWith(PREFIX)) {
                    continue;
                }
                
                // Skip permanent slots (image_1 through image_4) - never remove these
                if (isPermanentSlot(slot.name)) {
                    continue;
                }
                
                // This is a dynamic slot
                if (slot.link === null) {
                    // Empty dynamic slot - mark for removal unless it's the last dynamic slot
                    toRemove.push(idx);
                } else {
                    // Connected dynamic slot - update its name
                    dynamicSlotCount++;
                    slot.name = `${PREFIX}_${PERMANENT_IMAGES + dynamicSlotCount}`;
                }
            }
            
            // Keep the last empty dynamic slot (if any), remove the rest
            if (toRemove.length > 0) {
                // Keep the last one in the list (highest index)
                toRemove.pop();
            }
            
            // Remove empty slots from highest index to lowest
            toRemove.reverse();
            for (const removeIdx of toRemove) {
                this.removeInput(removeIdx);
            }

            // Count total connected image inputs (permanent + dynamic)
            const connectedImageCount = this.inputs.filter(
                slot => slot.name.startsWith(PREFIX) && slot.link !== null
            ).length;
            
            // Check if we need to add a new dynamic slot
            // Find the last image input
            let lastImageInput = null;
            for (let i = this.inputs.length - 1; i >= 0; i--) {
                if (this.inputs[i].name.startsWith(PREFIX)) {
                    lastImageInput = this.inputs[i];
                    break;
                }
            }
            
            // Add a new dynamic slot if:
            // 1. All permanent slots are connected, AND
            // 2. The last image slot is connected (or doesn't exist), AND
            // 3. We haven't reached the maximum
            const allPermanentConnected = this.inputs
                .filter(s => isPermanentSlot(s.name))
                .every(s => s.link !== null);
            
            if (allPermanentConnected && 
                (!lastImageInput || lastImageInput.link !== null || isPermanentSlot(lastImageInput.name)) && 
                connectedImageCount < MAX_IMAGES) {
                this.addInput(`${PREFIX}_${PERMANENT_IMAGES + dynamicSlotCount + 1}`, TYPE);
                // Set the unconnected slot to appear gray
                const newSlot = this.inputs[this.inputs.length - 1];
                if (newSlot) {
                    newSlot.color_off = "#666";
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