import gradio as gr
import torch
import numpy as np
import os
from PIL import Image, ImageDraw
from segment_anything import SamPredictor, sam_model_registry

# --- 1. Setup Imports ---
import sys
sys.path.append("notebook")
from inference import Inference, load_image

# --- 2. Configuration ---
SAM_CHECKPOINT = "checkpoints/sam_vit_h_4b8939.pth"
SAM3D_CONFIG = "checkpoints/hf/pipeline.yaml"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🚀 Starting SAM 3D Objects Application (Splat Mode)")
print(f"📍 Device: {device}")

print(f"[1/2] Loading SAM (Segmentation) from {SAM_CHECKPOINT}...")
sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT)
sam.to(device=device)
predictor = SamPredictor(sam)
print("✅ SAM loaded successfully!")

print(f"[2/2] Loading SAM 3D (Reconstruction) from {SAM3D_CONFIG}...")
inference_3d = Inference(SAM3D_CONFIG, compile=False)
print("✅ SAM 3D loaded successfully!")

# --- 3. Processing Functions ---

def add_point_and_segment(image_pil, evt: gr.SelectData, points_state, labels_state):
    """Handle point clicks and generate mask preview"""
    if image_pil is None:
        return None, None, points_state, labels_state

    if points_state is None: 
        points_state = []
    if labels_state is None: 
        labels_state = []

    # Add clicked point
    points_state.append([evt.index[0], evt.index[1]])
    labels_state.append(1)
    
    # Run SAM segmentation
    image_np = np.array(image_pil)
    predictor.set_image(image_np)
    
    masks, _, _ = predictor.predict(
        point_coords=np.array(points_state),
        point_labels=np.array(labels_state),
        multimask_output=False
    )
    best_mask = masks[0]

    # Draw points on image
    image_with_points = image_pil.copy()
    draw = ImageDraw.Draw(image_with_points)
    for pt in points_state:
        r = 5
        draw.ellipse((pt[0]-r, pt[1]-r, pt[0]+r, pt[1]+r), outline="cyan", width=3)

    # Create mask visualization
    mask_visual = image_np.copy()
    mask_visual[best_mask] = image_np[best_mask] * 0.5 + np.array([255, 0, 0]) * 0.5
    mask_visual_pil = Image.fromarray(mask_visual.astype(np.uint8))

    return image_with_points, mask_visual_pil, points_state, labels_state

def reset_selection(image_pil):
    """Reset all points and masks"""
    return image_pil, None, [], []

def generate_splat(image_pil, points_state, labels_state):
    """Generate Gaussian Splat (.ply) only"""
    if not points_state:
        yield None, "❌ Error: Please click on the object first."
        return

    yield None, "🔄 Step 1/3: Initializing..."

    try:
        # Step 1: Regenerate mask
        image_np = np.array(image_pil)
        predictor.set_image(image_np)
        masks, _, _ = predictor.predict(
            point_coords=np.array(points_state),
            point_labels=np.array(labels_state),
            multimask_output=False
        )
        best_mask = masks[0]

        # Step 2: Prepare input
        temp_img_path = "temp_input.png"
        image_pil.save(temp_img_path)
        sam3d_image = load_image(temp_img_path)

        yield None, "🔄 Step 2/3: Running 3D inference (~60 seconds)..."
        
        # --- CORRECTED CALL ---
        # We only pass the arguments that __call__ actually accepts.
        # The inference.py file handles the rest (vertex colors, etc.) internally.
        output = inference_3d(
            sam3d_image, 
            best_mask, 
            seed=42
        )

        yield None, "🔄 Step 3/3: Saving Gaussian Splat..."
        
        output_ply = "output_splat.ply"
        
        # --- EXPORT LOGIC ---
        if "gs" in output and output["gs"] is not None:
            # save_ply is a method on the Gaussian Splat object
            output["gs"].save_ply(output_ply)
            print(f"✅ Splat exported successfully to {output_ply}")
            
            yield output_ply, "✅ Done! Download your Gaussian Splat (.ply)."
        else:
            yield None, "❌ Error: No Gaussian Splat data generated."

    except Exception as e:
        print(f"❌ Error during generation: {e}")
        import traceback
        traceback.print_exc()
        yield None, f"❌ Error: {str(e)}"

# --- 4. Build UI ---
with gr.Blocks(
    title="SAM 3D Objects - Splat Mode",
    theme=gr.themes.Soft()
) as demo:
    gr.Markdown(
        """
        # ☁️ SAM 3D Objects: Gaussian Splat Generator
        
        **Instructions:**
        1. Upload an image
        2. Click on the object you want to reconstruct (multiple points = better mask)
        3. Check the mask preview (should be highlighted in red)
        4. Click "Generate Splat"
        5. View and Download the **.ply** file
        """
    )
    
    points_state = gr.State([])
    labels_state = gr.State([])

    with gr.Row():
        with gr.Column():
            input_img = gr.Image(
                type="pil", 
                label="📷 Input Image", 
                interactive=True,
                height=400
            )
            with gr.Row():
                reset_btn = gr.Button("🔄 Reset Selection", variant="secondary", scale=1)
                gen_btn = gr.Button("🚀 Generate Splat", variant="primary", scale=2)
            status_text = gr.Textbox(
                label="📊 Status", 
                value="Waiting for you to click on an object...",
                lines=2
            )
            
        with gr.Column():
            mask_preview = gr.Image(
                label="🎭 Mask Preview (Red = Selected)", 
                height=300, 
                interactive=False
            )
            # gr.Model3D natively supports displaying .ply files
            output_3d = gr.Model3D(
                label="☁️ 3D Gaussian Splat (.ply)", 
                clear_color=[0.0, 0.0, 0.0, 1.0], # Black background is usually better for splats
                height=400,
                interactive=True
            )

    # Event handlers
    input_img.select(
        fn=add_point_and_segment,
        inputs=[input_img, points_state, labels_state],
        outputs=[input_img, mask_preview, points_state, labels_state]
    )

    reset_btn.click(
        fn=reset_selection,
        inputs=[input_img],
        outputs=[input_img, mask_preview, points_state, labels_state]
    )

    gen_btn.click(
        fn=generate_splat,
        inputs=[input_img, points_state, labels_state],
        outputs=[output_3d, status_text]
    )

if __name__ == "__main__":
    print("\n" + "="*60)
    print("☁️ SAM 3D Objects - Gaussian Splat Mode")
    print("="*60)
    print("✅ All models loaded successfully!")
    print("🌐 Starting Gradio interface...")
    print("="*60 + "\n")
    
    demo.launch(
        server_name="0.0.0.0", 
        server_port=7860,
        share=False
    )