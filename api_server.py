#!/usr/bin/env python3
"""
HAIR RESTORATION API SERVER
============================

FastAPI backend for the hair restoration pipeline.

Run with:
    uvicorn api_server:app --host 0.0.0.0 --port 8000

Endpoints:
    POST /jobs                  - Create new job (upload video)
    GET  /jobs/{job_id}         - Get job status
    GET  /jobs/{job_id}/frame   - Get frame for mask drawing
    POST /jobs/{job_id}/mask    - Submit mask and start processing
    GET  /jobs/{job_id}/result  - Download result video
"""

import cv2
import numpy as np
import os
import sys
import time
import uuid
import shutil
import requests
import subprocess
import threading
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import base64

# ============================================================================
# CONFIGURATION
# ============================================================================
COMFYUI_URL = os.getenv("COMFYUI_URL", "https://3uu7kqlsr37ov6-8188.proxy.runpod.net")
JOBS_DIR = Path("jobs")
JOBS_DIR.mkdir(exist_ok=True)

WORK_WIDTH = 720
WORK_HEIGHT = 1280

# Job status
STATUS_PENDING = "pending"           # Video uploaded, waiting for mask
STATUS_PROCESSING = "processing"     # Pipeline running
STATUS_COMPLETE = "complete"         # Done
STATUS_FAILED = "failed"             # Error occurred

# In-memory job tracking (use Redis/DB in production)
jobs = {}

# ============================================================================
# FASTAPI APP
# ============================================================================
app = FastAPI(
    title="Hair Restoration API",
    description="AI-powered hair restoration for videos",
    version="1.0.0"
)

# CORS for mobile/web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# MODELS
# ============================================================================
class JobStatus(BaseModel):
    job_id: str
    status: str
    message: Optional[str] = None
    progress: Optional[int] = None
    frame_count: Optional[int] = None
    result_url: Optional[str] = None
    multi_frame: Optional[bool] = False
    keyframe_indices: Optional[list] = None
    current_keyframe: Optional[int] = None


class MaskSubmission(BaseModel):
    mask_data: str  # Base64 encoded PNG
    keyframe_index: Optional[int] = None  # For multi-frame mode


# ============================================================================
# COMFYUI API (from original pipeline)
# ============================================================================
def check_comfyui():
    try:
        r = requests.get(f"{COMFYUI_URL}/system_stats", timeout=10)
        return r.status_code == 200
    except:
        return False


def upload_image_to_comfy(image_path, filename):
    with open(image_path, 'rb') as f:
        files = {'image': (filename, f, 'image/png')}
        r = requests.post(f"{COMFYUI_URL}/upload/image", files=files, data={'overwrite': 'true'})
        if r.status_code != 200:
            raise RuntimeError(f"Upload failed: {r.text}")
        return r.json()


def queue_prompt(prompt):
    r = requests.post(f"{COMFYUI_URL}/prompt", json={"prompt": prompt})
    return r.json()


def get_history(prompt_id):
    r = requests.get(f"{COMFYUI_URL}/history/{prompt_id}")
    return r.json()


def wait_for_result(prompt_id, timeout=120):
    start = time.time()
    while time.time() - start < timeout:
        history = get_history(prompt_id)
        if prompt_id in history:
            outputs = history[prompt_id].get("outputs", {})
            for node_id, node_output in outputs.items():
                if "images" in node_output:
                    img = node_output["images"][0]
                    return f"{COMFYUI_URL}/view?filename={img['filename']}&type=output"
        time.sleep(1)
    raise RuntimeError("Timeout waiting for ComfyUI")


def generate_hair(frame_path, mask_path, prefix):
    input_name = f"{prefix}_input.png"
    mask_name = f"{prefix}_mask.png"

    upload_image_to_comfy(frame_path, input_name)
    upload_image_to_comfy(mask_path, mask_name)

    workflow = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "sd-v1-5-inpainting.ckpt"}},
        "2": {"class_type": "LoadImage", "inputs": {"image": input_name}},
        "3": {"class_type": "LoadImage", "inputs": {"image": mask_name}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {
            "text": "natural hair, realistic hair texture, detailed hair strands, photorealistic, matching surrounding hair color and style",
            "clip": ["1", 1]
        }},
        "5": {"class_type": "CLIPTextEncode", "inputs": {
            "text": "bald, scalp, skin, blurry, artificial, wrong color",
            "clip": ["1", 1]
        }},
        "6": {"class_type": "ImageToMask", "inputs": {"image": ["3", 0], "channel": "red"}},
        "7": {"class_type": "VAEEncodeForInpaint", "inputs": {
            "pixels": ["2", 0], "vae": ["1", 2], "mask": ["6", 0], "grow_mask_by": 6
        }},
        "8": {"class_type": "KSampler", "inputs": {
            "model": ["1", 0], "positive": ["4", 0], "negative": ["5", 0],
            "latent_image": ["7", 0], "seed": np.random.randint(0, 2**31),
            "steps": 30, "cfg": 7.0, "sampler_name": "euler_ancestral",
            "scheduler": "normal", "denoise": 0.85
        }},
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["1", 2]}},
        "10": {"class_type": "SaveImage", "inputs": {"images": ["9", 0], "filename_prefix": prefix}}
    }

    result = queue_prompt(workflow)
    prompt_id = result.get("prompt_id")
    if not prompt_id:
        raise RuntimeError(f"Queue failed: {result}")

    return wait_for_result(prompt_id)


# ============================================================================
# PIPELINE (from original)
# ============================================================================
def extract_red_mask(marked_frame_path):
    """Extract red markings from frame and create mask."""
    img = cv2.imread(str(marked_frame_path))
    if img is None:
        raise FileNotFoundError(f"Cannot load: {marked_frame_path}")

    h, w = img.shape[:2]
    b, g, r = cv2.split(img)

    red_mask = ((r > 150) & (g < 100) & (b < 100)).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    red_mask = cv2.dilate(red_mask, kernel, iterations=2)

    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise ValueError("No red markings found")

    best_contour = None
    best_score = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        score = circularity * np.sqrt(area)

        if score > best_score:
            best_score = score
            best_contour = cnt

    if best_contour is None:
        best_contour = max(contours, key=cv2.contourArea)

    clean_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(clean_mask, [best_contour], -1, 255, -1)
    clean_mask = cv2.dilate(clean_mask, kernel, iterations=3)

    return clean_mask


def run_pipeline(job_id, job_dir, video_path, mask_img):
    """Run the full hair restoration pipeline."""
    try:
        jobs[job_id]["status"] = STATUS_PROCESSING
        jobs[job_id]["message"] = "Loading video..."

        # Load video
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames, grays = [], []
        while True:
            ret, f = cap.read()
            if not ret:
                break
            f = cv2.resize(f, (WORK_WIDTH, WORK_HEIGHT))
            frames.append(f)
            grays.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
        cap.release()

        n = len(frames)
        ref_idx = n // 2
        jobs[job_id]["frame_count"] = n
        jobs[job_id]["message"] = f"Loaded {n} frames"

        # Resize mask
        mask = cv2.resize(mask_img, (WORK_WIDTH, WORK_HEIGHT))
        mask = (mask > 127).astype(np.uint8) * 255

        # Save reference frame and mask
        ref_path = job_dir / "ref_frame.png"
        mask_path = job_dir / "mask.png"
        cv2.imwrite(str(ref_path), frames[ref_idx])
        cv2.imwrite(str(mask_path), mask)

        # Generate hair
        jobs[job_id]["message"] = "Generating hair with AI..."
        jobs[job_id]["progress"] = 10

        prefix = f"hair_{job_id}"
        hair_url = generate_hair(str(ref_path), str(mask_path), prefix)

        hair_resp = requests.get(hair_url)
        hair_img = cv2.imdecode(np.frombuffer(hair_resp.content, np.uint8), cv2.IMREAD_COLOR)
        hair_img = cv2.resize(hair_img, (WORK_WIDTH, WORK_HEIGHT))

        # Sharpening disabled (testing all 4 options combined)
        # blurred = cv2.GaussianBlur(hair_img, (0, 0), 3)
        # hair_img = np.clip(cv2.addWeighted(hair_img, 1.5, blurred, -0.5, 0), 0, 255).astype(np.uint8)

        jobs[job_id]["message"] = "Processing video frames..."
        jobs[job_id]["progress"] = 30

        # DIS optical flow
        def compute_dis_flow(gray_src, gray_dst):
            dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
            dis.setFinestScale(0)
            dis.setVariationalRefinementIterations(5)
            return dis.calc(gray_dst, gray_src, None)

        def warp_with_flow(img, flow):
            h, w = img.shape[:2]
            map_x = np.arange(w, dtype=np.float32)[None, :] + flow[:, :, 0]
            map_y = np.arange(h, dtype=np.float32)[:, None] + flow[:, :, 1]
            if img.ndim == 3:
                return cv2.remap(img, map_x, map_y, cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)
            return cv2.remap(img, map_x, map_y, cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        # Create feather mask (minimal 7px for crisp edges)
        # Start with slightly eroded mask to prevent extension from flow drift
        erode_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask_tight = cv2.erode(mask, erode_k, iterations=2)
        feather_base = mask_tight.astype(np.float32)
        feather_base = cv2.GaussianBlur(feather_base, (15, 15), 3.5)

        # Propagate hair using CUMULATIVE flow (warp from original to avoid blur accumulation)
        hair_warped = [None] * n
        alpha_warped = [None] * n
        hair_warped[ref_idx] = hair_img.copy()
        alpha_warped[ref_idx] = feather_base.copy()

        # Forward - accumulate flow and warp from original
        cumulative_flow = None
        for i in range(ref_idx, n - 1):
            flow = compute_dis_flow(grays[i], grays[i + 1])
            if cumulative_flow is None:
                cumulative_flow = flow.copy()
            else:
                # Warp previous cumulative flow and add new flow
                warped_cum = warp_with_flow(cumulative_flow, flow)
                cumulative_flow = warped_cum + flow
            hair_warped[i + 1] = warp_with_flow(hair_img, cumulative_flow)
            alpha_warped[i + 1] = warp_with_flow(feather_base, cumulative_flow)
            jobs[job_id]["progress"] = 30 + int(20 * (i - ref_idx) / max(1, n - ref_idx - 1))

        # Backward - accumulate flow and warp from original
        cumulative_flow = None
        for i in range(ref_idx, 0, -1):
            flow = compute_dis_flow(grays[i], grays[i - 1])
            if cumulative_flow is None:
                cumulative_flow = flow.copy()
            else:
                warped_cum = warp_with_flow(cumulative_flow, flow)
                cumulative_flow = warped_cum + flow
            hair_warped[i - 1] = warp_with_flow(hair_img, cumulative_flow)
            alpha_warped[i - 1] = warp_with_flow(feather_base, cumulative_flow)
            jobs[job_id]["progress"] = 50 + int(20 * (ref_idx - i) / max(1, ref_idx))

        jobs[job_id]["message"] = "Color matching..."
        jobs[job_id]["progress"] = 70

        # Brightness matching
        mask_region = mask > 127
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
        dilated = cv2.dilate(mask, kernel_dilate)
        surround_region = (dilated > 127) & (mask < 127)

        hair_brightness = np.mean(hair_img[mask_region])
        surround_brightness = np.mean(frames[ref_idx][surround_region])

        darken_factor = (surround_brightness / hair_brightness) * 0.72 if hair_brightness > 0 else 1.0
        darken_factor = min(darken_factor, 1.0)

        for i in range(n):
            hair_warped[i] = (hair_warped[i].astype(np.float32) * darken_factor).clip(0, 255).astype(np.uint8)

        jobs[job_id]["message"] = "Compositing..."
        jobs[job_id]["progress"] = 80

        # Composite (tighter alpha: raised LO from 16->40 for crisper edges)
        ALPHA_LO, ALPHA_HI = 40, 240
        output_frames = []

        for i in range(n):
            wc = hair_warped[i]
            wa = alpha_warped[i].copy()

            wa = cv2.medianBlur(wa.astype(np.uint8), 3).astype(np.float32)

            wa = np.clip(wa, 0, ALPHA_HI)
            wa = np.where(wa < ALPHA_LO, 0, wa)

            af = wa / 255.0
            m3 = np.stack([af] * 3, axis=-1)
            result = np.clip(wc.astype(np.float32) * m3 + frames[i].astype(np.float32) * (1 - m3), 0, 255).astype(np.uint8)

            # Gentle sharpen in mask region (reduced from 1.5 to 1.15)
            blurred = cv2.GaussianBlur(result, (0, 0), 2)
            sharpened = cv2.addWeighted(result, 1.15, blurred, -0.15, 0)
            # Apply sharpening only where mask is active
            sharp_mask = (af > 0.1)[:, :, None].astype(np.float32)
            result = np.clip(sharpened * sharp_mask + result * (1 - sharp_mask), 0, 255).astype(np.uint8)

            output_frames.append(result)

        jobs[job_id]["message"] = "Encoding video..."
        jobs[job_id]["progress"] = 90

        # Save video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        tmp = job_dir / "tmp.mp4"
        vw = cv2.VideoWriter(str(tmp), fourcc, fps, (WORK_WIDTH, WORK_HEIGHT))
        for f in output_frames:
            vw.write(f)
        vw.release()

        result_path = job_dir / "result.mp4"
        subprocess.run(["ffmpeg", "-y", "-i", str(tmp), "-c:v", "libx264", "-crf", "18",
                        "-preset", "fast", "-pix_fmt", "yuv420p", str(result_path)], capture_output=True)
        tmp.unlink()

        jobs[job_id]["status"] = STATUS_COMPLETE
        jobs[job_id]["message"] = "Complete!"
        jobs[job_id]["progress"] = 100
        jobs[job_id]["result_url"] = f"/jobs/{job_id}/result"

    except Exception as e:
        jobs[job_id]["status"] = STATUS_FAILED
        jobs[job_id]["message"] = str(e)


def run_multi_pipeline(job_id, job_dir, video_path, masks_collected, keyframe_indices):
    """Run the multi-keyframe hair restoration pipeline."""
    try:
        jobs[job_id]["status"] = STATUS_PROCESSING
        jobs[job_id]["message"] = "Loading video..."

        # Load video
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames, grays = [], []
        while True:
            ret, f = cap.read()
            if not ret:
                break
            f = cv2.resize(f, (WORK_WIDTH, WORK_HEIGHT))
            frames.append(f)
            grays.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
        cap.release()

        n = len(frames)
        jobs[job_id]["frame_count"] = n
        jobs[job_id]["message"] = f"Loaded {n} frames, generating hair for {len(keyframe_indices)} keyframes..."

        # Generate hair for each keyframe
        keyframe_data = {}
        for i, idx in enumerate(keyframe_indices):
            jobs[job_id]["message"] = f"Generating hair for keyframe {i+1}/{len(keyframe_indices)}..."
            jobs[job_id]["progress"] = int(5 + 20 * i / len(keyframe_indices))

            mask = masks_collected[idx]
            # Apply same erosion as single-frame mode
            erode_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            mask_tight = cv2.erode(mask, erode_k, iterations=2)

            ref_path = job_dir / f"ref_frame_{idx}.png"
            mask_path = job_dir / f"mask_{idx}.png"
            cv2.imwrite(str(ref_path), frames[idx])
            cv2.imwrite(str(mask_path), mask)

            prefix = f"hair_{job_id}_{idx}"
            hair_url = generate_hair(str(ref_path), str(mask_path), prefix)

            hair_resp = requests.get(hair_url)
            hair_img = cv2.imdecode(np.frombuffer(hair_resp.content, np.uint8), cv2.IMREAD_COLOR)
            hair_img = cv2.resize(hair_img, (WORK_WIDTH, WORK_HEIGHT))

            # Feather mask
            feather = mask_tight.astype(np.float32)
            feather = cv2.GaussianBlur(feather, (15, 15), 3.5)

            keyframe_data[idx] = {'mask': mask_tight, 'hair': hair_img, 'feather': feather}

        jobs[job_id]["message"] = "Interpolating between keyframes..."
        jobs[job_id]["progress"] = 30

        # DIS optical flow (higher quality for multi-frame)
        def compute_dis_flow(gray_src, gray_dst):
            dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
            dis.setFinestScale(0)
            dis.setVariationalRefinementIterations(10)  # Increased for better accuracy
            dis.setPatchSize(8)  # Smaller patches for finer detail
            dis.setPatchStride(3)  # Denser sampling
            return dis.calc(gray_dst, gray_src, None)

        def warp_with_flow(img, flow):
            h, w = img.shape[:2]
            map_x = np.arange(w, dtype=np.float32)[None, :] + flow[:, :, 0]
            map_y = np.arange(h, dtype=np.float32)[:, None] + flow[:, :, 1]
            if img.ndim == 3:
                return cv2.remap(img, map_x, map_y, cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)
            return cv2.remap(img, map_x, map_y, cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        # Initialize arrays
        hair_all = [None] * n
        alpha_all = [None] * n

        for idx in keyframe_indices:
            hair_all[idx] = keyframe_data[idx]['hair'].copy()
            alpha_all[idx] = keyframe_data[idx]['feather'].copy()

        # Bidirectional propagation between keyframes
        for seg_idx in range(len(keyframe_indices) - 1):
            start_idx = keyframe_indices[seg_idx]
            end_idx = keyframe_indices[seg_idx + 1]
            seg_len = end_idx - start_idx

            jobs[job_id]["message"] = f"Processing segment {seg_idx+1}/{len(keyframe_indices)-1}..."
            jobs[job_id]["progress"] = 30 + int(40 * seg_idx / (len(keyframe_indices) - 1))

            # Forward propagation using cumulative flow
            fwd_hair = [hair_all[start_idx].copy()]
            fwd_alpha = [alpha_all[start_idx].copy()]
            cumulative_flow = None
            for i in range(seg_len):
                flow = compute_dis_flow(grays[start_idx + i], grays[start_idx + i + 1])
                if cumulative_flow is None:
                    cumulative_flow = flow.copy()
                else:
                    warped_cum = warp_with_flow(cumulative_flow, flow)
                    cumulative_flow = warped_cum + flow
                fwd_hair.append(warp_with_flow(keyframe_data[start_idx]['hair'], cumulative_flow))
                fwd_alpha.append(warp_with_flow(keyframe_data[start_idx]['feather'], cumulative_flow))

            # Backward propagation using cumulative flow
            bwd_hair = [None] * (seg_len + 1)
            bwd_alpha = [None] * (seg_len + 1)
            bwd_hair[seg_len] = hair_all[end_idx].copy()
            bwd_alpha[seg_len] = alpha_all[end_idx].copy()
            cumulative_flow = None
            for i in range(seg_len, 0, -1):
                flow = compute_dis_flow(grays[start_idx + i], grays[start_idx + i - 1])
                if cumulative_flow is None:
                    cumulative_flow = flow.copy()
                else:
                    warped_cum = warp_with_flow(cumulative_flow, flow)
                    cumulative_flow = warped_cum + flow
                bwd_hair[i - 1] = warp_with_flow(keyframe_data[end_idx]['hair'], cumulative_flow)
                bwd_alpha[i - 1] = warp_with_flow(keyframe_data[end_idx]['feather'], cumulative_flow)

            # Blend forward and backward based on position
            for i in range(1, seg_len):
                t = i / seg_len
                frame_idx = start_idx + i
                hair_all[frame_idx] = cv2.addWeighted(fwd_hair[i], 1 - t, bwd_hair[i], t, 0)
                alpha_all[frame_idx] = fwd_alpha[i] * (1 - t) + bwd_alpha[i] * t

        jobs[job_id]["message"] = "Color matching..."
        jobs[job_id]["progress"] = 70

        # Brightness matching using first keyframe
        first_key = keyframe_indices[0]
        mask_region = keyframe_data[first_key]['mask'] > 127
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
        dilated = cv2.dilate(keyframe_data[first_key]['mask'], kernel_dilate)
        surround_region = (dilated > 127) & (keyframe_data[first_key]['mask'] < 127)

        hair_brightness = np.mean(keyframe_data[first_key]['hair'][mask_region])
        surround_brightness = np.mean(frames[first_key][surround_region])
        darken_factor = (surround_brightness / hair_brightness) * 0.72 if hair_brightness > 0 else 1.0
        darken_factor = min(darken_factor, 1.0)

        for i in range(n):
            if hair_all[i] is not None:
                hair_all[i] = (hair_all[i].astype(np.float32) * darken_factor).clip(0, 255).astype(np.uint8)

        jobs[job_id]["message"] = "Compositing..."
        jobs[job_id]["progress"] = 80

        # Composite
        ALPHA_LO, ALPHA_HI = 40, 240
        output_frames = []

        for i in range(n):
            if hair_all[i] is None:
                output_frames.append(frames[i])
                continue

            wc = hair_all[i]
            wa = alpha_all[i].copy()

            wa = cv2.medianBlur(wa.astype(np.uint8), 3).astype(np.float32)
            wa = np.clip(wa, 0, ALPHA_HI)
            wa = np.where(wa < ALPHA_LO, 0, wa)

            af = wa / 255.0
            m3 = np.stack([af] * 3, axis=-1)
            result = np.clip(wc.astype(np.float32) * m3 + frames[i].astype(np.float32) * (1 - m3), 0, 255).astype(np.uint8)

            # Gentle sharpen in mask region (reduced from 1.5 to 1.15)
            blurred = cv2.GaussianBlur(result, (0, 0), 2)
            sharpened = cv2.addWeighted(result, 1.15, blurred, -0.15, 0)
            sharp_mask = (af > 0.1)[:, :, None].astype(np.float32)
            result = np.clip(sharpened * sharp_mask + result * (1 - sharp_mask), 0, 255).astype(np.uint8)

            output_frames.append(result)

        jobs[job_id]["message"] = "Encoding video..."
        jobs[job_id]["progress"] = 90

        # Save video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        tmp = job_dir / "tmp.mp4"
        vw = cv2.VideoWriter(str(tmp), fourcc, fps, (WORK_WIDTH, WORK_HEIGHT))
        for f in output_frames:
            vw.write(f)
        vw.release()

        result_path = job_dir / "result.mp4"
        subprocess.run(["ffmpeg", "-y", "-i", str(tmp), "-c:v", "libx264", "-crf", "18",
                        "-preset", "fast", "-pix_fmt", "yuv420p", str(result_path)], capture_output=True)
        tmp.unlink()

        jobs[job_id]["status"] = STATUS_COMPLETE
        jobs[job_id]["message"] = "Complete!"
        jobs[job_id]["progress"] = 100
        jobs[job_id]["result_url"] = f"/jobs/{job_id}/result"

    except Exception as e:
        jobs[job_id]["status"] = STATUS_FAILED
        jobs[job_id]["message"] = str(e)


# ============================================================================
# API ENDPOINTS
# ============================================================================
@app.get("/")
def root():
    return {"service": "Hair Restoration API", "status": "running"}


@app.get("/health")
def health():
    comfy_ok = check_comfyui()
    return {
        "api": "ok",
        "comfyui": "ok" if comfy_ok else "unavailable",
        "comfyui_url": COMFYUI_URL
    }


@app.post("/jobs", response_model=JobStatus)
async def create_job(video: UploadFile = File(...), multi: bool = False):
    """Upload a video and create a new job."""
    job_id = str(uuid.uuid4())[:8]
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # Save uploaded video
    video_path = job_dir / "input.mp4"
    with open(video_path, "wb") as f:
        content = await video.read()
        f.write(content)

    # Extract frames for mask drawing
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if multi:
        # Multi-frame mode: time-based keyframes (~1 per second, min 2, max 10)
        duration_seconds = total_frames / fps if fps > 0 else 5
        num_keyframes = max(2, min(10, int(duration_seconds)))  # 1 per second, capped at 10

        # Generate evenly spaced keyframes including start and end
        if num_keyframes == 2:
            keyframe_indices = [0, total_frames - 1]
        else:
            step = (total_frames - 1) / (num_keyframes - 1)
            keyframe_indices = [int(i * step) for i in range(num_keyframes)]
        keyframe_indices = sorted(set(keyframe_indices))  # Remove duplicates

        # Extract all keyframes
        for idx in keyframe_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (WORK_WIDTH, WORK_HEIGHT))
                cv2.imwrite(str(job_dir / f"frame_{idx}.png"), frame)
        cap.release()

        # Track job
        jobs[job_id] = {
            "job_id": job_id,
            "status": STATUS_PENDING,
            "message": f"Ready for mask drawing (frame 1/{len(keyframe_indices)})",
            "frame_count": total_frames,
            "fps": fps,
            "multi_frame": True,
            "keyframe_indices": keyframe_indices,
            "current_keyframe": 0,
            "masks_collected": {},
            "progress": 0
        }
    else:
        # Single-frame mode (original behavior)
        ref_idx = total_frames // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, ref_idx)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            shutil.rmtree(job_dir)
            raise HTTPException(status_code=400, detail="Could not read video")

        frame = cv2.resize(frame, (WORK_WIDTH, WORK_HEIGHT))
        frame_path = job_dir / "frame.png"
        cv2.imwrite(str(frame_path), frame)

        jobs[job_id] = {
            "job_id": job_id,
            "status": STATUS_PENDING,
            "message": "Ready for mask drawing",
            "frame_count": total_frames,
            "fps": fps,
            "ref_idx": ref_idx,
            "multi_frame": False,
            "progress": 0
        }

    return JobStatus(**jobs[job_id])


@app.get("/jobs/{job_id}", response_model=JobStatus)
def get_job(job_id: str):
    """Get job status."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatus(**jobs[job_id])


@app.get("/jobs/{job_id}/frame")
def get_frame(job_id: str, index: Optional[int] = None):
    """Get the frame image for mask drawing."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    job_dir = JOBS_DIR / job_id

    if job.get("multi_frame"):
        # Multi-frame mode: get specific keyframe
        if index is None:
            index = job.get("current_keyframe", 0)
        keyframe_indices = job.get("keyframe_indices", [])
        if index >= len(keyframe_indices):
            raise HTTPException(status_code=400, detail="Invalid keyframe index")
        frame_idx = keyframe_indices[index]
        frame_path = job_dir / f"frame_{frame_idx}.png"
    else:
        # Single-frame mode
        frame_path = job_dir / "frame.png"

    if not frame_path.exists():
        raise HTTPException(status_code=404, detail="Frame not found")

    return FileResponse(frame_path, media_type="image/png")


@app.post("/jobs/{job_id}/mask", response_model=JobStatus)
async def submit_mask(job_id: str, mask: MaskSubmission, background_tasks: BackgroundTasks):
    """Submit the mask (with red markings) and start processing."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]

    if job["status"] != STATUS_PENDING:
        raise HTTPException(status_code=400, detail="Job already processing or complete")

    job_dir = JOBS_DIR / job_id
    video_path = job_dir / "input.mp4"

    # Decode base64 mask image
    try:
        mask_data = base64.b64decode(mask.mask_data)
        mask_array = np.frombuffer(mask_data, np.uint8)
        mask_img_raw = cv2.imdecode(mask_array, cv2.IMREAD_COLOR)

        if mask_img_raw is None:
            raise ValueError("Could not decode mask image")

        # Resize to working resolution to match frame
        mask_img_raw = cv2.resize(mask_img_raw, (WORK_WIDTH, WORK_HEIGHT))

        # Extract red mask from the marked image
        mask_img = extract_red_mask_from_array(mask_img_raw)

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid mask data: {str(e)}")

    if job.get("multi_frame"):
        # Multi-frame mode: collect masks
        keyframe_index = mask.keyframe_index if mask.keyframe_index is not None else job.get("current_keyframe", 0)
        keyframe_indices = job.get("keyframe_indices", [])
        frame_idx = keyframe_indices[keyframe_index]

        # Save this mask
        cv2.imwrite(str(job_dir / f"mask_{frame_idx}.png"), mask_img)
        job["masks_collected"][frame_idx] = mask_img

        # Move to next keyframe
        next_keyframe = keyframe_index + 1
        job["current_keyframe"] = next_keyframe

        if next_keyframe >= len(keyframe_indices):
            # All masks collected - start multi-frame pipeline
            job["message"] = "All masks collected, starting processing..."
            background_tasks.add_task(run_multi_pipeline, job_id, job_dir, video_path, job["masks_collected"], keyframe_indices)
            job["status"] = STATUS_PROCESSING
        else:
            job["message"] = f"Mask {next_keyframe}/{len(keyframe_indices)} - draw on next frame"

        return JobStatus(**job)
    else:
        # Single-frame mode (original behavior)
        cv2.imwrite(str(job_dir / "received_mask.png"), mask_img)
        background_tasks.add_task(run_pipeline, job_id, job_dir, video_path, mask_img)

    jobs[job_id]["status"] = STATUS_PROCESSING
    jobs[job_id]["message"] = "Starting pipeline..."

    return JobStatus(**jobs[job_id])


def extract_red_mask_from_array(img):
    """Extract red markings from image array - use only the LARGEST red region."""
    h, w = img.shape[:2]
    b, g, r = cv2.split(img)

    # Detect all red pixels
    red_mask = ((r > 150) & (g < 100) & (b < 100)).astype(np.uint8) * 255

    if np.sum(red_mask > 0) < 100:
        raise ValueError("No red markings found")

    # Find contours and keep only the LARGEST one (user's drawing)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise ValueError("No red markings found")

    # Use the largest contour only
    largest_contour = max(contours, key=cv2.contourArea)

    clean_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(clean_mask, [largest_contour], -1, 255, -1)
    clean_mask = cv2.dilate(clean_mask, kernel, iterations=3)

    return clean_mask


@app.get("/jobs/{job_id}/result")
def get_result(job_id: str):
    """Download the result video."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    if jobs[job_id]["status"] != STATUS_COMPLETE:
        raise HTTPException(status_code=400, detail="Job not complete")

    result_path = JOBS_DIR / job_id / "result.mp4"
    if not result_path.exists():
        raise HTTPException(status_code=404, detail="Result not found")

    return FileResponse(result_path, media_type="video/mp4", filename=f"restored_{job_id}.mp4")


@app.delete("/jobs/{job_id}")
def delete_job(job_id: str):
    """Delete a job and its files."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job_dir = JOBS_DIR / job_id
    if job_dir.exists():
        shutil.rmtree(job_dir)

    del jobs[job_id]
    return {"message": "Job deleted"}


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    import uvicorn
    print("Starting Hair Restoration API Server...")
    print(f"ComfyUI URL: {COMFYUI_URL}")
    uvicorn.run(app, host="0.0.0.0", port=8000)
