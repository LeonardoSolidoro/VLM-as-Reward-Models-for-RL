import os
import json
import shutil
import cv2

def restructure_maniskill3():
    data_root = os.path.join(os.path.dirname(__file__), "..", "data", "maniskill3")
    
    if not os.path.exists(data_root):
        print(f"Directory {data_root} does not exist.")
        return

    tasks = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
    
    for task in tasks:
        print(f"Processing task: {task}")
        expert_dir = os.path.join(data_root, task, "expert")
        
        if not os.path.exists(expert_dir):
            continue
            
        trajectories = [d for d in os.listdir(expert_dir) if d.startswith("traj_")]
        
        for traj in trajectories:
            traj_path = os.path.join(expert_dir, traj)
            
            # Extract number from traj_XXX
            try:
                traj_num = int(traj.split("_")[1])
            except ValueError:
                continue
                
            rollout_name = f"rollout_{traj_num}"
            rollout_path = os.path.join(expert_dir, rollout_name)
            
            # Rename directory if not already renamed
            if traj_path != rollout_path:
                os.rename(traj_path, rollout_path)
                print(f"  Renamed {traj} -> {rollout_name}")
            
            # Process frames
            frames_dir = os.path.join(rollout_path, "frames")
            
            # Check if rewards.json already exists to avoid duplicate work if script is re-run
            if os.path.exists(os.path.join(rollout_path, "rewards.json")):
                continue
                
            num_frames = 0
            metadata_path = os.path.join(rollout_path, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    meta = json.load(f)
                    num_frames = meta.get("num_sampled_frames", 20)
            else:
                num_frames = 20 # Fallback
                
            # Create dummy rewards.json for prepare_in_context.py length check
            dummy_rewards = [{"score": 0.0} for _ in range(num_frames)]
            with open(os.path.join(rollout_path, "rewards.json"), "w") as f:
                json.dump(dummy_rewards, f)
                
            if os.path.exists(frames_dir):
                png_files = [f for f in os.listdir(frames_dir) if f.endswith(".png")]
                for png_file in png_files:
                    try:
                        # frame_0000.png
                        frame_num = int(png_file.replace("frame_", "").replace(".png", ""))
                        
                        src_img = os.path.join(frames_dir, png_file)
                        dst_img = os.path.join(rollout_path, f"topview_frame_{frame_num:03d}.jpg")
                        
                        # Convert PNG to JPG
                        img = cv2.imread(src_img)
                        if img is not None:
                            cv2.imwrite(dst_img, img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                    except Exception as e:
                        print(f"  Error processing {png_file}: {e}")
                
                # Delete the old frames directory
                shutil.rmtree(frames_dir)
                print(f"  Converted frames and removed frames/ dir for {rollout_name}")

if __name__ == "__main__":
    restructure_maniskill3()
    print("Maniskill3 restructuring complete!")
