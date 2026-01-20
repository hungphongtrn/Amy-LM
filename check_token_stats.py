from datasets import load_from_disk
import numpy as np

def check_token_duration():
    ds = load_from_disk("data/Amy-LM-Dataset")
    item = ds[0]
    llm_times = np.array(item["llm_times"])
    
    # llm_times is (N, 2) where col 0 is start, col 1 is end
    # Note: explicit -1 might define end of sequence/padding, handle carefully
    
    durations = []
    valid_times = llm_times[llm_times[:, 1] != -1]
    
    durations = valid_times[:, 1] - valid_times[:, 0]
    
    print(f"Number of tokens: {len(durations)}")
    print(f"Average Duration: {np.mean(durations):.4f}s")
    print(f"Min Duration: {np.min(durations):.4f}s")
    print(f"Max Duration: {np.max(durations):.4f}s")
    
    # Compare with Mimi Frame Duration
    mimi_frame_duration = 1.0 / 12.5 # 0.08s
    print(f"Mimi Frame Duration: {mimi_frame_duration:.4f}s")
    
    avg_frames_per_token = np.mean(durations) / mimi_frame_duration
    print(f"Average Frames per Token: {avg_frames_per_token:.2f}")

if __name__ == "__main__":
    check_token_duration()
