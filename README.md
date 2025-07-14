# ⚽ Player Re-Identification and Tracking Demo
## 📄 Project Description

This project demonstrates a complete **player re-identification and tracking pipeline** applied to a short sports video clip. The goal is to identify and consistently track each player throughout the video, even when they temporarily leave the frame and reappear later. This type of solution simulates how modern sports analytics and broadcast systems track players in real time to gather statistics and enhance the viewing experience.

The system starts with a **YOLOv5 object detector**, which detects all players in every frame of the video. These detections provide bounding boxes that locate the players on the field. Next, the system uses a **tracker** that combines simple motion-based tracking with basic visual features (appearance descriptors) to maintain player identities across frames. The Hungarian Algorithm is applied for optimal matching of detected players with existing tracks, using a cost matrix that combines spatial overlap (IOU) and appearance similarity.

A unique ID is assigned to each new player, and the same ID is preserved when that player reappears later. The solution also visualizes the tracking process by drawing **bounding boxes**, displaying **player IDs**, and rendering each player’s **trajectory** as they move around the field.

As output, the system saves:
- ✅ An annotated video showing the real-time tracking results.
- ✅ A JSON file containing detailed detection and tracking data for every frame.
- ✅ A statistics file summarizing total frames processed, the number of unique players detected, total detections, and average processing speed.

The solution is modular and extensible. For example, the feature extractor can be upgraded to use deep embeddings for more robust re-identification. This would improve ID consistency when players change pose or appearance.

This project is fully self-contained and reproducible. It serves as a practical demonstration of real-time multi-object tracking and basic re-ID, bridging computer vision theory with a real-world sports use case.

---

## ⚙️ Setup & Run

1. Clone this repository:
   ```bash
   git clone https://github.com/gauRAVNegi00007/Player_Re-identification.git
2. Create a virtual environment and activate it:
     python3 -m venv player_reid_env
     source player_reid_env/bin/activate
3. Install dependencies:
   pip install -r requirements.txt
4. Run the demo:
   python3 demo_script.py input_video.mp4
