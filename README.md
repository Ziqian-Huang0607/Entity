# Entity - Real-Time Threat Analysis Engine

A military-grade software prototype for real-time threat detection from aerial video feeds. This system uses a fine-tuned YOLOv8 model, persistent object tracking, and a multi-layered behavioral analysis engine to identify and prioritize potential security risks.

![Demo GIF](demo.gif)

---

## About The Project

The "Entity" project was developed as a proof-of-concept for an intelligent surveillance system that moves beyond simple object detection. Instead of just identifying *what* an object is, it analyzes its *behavior over time* to determine its intent.

The system is built on a clean, modular architecture:
*   **`realtime_analyzer.py` (The "Eyes"):** A high-performance visual perception layer that handles video processing, object detection, and the live GUI.
*   **`threat_algorithm.py` (The "Brain"):** A self-contained analytical core that tracks objects, scores their behavior for anomalies, and matches them against pre-defined threat "playbooks" to calculate a final threat probability.

## Key Features

*   **Real-Time Detection:** Identifies vehicles and pedestrians in live or pre-recorded video.
*   **Persistent Object Tracking:** Assigns and maintains unique IDs for each detected entity.
*   **Behavioral Playbook Analysis:** Matches entity behavior against known threat patterns (e.g., VBIED drop-off).
*   **Probabilistic Threat Scoring:** Fuses evidence over time to generate a high-confidence threat assessment.
*   **Live GUI:** Provides a real-time command-and-control interface, highlighting alerts as they happen.

## Tech Stack

*   **Python 3**
*   **PyTorch**
*   **Ultralytics YOLOv8**
*   **OpenCV**
*   **Supervision**

---

## Getting Started

To get a local copy up and running, follow these steps.

### Prerequisites

*   Python 3.10+
*   Git

### Installation & Setup

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/your_username/your_repository_name.git
    cd your_repository_name
    ```

2.  **Create and activate a virtual environment:**
    ```sh
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required packages:**
    ```sh
    pip install ultralytics supervision opencv-python
    ```

4.  **Download the Pre-trained Model:**
    This project uses a YOLOv8 model fine-tuned on the VisDrone dataset.
    *   Go to the model's Hugging Face repository: **[Mahadih534/YoloV8-VisDrone](https://huggingface.co/Mahadih534/YoloV8-VisDrone)**
    *   Download the `best.pt` file.
    *   Place the downloaded file in the root of your project folder.
    *   **IMPORTANT:** Rename the file from `best.pt` to **`visdrone_pretrained.pt`**.

### Usage

1.  Ensure a test video (e.g., `test_parking_lot.mp4`) is in the project directory.
2.  Run the main analyzer application:
    ```sh
    python realtime_analyzer.py
    ```
3.  The GUI window will appear, showing the real-time analysis.
4.  Press **'q'** on your keyboard to quit the application.

---

## Model Credits

*   The fine-tuned VisDrone model used in this project was created by **Mahadi Hasan** and is available on Hugging Face. All credit for the model training and performance goes to the original author.

## License

Distributed under the MIT License.
