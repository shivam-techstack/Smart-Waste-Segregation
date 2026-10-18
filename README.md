# ♻️ Smart Waste Segregation Using YOLO

An AI-powered waste detection and segregation project that uses **YOLO object detection and computer vision** to identify waste items in images and recommend suitable waste bins.

The project includes a Streamlit web application, model training and evaluation scripts, a Jupyter notebook, and rule-based bin recommendations.

## 📌 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Technologies Used](#-technologies-used)
- [Waste Categories](#-waste-categories)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Run the Application](#-run-the-application)
- [How It Works](#-how-it-works)
- [Model Training and Evaluation](#-model-training-and-evaluation)
- [Dataset](#-dataset)
- [Future Improvements](#-future-improvements)
- [Limitations](#-limitations)
- [Contributing](#-contributing)
- [License](#-license)

## 📖 Overview

Improper waste disposal is a major environmental challenge. Separating waste into appropriate categories can help improve recycling, waste management, and environmental sustainability.

**Smart Waste Segregation** uses a YOLO-based object detection model to analyze an uploaded image, detect waste objects, display the predicted classes and confidence scores, and suggest suitable bins using predefined rules.

The application provides a simple interface built with Streamlit, making it easy to test waste detection without requiring a separate frontend application.

## ✨ Features

- **Waste Object Detection:** Detect waste objects in uploaded images using a trained YOLO model.
- **Image Upload:** Upload JPG, JPEG, or PNG images for analysis.
- **Visual Detection Results:** View the original image alongside the annotated detection image.
- **Confidence Threshold:** Adjust the detection threshold to control which predictions are displayed.
- **Detection Details:** View detected classes, confidence scores, bounding-box coordinates, and suggested bins.
- **Class-Wise Summary:** Display the number of detected objects in each category.
- **Waste Bin Recommendations:** Get rule-based bin suggestions for detected waste categories.
- **Session History:** View previous image analyses during the current application session.
- **Model Training:** Use the provided training script to experiment with model training and fine-tuning.
- **Model Evaluation:** Use the evaluation script to assess model performance.
- **Jupyter Notebook:** Explore the project workflow through `Waste_Detection.ipynb`.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| YOLO / Ultralytics | Object detection |
| PyTorch | Deep learning framework used by the model |
| Streamlit | Interactive web application |
| OpenCV and image-processing tools | Image processing and visualization |
| Pandas | Organizing detection results into tables |
| Pillow (PIL) | Loading and processing uploaded images |
| Jupyter Notebook | Experimentation and analysis |

## ♻️ Waste Categories

The application is configured to detect the following classes:

| Category | Description |
|---|---|
| Glass | Glass-based waste items |
| Metal | Metal waste items |
| Paper | Paper-based waste |
| Plastic | Plastic waste items |
| Waste | Other or general waste represented by the model's `Waste` class |

The actual detection capability depends on the model's training data and learned classes.

**Note:** Bin recommendations are generated using predefined rules in `bins.py`. They are separate from the YOLO model's predictions and should be treated as guidance rather than a guarantee of correct disposal.

## 📂 Project Structure

```text
Smart-Waste-Segregation/
│
├── Image/                  # Project images or image assets
├── data/
│   └── waste_detection/    # Dataset configuration and related files
│
├── Waste_Detection.ipynb   # Jupyter notebook
├── app.py                  # Streamlit application
├── bins.py                 # Waste bin recommendation rules
├── train.py                # Model training script
├── evaluate.py             # Model evaluation script
├── requirements            # Python dependency list
├── best_model.pt           # Model weights used by the application
├── best_finetuned.pt       # Fine-tuned model weights, if applicable
├── LICENSE                 # Project license
├── README.md               # Project documentation
└── .gitignore              # Files excluded from Git
```

Some generated training outputs, caches, and dataset archives may be excluded from the repository. The structure above describes the main project files; your local folder may contain additional files.

## 🚀 Installation

### Prerequisites

- Python installed on your system
- Git
- A compatible environment for the project's Python dependencies
- The trained model weights required by the application

### 1. Clone the repository

```bash
git clone https://github.com/shivam-techstack/Smart-Waste-Segregation.git
```

### 2. Navigate to the project directory

```bash
cd Smart-Waste-Segregation
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

If you use Command Prompt instead:

```cmd
.venv\Scripts\activate.bat
```

### 4. Install dependencies

The dependency file in this repository is named `requirements`, without a `.txt` extension.

Run:

```bash
python -m pip install -r requirements
```

If installation fails, check the Python version and the package versions specified in the dependency file.

## ▶️ Run the Application

Start the Streamlit application from the project directory:

```bash
streamlit run app.py
```

Streamlit will display a local URL in your terminal, usually:

```text
http://localhost:8501
```

Open that URL in your browser.

### Using the application

1. Start the application.
2. Upload a waste image in JPG, JPEG, or PNG format.
3. Adjust the confidence threshold if required.
4. Click **Detect Waste**.
5. Review the original and annotated images.
6. Inspect the detected objects, confidence scores, and bounding boxes.
7. Review the class-wise counts and suggested waste bins.

**Important:** The application loads `best_model.pt` by default. Make sure this file is available in the project directory, or configure the `WASTE_MODEL` environment variable to point to compatible model weights.

## ⚙️ How It Works

The project follows this general workflow:

```text
Input Waste Image
        |
        v
Image Loading and Validation
        |
        v
YOLO Object Detection Model
        |
        v
Predicted Classes and Bounding Boxes
        |
        v
Detection Results and Class Counts
        |
        v
Rule-Based Waste Bin Recommendations
        |
        v
Display Results in Streamlit
```

### Step 1: Image Input

The user uploads an image through the Streamlit interface.

### Step 2: Object Detection

The trained YOLO model processes the image and predicts the locations and classes of recognizable waste objects.

### Step 3: Detection Results

The application displays bounding boxes, predicted categories, confidence scores, and a summary of detected objects.

### Step 4: Bin Recommendation

The application uses the rules defined in `bins.py` to recommend a bin for each detected category.

### Step 5: Results Visualization

The original image, annotated image, detection table, class counts, and suggested disposal categories are displayed in the application.

## 🧠 Model Training and Evaluation

The repository includes scripts for training and evaluating the model.

### Training

The training script is:

```text
train.py
```

Before training, verify that the dataset is available and that the dataset configuration points to the correct image and label directories.

Review the training script to identify its supported arguments and settings before running it.

### Evaluation

The evaluation script is:

```text
evaluate.py
```

Use it to assess the trained model against the configured evaluation dataset. Check the script for the supported command-line arguments and evaluation metrics.

### Jupyter Notebook

Open `Waste_Detection.ipynb` in Jupyter Notebook or VS Code to explore the notebook-based workflow and experiments.

**Reproducibility note:** Training results depend on the dataset, model weights, configuration, hardware, and training parameters. No specific accuracy or performance value is claimed here.

## 📊 Dataset

The project uses a waste detection dataset configured through the files in `data/waste_detection/`.

Dataset archives such as `data.zip` and `dataset.zip` may be too large to store directly in a standard GitHub repository.

If these files are not included in the repository:

1. Obtain the dataset from its original source or your own local backup.
2. Extract it into the expected directory structure.
3. Check the dataset configuration file, such as `data.yaml`.
4. Confirm that image and label paths are valid before training or evaluation.
5. Follow the dataset's applicable license and usage terms.

Do not assume that cloning the repository automatically downloads the dataset.

## 🔮 Future Improvements

Potential improvements include:

- Expanding the dataset with more waste categories and real-world images.
- Improving detection performance on cluttered backgrounds and overlapping objects.
- Evaluating model performance across different lighting and environmental conditions.
- Adding webcam or real-time video detection.
- Improving the waste classification and disposal recommendation rules.
- Supporting recyclable, organic, hazardous, and electronic waste where suitable training data is available.
- Deploying the application to a suitable hosting platform.
- Adding more detailed performance reports and visual analytics.
- Exploring model optimization for edge devices and waste collection systems.

These are proposed enhancements and may not be implemented in the current version.

## ⚠️ Limitations

- Detection quality depends on the training dataset and model weights.
- Objects outside the model's learned categories may be misclassified or missed.
- Confidence scores do not guarantee that a prediction is correct.
- Bin recommendations are rule-based and may not account for local waste-disposal regulations.
- Training requires the appropriate dataset, configuration, dependencies, and computational resources.
- Model weights and datasets may need to be obtained separately if they are not distributed with the source code.

## 🤝 Contributing

Contributions and suggestions are welcome.

1. Fork the repository.
2. Create a new branch for your changes.
3. Make and test your modifications.
4. Commit your changes with a descriptive message.
5. Open a pull request describing your improvements.

Please ensure that large datasets, generated files, private credentials, and unnecessary model artifacts are not committed accidentally.

## 📄 License

This project includes an MIT License file. Refer to [`LICENSE`](LICENSE) for the applicable terms.

## 👨‍💻 Author

**Shivam Kumar**

GitHub: [@shivam-techstack](https://github.com/shivam-techstack)

## ⭐ Support

If you find this project useful for learning about computer vision, YOLO, and sustainable waste management, consider giving the repository a star on GitHub.

**Repository:** [Smart-Waste-Segregation](https://github.com/shivam-techstack/Smart-Waste-Segregation)

