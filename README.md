🖼️ Image Classification with Grad-CAM

This project is a Streamlit-based web application for image classification using a pre-trained PyTorch model.
Additionally, the app generates Grad-CAM visualizations to highlight important regions in the image that contributed to the model's decision.

🚀 Features

-📂 Upload an image (.jpg, .jpeg, .png).

-🔍 Perform image classification with a trained PyTorch model.

-🌈 Generate Grad-CAM heatmaps for model explainability.

-📊 Display predictions with confidence scores.

-🖥️ Simple and interactive Streamlit UI.


📂 Project Structure
├── app.py               # Main Streamlit application
├── model.pth            # Trained PyTorch model (ResNet, DenseNet, etc.)
├── requirements.txt     # Required dependencies
└── README.md            # Project documentation


🖼️ Example Output

✅ Prediction of the uploaded image.

🌈 Grad-CAM heatmap overlayed on the image for interpretability.



🧠 Model

-The model is loaded from a saved .pth checkpoint.

-Compatible with ResNet, DenseNet, and other torchvision models.

-The final layer is adjusted for your number of classes.



📌 Notes

-Make sure your model.pth file matches the architecture defined in gui_final_project.py.

If you face BidirectionalMap errors, ensure the model was saved/loaded correctly with torch.save(model.state_dict()) and model.load_state_dict().

Grad-CAM is attached to the last convolutional layer of the model (changeable in code).


