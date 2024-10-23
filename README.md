<h1>🖊️ Name Image Classification System</h1>

<p>This project generates handwritten and digital images of names and classifies them using a trained model. The system uses a combination of image datasets and a neural network model to categorize names with high accuracy and provides a user-friendly GUI for testing and logging results.</p>

<h2>📂 Key Files Overview</h2>

<ul>
  <li><b>dataset_generator.py</b>, <b>dataset_generator2.py</b>, and <b>dataset_generator3.py</b>: <br/>
    These scripts generate images for each class of names: Emily, Michael, Sofia, and Jacob. The generated images vary in handwriting style, background, and visual artifacts such as smudges and ink blots. These will not run as I have removed the open ai API key for obvious security reasons. however, the dataset is already sufficient.
    <ul>
      <li><b>dataset_generator.py:</b> Generates 192 images each run, including potential misspellings and other variations.</li>
      <li><b>dataset_generator2.py:</b> Generates 144 images per run, but without any misspellings.</li>
      <li><b>dataset_generator3.py:</b> Generates 8 digital images for each name class using two fonts.</li>
    </ul>
  </li>

  <li><b>name_classifier2.py</b>: <br/>
    This file builds and trains the name classification model using the generated image data. It produces a visual plot of accuracy and validation accuracy after the final test. The best performing model is saved as <b>best_model.keras</b> for later use in the GUI. You can delete the <b>output2</b> folder to retrain the model from scratch, but this will extend training time.
  </li>

  <li><b>name_gui.py</b>: <br/>
    This is the GUI for the letter notification system, which uses <b>best_model.keras</b> for name classification. You can select a date, click "Check", and the system will classify 1 to 10 randomly selected images from the dataset. The classified names are logged in <b>Letter_Notification.log</b> under the chosen date along with the current time. The model achieves approximately 90% accuracy in classifying names.
  </li>
</ul>

<h2>Installation</h2> 

To set up and run this project on your local machine, follow the steps below:

### 1. Clone the Repository

Copy the repository to a convenient location on your computer:

```bash
git clone <repository-url>
cd <repository-directory>
```

### 2. Create a Virtual Environment

Inside the cloned repository, create a virtual environment using the following command:

```bash
python -m venv venv
```

### 3. Activate the Virtual Environment

Activate the virtual environment using:

  ```bash
  .\venv\Scripts\activate
  ```

### 4. Install Dependencies

After activating the virtual environment, install the necessary dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```
<h2>👨‍💻 Steps to Run the Project</h2>

<ol>
  <li><b>Dataset Generation:</b> If you wish to expand the dataset,  you will need to add your own open ai API key to run any of the dataset_generator scripts, though it's unnecessary as the current dataset is adequate.</li>
  <li><b>Model Training:</b> Run <b>name_classifier2.py</b> to train the model. Delete the <b>output2</b> folder to start fresh if needed.</li>
  <li><b>GUI Testing:</b> Run <b>name_gui.py</b> to test the classification via the letter notification system. The results will be logged automatically.</li>
</ol>

<h2>📝 Example Use Cases</h2>

<ul>
  <li>A mailroom assistant can use the GUI to quickly classify handwritten letters based on the names they are addressed to, providing a streamlined workflow for managing received mail.</li>
  <li>Researchers can expand on this model to classify additional names or develop similar systems for other types of handwritten text recognition.</li>
</ul>

<h2>🤳 Connect with me:</h2>

<a href="https://linkedin.com/in/yourprofile"><img align="left" alt="LinkedIn" width="22px" src="https://cdn.jsdelivr.net/npm/simple-icons@v3/icons/linkedin.svg" /></a> 
