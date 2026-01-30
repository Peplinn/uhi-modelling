# Setup Python Environment

Create a virtual Python environment and install the necessary packages from the `uhi_requriements.txt` file.

```bash
python -m venv myenv 
source myenv/bin/activate # On Windows: myenv\Scripts\activate
pip install -r uhi_requirements.txt
```
# Authenticate Google Earth Engine

This project uses the Google Earth Engine Python API. You will need to authenticate and initialize it.
Run the following cell in the `0873107-uhi-project.ipynb` notebook:
```Python
import ee
ee.Authenticate()
ee.Initialize()
```
You may be prompted to sign in to a Google account. Simply follow the browser prompts.  
If this fails in Jupyter notebook, try running `ee.Authenticate()` in a standalone Python script.

# You are good to go!

After authenticating, you can run the rest of the code cells in the `0873107-uhi-project.ipynb` notebook to replicate my workflow.