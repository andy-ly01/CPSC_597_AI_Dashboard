# CPSC 597: Final Project AI Sentiment & Trust Dashboard

Andy Ly andy_ly01@csu.fullerton.edu

Installation Instructions
1.	Install Python
•	Go to the official Python website: https://www.python.org/downloads/
•	Download Python 3.10 or newer
•	During Installation check the box “Add Python to PATH” and install now

2.	Download Project Files
•	Download the AI Sentiment and Trust Dashboard .zip file
•	Right click the .zip file and select “Extract All” to unzip
•	Open the extracted folder

3.	Open a Terminal (Command Prompt)
•	Inside the unzipped project folder hold Shift and right click
•	Click “Open in Terminal”

4.	Create and Activate a Virtual Environment
•	In the terminal type in the command “python -m venv venv” then hit enter
•	Then activate it by entering “venv\Scripts\activate”
•	Once activated, users should see (venv) at the beginning of the command line to know they are working within the virtual environment

 
5.	Install Required Libraries and Packages
•	Inside the virtual environment run the command “pip install -r requirements.txt”
 
•	OPTIONAL: Use a GPU (If available)
o	Make sure GPU drivers are up to date
o	Install PyTorch 12.8 with CUDA support: https://pytorch.org/get-started/locally/
o	The Dashboard will automatically detect and use your GPU

6.	Run the Dashboard
•	To launch the dashboard, type the command “streamlit run dashboard.py” and hit enter
•	A browser tab should open automatically with the Dashboard running (if not it should give a clickable URL in the terminal
