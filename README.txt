The following code is the minimal reference implementation of the signal processing presented in 10.1021/acs.analchem.3c00321. 
Please have a look at the branch `dev` for a more structured implementation as well as visualization scripts.

Developed by Henning Bonart, Florian Gebard, and Lukas Hecht.

Execute the code online in your browser:
1. Go to https://mybinder.org/v2/gh/bonh/sami-detection-itp/HEAD. After a short while JupyterLab will be started. 
2. In the top left corner is a blue button with a plus sign. Two symbols to the left is a button to upload files. Klick that button.
3. Upload the notebook ReferenceImplementation.ipynb from your computer and open it from the folder view on the left.
4. Klick on the upload button again. Upload a *.nd2 file of your choice. You can download the images from https://doi.org/10.48328/tudatalib-914.
5. Change the "inname" parameter in ReferenceImplementation.ipnyb to match the path and name of the *.nd2 file you just uploaded.
6. Execute the notebook by repeatedly clicking on the "play" button on the top. This might take a while because binder is not superfast.

Build a local docker and execute on your computer:
1. Use https://github.com/bonh/sami-detection-itp and jupyter-repo2docker. 
2. Follow the steps above to load the notebook and data into your local docker.

Execute code directly:
1. Use the environment.yml from the github repo above for your conda environment.
2. Obtain the notebook and images.
3. Execute.
