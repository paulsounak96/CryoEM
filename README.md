# CryoEM
Research on reconstruction from Cryo EM images. 1000 simulated cryo-EM images (51 X 51 pixels each) can be found at the following link:
https://drive.google.com/file/d/1T0WGdufvtBgyA3mXnXw4TMTkVS-yqYTb/view?usp=sharing .
To load the images in python, please import the packages numpy and pickle, and enter the following code:

with open('protein51x51_1000imgs.pkl', 'rb') as inp:\
&nbsp;&nbsp; q = pickle.load(inp)\
&nbsp;&nbsp; ground_state = pickle.load(inp)\
&nbsp;&nbsp; notfree = pickle.load(inp)\
&nbsp;&nbsp; atom_coord = pickle.load(inp)\
&nbsp;&nbsp; rotations = pickle.load(inp)\
&nbsp;&nbsp; cryoem_imgs = pickle.load(inp)
    
The variable 'cryoem_imgs' will be a list of length 1000, containing the images in a 51 X 51 numpy array format.
