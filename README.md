

TASK - Create a process for turning documents into book orientation
- Using an ML model that will __determine the angle from -30 to 30 degrees__.
- Libraries: __pytorch__ and tf and any supporting libraries.
- Time constraint: the model should run for __<5 seconds__, but this is not a critical requirement.
- Document the process sufficiently so it's and understandable during the review.
- requirements.txt file, which lists all the libraries used in the process with their versions
- deadline __11.09.2023__

curl https://guillaumejaume.github.io/FUNSD/dataset.zip > funsd.zip

./venv/bin/python3.10 -m pip freeze > requirements.txt

Deskew plan
1. Research
   1. Lookup best practices for detecting skew angle
2. Dataset
   1. Find document dataset
      1. https://paperswithcode.com/dataset/funsd 
      2. Deskew with pytesseract
   2. Skew images and put skew angle in name
3. Preprocess
   1. Fast Fourier transform.
4. Train CNN
   1. regression
      1. determine layers
         1. convolution L - 1 layer (filter 3x3)
                learning features from the input image.
         2. pooling L - 1 layer (filter 3x3)
                reduce the spatial dimensions of the feature maps
         3. fully Connected L - 0 for now
                reduce the spatial dimensions of the feature maps
         4. output L - continuous range 0...1, where 0 -> -180; 1 -> 180
      2. test performance
      3. add layers 
      4. tune hyperparams
   2. *classification (range(-30, 30) -> 61 class)
   3. *?VAE
5. Download skewed dataset
6. Display skewed images in grid
7. 
8. 

9.  Create docker container to isolate application on system level
10. Create a virtual environment to isolate and specify Python dependencies and versions
