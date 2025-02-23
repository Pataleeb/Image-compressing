# Image-compressing
Image compressing using clustering

Input
• pixels: the input image representation. Each row contains one data point (pixel). For image dataset, it
contains 3 columns, each column corresponding to Red, Green, and Blue components. Each component
has an integer value between 0 and 255. \\

• k: the number of desired clusters.\\

Output

• class: cluster assignment of each data point in pixels. The assignment should be 1, 2, 3, etc. For
k = 5, for example, each cell of the class should be either 1, 2, 3, 4, or 5. The output should be a
column vector with size(pixels, 1) elements.
• centroid: With images, each centroid corresponds to the representative color of each cluster. 
