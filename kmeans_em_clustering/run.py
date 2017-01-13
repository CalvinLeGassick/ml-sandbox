import os

from cluster import runKMeans, runKernelMeans, runGMM

inputFileOne = "hw5_blob.csv"
inputFileTwo = "hw5_circle.csv"
imageDirectory = "images"

def createImageDirectory(imageDirectory):
    if not os.path.exists(imageDirectory):
        os.makedirs(imageDirectory)

def main():
    createImageDirectory(imageDirectory)
    for k in [2, 3, 5]:
        runKMeans(inputFileOne, imageDirectory, k)
        runKMeans(inputFileTwo, imageDirectory, k)
    runKernelMeans(inputFileTwo, imageDirectory, k=2)

    runGMM(inputFileOne, imageDirectory, k=3)

if __name__ == "__main__":
    main()