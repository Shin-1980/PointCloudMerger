import cv2
import csv
import sys
import numpy as np
from matplotlib import pyplot as plt
from typing import List, Tuple
from XYZgenerator import XYZgenerator
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R

def mse(params, model1, model2):
    translation = params[:3]
    rotation = params[3:]
    
    rotation_matrix = R.from_euler('xyz', rotation).as_matrix()
    rotated_model1 = np.dot(model1, rotation_matrix.T)
    
    translated_model1 = rotated_model1 + translation    
    squared_diff = np.sum((translated_model1 - model2)**2, axis=1)
    
    mean_squared_error = np.mean(squared_diff)
    
    return mean_squared_error

def calTranslatedXYZ(fileToRead1: str, fileToRead2: str):

    model1 = []
    model2 = []

    try:
        with open(fileToRead1, 'r') as fr1:
            header = next(fr1)        
            for line in fr1:
                s1,s2,s3,s4,s5,s6 = line.split(",") 
                x = float(s1)
                y = float(s2)
                z = float(s3)
                xyz = [x,y,z]
                model2.append(xyz)
    except FileNotFoundError:
        print(f"Error: {fileToRead1} not found.")
        return

    try:
        with open(fileToRead2, 'r') as fr2:
            header = next(fr2)        
            for line in fr2:
                s1,s2,s3,s4,s5,s6 = line.split(",") 
                x = float(s1)
                y = float(s2)
                z = float(s3)
                xyz = [x,y,z]
                model1.append(xyz)
    except FileNotFoundError:
        print(f"Error: {fileToRead2} not found.")
        return

    initial_params = np.zeros(6)
    result = minimize(mse, initial_params, args=(model1, model2))

    optimal_params = result.x
    optimal_translation = optimal_params[:3]
    optimal_rotation = optimal_params[3:]
    rotation_matrix = R.from_euler('xyz', optimal_rotation).as_matrix()

    print("Optimal translation:", optimal_translation)
    print("Optimal rotation (degrees):", optimal_rotation)

    fileToRead3 = fileToRead2.replace('_matchXYZ.csv','.xyz')


    try:
        with open(fileToRead3, 'r') as fr3:
            print(f"Read {fileToRead3}")

            try: 
                with open(fileToRead3.replace('.xyz','translated.xyz'), 'w') as fw:
                    header = next(fr3)

                    for line in fr3:
                        s1,s2,s3,s4,s5,s6 = line.split(" ") 
                        x = float(s1)
                        y = float(s2)
                        z = float(s3)
                        r = int(s4)
                        g = int(s5)
                        b = int(s6)

                        xyz = [x,y,z]
                        model = []
                        model.append(xyz)

                        rotated_model = np.dot(model, rotation_matrix.T)    
                        translated_model = rotated_model + optimal_translation
                        text = f"{translated_model[0][0]} {translated_model[0][1]} {translated_model[0][2]} {r} {g} {b}\n"
                        fw.write(text)

            except FileNotFoundError:
                print(f"Error: {fileToRead3.replace('.xyz','translated.xyz')} not found.")
                return

    except FileNotFoundError:
        print(f"Error: {fileToRead3} not found.")
        return

def extractFetureXYZ(path: str, fileToRead: str, fileToWrite: str):
    xyzGen = XYZgenerator()
    xyzGen.setConf()

    print("xyzGen.getAndConfirmFiles")
    if xyzGen.getAndConfirmFiles(path) and xyzGen.calcParam():
        xyzGen.extractXYZbasedOnCSV(fileToRead, fileToWrite)
    else:
        print(f"File cannot be opened")

    print("done")

def extractFetureXY(path1: str, path2: str, maxPoints: int, fileToWrite1: str, fileToWrite2: str):
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)

    if img1 is None:
        print(f"Error: Could not read {path1}")
        return

    if img2 is None:
        print(f"Error: Could not read {path2}")
        return

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    akaze = cv2.AKAZE_create()
    kp1, des1 = akaze.detectAndCompute(gray1, None)
    kp2, des2 = akaze.detectAndCompute(gray2, None)
    img1_akaze = cv2.drawKeypoints(gray1, kp1, None, flags=4)
    img2_akaze = cv2.drawKeypoints(gray2, kp2, None, flags=4)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:maxPoints], None, 
                            matchColor=(255, 0, 0), 
                            singlePointColor=(255, 0, 0), 
                            flags=2)

    print("the result of matching")
    pngFile = path2.replace('.jpeg', '_match.png')
    print(pngFile)
    plt.imsave(pngFile, img3)

    try:
        with open(fileToWrite1, 'w') as fw1:
            text = "x, y\n" 
            fw1.write(text)

            for i in range(maxPoints):
                text = str(int(kp1[matches[i].queryIdx].pt[0])) + ','
                text = text + str(int(kp1[matches[i].queryIdx].pt[1])) 
                text = text + "\n"
                fw1.write(text)

    except FileNotFoundError:
        print(f"cannot open {fileToWrite1}")

    try:
        with open(fileToWrite2, 'w') as fw2:
            fw2 = open(fileToWrite2, 'w')
            text = "x, y\n" 
            fw2.write(text)

            for i in range(maxPoints):
                text = str(int(kp2[matches[i].trainIdx].pt[0])) + ','
                text = text + str(int(kp2[matches[i].trainIdx].pt[1])) 
                text = text + "\n"
                fw2.write(text)

    except FileNotFoundError:
        print(f"cannot open {fileToWrite2}")


def main():
    path1 = sys.argv[1]
    path2 = sys.argv[2]
    maxPoints = int(sys.argv[3]) 

    print("files to read:")
    print(path1)
    print(path2)

    print("files to write")
    fileToWriteXY1 = path1.replace('.jpeg','_matchXY.csv')
    fileToWriteXY2 = path2.replace('.jpeg','_matchXY.csv')
    print(fileToWriteXY1)    
    print(fileToWriteXY2)

    extractFetureXY(path1, path2, maxPoints, fileToWriteXY1, fileToWriteXY2)

    print("extractFetureXYZ")
    fileToWriteXYZ1 = fileToWriteXY1.replace('_matchXY.csv','_matchXYZ.csv')        
    fileToWriteXYZ2 = fileToWriteXY2.replace('_matchXY.csv','_matchXYZ.csv')        

    extractFetureXYZ(path1, fileToWriteXY1, fileToWriteXYZ1)
    extractFetureXYZ(path2, fileToWriteXY2, fileToWriteXYZ2)

    calTranslatedXYZ(fileToWriteXYZ1, fileToWriteXYZ2)

if __name__ == "__main__":
    main()