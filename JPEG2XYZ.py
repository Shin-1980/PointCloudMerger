import glob
import copy
import argparse
from XYZgenerator import XYZgenerator

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    # fmt: off
    parser.add_argument(
        "--input", default="input",
        help="image path or directory path which contains images to infer",
    )
    # fmt: on
    args = parser.parse_args()

    return args

class JPEG2XYZ:
                                        
    def execProg(self, folder_name: str):

        filenamesList = glob.glob(folder_name + '/*.jpeg')
        
        for pictFilename in filenamesList:
            print("execute {pictFilename}")

            xyzGen = XYZgenerator()
            xyzGen.setConf()

            if not xyzGen.getAndConfirmFiles(pictFilename) or not xyzGen.calcParam():
                continue

            xyzGen.generateXYZ()
                    
def main():
    args = parse_args()
    
    if len(args.input) > 0:
        folder_name = str(args.input)
    else:
        print("Error: The path is incorrect.")
        return

    pic2xyz = JPEG2XYZ()
    pic2xyz.execProg(folder_name)

    print("done")

if __name__ == "__main__":
    main()
