"""detect_species.py: 

    Given a template in PNG and library files in tiff, detect the template.

"""
    
__author__           = "Dilawar Singh"
__copyright__        = "Copyright 2017-, Dilawar Singh"
__version__          = "1.0.0"
__maintainer__       = "Dilawar Singh"
__email__            = "dilawars@ncbs.res.in"
__status__           = "Development"

import sys
import os
import glob
from libtiff import TIFF
import cv2
import numpy as np

FLANN_INDEX_KDTREE = 0
MIN_MATCH_COUNT = 20

def show_frame( frame, block = False ):
    cv2.imshow( 'Frame', frame )
    if not block:
        cv2.waitKey( 1 )
    else:
        cv2.waitKey( -1 )

def detectTemplate( template, library):
    template = cv2.imread( template, 0 )
    sift = cv2.xfeatures2d.SIFT_create( )
    kpTemp, desTemp = sift.detectAndCompute( template, None )

    print( 'Template is loaded ' )
    alltifs = glob.glob( os.path.join( library, '*.tif' ) )

    frames = [ ]
    for f in alltifs:
        fh = TIFF.open( f, mode="r")
        for frame in fh.iter_images( ):
            frames.append( frame )
        fh.close( )

    print( 'Total frames in library %d' % len( frames ) )
    for i, f in enumerate( frames ):
        kp, des = sift.detectAndCompute( f, None )
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(desTemp, des, k=2)

        # store all the good matches as per Lowe's ratio test.
        good = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)

        if len(good) > MIN_MATCH_COUNT:
            print( 'Found template. Good points %d' % len(good) )
            ## src_pts = np.float32([ kpTemp[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            ## dst_pts = np.float32([ kp[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

            ## M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            ## matchesMask = mask.ravel().tolist()
            ## h,w = template.shape
            ## pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)

            ## print( pts.shape )
            ## dst = cv2.perspectiveTransform(pts,M)
            ## f = cv2.polylines(f,[np.int32(dst)],True,255,3, cv2.LINE_AA)
            img3 = cv2.drawMatchesKnn(template, kpTemp, f, kp, good, flags=2)
            show_frame( img3, True )
        else:
            print( '.', end='' )
            sys.stdout.flush( )

def main( ):
    templateFile = sys.argv[1]
    libraryTiff = sys.argv[2]
    detectTemplate( templateFile, libraryTiff )

if __name__ == '__main__':
    main()
