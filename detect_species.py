"""detect_species.py: 

    Given a template in PNG and library files in tiff, detect the template.

"""
from __future__ import print_function
    
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
MIN_MATCH_COUNT = 10

kpTemp_, desTemp_ = None, None
debug_ = False
templdir = "."
resdir = "_result"
current_f_index_ = 0

if not os.path.isdir( resdir ):
    os.makedirs( resdir )

def show_frame( frame, block = False ):
    cv2.imshow( 'Frame', frame )
    if not block:
        cv2.waitKey( 1 )
    else:
        cv2.waitKey( -1 )

def find_all_files( library ):
    tiffs = [ ]
    for d, sd, fs in os.walk( library ):
        for f in fs:
            ext = f.split( '.' )[-1]
            if ext in [ 'tiff', 'TIFF', 'tif', 'TIF' ]:
                tiffs.append( os.path.join( d, f ) )
    print( 'Found %d tiff files in library' % len( tiffs ) )
    return tiffs


def detectTemplate( templ_path, library):
    global kpTemp_, desTemp_ 
    global templdir
    global current_f_index_ 

    template = cv2.imread( templ_path, 0 )

    # Create template directory.
    tempName = os.path.basename( templ_path )
    templdir = os.path.join( resdir, tempName )
    if not os.path.isdir( templdir ):
        os.makedirs( templdir )

    sift = cv2.xfeatures2d.SIFT_create( sigma = 1.2, nOctaveLayers = 5 )
    #sift = cv2.xfeatures2d.SURF_create( )

    kpTemp_, desTemp_ = sift.detectAndCompute( template, None )
    assert desTemp_ is not None

    print( 'Template is loaded ' )
    alltifs = find_all_files( library ) 

    for f in alltifs:
        current_f_index_ += 1
        fh = TIFF.open( f, mode="r")
        for frame in fh.iter_images( ):
            # Make frame b&w
            frame = cv2.cvtColor( frame, cv2.COLOR_RGB2GRAY )
            searchForTemplate( sift, template, frame )
        fh.close( )

def searchForTemplate( sift, template, f ):
    global kpTemp_, desTemp_
    global templdir
    global current_f_index_

    f = cv2.bilateralFilter( f, 13, 5, 7 )
    kp, des = sift.detectAndCompute( f, None )
    bf = cv2.BFMatcher( )
    matches = bf.knnMatch(desTemp_, des, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) > MIN_MATCH_COUNT:
        print( 'x', end='')
        sys.stdout.flush( )
        framePath = os.path.join( templdir, 'frame_%04d.png' % current_f_index_ )
        newF = np.zeros_like( f )
        newF = cv2.drawMatches( template, kpTemp_, f, kp, good, newF )
        cv2.imwrite( framePath, newF )
    else:
        print( '.', end='' )
        sys.stdout.flush( )


def main( ):
    templateFile = sys.argv[1]
    libraryTiff = sys.argv[2]
    detectTemplate( templateFile, libraryTiff )

if __name__ == '__main__':
    main()
