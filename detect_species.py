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
MIN_MATCH_COUNT = 13

resdir = "_result"
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
    template = cv2.imread( templ_path, 0 )
    # Create template directory.
    tempName = os.path.basename( templ_path )
    templdir = os.path.join( resdir, tempName )
    if not os.path.isdir( templdir ):
        os.makedirs( templdir )

    sift = cv2.SIFT( )
    kpTemp, desTemp = sift.detectAndCompute( template, None )

    print( 'Template is loaded ' )
    alltifs = find_all_files( library ) 

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
            # draw matches.
            print( 'x', end='')
            sys.stdout.flush( )
            framePath = os.path.join( templdir, 'frame_%04d.png' % i )
            cv2.imwrite( framePath, f )
        else:
            print( '.', end='' )
            sys.stdout.flush( )

def main( ):
    templateFile = sys.argv[1]
    libraryTiff = sys.argv[2]
    detectTemplate( templateFile, libraryTiff )

if __name__ == '__main__':
    main()
