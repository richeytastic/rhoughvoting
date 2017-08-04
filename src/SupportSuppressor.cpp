/************************************************************************
 * Copyright (C) 2017 Richard Palmer
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 ************************************************************************/

#include "SupportSuppressor.h"
using HoughVoting::SupportSuppressor;
using HoughVoting::Support; //struct { const float probability; const cv::Vec2i offset;}
#include <algorithm>
#include <iostream>
#include <cstdlib>
#include <cmath>


SupportSuppressor::SupportSuppressor( const ForestScanner* scanner)
    : _scanner(scanner)
{}   // end ctor


// Get the maximum symmetric bounds of the rectangle centred on focal pt
cv::Rect calcSupportBounds( const cv::Point& pt, const list<Support>& support)
{
    int hwidth = 1;
    int hheight = 1;
    BOOST_FOREACH ( const Support& s, support)
    {
        if ( s.probability < 1)
            continue;

        hwidth = std::max<int>( hwidth, abs(s.offset[0]));
        hheight = std::max<int>( hheight, abs(s.offset[1]));
    }   // end foreach
    return cv::Rect(pt.x - hwidth, pt.y - hheight, 2*hwidth, 2*hheight);
}   // end calcSupportBounds



// public
int SupportSuppressor::calcBoundingBoxes( int maxBoxes, float threshVal, vector<cv::Rect>& bboxes, vector<float>& voteVals) const
{
    if ( maxBoxes < 1)
        maxBoxes = 1;

    cv::Mat_<float> cvotes = _scanner->getResponseMap().clone();
    const cv::Rect imgRct(0,0, cvotes.cols, cvotes.rows);

    // Get the location of the maximum
    double mn, mx;
    cv::Point pt;
    cv::minMaxLoc( cvotes, &mn, &mx, NULL, &pt);

    cv::Rect objRct;
    int numBoxes = 0;
    while ( mx > threshVal && numBoxes < maxBoxes)
    {
        list<Support> support; // Get the vote support for this point
        _scanner->getSupport( pt.y, pt.x, support);

        // Get the minimum containing rectangle of the supporting votes
        cv::Rect objRct = calcSupportBounds( pt, support);
        objRct &= imgRct;   // Ensure contained in image

        bboxes.push_back( objRct);
        voteVals.push_back( (float)mx);
        numBoxes++;

        cv::rectangle( cvotes, objRct, 0, CV_FILLED); // Prevent intersecting votes from being found again
        cv::minMaxLoc( cvotes, &mn, &mx, NULL, &pt);    // Find next highest vote point
    }   // end while

    return numBoxes;
}   // end calcBoundingBoxes



struct ObjStats
{
    ObjStats( const cv::Point& cpt) : centrePt(cpt), minDepth(FLT_MAX), maxDepth(-FLT_MAX), meanDepth(0), bbox(cpt.x, cpt.y, 0, 0) {}

    const cv::Point centrePt;
    float minDepth;
    float maxDepth;
    float meanDepth;

    cv::Rect bbox;  // Object's bounding box
};  // end struct


// Get the valid support points (absolute location) for a detection at cpt and return object's mean depth
ObjStats rangeFilterObjectSupport( const cv::Point& cpt, const cv::Mat_<float>& rngMap,
                                   const list<Support>* support[], int numSupportLists,
                                   vector<cv::Point>& dpoints)
{
    vector<cv::Point> spts;  // Corresponding support points
    spts.push_back(cpt);

    cv::Mat_<float> kdata;  // For k-means clustering on depth
    kdata.push_back( rngMap.at<float>(cpt));    // Range at centre point

    int i = 0;
    for ( int j = 0; j < numSupportLists; ++j)
    {
        const list<Support>* supportList = support[j];
        BOOST_FOREACH ( const Support& s, *supportList)
        {
            const cv::Point pt( cpt.x + s.offset[0], cpt.y + s.offset[1]);
            const float srng = rngMap.at<float>(pt);   // Range at support point

            kdata.push_back(srng);
            spts.push_back(pt);
        }   // end foreach
    }   // end for

    const int N = kdata.rows;   // Number of samples

    // Do K-means on depth values
    const int K = 5;
    cv::Mat_<int> labels( N, 1);   // Cluster label per sample
    cv::Mat_<float> centers( K, 1);
    cv::kmeans( kdata, K, labels,
                cv::TermCriteria( cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 15, 0.1),
                5, // Number of attempts
                cv::KMEANS_PP_CENTERS,
                centers);

    // Find the largest cluster
    int kcounts[K] = {0,0,0};
    vector<int> cidxs[K];   // Indices of entries for the 3 clusters
    for ( i = 0; i < N; ++i)
    {
        const int label = *labels.ptr<int>(i);
        kcounts[label]++;
        cidxs[label].push_back(i);
    }   // end for

    // Find the most populated cluster
    int topBin = 0;
    int M = kcounts[0];
    for ( i = 1; i < K; ++i)
    {
        if ( kcounts[i] > M)
        {
            topBin = i;
            M = kcounts[i];
        }   // end if
    }   // end for

    // Only want the support vote locations from within this most populated bin.
    // Also record the min and max depths from this bin and while iterating
    // over the locations of the support votes, extract the bounding box dims.
    ObjStats ds( cpt);
    ds.meanDepth = *centers.ptr<float>(topBin);

    const vector<int>& topIdxs = cidxs[topBin];
    for ( i = 0; i < M; ++i)
    {
        const int idx = topIdxs[i];
        const float depth = *kdata.ptr<float>(idx);
        const cv::Point& p = spts[idx];

        // Store the support points for the object
        dpoints.push_back( p);

        ds.minDepth = std::min<float>( ds.minDepth, depth);
        ds.maxDepth = std::max<float>( ds.maxDepth, depth);

        // Update the object's bounding box from the support point
        ds.bbox.width = std::max<int>( ds.bbox.width, abs(p.x - ds.bbox.x));
        ds.bbox.height = std::max<int>( ds.bbox.height, abs(p.y - ds.bbox.y));
    }   // end for

    ds.bbox.x = ds.bbox.x - ds.bbox.width;
    ds.bbox.y = ds.bbox.y - ds.bbox.height;
    ds.bbox.width *= 2;
    ds.bbox.height *= 2;
    ds.bbox.width += 1;
    ds.bbox.height += 1;

    // Finally, ensure the object bounding box doesn't lie outside the image rectangle
    ds.bbox &= cv::Rect( 0, 0, rngMap.cols, rngMap.rows);

    return ds;
}   // end rangeFilterObjectSupport


const int MARK_UNKNOWN = 0;
const int MARK_BACKGROUND = 1;
const int MARK_OBJECT = 2;



// Create object markers from a bounding box and supporting object points
cv::Mat_<int> createMarkers( const cv::Rect& bbox, const vector<cv::Point>& objPoints, const cv::Mat_<float>& rngMap, float minRng, float maxRng)
{
    cv::Mat_<int> markers = cv::Mat_<int>::ones( rngMap.size()) * MARK_BACKGROUND;
    // Mark the areas having valid depth within the expanded bounding box as possibly being part of the object.
    for ( int i = bbox.y; i < bbox.y + bbox.height; ++i)
    {
        int* mrow = markers.ptr<int>(i);    // Output markers row
        const float* frow = rngMap.ptr<float>(i); // valid depth row

        for ( int j = bbox.x; j < bbox.x + bbox.width; ++j)
        {
            if ( frow[j] >= minRng && frow[j] <= maxRng)
                mrow[j] = MARK_UNKNOWN;    // Possibly part of the object
        }   // end for - cols
    }   // end for - rows

    // Plot the definite object points (use 2 as indicator for object)
    BOOST_FOREACH ( const cv::Point& p, objPoints)
        markers.at<int>(p) = MARK_OBJECT;

    return markers;
}   // end createMarkers



cv::Rect expandBoundingBox( const ObjStats& objStats, int rows, int cols)
{
    // The bounding box is initially given by the extent of its support votes.
    cv::Rect bbox = objStats.bbox;
    // For watershedding, we expand it to four times its size since some of the object may be even further out
    bbox.x -= bbox.width/4;
    bbox.width *= 1.5;
    bbox.y -= bbox.height/4;
    bbox.height *= 1.5;
    bbox &= cv::Rect(0,0,cols,rows); // Ensure newly expanded bounding box is contained in image
    return bbox;
}   // end expandBoundingBox



cv::Rect calcExtents( const cv::Mat_<byte>& mask, const cv::Point& cpt, const cv::Rect& bbox)
{
    // Find the extents of this mask to find the new bounding box
    int x1 = mask.cols, x2 = 0, y1 = mask.rows, y2 = 0;

    for ( int i = bbox.y; i < bbox.y + bbox.height; ++i)
    {
        const byte* mrow = mask.ptr<byte>(i);   // Mask row
        for ( int j = bbox.x; j < bbox.x + bbox.width; ++j)
        {
            if ( mrow[j])
            {
                if ( j < x1)
                    x1 = j;
                if ( j > x2)
                    x2 = j;
                if ( i < y1)
                    y1 = i;
                if ( i > y2)
                    y2 = i;
            }   // end if
        }   // end for - cols
    }   // end for - rows

    const int hw = std::min<int>( abs(cpt.x - x1), abs(cpt.x - x2));
    const int hh = std::min<int>( abs(cpt.y - y1), abs(cpt.y - y2));
    return cv::Rect( cpt.x - hw, cpt.y - hh, 2*hw+1, 2*hh+1);
}   // end calcExtents



// Given an object's stats (bounding box, depth info) and a list of supporting points,
// segment out that part of the depth map having similar depth that is also connected
// to one or more of the supporting points.
cv::Mat_<byte> segmentObject( ObjStats& objStats, const vector<cv::Point>& objPoints, const cv::Mat_<float>& rngMap, const cv::Mat_<cv::Vec3b>& cimg)
{
    const cv::Rect bbox = objStats.bbox;//expandBoundingBox( objStats, rngMap.rows, rngMap.cols);
    /*
    cv::Mat_<int> rmarkers = createMarkers( bbox, objPoints, rngMap, objStats.minDepth, objStats.maxDepth);
    cv::Mat_<int> cmarkers = rmarkers.clone();  // Markers for colour map watershed
    const cv::Mat_<cv::Vec3b> rmap = RFeatures::makeCV_8UC3( rngMap);
    cv::watershed( rmap, rmarkers);
    cv::watershed( cimg, cmarkers);

    // Object mask is where marker pixels == MARK_OBJECT on both rmarkers AND cmarkers
    const cv::Mat_<byte> mask = (rmarkers == MARK_OBJECT) & (cmarkers == MARK_OBJECT);
    //const cv::Mat_<byte> mask = rmarkers == MARK_OBJECT;
    //const cv::Mat_<byte> mask = cmarkers == MARK_OBJECT;

    objStats.bbox = calcExtents( mask, objStats.centrePt, bbox);   // Set new bounding box for object
    */
    const cv::Mat_<byte> mask = rngMap >= objStats.minDepth & rngMap <= objStats.maxDepth;

    return mask( objStats.bbox);
}   // end segmentObject



// public
int SupportSuppressor::calcRangeFilteredBoundingBoxes( int maxBoxes, float threshVal,
                                           const cv::Mat_<float>& rngMap, const cv::Mat_<cv::Vec3b>& colImg,
                                           vector<cv::Rect>& bboxes, vector<cv::Point>& dpoints,
                                           vector<cv::Mat_<byte> >& dmaps, vector<float>& voteVals) const
{
    if ( maxBoxes < 1)
        maxBoxes = 1;

    cv::Mat_<int> segImg = cv::Mat_<int>::zeros( rngMap.size());

    cv::Mat_<float> cvotes = _scanner->getResponseMap().clone();
    const cv::Rect imgRct(0,0, cvotes.cols, cvotes.rows);   // Convenience

    // Get the location of the first maximum
    double mn, mx;
    cv::Point cpt;
    cv::minMaxLoc( cvotes, &mn, &mx, NULL, &cpt);

    const int numTrees = _scanner->getNumTrees();
    const list<Support>* support[ numTrees]; // Holds the vote support lists for a point

    cv::Rect objRct;
    int numBoxes = 0;
    while ( mx > threshVal && numBoxes < maxBoxes)
    {
        const int numLists = _scanner->getSupport( cpt.y, cpt.x, support);
        if ( numLists < numTrees)   // All trees must vote or can't have been a significant enough point
            break;

        // Do K-means to keep only support points having similar depth
        vector<cv::Point> dpts;
        ObjStats objStats = rangeFilterObjectSupport( cpt, rngMap, support, numLists, dpts);
        if ( objStats.meanDepth <= 0)
            continue;

        dpoints.insert(dpoints.end(), dpts.begin(), dpts.end());
        voteVals.push_back( (float)mx);
        // Segment out the object and update the bounding box extents
        const cv::Mat_<byte> objMask = segmentObject( objStats, dpts, rngMap, colImg);
        bboxes.push_back( objStats.bbox);
        dmaps.push_back( objMask);

        numBoxes++;

        cv::rectangle( cvotes, objStats.bbox, 0, CV_FILLED); // Prevent intersecting votes from being found again
        cv::minMaxLoc( cvotes, &mn, &mx, NULL, &cpt);    // Find next highest vote point
    }   // end while

    return numBoxes;
}   // end calcRangeFilteredBoundingBoxes


