#include "NonMaxSuppressor.h"
using HoughVoting::NonMaxSuppressor;
#include <cassert>
#include <cmath>


NonMaxSuppressor::NonMaxSuppressor( const cv::Size2f& modSz, const cv::Point2f& oset,
                                    const cv::Mat_<float>& rngMap, const cv::Mat_<float>& vMap)
    : _modelSz(modSz), _objOffset(oset), _rngMap(rngMap), _voteMap(vMap)
{
    assert( rngMap.size() == vMap.size());
}   // end ctor



// public
int NonMaxSuppressor::calcSimpleBoxes( int maxBoxes, float threshVal, vector<cv::Rect>& bboxes, vector<float>& voteVals) const
{
    if ( maxBoxes < 1)
        maxBoxes = 1;

    cv::Mat_<float> cvotes = _voteMap.clone();  // Will modify to suppress values
    const cv::Rect imgRct(0,0, cvotes.cols, cvotes.rows);

    double mn, mx;
    cv::Point pt;
    cv::minMaxLoc( cvotes, &mn, &mx, NULL, &pt);

    RFeatures::PatchRanger patchRanger( _rngMap);
    cv::Rect objRct;
    int numBoxes = 0;
    while ( mx > threshVal && numBoxes < maxBoxes)
    {
        patchRanger.calcPatchRect( pt, _modelSz, objRct);   // Scaled rectangle centred over pt
        cv::rectangle( cvotes, (objRct & imgRct), 0, CV_FILLED); // Prevent intersecting votes from being found again

        // Translate according to object offset
        objRct.x += (0.5 - _objOffset.x) * objRct.width;
        objRct.y += (0.5 - _objOffset.y) * objRct.height;
        objRct &= imgRct;   // Ensure contained in image

        voteVals.push_back( (float)mx);

        numBoxes++;
        bboxes.push_back( objRct);

        cv::minMaxLoc( cvotes, &mn, &mx, NULL, &pt);  // Find next box
    }   // end while

    return numBoxes;
}   // end calcSimpleBoxes


