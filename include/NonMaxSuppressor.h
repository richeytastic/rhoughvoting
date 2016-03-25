#pragma once
#ifndef HOUGHVOTING_NON_MAX_SUPPRESSOR_H
#define HOUGHVOTING_NON_MAX_SUPPRESSOR_H

#include <AdaptiveDepthPatchScanner.h>  // RFeatures::PatchRanger
#include <opencv2/opencv.hpp>
#include <vector>
using std::vector;


namespace HoughVoting
{

class NonMaxSuppressor
{
public:
    NonMaxSuppressor( const cv::Size2f& modelSz, const cv::Point2f& objOffset,
                      const cv::Mat_<float>& rngMap, const cv::Mat_<float>& voteMap);

    int calcSimpleBoxes( int maxBoxes, float threshVal, vector<cv::Rect>& bboxes, vector<float>& voteVals) const;

private:
    const cv::Size2f _modelSz;
    const cv::Point2f _objOffset;
    const cv::Mat_<float> _rngMap;
    const cv::Mat_<float> _voteMap;
};  // end class

}   // end namespace

#endif
