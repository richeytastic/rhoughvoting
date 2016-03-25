#pragma once
#ifndef HOUGHVOTING_SUPPORT_SUPPRESSOR_H
#define HOUGHVOTING_SUPPORT_SUPPRESSOR_H

#include "ForestScanner.h"
using HoughVoting::ForestScanner;
#include "DepthSegmenter.h"
#include <FeatureUtils.h>   // RFeatures
#include <vector>
using std::vector;


namespace HoughVoting
{

class SupportSuppressor
{
public:
    explicit SupportSuppressor( const ForestScanner* scanner);

    int calcBoundingBoxes( int maxBoxes, float threshVal, vector<cv::Rect>& bboxes, vector<float>& voteVals) const;

    int calcRangeFilteredBoundingBoxes( int maxBoxes, float threshVal, const cv::Mat_<float>& rngMap, const cv::Mat_<cv::Vec3b>& colImg,
                                        vector<cv::Rect>& bboxes, vector<cv::Point>& dpoints,
                                        vector<cv::Mat_<byte> >& dmaps, vector<float>& voteVals) const;

private:
    const ForestScanner* _scanner;
};  // end class

}   // end namespace

#endif
