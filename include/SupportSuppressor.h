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
