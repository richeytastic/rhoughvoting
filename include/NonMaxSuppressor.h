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
