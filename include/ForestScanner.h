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
#ifndef HOUGHVOTING_FOREST_SCANNER_H
#define HOUGHVOTING_FOREST_SCANNER_H

#include <AdaptiveDepthPatchScanner.h>
#include <FeatureExtractor.h>
using RFeatures::FeatureExtractor;
#include "RandomForest.h"
using HoughVoting::RandomForest;
#include "TreeScanner.h"
using HoughVoting::TreeScanner;
using HoughVoting::Support;

#include <list>
using std::list;
#include <iterator>
using std::iterator;
#include <boost/foreach.hpp>


namespace HoughVoting
{

class ForestScanner : private RFeatures::PatchProcessor
{
public:
    // fxs: test image pre-processed feature extractors
    ForestScanner( const RandomForest::Ptr forest, const vector<FeatureExtractor::Ptr>& fxs,
                   const cv::Mat_<float>& rngMap, const cv::Mat_<byte>& mask, bool useDepthWeighting=false);
    ~ForestScanner();

    // Depth adaptively scan image and return response map (patch sizes are scaled with depth).
    const cv::Mat_<float> scaleScan( cv::Size2f realPatchDims);

    // Scan the target image using a fixed pixel patch size and return the response map.
    const cv::Mat_<float> fixedScan( cv::Size2i pixelDims);

    const cv::Mat_<float> getResponseMap() const { return _responseMap;}

    inline int getNumTrees() const { return _forest->getNumTrees();}

    /*
    // Get the supporting votes (if any) for the given pixel. Returns
    // the number of supporting votes added to parameter list.
    int getSupport( int row, int col, list<Support>& support) const;

    // Get an array of pointers to Support object lists. One entry per
    // tree. Client should provide an array with enough room
    // for the number of trees in the forest. Since not every tree will have
    // vote support for the requested point, the function returns the number
    // of actual support lists set in the provided array.
    int getSupport( int row, int col, const list<Support>* support[]) const;

    // Find locations within bbox giving supporting votes (as absolute points) to any location within the inner
    // quadrant of bbox where the point being voted for is closer to the centre than the location of the voting
    // location in bbox.
    // Returns a matrix with non-zero values (255) denoting locations of votes for the inner quadrant of bbox.
    // Votes coming from locations outside of bbox are discounted.
    // The returned matrix must be translated to the correct x,y position of bbox.
    cv::Mat_<byte> findSupportingPoints( const cv::Rect& bbox, list<cv::Point>& supportingPoints) const;
    */

private:
    const RandomForest::Ptr _forest;
    const vector<FeatureExtractor::Ptr>& _fxs;
    const cv::Mat_<float>& _rngMap;
    const cv::Mat_<byte>& _mask;
    const cv::Rect _imgRct;
    const bool _useDepthWeighting;
    const cv::Rect _responseRect;
    cv::Size _minSamplingDims;  // For FXs

    cv::Mat_<float> _responseMap;
    virtual void process( const cv::Point&, float depth, const cv::Rect&);

    void addToResponseMap( const cv::Point& patchCentre, const cv::Rect& patchRect,
                           float prob, float depth, const PatchSet* pmatches);
};  // end class

}   // end namespace

#endif

