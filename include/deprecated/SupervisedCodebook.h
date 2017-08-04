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
#ifndef HOUGHPROHOG_SUPERVISED_CODEBOOK_H
#define HOUGHPROHOG_SUPERVISED_CODEBOOK_H

#include <FeatureExtractor.h>
using RFeatures::FeatureExtract;
using RFeatures::ObjectVote;
#include <boost/unordered_map.hpp>
using boost::unordered_map;
#include <boost/foreach.hpp>
#include <vector>
using std::vector;


namespace HoughVoting
{


class SupervisedCodebook
{
public:
    SupervisedCodebook( const unordered_set<FeatureExtract*> *extracts);

    // Returns a single offset vote for a given descriptor
    float lookupVote( int classId, const cv::Mat& fv, cv::Vec2f& offset) const;

    // Returns all offsets from the foreground extracts for a given descriptor
    float lookupVotes( int classId, const cv::Mat& fv, const vector<cv::Vec2f>** offsets) const;

    // Returns a max of numOffsets
    float lookupVotes( int classId, const cv::Mat& fv, int numOffsets, vector<cv::Vec2f>& offsets) const;

private:
    float pF_;              // Prior prob of foreground object P(F=1) where F is one of {0,1}
    vector<float> pQi_F_;   // P(Qi|F=1)
    vector<float> pQi_;     // P(Qi)

    vector<cv::Mat_<float> > meanFVs_;
    vector<cv::Vec2f> offsets_;
    vector< vector<cv::Vec2f> > moffsets_;

    int findClosestCluster( const cv::Mat_<float>&, double* ssd=NULL) const;

    float calcProb( int, const cv::Mat&, int&) const;

    cv::Vec2f calcMeans( int clusterId, const vector<const FeatureExtract*>&, cv::Mat_<float>&);
};  // end class

}   // end namespace

#endif
