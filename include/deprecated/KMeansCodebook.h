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

/**
 * Stores a codebook of "mean" Pro-HOG features derived from example data.
 * Richard Palmer
 * March - May 2013
 */

#pragma once
#ifndef HOUGHVOTING_KMEANS_CODEBOOK_H
#define HOUGHVOTING_KMEANS_CODEBOOK_H

#include <FeatureExtractor.h>   // RFeatures
#include <boost/unordered_map.hpp>
using boost::unordered_map;
#include <boost/unordered_set.hpp>
using boost::unordered_set;
#include <vector>
using std::vector;


namespace HoughVoting
{

typedef RFeatures::ObjectVote::Ptr Vote;
typedef unordered_set<Vote> VoteSet;
typedef RFeatures::FeatureExtract Extract;


class KMeansCodebook
{
public:
    // Partition the extracts from all the examples into K representative codebook extries.
    // Clusters 10 times and uses the best clustering (minimises the compactness measure
    // from each clustering).
    KMeansCodebook( const unordered_set<Extract*> *extracts, int k=30);
    ~KMeansCodebook();

    double getCompactness() const { return compactness_;}

    // Given a descriptor vector (fv) of type CV_32FC1, get the VoteSet from the
    // corresponding codebook descriptor that best matches the provided descriptor.
    // The match weight is calculated as 1/(d+1) where d is the euclidean distance
    // to the nearest codebook entry. This gives a weight between 1 and 0 with values
    // closer to 1 denoting a better match. A pointer to the VoteSet is returned.
    const VoteSet* lookup( int classId, const cv::Mat_<float> &fv, float &matchWeight) const;

private:
    const int totExtracts_;     // Total number of extracts from constructor
    vector<int> classCounts_;   // Count of extracts of different classes (0 = background, 1 = foreground1 etc)

    unordered_map<int, VoteSet*> codes_;    // Each cluster's votes (VoteSet contains votes of all object classes)

    // A mapping to each cluster ID of sets of votes for the same object. That is,
    // the vector mapped to the the cluster contains votes where each element in the
    // vector contains votes for the same object class. Note that index 0 is reserved
    // for non-foreground objects (foreground class indices start at 1).
    unordered_map<int, vector<VoteSet*> > clusterVotes_;

    // P(X_i|F_j) probabilities
    unordered_map<int, vector<double> > clusterClassProbs_; // clusterClassProbs_[3][1] == P(X3|F1)

    cv::Mat_<float> clusters_;  // K rows map to codes_ indices (entries are CV_32FC1)
    double compactness_;

    double lookupProb( int j, int i, double pX_i) const;    // cluster i, class j
};  // end class

}   // end namespace

#endif
