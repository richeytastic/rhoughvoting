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

#include "KMeansCodebook.h"
using HoughVoting::KMeansCodebook;
#include <algorithm>
#include <cassert>
#include <iostream>
#include <iomanip>
#include <cmath>

typedef unsigned char byte;


KMeansCodebook::KMeansCodebook( const unordered_set<Extract*> *extracts, int k)
    : totExtracts_(extracts->size())
{
    // Copy Pro-HOG feature vectors to cv::Mat for cv::kmeans
    // and tally up the number of objects of each different class
    const int fvSz = (*extracts->begin())->fv.total();  // Columns per extract
    cv::Mat_<float> samples( 0, fvSz);
    BOOST_FOREACH ( const Extract *x, *extracts)
    {
        const int classId = x->objVote->classId;
        samples.push_back( x->fv);
        classCounts_.resize( std::max( classId+1, int(classCounts_.size())));
        classCounts_[classId]++;
    }   // end foreach

    const int numClasses = classCounts_.size();

    // Print the number of different object classes received
    std::cout << totExtracts_ << " total extracts with breakdown:" << std::endl;
    for ( int j = 0; j < numClasses; ++j)
        std::cout << "Class " << j << ": " << classCounts_[j] << std::endl;

    // Ensure k is within allowable range
    if ( k < 2)
        k = 2;
    if ( k > totExtracts_/2)
        k = totExtracts_/2;

    // Do the clustering
    std::cout << "Doing " << k << "-means clustering on all extracts...";
    std::cout.flush();
    cv::Mat_<int> labels( totExtracts_, 1);
    clusters_ = cv::Mat_<float>( k, fvSz);  // Cluster centres
    const int flags = cv::KMEANS_PP_CENTERS;
    const int nattempts = 10;
    const cv::TermCriteria tc( cv::TermCriteria::COUNT, nattempts, 0/*epsilon*/);
    compactness_ = cv::kmeans( samples, k, labels, tc, nattempts, flags, clusters_);
    std::cout << " done" << std::endl;

    for ( int i = 0; i < k; ++i)
    {
        codes_[i] = new VoteSet;
        clusterVotes_[i].resize( numClasses);
        clusterClassProbs_[i].resize( numClasses);
        for ( int j = 0; j < numClasses; ++j)
            clusterVotes_[i][j] = new VoteSet;
    }   // end for

    // Now we have all the cluster centres, we must index them
    // to associated object detection info (class IDs, angles, distances)
    int i = 0;
    BOOST_FOREACH ( const Extract *x, *extracts)
    {
        // Gather distances to every cluster j for example i
        vector<double> ssds( k);    // SSD to each cluster (index j)
        double totSSDs = 0;
        int clusterIdx = 0; // Will be cluster this extract belongs to
        for ( int j = 0; j < k; ++j)
        {
            const cv::Mat_<float> d = x->fv - clusters_.row(j);    // Distance of example to cluster mean
            ssds[j] = cv::norm( d, cv::NORM_L2);
            totSSDs += ssds[j];
            if ( ssds[j] < ssds[clusterIdx])
                clusterIdx = j;
        }   // end for
        x->objVote->cweight = 1. - ssds[clusterIdx]/totSSDs;
        //x->objVote->cweight = 1;

        //int clusterIdx = labels.at<int>(i++);
        const int classId = x->objVote->classId;
        // Set the mapping of the object reference to the cluster ID
        codes_[clusterIdx]->insert( x->objVote);
        clusterVotes_[clusterIdx][classId]->insert( x->objVote);
    }   // end foreach

    // Print class breakdown and calculate the per cluster P(X_i|F_j) = P(X_i)P(F_j|X_i)/P(F_j)
    // = |{F_j : F_j e X_i}| / |F_j|
    for ( int i = 0; i < k; ++i)
    {
        std::cout << "Cluster X" << i << ":";
        for ( int j = 0; j < numClasses; ++j)
        {
            clusterClassProbs_[i][j] = (double)(clusterVotes_[i][j]->size() + 1) / (classCounts_[j] + k);
            std::cout << " F" << j << " [" << clusterVotes_[i][j]->size() << "]";
            std::cout << " P(X" << i << "|F" << j << ") = " << clusterClassProbs_[i][j];
        }   // end for
        std::cout << "  TOTAL = " << codes_[i]->size() << std::endl;
    }   // end for
}   // end ctor



KMeansCodebook::~KMeansCodebook()
{
    const int ccnt = clusters_.rows;
    assert( ccnt == clusterClassProbs_.size());
    for ( int i = 0; i < ccnt; ++i)
    {
        delete codes_[i];
        const vector<VoteSet*>& classVotes = clusterVotes_[i];
        for ( int j = 0; j < classVotes.size(); ++j)
            delete classVotes[j];
    }   // end for
}   // end dtor



const HoughVoting::VoteSet* KMeansCodebook::lookup( int classId, const cv::Mat_<float> &fv, float &matchWeight) const
{
    assert( fv.rows == 1);
    assert( clusters_.cols == fv.cols);

    const int numClusters = clusters_.rows;

    // Calculate distance to each cluster for given fv
    int closestCluster = 0;
    double totSSD = 0;
    vector<double> ssds( numClusters);
    for ( int i = 0; i < numClusters; ++i)
    {
        const cv::Mat_<float> d = fv - clusters_.row(i);
        ssds[i] = cv::norm( d, cv::NORM_L2);  // SSD
        totSSD += ssds[i];
        if ( ssds[i] < ssds[closestCluster])
            closestCluster = i;
    }   // end for

    /*
    // Choose the best cluster based on products of these pseudo probability distances and the cluster weights
    double maxLogProb = -DBL_MAX;
    for ( int i = 0; i < numClusters; ++i)
    {
        ssds[i] = 1. - ssds[i]/totSSD;  // Pseudo probability based on distance to cluster

        const double logprob = log(ssds[i]) + log(clusterClassProbs_.at(i)[classId]);
        if ( logprob > maxLogProb)
        {
            maxLogProb = logprob;
            closestCluster = i;
        }   // end if
    }   // end for
    */

    matchWeight = float((1. - ssds[closestCluster]/totSSD) * (clusterClassProbs_.at(closestCluster)[classId]));
    const VoteSet* vs = clusterVotes_.at(closestCluster)[classId];
    //std::cout << "P(X_" << closestCluster << "|F_" << classId << ") = " << matchWeight << " (" << vs->size() << ")" << std::endl;
    return vs;
}   // end lookup

